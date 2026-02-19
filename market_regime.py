"""
Market Regime Engine — Piyasa Rejimi ve Rölatif Güç Analizi
============================================================
KATMAN 1: Makro rejim tespiti (BTC trend + BTC.D proxy + USDT.D proxy)
KATMAN 2: Rölatif güç sıralaması (her coin vs BTC)
KATMAN 3: Fırsat filtreleme (Top adaylar → ICT/QPA'ya gönder)

OKX Public API kullanır, API key gerekmez.
"""

import logging
import time
import numpy as np
from data_fetcher import data_fetcher
from config import INST_TYPE

logger = logging.getLogger("ICT-Bot.Regime")

# ─────────────────────────────────────────────────────
# REGIME CONFIG
# ─────────────────────────────────────────────────────
REGIME_CONFIG = {
    # BTC trend tespiti
    "btc_trend_fast_period": 8,      # Hızlı EMA (kısa vadeli yön)
    "btc_trend_slow_period": 21,     # Yavaş EMA (orta vadeli yön)
    "btc_trend_lookback": 6,         # Son N mumu kontrol et (momentum)

    # BTC.D proxy (BTC vs Altcoin performans farkı)
    "btc_d_lookback_candles": 16,    # Son N mum karşılaştırma (4H → 16 mum = ~2.5 gün)
    "btc_d_threshold": 1.5,          # BTC %1.5 daha iyi → dominans artıyor

    # USDT.D proxy (Hacim + fiyat yönü)
    "usdt_d_volume_drop_pct": -15,   # Ortalamadan %15 düşüş → para çıkıyor
    "usdt_d_volume_surge_pct": 20,   # Ortalamadan %20 artış → ilgi artıyor

    # Rölatif Güç
    "rs_periods": [4, 16, 48],       # 1h, 4h, 12h (15m mumları bazında)
    "rs_weights": [0.5, 0.3, 0.2],   # Kısa vade ağırlıklı
    "rs_min_candles": 50,            # Minimum mum sayısı

    # Fırsat filtreleme
    "max_long_candidates": 3,        # Max LONG aday
    "max_short_candidates": 3,       # Max SHORT aday
    "rs_long_threshold": 0.5,        # RS > 0.5 = güçlü (LONG aday)
    "rs_short_threshold": -0.5,      # RS < -0.5 = zayıf (SHORT aday)
    "min_volume_confirmation": 0.8,  # Hacim ortalamanın %80'i üstünde olmalı

    # Rejim geçerlilik
    "regime_cache_seconds": 120,     # Rejim tespiti 2dk cache
}

# Rejim tipleri
REGIME_RISK_ON = "RISK_ON"           # BTC↑ + Altlar güçlü → LONG fırsatları
REGIME_RISK_OFF = "RISK_OFF"         # BTC↓ + Altlar zayıf → SHORT fırsatları
REGIME_ALT_SEASON = "ALT_SEASON"     # Altlar BTC'den iyi → Altcoin LONG
REGIME_CAPITULATION = "CAPITULATION" # Her şey düşüyor, altlar çöküyor → Dikkatli SHORT
REGIME_NEUTRAL = "NEUTRAL"           # Belirsiz → Coin bazlı fırsat ara


class MarketRegime:
    """Piyasa rejimi tespit ve rölatif güç sıralama motoru"""

    def __init__(self):
        self._regime_cache = None
        self._regime_ts = 0
        self._rs_cache = {}
        self._rs_ts = 0

    @staticmethod
    def _btc_symbol():
        """Aktif enstrüman tipine göre BTC sembolü döndür"""
        return "BTC-USDT-SWAP" if INST_TYPE == "SWAP" else "BTC-USDT"

    @staticmethod
    def _is_btc(symbol):
        """Sembol BTC mi kontrol et"""
        return symbol in ("BTC-USDT", "BTC-USDT-SWAP")

    # ═════════════════════════════════════════════════
    # ANA FONKSİYON: Tam analiz döngüsü
    # ═════════════════════════════════════════════════
    def analyze_market(self, coin_list):
        """
        Tam piyasa analizi yap. scan_markets() her çalıştığında çağrılır.

        Returns:
            dict: {
                "regime": str,                # RISK_ON / RISK_OFF / ALT_SEASON / CAPITULATION / NEUTRAL
                "regime_details": dict,        # BTC trend, BTC.D, USDT.D detayları
                "btc_bias": "LONG" | "SHORT" | "NEUTRAL",
                "rs_rankings": list[dict],     # Tüm coinler RS skoru ile sıralı
                "long_candidates": list[str],  # LONG aday coinler (en güçlüler)
                "short_candidates": list[str], # SHORT aday coinler (en zayıflar)
                "filtered_coins": dict,        # {symbol: {"allowed_directions": ["LONG"], "rs_score": 2.1, ...}}
            }
        """
        now = time.time()
        cfg = REGIME_CONFIG

        # ── 1. BTC verilerini çek ──
        btc_symbol = self._btc_symbol()
        btc_4h = data_fetcher.get_candles(btc_symbol, "4H", 100)
        btc_1h = data_fetcher.get_candles(btc_symbol, "1H", 100)
        btc_15m = data_fetcher.get_candles(btc_symbol, "15m", 100)

        if btc_15m is None or len(btc_15m) < cfg["rs_min_candles"]:
            logger.warning("BTC verisi yetersiz, rejim tespiti yapılamıyor")
            return self._neutral_result(coin_list)

        # ── 2. BTC Trend Analizi ──
        btc_trend = self._analyze_btc_trend(btc_4h, btc_1h, btc_15m)

        # ── 3. BTC Dominans Proxy ──
        btc_d_signal = self._analyze_btc_dominance_proxy(btc_15m, coin_list)

        # ── 4. USDT.D Proxy (hacim bazlı para akışı) ──
        usdt_d_signal = self._analyze_usdt_flow_proxy(btc_15m, coin_list)

        # ── 5. Rejim Tespiti ──
        regime = self._determine_regime(btc_trend, btc_d_signal, usdt_d_signal)

        # ── 6. Rölatif Güç Hesaplama ──
        rs_rankings = self._calculate_all_relative_strength(btc_15m, coin_list)

        # ── 7. Fırsat Filtreleme ──
        long_candidates, short_candidates = self._filter_opportunities(
            rs_rankings, regime, btc_trend
        )

        # ── 8. Her coin için izin verilen yönleri belirle ──
        filtered_coins = self._build_filtered_map(
            coin_list, rs_rankings, long_candidates, short_candidates, regime
        )

        result = {
            "regime": regime,
            "regime_details": {
                "btc_trend": btc_trend,
                "btc_dominance": btc_d_signal,
                "usdt_flow": usdt_d_signal,
            },
            "btc_bias": btc_trend["bias"],
            "rs_rankings": rs_rankings,
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "filtered_coins": filtered_coins,
            "timestamp": time.time(),
        }

        self._regime_cache = result
        self._regime_ts = now

        # Logla
        n_long = len(long_candidates)
        n_short = len(short_candidates)
        logger.info(
            f"📊 Rejim: {regime} | BTC: {btc_trend['bias']} ({btc_trend['strength']}) | "
            f"BTC.D: {btc_d_signal['direction']} | Para Akışı: {usdt_d_signal['direction']} | "
            f"Fırsatlar: {n_long} LONG, {n_short} SHORT aday"
        )

        return result

    # ═════════════════════════════════════════════════
    # KATMAN 1: BTC TREND ANALİZİ
    # ═════════════════════════════════════════════════
    def _analyze_btc_trend(self, btc_4h, btc_1h, btc_15m):
        """
        BTC'nin multi-timeframe trend yönünü belirle.
        EMA kullanıyoruz ama bu BTC filtreleme için — coin sinyali için değil.
        """
        cfg = REGIME_CONFIG
        result = {"bias": "NEUTRAL", "strength": "WEAK", "momentum": 0, "change_pcts": {}}

        try:
            # 4H trend (ana yön)
            if btc_4h is not None and len(btc_4h) >= cfg["btc_trend_slow_period"] + 5:
                closes_4h = btc_4h["close"].values.astype(float)
                ema_fast_4h = self._ema(closes_4h, cfg["btc_trend_fast_period"])
                ema_slow_4h = self._ema(closes_4h, cfg["btc_trend_slow_period"])
                trend_4h = 1 if ema_fast_4h > ema_slow_4h else -1

                # 4H değişim yüzdesi
                period_4h = min(6, len(closes_4h) - 1)
                change_4h = ((closes_4h[-1] - closes_4h[-period_4h - 1]) / closes_4h[-period_4h - 1]) * 100
                result["change_pcts"]["4h"] = round(change_4h, 2)
            else:
                trend_4h = 0
                change_4h = 0

            # 1H trend (orta vade)
            if btc_1h is not None and len(btc_1h) >= cfg["btc_trend_slow_period"] + 5:
                closes_1h = btc_1h["close"].values.astype(float)
                ema_fast_1h = self._ema(closes_1h, cfg["btc_trend_fast_period"])
                ema_slow_1h = self._ema(closes_1h, cfg["btc_trend_slow_period"])
                trend_1h = 1 if ema_fast_1h > ema_slow_1h else -1

                change_1h = ((closes_1h[-1] - closes_1h[-5]) / closes_1h[-5]) * 100
                result["change_pcts"]["1h"] = round(change_1h, 2)
            else:
                trend_1h = 0
                change_1h = 0

            # 15m momentum (kısa vade)
            closes_15m = btc_15m["close"].values.astype(float)
            lookback = cfg["btc_trend_lookback"]
            momentum_pct = ((closes_15m[-1] - closes_15m[-lookback - 1]) / closes_15m[-lookback - 1]) * 100
            result["change_pcts"]["15m_momentum"] = round(momentum_pct, 2)
            result["momentum"] = momentum_pct

            # Ağırlıklı trend skoru: 4H en önemli
            trend_score = (trend_4h * 0.5) + (trend_1h * 0.3) + (np.sign(momentum_pct) * 0.2)

            if trend_score > 0.3:
                result["bias"] = "LONG"
                result["strength"] = "STRONG" if trend_score > 0.7 else "MODERATE"
            elif trend_score < -0.3:
                result["bias"] = "SHORT"
                result["strength"] = "STRONG" if trend_score < -0.7 else "MODERATE"
            else:
                result["bias"] = "NEUTRAL"
                result["strength"] = "WEAK"

            result["trend_score"] = round(trend_score, 2)

        except Exception as e:
            logger.error(f"BTC trend analiz hatası: {e}")

        return result

    # ═════════════════════════════════════════════════
    # KATMAN 1: BTC DOMİNANS PROXY
    # ═════════════════════════════════════════════════
    def _analyze_btc_dominance_proxy(self, btc_15m, coin_list):
        """
        BTC.D direkt çekilemez (OKX'te yok).
        PROXY: BTC performansı vs ortalama altcoin performansı
        BTC daha iyi → BTC.D yükseliyor (para BTC'ye akıyor)
        Altlar daha iyi → BTC.D düşüyor (para altlara akıyor)
        """
        cfg = REGIME_CONFIG
        lookback = cfg["btc_d_lookback_candles"]
        result = {"direction": "NEUTRAL", "spread": 0, "btc_change": 0, "alt_avg_change": 0}

        try:
            btc_closes = btc_15m["close"].values.astype(float)
            if len(btc_closes) < lookback + 1:
                return result

            btc_change = ((btc_closes[-1] - btc_closes[-lookback - 1]) / btc_closes[-lookback - 1]) * 100

            # Top 5-6 büyük altcoini karşılaştır (hız için hepsini değil)
            major_alts = [s for s in coin_list if s != self._btc_symbol()][:8]
            alt_changes = []

            for alt_symbol in major_alts:
                try:
                    alt_df = data_fetcher.get_candles(alt_symbol, "15m", lookback + 10)
                    if alt_df is not None and len(alt_df) >= lookback + 1:
                        alt_c = alt_df["close"].values.astype(float)
                        alt_chg = ((alt_c[-1] - alt_c[-lookback - 1]) / alt_c[-lookback - 1]) * 100
                        alt_changes.append(alt_chg)
                except Exception:
                    continue

            if not alt_changes:
                return result

            alt_avg = np.mean(alt_changes)
            spread = btc_change - alt_avg  # Pozitif → BTC daha iyi → BTC.D yükseliyor

            result["btc_change"] = round(btc_change, 2)
            result["alt_avg_change"] = round(alt_avg, 2)
            result["spread"] = round(spread, 2)

            threshold = cfg["btc_d_threshold"]
            if spread > threshold:
                result["direction"] = "RISING"  # BTC.D artıyor
            elif spread < -threshold:
                result["direction"] = "FALLING"  # BTC.D düşüyor (alt season sinyali)
            else:
                result["direction"] = "NEUTRAL"

        except Exception as e:
            logger.error(f"BTC.D proxy hatası: {e}")

        return result

    # ═════════════════════════════════════════════════
    # KATMAN 1: USDT AKIŞI PROXY
    # ═════════════════════════════════════════════════
    def _analyze_usdt_flow_proxy(self, btc_15m, coin_list):
        """
        USDT.D direkt çekilemez.
        PROXY: Toplam piyasa hacmi + fiyat yönü analizi
        - Hacim düşüyor + fiyatlar düşüyor → Para çıkıyor (USDT.D yükseliyor)
        - Hacim artıyor + fiyatlar çıkıyor → Para giriyor (USDT.D düşüyor)
        """
        result = {"direction": "NEUTRAL", "volume_change_pct": 0, "price_direction": "NEUTRAL"}

        try:
            # BTC hacim analizi (piyasa proxy'si)
            volumes = btc_15m["volume"].values.astype(float)
            closes = btc_15m["close"].values.astype(float)

            if len(volumes) < 20:
                return result

            # Son 4 mum hacmi vs son 20 mum ortalaması
            recent_vol = np.mean(volumes[-4:])
            avg_vol = np.mean(volumes[-20:])
            vol_change_pct = ((recent_vol - avg_vol) / avg_vol) * 100

            # Fiyat yönü (son 8 mum)
            price_change = ((closes[-1] - closes[-8]) / closes[-8]) * 100

            result["volume_change_pct"] = round(vol_change_pct, 1)

            if price_change > 0.3:
                result["price_direction"] = "UP"
            elif price_change < -0.3:
                result["price_direction"] = "DOWN"

            # Para akışı tespiti
            cfg = REGIME_CONFIG
            if vol_change_pct < cfg["usdt_d_volume_drop_pct"] and price_change < -0.2:
                # Hacim düşüyor + fiyat düşüyor → Likidite azalıyor → para çıkış
                result["direction"] = "OUTFLOW"  # USDT.D yükseliyor
            elif vol_change_pct > cfg["usdt_d_volume_surge_pct"] and price_change > 0.2:
                # Hacim artıyor + fiyat çıkıyor → Para giriyor
                result["direction"] = "INFLOW"  # USDT.D düşüyor
            elif vol_change_pct > cfg["usdt_d_volume_surge_pct"] and price_change < -0.5:
                # Hacim artıyor + fiyat düşüyor → Panik satışı
                result["direction"] = "PANIC_SELL"
            else:
                result["direction"] = "NEUTRAL"

        except Exception as e:
            logger.error(f"USDT flow proxy hatası: {e}")

        return result

    # ═════════════════════════════════════════════════
    # REJİM TESPİTİ
    # ═════════════════════════════════════════════════
    def _determine_regime(self, btc_trend, btc_d, usdt_flow):
        """
        3 sinyali birleştirip piyasa rejimini belirle.

        REJİMLER:
        RISK_ON       → Piyasa sağlıklı yükseliyor (BTC↑, para giriyor)
        ALT_SEASON    → Altcoinler BTC'den iyi (BTC.D↓, altlar güçlü)
        RISK_OFF      → Piyasa düşüşte (BTC↓, para çıkıyor)
        CAPITULATION  → Her şey çöküyor, panik (BTC↓↓, panik satış, BTC.D↑)
        NEUTRAL       → Belirsiz, karışık sinyaller
        """
        btc_bias = btc_trend["bias"]
        btc_strength = btc_trend["strength"]
        btc_d_dir = btc_d["direction"]
        flow_dir = usdt_flow["direction"]

        # KAPITÜLASYON: BTC düşüyor + panik satış + BTC.D yükseliyor
        if btc_bias == "SHORT" and flow_dir == "PANIC_SELL":
            return REGIME_CAPITULATION
        if btc_bias == "SHORT" and btc_strength == "STRONG" and btc_d_dir == "RISING":
            return REGIME_CAPITULATION

        # ALT SEASON: BTC.D düşüyor (altlar BTC'den iyi)
        if btc_d_dir == "FALLING" and btc_bias != "SHORT":
            return REGIME_ALT_SEASON
        if btc_d_dir == "FALLING" and btc_bias == "SHORT" and btc_strength == "WEAK":
            # BTC hafif düşüyor ama altlar direniyor → altseason başlangıcı olabilir
            return REGIME_ALT_SEASON

        # RISK ON: BTC yükseliyor + para giriyor
        if btc_bias == "LONG" and flow_dir in ("INFLOW", "NEUTRAL"):
            return REGIME_RISK_ON
        if btc_bias == "LONG" and btc_strength == "STRONG":
            return REGIME_RISK_ON

        # RISK OFF: BTC düşüyor + para çıkıyor
        if btc_bias == "SHORT" and flow_dir == "OUTFLOW":
            return REGIME_RISK_OFF
        if btc_bias == "SHORT" and btc_strength in ("STRONG", "MODERATE"):
            return REGIME_RISK_OFF

        # Belirsiz
        return REGIME_NEUTRAL

    # ═════════════════════════════════════════════════
    # KATMAN 2: RÖLATİF GÜÇ HESAPLAMA
    # ═════════════════════════════════════════════════
    def _calculate_all_relative_strength(self, btc_15m, coin_list):
        """
        Her coinin BTC'ye göre rölatif gücünü hesapla.
        RS > 0 → BTC'den güçlü
        RS < 0 → BTC'den zayıf
        """
        cfg = REGIME_CONFIG
        btc_closes = btc_15m["close"].values.astype(float)
        rankings = []

        for symbol in coin_list:
            if self._is_btc(symbol):
                continue

            try:
                coin_df = data_fetcher.get_candles(symbol, "15m", 100)
                if coin_df is None or len(coin_df) < cfg["rs_min_candles"]:
                    continue

                coin_closes = coin_df["close"].values.astype(float)
                coin_volumes = coin_df["volume"].values.astype(float)

                # Multi-period RS skoru
                rs_score = 0
                valid_periods = 0

                for period, weight in zip(cfg["rs_periods"], cfg["rs_weights"]):
                    if len(coin_closes) > period and len(btc_closes) > period:
                        coin_chg = ((coin_closes[-1] - coin_closes[-period - 1]) / coin_closes[-period - 1]) * 100
                        btc_chg = ((btc_closes[-1] - btc_closes[-period - 1]) / btc_closes[-period - 1]) * 100
                        rs_score += (coin_chg - btc_chg) * weight
                        valid_periods += 1

                if valid_periods == 0:
                    continue

                # Hacim doğrulaması
                recent_vol = float(np.asarray(coin_volumes[-4:], dtype=float).mean())
                avg_vol = float(np.asarray(coin_volumes[-20:], dtype=float).mean())
                vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0

                # Hacim artıyorsa RS güçlenir, düşüyorsa zayıflar
                vol_multiplier = 1.0
                if vol_ratio > 1.5:
                    vol_multiplier = 1.2  # Hacim patlıyor → sinyal güçlü
                elif vol_ratio < 0.5:
                    vol_multiplier = 0.7  # Hacim çok düşük → sinyal zayıf

                adjusted_rs = rs_score * vol_multiplier

                # Momentum hız farkı (son 4 mum, ~1 saat)
                if len(coin_closes) >= 5 and len(btc_closes) >= 5:
                    coin_mom = ((coin_closes[-1] - coin_closes[-5]) / coin_closes[-5]) * 100
                    btc_mom = ((btc_closes[-1] - btc_closes[-5]) / btc_closes[-5]) * 100
                    short_term_rs = coin_mom - btc_mom
                else:
                    short_term_rs = 0

                rankings.append({
                    "symbol": symbol,
                    "rs_score": round(adjusted_rs, 3),
                    "raw_rs": round(rs_score, 3),
                    "vol_ratio": round(vol_ratio, 2),
                    "vol_multiplier": vol_multiplier,
                    "short_term_rs": round(short_term_rs, 3),
                    "price_change_1h": round(
                        ((coin_closes[-1] - coin_closes[-5]) / coin_closes[-5]) * 100, 2
                    ) if len(coin_closes) >= 5 else 0,
                })

            except Exception as e:
                logger.debug(f"RS hesaplama hatası {symbol}: {e}")
                continue

        # RS skoruna göre sırala (en güçlü başta)
        rankings.sort(key=lambda x: x["rs_score"], reverse=True)
        return rankings

    # ═════════════════════════════════════════════════
    # KATMAN 3: FIRSAT FİLTRELEME
    # ═════════════════════════════════════════════════
    def _filter_opportunities(self, rs_rankings, regime, btc_trend):
        """
        Rejime göre LONG ve SHORT adaylarını belirle.
        Çorba yerine sadece gerçek fırsatları seç.
        """
        cfg = REGIME_CONFIG
        long_candidates = []
        short_candidates = []

        for coin in rs_rankings:
            symbol = coin["symbol"]
            rs = coin["rs_score"]
            vol = coin["vol_ratio"]

            # Hacim kontrolü — düşük hacimli coinleri atla
            if vol < cfg["min_volume_confirmation"]:
                continue

            # ── LONG adayları ──
            if rs > cfg["rs_long_threshold"]:
                if regime in (REGIME_RISK_ON, REGIME_ALT_SEASON, REGIME_NEUTRAL):
                    long_candidates.append(symbol)
                elif regime == REGIME_RISK_OFF and rs > 2.0:
                    # Risk-off'ta sadece ÇOK güçlü olanlar (BTC'ye rağmen yükselen)
                    long_candidates.append(symbol)
                # CAPITULATION'da LONG yok (çok riskli)

            # ── SHORT adayları ──
            if rs < cfg["rs_short_threshold"]:
                if regime in (REGIME_RISK_OFF, REGIME_CAPITULATION, REGIME_NEUTRAL):
                    short_candidates.append(symbol)
                elif regime == REGIME_RISK_ON and rs < -2.0:
                    # Risk-on'da sadece ÇOK zayıf olanlar (BTC çıkarken düşen)
                    short_candidates.append(symbol)
                # ALT_SEASON'da SHORT yok (altlar güçlü)

        # Limit uygula
        long_candidates = long_candidates[:cfg["max_long_candidates"]]
        short_candidates = short_candidates[-cfg["max_short_candidates"]:]  # En zayıflar

        return long_candidates, short_candidates

    # ═════════════════════════════════════════════════
    # FİLTRE HARİTASI OLUŞTUR
    # ═════════════════════════════════════════════════
    def _build_filtered_map(self, coin_list, rs_rankings, long_cands, short_cands, regime):
        """
        Her coin için izin verilen yönleri belirle.
        Stratejiler (ICT/QPA) bu haritaya bakarak sadece izinli yönde sinyal üretecek.
        """
        rs_lookup = {r["symbol"]: r for r in rs_rankings}
        filtered = {}

        for symbol in coin_list:
            if self._is_btc(symbol):
                continue

            rs_data = rs_lookup.get(symbol)
            rs_score = rs_data["rs_score"] if rs_data else 0

            allowed = []
            if symbol in long_cands:
                allowed.append("LONG")
            if symbol in short_cands:
                allowed.append("SHORT")

            # Eğer hiçbir listeye girmediyse → bu coinde sinyal yok
            # Ama NEUTRAL rejimde RS skoru ortalama ise yine de şans ver
            if not allowed and regime == REGIME_NEUTRAL:
                if abs(rs_score) < 0.3:
                    # Nötr coin, nötr rejim — her iki yöne de bakılabilir (düşük öncelik)
                    allowed = ["LONG", "SHORT"]

            filtered[symbol] = {
                "allowed_directions": allowed,
                "rs_score": rs_score,
                "rs_data": rs_data,
                "is_candidate": len(allowed) > 0,
            }

        return filtered

    # ═════════════════════════════════════════════════
    # CACHE'Lİ REJİM OKUMA
    # ═════════════════════════════════════════════════
    def get_cached_regime(self):
        """Son analiz sonucunu döndür (cache)"""
        return self._regime_cache

    def get_regime_summary(self):
        """UI için özet bilgi"""
        if not self._regime_cache:
            return {
                "regime": "UNKNOWN",
                "regime_label": "Veri Bekleniyor",
                "regime_emoji": "❓",
                "btc_bias": "UNKNOWN",
                "btc_details": {"bias": "UNKNOWN", "strength": "WEAK", "momentum": 0, "change_pcts": {}},
                "btc_dominance": {"direction": "UNKNOWN", "spread": 0, "btc_change": 0, "alt_avg_change": 0},
                "usdt_flow": {"direction": "UNKNOWN", "volume_change_pct": 0, "price_direction": "NEUTRAL"},
                "long_candidates": [],
                "short_candidates": [],
                "long_count": 0,
                "short_count": 0,
                "rs_rankings": [],
                "total_coins": 0,
                "timestamp": 0,
            }

        r = self._regime_cache
        return {
            "regime": r["regime"],
            "regime_label": self._regime_label(r["regime"]),
            "regime_emoji": self._regime_emoji(r["regime"]),
            "btc_bias": r["btc_bias"],
            "btc_details": r["regime_details"]["btc_trend"],
            "btc_dominance": r["regime_details"]["btc_dominance"],
            "usdt_flow": r["regime_details"]["usdt_flow"],
            "long_candidates": r["long_candidates"],
            "short_candidates": r["short_candidates"],
            "long_count": len(r["long_candidates"]),
            "short_count": len(r["short_candidates"]),
            "rs_rankings": r["rs_rankings"][:10],  # Top 10
            "total_coins": len(r["rs_rankings"]),
            "timestamp": r.get("timestamp", 0),
        }

    # ═════════════════════════════════════════════════
    # YARDIMCI FONKSİYONLAR
    # ═════════════════════════════════════════════════
    def _ema(self, data, period):
        """Exponential Moving Average — son değeri döndürür"""
        if len(data) < period:
            return data[-1]
        multiplier = 2 / (period + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val

    def _neutral_result(self, coin_list):
        """Veri yetersizse nötr sonuç döndür — tüm coinlere izin ver ama aday listesi boş"""
        filtered = {}
        for symbol in coin_list:
            if not self._is_btc(symbol):
                filtered[symbol] = {
                    "allowed_directions": ["LONG", "SHORT"],
                    "rs_score": 0,
                    "rs_data": None,
                    "is_candidate": False,
                }
        return {
            "regime": REGIME_NEUTRAL,
            "regime_details": {
                "btc_trend": {"bias": "NEUTRAL", "strength": "WEAK", "momentum": 0, "change_pcts": {}},
                "btc_dominance": {"direction": "NEUTRAL", "spread": 0, "btc_change": 0, "alt_avg_change": 0},
                "usdt_flow": {"direction": "NEUTRAL", "volume_change_pct": 0, "price_direction": "NEUTRAL"},
            },
            "btc_bias": "NEUTRAL",
            "rs_rankings": [],
            "long_candidates": [],
            "short_candidates": [],
            "filtered_coins": filtered,
            "timestamp": time.time(),
        }

    @staticmethod
    def _regime_label(regime):
        labels = {
            REGIME_RISK_ON: "Risk-On (Yükseliş)",
            REGIME_RISK_OFF: "Risk-Off (Düşüş)",
            REGIME_ALT_SEASON: "Alt Season",
            REGIME_CAPITULATION: "Kapitülasyon",
            REGIME_NEUTRAL: "Nötr",
        }
        return labels.get(regime, regime)

    @staticmethod
    def _regime_emoji(regime):
        emojis = {
            REGIME_RISK_ON: "🟢",
            REGIME_RISK_OFF: "🔴",
            REGIME_ALT_SEASON: "🚀",
            REGIME_CAPITULATION: "☠️",
            REGIME_NEUTRAL: "⚪",
        }
        return emojis.get(regime, "❓")


# Singleton
market_regime = MarketRegime()
