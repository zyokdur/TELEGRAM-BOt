# =====================================================
# ICT Trading Bot - Akıllı Para Strateji Motoru v2.0
# (Smart Money Concepts - Sequential Gate Protocol)
# =====================================================
#
# PROTOKOL — KATI SIRALI ICT MODELİ:
# ====================================
# Adım 1  HTF Bias (4H)          → HARD GATE
#          4 saatlik grafikteki BOS/CHoCH ile yön belirlenir.
#          BULLISH yapı → SADECE LONG,  BEARISH yapı → SADECE SHORT.
#
# Adım 2  Liquidity Sweep (15m)  → HARD GATE
#          LTF'de eski bir Swing High/Low seviyesinin
#          fitille temizlenip (wick beyond) geri kapanmasını bekle.
#          Bu "stop hunt / likidite avı" paternidir.
#
# Adım 3  Displacement + MSS     → HARD GATE
#          Sweep sonrası ters yöne güçlü hacimli mum (displacement)
#          ve Market Structure Shift (BOS veya CHoCH) tespit et.
#
# Adım 4  FVG Entry Zone         → GİRİŞ BELİRLEME
#          Displacement mumunun oluşturduğu Fair Value Gap
#          tespit edilir. Bu FVG "Giriş Bölgesi" olur.
#          Entry = FVG'nin CE (Consequent Encroachment = orta noktası).
#
# SL → Sweep yapısının invalidation noktası (yapısal seviye)
# TP → Karşı taraf likidite havuzu (Draw on Liquidity)
#
# RSI, MACD gibi retail indikatörler KULLANILMAZ.
# Tüm kararlar Price Action & Market Structure üzerine kuruludur.
# =====================================================

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
from config import ICT_PARAMS
from database import get_bot_param

logger = logging.getLogger("ICT-Bot.Strategy")


class ICTStrategy:
    """
    Akıllı Para (Smart Money) Strateji Motoru.
    Yukarıda anlatılan 4 adımlı katı sıralı protokolü uygular.
    """

    def __init__(self):
        self.params = self._load_params()

    def _load_params(self):
        """Veritabanından güncel parametreleri yükle, yoksa config varsayılanı kullan."""
        params = {}
        for key, default_val in ICT_PARAMS.items():
            db_val = get_bot_param(key)
            params[key] = db_val if db_val is not None else default_val
        return params

    def reload_params(self):
        """Parametreleri yeniden yükle (optimizer güncellemesi sonrası)."""
        self.params = self._load_params()

    # =================================================================
    #  BÖLÜM 1 — SESSION / KILLZONE
    # =================================================================

    def get_session_info(self):
        """
        ICT Killzone (oturum) bilgisi.
        Kurumsal aktivite belirli UTC saatlerinde yoğunlaşır:
          London Killzone  07-10 UTC  (yüksek volatilite, ana hareketler)
          NY Killzone      12-15 UTC  (yüksek volatilite, trend devamı)
          London Close     15-17 UTC  (geri çekilmeler)
          Asian Session    00-07 UTC  (düşük volatilite, likidite oluşumu)
          Off-Hours        17-00 UTC  (düşük volatilite)
        """
        now = datetime.now(timezone.utc)
        hour = now.hour

        if 7 <= hour < 10:
            return {"session": "LONDON_KILLZONE", "quality": 1.0, "label": "London Killzone"}
        elif 12 <= hour < 15:
            return {"session": "NY_KILLZONE", "quality": 1.0, "label": "NY Killzone"}
        elif 10 <= hour < 12:
            return {"session": "LONDON_NY_OVERLAP_PREP", "quality": 0.8, "label": "London-NY Geçiş"}
        elif 15 <= hour < 17:
            return {"session": "LONDON_CLOSE", "quality": 0.7, "label": "London Kapanış"}
        elif 0 <= hour < 7:
            return {"session": "ASIAN", "quality": 0.5, "label": "Asya Oturumu"}
        else:
            return {"session": "OFF_HOURS", "quality": 0.3, "label": "Düşük Aktivite"}

    # =================================================================
    #  BÖLÜM 2 — YATAY PİYASA TESPİTİ
    # =================================================================

    def detect_ranging_market(self, df, lookback=20):
        """
        Yatay (ranging) piyasayı tespit et.
        Range-bound piyasalarda ICT sinyalleri düşük kalitelidir.
        Efficiency ratio + range genişliği kontrolü uygular.
        Returns: True = ranging → sinyal üretme.
        """
        if len(df) < lookback:
            return False

        recent = df.tail(lookback)
        closes = recent["close"].values
        highs = recent["high"].values
        lows = recent["low"].values
        avg_price = np.mean(closes)

        # Net hareket / toplam hareket (efficiency ratio)
        net_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i - 1]) for i in range(1, len(closes)))
        efficiency = net_move / total_move if total_move > 0 else 0

        # Toplam high-low range genişliği
        total_range_pct = (np.max(highs) - np.min(lows)) / avg_price if avg_price > 0 else 0

        is_ranging = (efficiency < 0.15 and total_range_pct < 0.02) or efficiency < 0.10

        if is_ranging:
            logger.debug(f"  📊 Ranging market: eff={efficiency:.3f}, range={total_range_pct:.4f}")

        return is_ranging

    # =================================================================
    #  BÖLÜM 3 — SWING POINTS (Yapı Taşları)
    # =================================================================

    def find_swing_points(self, df, lookback=None):
        """
        Swing High ve Swing Low noktalarını tespit et.
        Swing High: lookback kadar sağ ve soldaki mumlardan yüksek olan tepe.
        Swing Low:  lookback kadar sağ ve soldaki mumlardan düşük olan dip.
        Bunlar piyasanın iskelet yapısını oluşturur.
        """
        if lookback is None:
            lookback = int(self.params["swing_lookback"])

        highs = df["high"].values
        lows = df["low"].values
        n = len(df)
        swing_highs = []
        swing_lows = []

        for i in range(lookback, n - lookback):
            # Swing High: merkez mum sağ ve soldakilerin hepsinden yüksek mi?
            is_sh = all(highs[i] > highs[i - j] and highs[i] > highs[i + j]
                        for j in range(1, lookback + 1))
            if is_sh:
                swing_highs.append({
                    "index": i,
                    "price": highs[i],
                    "timestamp": df["timestamp"].iloc[i]
                })

            # Swing Low: merkez mum sağ ve soldakilerin hepsinden düşük mü?
            is_sl = all(lows[i] < lows[i - j] and lows[i] < lows[i + j]
                        for j in range(1, lookback + 1))
            if is_sl:
                swing_lows.append({
                    "index": i,
                    "price": lows[i],
                    "timestamp": df["timestamp"].iloc[i]
                })

        return swing_highs, swing_lows

    # =================================================================
    #  BÖLÜM 4 — MARKET STRUCTURE (BOS / CHoCH)
    # =================================================================

    def detect_market_structure(self, df):
        """
        Piyasa yapısını analiz et:
        - BOS  (Break of Structure):  Mevcut trend yönünde yapı kırılımı.
        - CHoCH (Change of Character): Trendden ters yöne yapı değişimi.
        - Trend tespiti: HH+HL = Bullish,  LH+LL = Bearish.
        """
        swing_highs, swing_lows = self.find_swing_points(df)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                "trend": "NEUTRAL", "bos_events": [], "choch_events": [],
                "swing_highs": swing_highs, "swing_lows": swing_lows,
                "last_swing_high": None, "last_swing_low": None
            }

        bos_events = []
        choch_events = []
        current_trend = "NEUTRAL"
        min_displacement = self.params["bos_min_displacement"]

        # Tüm swing noktalarını index sırasına göre birleştir
        all_swings = []
        for sh in swing_highs:
            all_swings.append({"type": "HIGH", **sh})
        for sl in swing_lows:
            all_swings.append({"type": "LOW", **sl})
        all_swings.sort(key=lambda x: x["index"])

        # Yapı kırılımlarını tespit et
        for i in range(2, len(all_swings)):
            current = all_swings[i]
            # Aynı türden bir önceki swing'i bul
            prev_same = None
            for j in range(i - 1, -1, -1):
                if all_swings[j]["type"] == current["type"]:
                    prev_same = all_swings[j]
                    break
            if prev_same is None:
                continue

            if current["type"] == "HIGH":
                if current["price"] > prev_same["price"]:
                    displacement = (current["price"] - prev_same["price"]) / prev_same["price"]
                    if displacement > min_displacement:
                        if current_trend == "BEARISH":
                            choch_events.append({
                                "type": "BULLISH_CHOCH", "index": current["index"],
                                "price": current["price"], "prev_price": prev_same["price"],
                                "timestamp": current["timestamp"]
                            })
                        else:
                            bos_events.append({
                                "type": "BULLISH_BOS", "index": current["index"],
                                "price": current["price"], "prev_price": prev_same["price"],
                                "timestamp": current["timestamp"]
                            })
                        current_trend = "BULLISH"
                else:
                    if current_trend == "BULLISH":
                        current_trend = "WEAKENING_BULL"

            elif current["type"] == "LOW":
                if current["price"] < prev_same["price"]:
                    displacement = (prev_same["price"] - current["price"]) / prev_same["price"]
                    if displacement > min_displacement:
                        if current_trend == "BULLISH":
                            choch_events.append({
                                "type": "BEARISH_CHOCH", "index": current["index"],
                                "price": current["price"], "prev_price": prev_same["price"],
                                "timestamp": current["timestamp"]
                            })
                        else:
                            bos_events.append({
                                "type": "BEARISH_BOS", "index": current["index"],
                                "price": current["price"], "prev_price": prev_same["price"],
                                "timestamp": current["timestamp"]
                            })
                        current_trend = "BEARISH"
                else:
                    if current_trend == "BEARISH":
                        current_trend = "WEAKENING_BEAR"

        return {
            "trend": current_trend,
            "bos_events": bos_events[-5:] if bos_events else [],
            "choch_events": choch_events[-3:] if choch_events else [],
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "last_swing_high": swing_highs[-1] if swing_highs else None,
            "last_swing_low": swing_lows[-1] if swing_lows else None
        }

    # =================================================================
    #  BÖLÜM 5 — ORDER BLOCKS
    # =================================================================

    def find_order_blocks(self, df, structure):
        """
        Order Block tespiti:
        - Bullish OB: BOS/CHoCH öncesi son bearish mum
          (Kurumlar burada büyük alım yaptı → fiyat geri gelirse destek olur)
        - Bearish OB: BOS/CHoCH öncesi son bullish mum
          (Kurumlar burada büyük satış yaptı → fiyat geri gelirse direnç olur)
        """
        order_blocks = []
        max_age = int(self.params["ob_max_age_candles"])
        min_body_ratio = self.params["ob_body_ratio_min"]
        current_idx = len(df) - 1
        events = structure.get("bos_events", []) + structure.get("choch_events", [])

        for event in events:
            event_idx = event["index"]
            if current_idx - event_idx > max_age:
                continue

            is_bullish_event = "BULLISH" in event["type"]

            # Olay öncesi karşı yönlü mumu bul
            for j in range(event_idx - 1, max(event_idx - 10, 0), -1):
                if j >= len(df):
                    continue
                candle = df.iloc[j]
                body = abs(candle["close"] - candle["open"])
                total_range = candle["high"] - candle["low"]
                if total_range <= 0:
                    continue
                body_ratio = body / total_range

                if is_bullish_event:
                    # Bullish event öncesi bearish mum
                    if candle["close"] < candle["open"] and body_ratio >= min_body_ratio:
                        order_blocks.append({
                            "type": "BULLISH_OB", "index": j,
                            "high": candle["high"], "low": candle["low"],
                            "open": candle["open"], "close": candle["close"],
                            "timestamp": candle["timestamp"],
                            "mitigated": False, "strength": body_ratio
                        })
                        break
                else:
                    # Bearish event öncesi bullish mum
                    if candle["close"] > candle["open"] and body_ratio >= min_body_ratio:
                        order_blocks.append({
                            "type": "BEARISH_OB", "index": j,
                            "high": candle["high"], "low": candle["low"],
                            "open": candle["open"], "close": candle["close"],
                            "timestamp": candle["timestamp"],
                            "mitigated": False, "strength": body_ratio
                        })
                        break

        # Mitigated kontrol — fiyat OB bölgesinden geçtiyse artık geçersiz
        for ob in order_blocks:
            after_candles = df.iloc[ob["index"] + 1:]
            if len(after_candles) == 0:
                continue
            if ob["type"] == "BULLISH_OB" and after_candles["low"].min() < ob["low"]:
                ob["mitigated"] = True
            elif ob["type"] == "BEARISH_OB" and after_candles["high"].max() > ob["high"]:
                ob["mitigated"] = True

        active_obs = [ob for ob in order_blocks if not ob["mitigated"]]
        return active_obs, order_blocks

    # =================================================================
    #  BÖLÜM 6 — BREAKER BLOCKS
    # =================================================================

    def find_breaker_blocks(self, all_order_blocks, df):
        """
        Breaker Block: Mitigate olmuş OB'nin karşı yönde güçlü S/R haline gelmesi.
        - Kırılmış Bullish OB → Bearish Breaker (direnç)
        - Kırılmış Bearish OB → Bullish Breaker (destek)
        ICT'de yüksek olasılıklı setup'lardan biri.
        """
        breaker_blocks = []
        current_price = df["close"].iloc[-1]
        current_idx = len(df) - 1

        for ob in all_order_blocks:
            if not ob["mitigated"]:
                continue
            if current_idx - ob["index"] > 40:
                continue

            if ob["type"] == "BULLISH_OB":
                if current_price >= ob["low"] * 0.998 and current_price <= ob["high"] * 1.005:
                    breaker_blocks.append({
                        "type": "BEARISH_BREAKER",
                        "high": ob["high"], "low": ob["low"],
                        "index": ob["index"], "timestamp": ob["timestamp"]
                    })
            elif ob["type"] == "BEARISH_OB":
                if current_price >= ob["low"] * 0.995 and current_price <= ob["high"] * 1.002:
                    breaker_blocks.append({
                        "type": "BULLISH_BREAKER",
                        "high": ob["high"], "low": ob["low"],
                        "index": ob["index"], "timestamp": ob["timestamp"]
                    })

        return breaker_blocks

    # =================================================================
    #  BÖLÜM 7 — FAIR VALUE GAPS (FVG)
    # =================================================================

    def find_fvg(self, df):
        """
        Fair Value Gap tespiti (3 mumlu imbalance paterni):
        - Bullish FVG: mum[i-1].high < mum[i+1].low → arada boşluk (fiyat geri dönüp doldurmaya çalışır)
        - Bearish FVG: mum[i-1].low > mum[i+1].high → arada boşluk (fiyat geri çıkıp doldurmaya çalışır)
        Doldurulmamış FVG'ler güçlü giriş noktalarıdır — kurumsal emir boşluğu.
        """
        fvgs = []
        max_age = int(self.params["fvg_max_age_candles"])
        min_size_pct = self.params["fvg_min_size_pct"]
        n = len(df)
        current_idx = n - 1

        for i in range(1, n - 1):
            if current_idx - i > max_age:
                continue

            prev_c = df.iloc[i - 1]
            curr_c = df.iloc[i]
            next_c = df.iloc[i + 1]
            mid_price = curr_c["close"]
            if mid_price <= 0:
                continue

            # Bullish FVG
            if prev_c["high"] < next_c["low"]:
                gap = next_c["low"] - prev_c["high"]
                if gap / mid_price >= min_size_pct:
                    filled = False
                    if i + 2 < n:
                        if df.iloc[i + 2:]["low"].min() <= prev_c["high"]:
                            filled = True
                    if not filled:
                        fvgs.append({
                            "type": "BULLISH_FVG", "index": i,
                            "high": next_c["low"], "low": prev_c["high"],
                            "size_pct": round((gap / mid_price) * 100, 4),
                            "timestamp": curr_c["timestamp"], "filled": False
                        })

            # Bearish FVG
            if prev_c["low"] > next_c["high"]:
                gap = prev_c["low"] - next_c["high"]
                if gap / mid_price >= min_size_pct:
                    filled = False
                    if i + 2 < n:
                        if df.iloc[i + 2:]["high"].max() >= prev_c["low"]:
                            filled = True
                    if not filled:
                        fvgs.append({
                            "type": "BEARISH_FVG", "index": i,
                            "high": prev_c["low"], "low": next_c["high"],
                            "size_pct": round((gap / mid_price) * 100, 4),
                            "timestamp": curr_c["timestamp"], "filled": False
                        })

        return fvgs

    # =================================================================
    #  BÖLÜM 8 — LIQUIDITY LEVELS (Equal Highs / Lows)
    # =================================================================

    def find_liquidity_levels(self, df):
        """
        Likidite seviyelerini tespit et:
        - Equal Highs: Aynı seviyede biriken tepeler → üstte BSL (buy-side liq.)
        - Equal Lows:  Aynı seviyede biriken dipler  → altta SSL (sell-side liq.)
        Bu seviyeler, kurumların stopları temizlemek için hedeflediği bölgelerdir.
        """
        tolerance = self.params["liquidity_equal_tolerance"]
        min_touches = int(self.params["liquidity_min_touches"])
        swing_highs, swing_lows = self.find_swing_points(df)
        liquidity_levels = []

        # Equal Highs → BSL (Buy-Side Liquidity)
        for i, sh in enumerate(swing_highs):
            touches = 1
            touched_indices = [sh["index"]]
            for j in range(i + 1, len(swing_highs)):
                if abs(swing_highs[j]["price"] - sh["price"]) / sh["price"] <= tolerance:
                    touches += 1
                    touched_indices.append(swing_highs[j]["index"])
            if touches >= min_touches:
                exists = any(
                    ll["type"] == "EQUAL_HIGHS" and
                    abs(ll["price"] - sh["price"]) / sh["price"] <= tolerance
                    for ll in liquidity_levels
                )
                if not exists:
                    swept = False
                    max_idx = max(touched_indices)
                    if max_idx + 1 < len(df):
                        if df.iloc[max_idx + 1:]["high"].max() > sh["price"] * (1 + tolerance):
                            swept = True
                    liquidity_levels.append({
                        "type": "EQUAL_HIGHS", "price": sh["price"],
                        "touches": touches, "indices": touched_indices,
                        "swept": swept, "side": "SELL"
                    })

        # Equal Lows → SSL (Sell-Side Liquidity)
        for i, sl in enumerate(swing_lows):
            touches = 1
            touched_indices = [sl["index"]]
            for j in range(i + 1, len(swing_lows)):
                if abs(swing_lows[j]["price"] - sl["price"]) / sl["price"] <= tolerance:
                    touches += 1
                    touched_indices.append(swing_lows[j]["index"])
            if touches >= min_touches:
                exists = any(
                    ll["type"] == "EQUAL_LOWS" and
                    abs(ll["price"] - sl["price"]) / sl["price"] <= tolerance
                    for ll in liquidity_levels
                )
                if not exists:
                    swept = False
                    max_idx = max(touched_indices)
                    if max_idx + 1 < len(df):
                        if df.iloc[max_idx + 1:]["low"].min() < sl["price"] * (1 - tolerance):
                            swept = True
                    liquidity_levels.append({
                        "type": "EQUAL_LOWS", "price": sl["price"],
                        "touches": touches, "indices": touched_indices,
                        "swept": swept, "side": "BUY"
                    })

        return liquidity_levels

    # =================================================================
    #  BÖLÜM 9 — DISPLACEMENT (Güçlü Momentum Mumları)
    # =================================================================

    def detect_displacement(self, df, lookback=10):
        """
        Displacement: Kurumsal aktivitenin izini gösteren güçlü momentum mumları.
        Büyük gövdeli, küçük fitilli, tek yönlü hareket.
        ICT'de "displacement" olmadan giriş yapılmaz — kurumsal onay eksik demektir.
        """
        displacements = []
        min_body_ratio = self.params["displacement_min_body_ratio"]
        min_size_pct = self.params["displacement_min_size_pct"]
        n = len(df)

        for i in range(max(0, n - lookback), n):
            candle = df.iloc[i]
            body = abs(candle["close"] - candle["open"])
            total_range = candle["high"] - candle["low"]
            mid_price = (candle["high"] + candle["low"]) / 2
            if total_range <= 0 or mid_price <= 0:
                continue
            body_ratio = body / total_range
            size_pct = body / mid_price

            if body_ratio >= min_body_ratio and size_pct >= min_size_pct:
                direction = "BULLISH" if candle["close"] > candle["open"] else "BEARISH"
                displacements.append({
                    "type": f"{direction}_DISPLACEMENT", "index": i,
                    "body_ratio": round(body_ratio, 3),
                    "size_pct": round(size_pct * 100, 3),
                    "direction": direction, "timestamp": candle["timestamp"]
                })

        return displacements

    # =================================================================
    #  BÖLÜM 10 — PREMIUM / DISCOUNT + OTE
    # =================================================================

    def calculate_premium_discount(self, df, structure):
        """
        Premium/Discount bölgeleri:
        Son swing high-low arasının %50 seviyesi = Equilibrium (denge noktası).
        Premium (üst yarı) = Satış bölgesi — SHORT için ideal.
        Discount (alt yarı) = Alış bölgesi — LONG için ideal.
        OTE (Optimal Trade Entry) = Fibonacci 0.618-0.786 arası → en ideal giriş.
        """
        last_high = structure.get("last_swing_high")
        last_low = structure.get("last_swing_low")
        if not last_high or not last_low:
            return None

        high_price = last_high["price"]
        low_price = last_low["price"]
        equilibrium = (high_price + low_price) / 2
        current_price = df["close"].iloc[-1]

        fib_range = high_price - low_price
        ote_high = low_price + fib_range * 0.786
        ote_low = low_price + fib_range * 0.618

        zone = "PREMIUM" if current_price > equilibrium else "DISCOUNT"

        return {
            "high": high_price, "low": low_price,
            "equilibrium": equilibrium, "current_price": current_price,
            "zone": zone, "ote_high": ote_high, "ote_low": ote_low,
            "in_ote": ote_low <= current_price <= ote_high,
            "premium_level": round(
                (current_price - low_price) / (high_price - low_price) * 100, 1
            ) if high_price != low_price else 50
        }

    # =================================================================
    #  BÖLÜM 11 — GATE 1: HTF BIAS (4 Saatlik Yapı Analizi)
    # =================================================================

    def _analyze_htf_bias(self, multi_tf_data):
        """
        ★ GATE 1 — HTF (4H) yapısından KESTİN yön tespiti.

        4H BOS/CHoCH yukarıysa → SADECE LONG aranır.
        4H BOS/CHoCH aşağıysa → SADECE SHORT aranır.
        Belirsizse (NEUTRAL) → İŞLEM YAPILMAZ.

        Bu en kritik filtredir — 4H trendi karşısına işlem açmak
        bireysel yatırımcıların en büyük hatasıdır.

        Returns: {"bias": "LONG"/"SHORT", "htf_trend": str, "structure": dict}
                 veya None (belirsiz → işlem yok)
        """
        if not multi_tf_data or "4H" not in multi_tf_data:
            return None

        htf_df = multi_tf_data["4H"]
        if htf_df is None or htf_df.empty or len(htf_df) < 30:
            return None

        structure = self.detect_market_structure(htf_df)
        htf_liquidity = self.find_liquidity_levels(htf_df)

        result_base = {"structure": structure, "liquidity": htf_liquidity}

        if structure["trend"] == "BULLISH":
            return {**result_base, "bias": "LONG", "htf_trend": "BULLISH", "weak": False}
        elif structure["trend"] == "BEARISH":
            return {**result_base, "bias": "SHORT", "htf_trend": "BEARISH", "weak": False}
        elif structure["trend"] == "WEAKENING_BEAR":
            # Düşüş zayıflıyor → potansiyel LONG (dikkatli)
            return {**result_base, "bias": "LONG", "htf_trend": "WEAKENING_BEAR", "weak": True}
        elif structure["trend"] == "WEAKENING_BULL":
            # Yükseliş zayıflıyor → potansiyel SHORT (dikkatli)
            return {**result_base, "bias": "SHORT", "htf_trend": "WEAKENING_BULL", "weak": True}

        return None  # NEUTRAL → NET YÖN YOK → İŞLEM YAPILMAZ

    # =================================================================
    #  BÖLÜM 12 — GATE 2: LIQUIDITY SWEEP (Likidite Avı)
    # =================================================================

    def _find_sweep_event(self, df, bias, lookback=30):
        """
        ★ GATE 2 — Likidite Avı (Stop Hunt) Tespiti.

        Bu ICT'nin kalbindeki kavramdır: Kurumlar, bireysel yatırımcıların
        stop-loss emirlerini tetiklemek için fiyatı kasıtlı olarak eski
        swing noktalarının ötesine iter, sonra asıl yöne döner.

        LONG bias → Fiyat eski bir Swing Low'un ALTINA fitil atar ve
                     ÜSTÜNDE kapanır (stop hunt → smart money alım)
        SHORT bias → Fiyat eski bir Swing High'ın ÜSTÜNE fitil atar ve
                      ALTINDA kapanır (stop hunt → smart money satış)

        Returns: {"swept_level": float, "sweep_candle_idx": int, ...}
                 veya None (sweep yok)
        """
        swing_highs, swing_lows = self.find_swing_points(df)
        n = len(df)

        if bias == "LONG":
            # LONG → SSL (Sell-Side Liquidity) avı → eski swing low altına fitil
            for sw in reversed(swing_lows):
                sw_price = sw["price"]
                sw_idx = sw["index"]
                # Sweep sonraki mumlarda olmalı
                for i in range(sw_idx + 1, min(sw_idx + lookback + 1, n)):
                    candle = df.iloc[i]
                    # Wick sw_price altına inip, close sw_price üstünde mi?
                    if candle["low"] < sw_price and candle["close"] > sw_price:
                        # Sweep çok eski olmamalı
                        if n - 1 - i <= lookback:
                            return {
                                "swept_level": sw_price,
                                "sweep_candle_idx": i,
                                "sweep_wick": candle["low"],
                                "sweep_type": "SSL_SWEEP",
                                "swing_index": sw_idx
                            }

        elif bias == "SHORT":
            # SHORT → BSL (Buy-Side Liquidity) avı → eski swing high üstüne fitil
            for sw in reversed(swing_highs):
                sw_price = sw["price"]
                sw_idx = sw["index"]
                for i in range(sw_idx + 1, min(sw_idx + lookback + 1, n)):
                    candle = df.iloc[i]
                    # Wick sw_price üstüne çıkıp, close sw_price altında mı?
                    if candle["high"] > sw_price and candle["close"] < sw_price:
                        if n - 1 - i <= lookback:
                            return {
                                "swept_level": sw_price,
                                "sweep_candle_idx": i,
                                "sweep_wick": candle["high"],
                                "sweep_type": "BSL_SWEEP",
                                "swing_index": sw_idx
                            }

        return None

    # =================================================================
    #  BÖLÜM 13 — GATE 3: DISPLACEMENT + MSS (Onay)
    # =================================================================

    def _find_post_sweep_confirmation(self, df, sweep, bias):
        """
        ★ GATE 3 — Sweep Sonrası Displacement + Market Structure Shift.

        Sweep tek başına yeterli değil — ardından gerçek dönüş onayı gerekli:

        1. DISPLACEMENT: Sweep'ten sonra bias yönünde güçlü hacimli mum.
           Bu mum kurumsal aktivitenin "ayak izi"dir.

        2. MSS (Market Structure Shift): Displacement sonrası yapı kırılımı.
           LONG → Son swing high kırılır (Bullish BOS/CHoCH)
           SHORT → Son swing low kırılır (Bearish BOS/CHoCH)

        Minimum displacement ZORUNLU. MSS güven bonus'u sağlar.

        Returns: {"displacement": dict, "mss_confirmed": bool}
                 veya None (onay yok)
        """
        sweep_idx = sweep["sweep_candle_idx"]
        n = len(df)
        max_lookahead = 15  # Sweep sonrası max 15 mum içinde olmalı

        # --- Displacement Tespiti ---
        min_body_ratio = self.params.get("displacement_min_body_ratio", 0.7)
        min_size_pct = self.params.get("displacement_min_size_pct", 0.005)
        displacement = None

        for i in range(sweep_idx + 1, min(sweep_idx + max_lookahead + 1, n)):
            candle = df.iloc[i]
            body = abs(candle["close"] - candle["open"])
            total_range = candle["high"] - candle["low"]
            mid_price = (candle["high"] + candle["low"]) / 2
            if total_range <= 0 or mid_price <= 0:
                continue

            body_ratio = body / total_range
            size_pct = body / mid_price

            if body_ratio >= min_body_ratio and size_pct >= min_size_pct:
                candle_dir = "BULLISH" if candle["close"] > candle["open"] else "BEARISH"
                # LONG → displacement BULLISH olmalı
                if bias == "LONG" and candle_dir == "BULLISH":
                    displacement = {
                        "index": i, "direction": "BULLISH",
                        "body_ratio": round(body_ratio, 3),
                        "size_pct": round(size_pct * 100, 3)
                    }
                    break
                elif bias == "SHORT" and candle_dir == "BEARISH":
                    displacement = {
                        "index": i, "direction": "BEARISH",
                        "body_ratio": round(body_ratio, 3),
                        "size_pct": round(size_pct * 100, 3)
                    }
                    break

        if displacement is None:
            return None

        # --- MSS (Market Structure Shift) Tespiti ---
        # Displacement mumunun sweep öncesi yapıyı kırıp kırmadığını kontrol et
        mss_confirmed = False
        disp_idx = displacement["index"]

        if bias == "LONG":
            # Sweep öncesi son swing high bulunmalı ve kırılmalı
            pre_sweep_highs = [sh for sh in self.find_swing_points(df)[0]
                               if sh["index"] < sweep_idx]
            if pre_sweep_highs:
                target_high = pre_sweep_highs[-1]["price"]
                for i in range(disp_idx, min(disp_idx + 8, n)):
                    if df.iloc[i]["high"] > target_high:
                        mss_confirmed = True
                        break
        elif bias == "SHORT":
            pre_sweep_lows = [sl for sl in self.find_swing_points(df)[1]
                              if sl["index"] < sweep_idx]
            if pre_sweep_lows:
                target_low = pre_sweep_lows[-1]["price"]
                for i in range(disp_idx, min(disp_idx + 8, n)):
                    if df.iloc[i]["low"] < target_low:
                        mss_confirmed = True
                        break

        return {
            "displacement": displacement,
            "mss_confirmed": mss_confirmed,
            "confidence_boost": 10 if mss_confirmed else 0
        }

    # =================================================================
    #  BÖLÜM 14 — GATE 4: DISPLACEMENT FVG (Giriş Bölgesi)
    # =================================================================

    def _find_displacement_fvg(self, df, displacement_idx, bias):
        """
        ★ GATE 4 — Displacement mumunun oluşturduğu FVG'yi bul.

        Displacement mumu tek yönlü güçlü hareket yapar ve MUTLAKA
        bir FVG (Fair Value Gap) bırakır. Bu FVG kurumsal emir
        boşluğudur — fiyat buraya geri döner (fill) ve bu bizim
        GİRİŞ BÖLGEMİZDİR.

        Arama: displacement mumunun kendisi ve çevresindeki 3 mumda FVG ara.
        Bulamazsa, displacement sonrası oluşan tüm FVG'leri kontrol et.

        Returns: FVG dict veya None
        """
        n = len(df)
        search_start = max(1, displacement_idx - 1)
        search_end = min(n - 1, displacement_idx + 4)
        min_fvg_size = self.params.get("fvg_min_size_pct", 0.001)
        best_fvg = None

        for i in range(search_start, search_end):
            if i < 1 or i >= n - 1:
                continue
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            next_ = df.iloc[i + 1]
            mid_price = curr["close"]
            if mid_price <= 0:
                continue

            if bias == "LONG":
                # Bullish FVG: prev.high < next.low
                if prev["high"] < next_["low"]:
                    gap = next_["low"] - prev["high"]
                    if gap / mid_price >= min_fvg_size:
                        filled = False
                        if i + 2 < n and df.iloc[i + 2:]["low"].min() <= prev["high"]:
                            filled = True
                        if not filled:
                            fvg = {
                                "type": "BULLISH_FVG", "index": i,
                                "high": next_["low"], "low": prev["high"],
                                "size_pct": round((gap / mid_price) * 100, 4),
                                "timestamp": curr["timestamp"]
                            }
                            if best_fvg is None or abs(i - displacement_idx) < abs(best_fvg["index"] - displacement_idx):
                                best_fvg = fvg

            elif bias == "SHORT":
                # Bearish FVG: prev.low > next.high
                if prev["low"] > next_["high"]:
                    gap = prev["low"] - next_["high"]
                    if gap / mid_price >= min_fvg_size:
                        filled = False
                        if i + 2 < n and df.iloc[i + 2:]["high"].max() >= prev["low"]:
                            filled = True
                        if not filled:
                            fvg = {
                                "type": "BEARISH_FVG", "index": i,
                                "high": prev["low"], "low": next_["high"],
                                "size_pct": round((gap / mid_price) * 100, 4),
                                "timestamp": curr["timestamp"]
                            }
                            if best_fvg is None or abs(i - displacement_idx) < abs(best_fvg["index"] - displacement_idx):
                                best_fvg = fvg

        # Displacement yakınında FVG bulunamadıysa, displacement sonrası tüm FVG'leri kontrol et
        if best_fvg is None:
            all_fvgs = self.find_fvg(df)
            target_type = "BULLISH_FVG" if bias == "LONG" else "BEARISH_FVG"
            relevant = [f for f in all_fvgs
                        if f["type"] == target_type and f["index"] >= displacement_idx - 3]
            if relevant:
                best_fvg = min(relevant, key=lambda f: abs(f["index"] - displacement_idx))

        return best_fvg

    # =================================================================
    #  BÖLÜM 15 — YAPISAL STOP LOSS HESAPLAMA
    # =================================================================

    def _calc_structural_sl(self, df, sweep, bias, structure):
        """
        Yapısal (Structural) Stop Loss hesaplama.
        SABIT YÜZDE KULLANILMAZ — her zaman piyasa yapısına göre hesaplanır.

        LONG SL sırası:
          1. Sweep mumunun wick'inin altı (sweep invalidation)
          2. Sweep edilen swing low'un altı
          3. Son swing low'un altı
        SHORT SL sırası:
          1. Sweep mumunun wick'inin üstü (sweep invalidation)
          2. Sweep edilen swing high'ın üstü
          3. Son swing high'ın üstü

        SL mesafesi çok uzaksa → sinyal üretilmez (None döner).
        """
        candidates = []

        if bias == "LONG":
            # 1. Sweep mumunun wick altı (en kesin invalidation noktası)
            sweep_wick = sweep.get("sweep_wick", sweep.get("sweep_low"))
            if sweep_wick:
                candidates.append(("SWEEP_WICK", sweep_wick * 0.998))

            # 2. Sweep edilen seviyenin altı
            candidates.append(("SWEPT_LEVEL", sweep["swept_level"] * 0.997))

            # 3. Son swing low
            if structure["last_swing_low"]:
                candidates.append(("SWING_LOW", structure["last_swing_low"]["price"] * 0.997))

            # En yakın (entry'ye en yakın) geçerli SL'yi seç
            valid = [(name, price) for name, price in candidates if price > 0]
            if not valid:
                return None

            # En yakın olanı seç (unnecessarily geniş SL'den kaçın)
            best = max(valid, key=lambda x: x[1])

            logger.debug(f"  LONG SL: {best[0]} @ {best[1]:.8f}")
            return best[1]

        elif bias == "SHORT":
            sweep_wick = sweep.get("sweep_wick", sweep.get("sweep_high"))
            if sweep_wick:
                candidates.append(("SWEEP_WICK", sweep_wick * 1.002))

            candidates.append(("SWEPT_LEVEL", sweep["swept_level"] * 1.003))

            if structure["last_swing_high"]:
                candidates.append(("SWING_HIGH", structure["last_swing_high"]["price"] * 1.003))

            valid = [(name, price) for name, price in candidates if price > 0]
            if not valid:
                return None

            best = min(valid, key=lambda x: x[1])

            logger.debug(f"  SHORT SL: {best[0]} @ {best[1]:.8f}")
            return best[1]

        return None

    # =================================================================
    #  BÖLÜM 16 — KARŞI LİKİDİTE TP HESAPLAMA (Draw on Liquidity)
    # =================================================================

    def _calc_opposing_liquidity_tp(self, df, multi_tf_data, entry, sl, bias, structure):
        """
        Draw on Liquidity — Karşı taraftaki likidite havuzunu hedefle.
        SABİT R:R KULLANILMAZ — her zaman yapısal hedef aranır.

        LONG TP sırası:
          1. HTF (4H) equal highs → en güçlü mıknatıs
          2. LTF (15m) equal highs → ana hedef
          3. Karşı (bearish) Order Block → fiyat burada tepki verir
          4. Son swing high → yapısal direnç
          5. Minimum R:R fallback (sadece hiçbir hedef bulunamazsa)

        SHORT TP sırası:
          1. HTF (4H) equal lows
          2. LTF (15m) equal lows
          3. Karşı (bullish) Order Block
          4. Son swing low
          5. Minimum R:R fallback
        """
        tp_candidates = []

        # HTF likidite
        htf_liquidity = []
        if multi_tf_data and "4H" in multi_tf_data and not multi_tf_data["4H"].empty:
            htf_liquidity = self.find_liquidity_levels(multi_tf_data["4H"])

        # LTF likidite
        ltf_liquidity = self.find_liquidity_levels(df)

        # LTF yapı ve order blocks
        ltf_structure = self.detect_market_structure(df)
        active_obs, _ = self.find_order_blocks(df, ltf_structure)

        if bias == "LONG":
            risk = entry - sl if sl and sl < entry else entry * 0.015

            # HTF Draw on Liquidity
            for liq in htf_liquidity:
                if liq["type"] == "EQUAL_HIGHS" and not liq["swept"] and liq["price"] > entry:
                    tp_candidates.append(("HTF_DRAW_LIQ", liq["price"] * 0.999))

            # LTF BSL (equal highs)
            for liq in ltf_liquidity:
                if liq["type"] == "EQUAL_HIGHS" and not liq["swept"] and liq["price"] > entry:
                    tp_candidates.append(("LTF_BSL", liq["price"] * 0.999))

            # Karşı OB
            for ob in active_obs:
                if ob["type"] == "BEARISH_OB" and ob["low"] > entry:
                    tp_candidates.append(("OPPOSING_OB", ob["low"]))

            # Son swing high
            if structure["last_swing_high"] and structure["last_swing_high"]["price"] > entry:
                tp_candidates.append(("SWING_HIGH", structure["last_swing_high"]["price"] * 0.998))

            # Önceki swing high'lar
            for sh in structure.get("swing_highs", []):
                if sh["price"] > entry * 1.005:
                    tp_candidates.append(("PREV_SH", sh["price"] * 0.998))

        elif bias == "SHORT":
            risk = sl - entry if sl and sl > entry else entry * 0.015

            for liq in htf_liquidity:
                if liq["type"] == "EQUAL_LOWS" and not liq["swept"] and liq["price"] < entry:
                    tp_candidates.append(("HTF_DRAW_LIQ", liq["price"] * 1.001))

            for liq in ltf_liquidity:
                if liq["type"] == "EQUAL_LOWS" and not liq["swept"] and liq["price"] < entry:
                    tp_candidates.append(("LTF_SSL", liq["price"] * 1.001))

            for ob in active_obs:
                if ob["type"] == "BULLISH_OB" and ob["high"] < entry:
                    tp_candidates.append(("OPPOSING_OB", ob["high"]))

            if structure["last_swing_low"] and structure["last_swing_low"]["price"] < entry:
                tp_candidates.append(("SWING_LOW", structure["last_swing_low"]["price"] * 1.002))

            for sl_p in structure.get("swing_lows", []):
                if sl_p["price"] < entry * 0.995:
                    tp_candidates.append(("PREV_SL", sl_p["price"] * 1.002))

        if not tp_candidates:
            # Son çare: minimum R:R ile hesapla — ama bu ideal DEĞİL
            min_rr = self.params.get("default_tp_ratio", 2.5)
            if bias == "LONG":
                return entry + (risk * min_rr)
            else:
                return entry - (risk * min_rr)

        # Minimum 1.5 R:R sağlayan en yakın yapısal hedefi seç
        min_reward = risk * 1.5

        if bias == "LONG":
            valid = [(n, p) for n, p in tp_candidates if (p - entry) >= min_reward]
            if valid:
                best = min(valid, key=lambda x: x[1])  # En yakın geçerli hedef
                logger.debug(f"  LONG TP: {best[0]} @ {best[1]:.8f}")
                return best[1]
            # 1.5 RR sağlayan hedef yoksa en uzak olanı dene
            if tp_candidates:
                best = max(tp_candidates, key=lambda x: x[1])
                if (best[1] - entry) > risk:
                    return best[1]
        else:
            valid = [(n, p) for n, p in tp_candidates if (entry - p) >= min_reward]
            if valid:
                best = max(valid, key=lambda x: x[1])  # En yakın geçerli hedef (SHORT için en yüksek)
                logger.debug(f"  SHORT TP: {best[0]} @ {best[1]:.8f}")
                return best[1]
            if tp_candidates:
                best = min(tp_candidates, key=lambda x: x[1])
                if (entry - best[1]) > risk:
                    return best[1]

        # Gerçekten hiçbir hedef yoksa minimum R:R
        min_rr = self.params.get("default_tp_ratio", 2.5)
        if bias == "LONG":
            return entry + (risk * min_rr)
        return entry - (risk * min_rr)

    # =================================================================
    #  BÖLÜM 17 — CONFLUENCE SCORING (Geriye Uyumlu)
    # =================================================================

    def calculate_confluence(self, df, multi_tf_data=None):
        """
        Tüm ICT bileşenlerini analiz edip confluent skor hesapla.

        Bu metod hem generate_signal() tarafından hem de
        izleme listesi onayı (check_watchlist) ve API tarafından kullanılır.

        Sıralı Ağırlıklandırma:
          HTF Bias uyumu:       25 puan (veya -15 ceza)
          Liquidity Sweep:      20 puan
          Displacement:         15 puan (yoksa -8 ceza)
          FVG giriş bölgesi:    15 puan
          Market Structure:     10 puan
          Premium/Discount:     10 puan
          Session (Killzone):    5 puan
          Order Block:           5 bonus
          Breaker Block:         5 bonus
          Sweep+MSS (A+):      10 bonus
        """
        analysis = {}
        components = []
        score = 0
        penalties = []

        current_price = df["close"].iloc[-1]
        current_idx = len(df) - 1
        analysis["current_price"] = current_price

        # === RANGING MARKET ===
        is_ranging = self.detect_ranging_market(df)
        analysis["is_ranging"] = is_ranging

        # === SESSION / KILLZONE ===
        session_info = self.get_session_info()
        analysis["session"] = session_info

        # === LTF MARKET STRUCTURE (15m) ===
        structure = self.detect_market_structure(df)
        analysis["structure"] = structure

        if structure["trend"] in ["BULLISH", "BEARISH"]:
            score += 10
            components.append("MARKET_STRUCTURE")
        elif structure["trend"] in ["WEAKENING_BULL", "WEAKENING_BEAR"]:
            score += 3
            penalties.append("WEAKENING_TREND(-7)")

        # === HTF BIAS (4H) ===
        htf_bias_block = False
        htf_result = self._analyze_htf_bias(multi_tf_data)
        analysis["htf_result"] = htf_result

        if htf_result:
            analysis["htf_trend"] = htf_result["htf_trend"]
            analysis["htf_structure"] = htf_result["structure"]
            analysis["htf_liquidity"] = htf_result.get("liquidity", [])

            # HTF ve LTF aynı yönde mi?
            if htf_result["bias"] == "LONG" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                score += 25
                components.append("HTF_CONFIRMATION")
            elif htf_result["bias"] == "SHORT" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                score += 25
                components.append("HTF_CONFIRMATION")
            elif htf_result["bias"] == "LONG" and structure["trend"] == "BEARISH":
                # HTF LONG ama LTF BEARISH → HARD BLOCK
                htf_bias_block = True
                score -= 15
                penalties.append("HTF_BIAS_BLOCK(-15)")
            elif htf_result["bias"] == "SHORT" and structure["trend"] == "BULLISH":
                htf_bias_block = True
                score -= 15
                penalties.append("HTF_BIAS_BLOCK(-15)")
            else:
                # HTF var ama kısmi uyum
                score += 10
        else:
            analysis["htf_trend"] = "UNKNOWN"
            analysis["htf_structure"] = None
            analysis["htf_liquidity"] = []

        analysis["htf_bias_block"] = htf_bias_block

        # === MTF (1H) ONAY ===
        if multi_tf_data and "1H" in multi_tf_data and not multi_tf_data["1H"].empty:
            mtf_struct = self.detect_market_structure(multi_tf_data["1H"])
            analysis["mtf_trend"] = mtf_struct["trend"]
            if mtf_struct["trend"] == structure["trend"]:
                score += 3
                components.append("MTF_CONFIRMATION")
        else:
            analysis["mtf_trend"] = "UNKNOWN"

        # === YÖN === 
        direction = None
        if structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
            direction = "LONG"
        elif structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
            direction = "SHORT"
        analysis["direction"] = direction

        # === LIQUIDITY SWEEP ===
        bias_for_sweep = direction or (htf_result["bias"] if htf_result else None)
        sweep_detected = False
        sweep_mss_detected = False

        if bias_for_sweep:
            sweep = self._find_sweep_event(df, bias_for_sweep)
            analysis["sweep"] = sweep
            if sweep:
                score += 20
                components.append("LIQUIDITY_SWEEP")
                sweep_detected = True

                # Sweep sonrası displacement + MSS?
                confirmation = self._find_post_sweep_confirmation(df, sweep, bias_for_sweep)
                analysis["post_sweep_confirmation"] = confirmation
                if confirmation:
                    score += 15
                    components.append("DISPLACEMENT")

                    if confirmation["mss_confirmed"]:
                        score += 10
                        components.append("SWEEP_MSS_A_PLUS")
                        sweep_mss_detected = True

                    # Displacement FVG?
                    disp_fvg = self._find_displacement_fvg(df, confirmation["displacement"]["index"], bias_for_sweep)
                    analysis["displacement_fvg"] = disp_fvg
                    if disp_fvg:
                        score += 15
                        components.append("FVG")
                else:
                    analysis["post_sweep_confirmation"] = None
                    analysis["displacement_fvg"] = None
            else:
                analysis["sweep"] = None
                analysis["post_sweep_confirmation"] = None
                analysis["displacement_fvg"] = None
        else:
            analysis["sweep"] = None
            analysis["post_sweep_confirmation"] = None
            analysis["displacement_fvg"] = None

        analysis["sweep_mss"] = sweep_mss_detected

        # === ORDER BLOCKS (bonus) ===
        active_obs, all_obs = self.find_order_blocks(df, structure)
        analysis["order_blocks"] = active_obs
        analysis["all_order_blocks"] = all_obs

        relevant_obs = []
        for ob in active_obs:
            age = current_idx - ob["index"]
            recency = 1.0 if age <= 5 else (0.8 if age <= 15 else 0.5)

            if direction == "LONG" and ob["type"] == "BULLISH_OB":
                if ob["low"] <= current_price <= ob["high"] * 1.005:
                    relevant_obs.append(ob)
                    score += 5 * recency
                    components.append("ORDER_BLOCK")
                    break
            elif direction == "SHORT" and ob["type"] == "BEARISH_OB":
                if ob["low"] <= current_price <= ob["high"] * 1.005:
                    relevant_obs.append(ob)
                    score += 5 * recency
                    components.append("ORDER_BLOCK")
                    break
        analysis["relevant_obs"] = relevant_obs

        # === BREAKER BLOCKS (bonus) ===
        breaker_blocks = self.find_breaker_blocks(all_obs, df)
        analysis["breaker_blocks"] = breaker_blocks
        for bb in breaker_blocks:
            if (direction == "LONG" and bb["type"] == "BULLISH_BREAKER") or \
               (direction == "SHORT" and bb["type"] == "BEARISH_BREAKER"):
                score += 5
                components.append("BREAKER_BLOCK")
                break

        # === GENEL FVG KONTROLÜ (displacement FVG bulunamadıysa) ===
        fvgs = self.find_fvg(df)
        analysis["fvgs"] = fvgs

        relevant_fvgs = []
        if "FVG" not in components:
            for fvg in fvgs:
                fvg_age = current_idx - fvg["index"]
                fvg_recency = 1.0 if fvg_age <= 8 else 0.6
                if direction == "LONG" and fvg["type"] == "BULLISH_FVG":
                    if fvg["low"] * 0.998 <= current_price <= fvg["high"] * 1.002:
                        relevant_fvgs.append(fvg)
                        score += 10 * fvg_recency
                        components.append("FVG")
                        break
                elif direction == "SHORT" and fvg["type"] == "BEARISH_FVG":
                    if fvg["low"] * 0.998 <= current_price <= fvg["high"] * 1.002:
                        relevant_fvgs.append(fvg)
                        score += 10 * fvg_recency
                        components.append("FVG")
                        break
        analysis["relevant_fvgs"] = relevant_fvgs

        # === LIQUIDITY LEVELS ===
        liquidity = self.find_liquidity_levels(df)
        analysis["liquidity"] = liquidity

        # === DISPLACEMENT (genel — sweep bağımsız) ===
        if "DISPLACEMENT" not in components:
            displacements = self.detect_displacement(df)
            analysis["displacements"] = displacements
            if displacements:
                last_d = displacements[-1]
                if (direction == "LONG" and last_d["direction"] == "BULLISH") or \
                   (direction == "SHORT" and last_d["direction"] == "BEARISH"):
                    score += 8
                    components.append("DISPLACEMENT")
            if "DISPLACEMENT" not in components:
                score -= 8
                penalties.append("NO_DISPLACEMENT(-8)")
        else:
            analysis["displacements"] = self.detect_displacement(df)

        # === PREMIUM / DISCOUNT + OTE ===
        pd_zone = self.calculate_premium_discount(df, structure)
        analysis["premium_discount"] = pd_zone
        if pd_zone:
            if direction == "LONG" and pd_zone["zone"] == "DISCOUNT":
                score += 7
                components.append("DISCOUNT_ZONE")
                if pd_zone["in_ote"]:
                    score += 3
                    components.append("OTE")
            elif direction == "SHORT" and pd_zone["zone"] == "PREMIUM":
                score += 7
                components.append("PREMIUM_ZONE")
                if pd_zone["in_ote"]:
                    score += 3
                    components.append("OTE")

        # === SESSION KALİTESİ ===
        if session_info["quality"] >= 0.8:
            score += 5
            components.append("KILLZONE_ACTIVE")
        elif session_info["quality"] <= 0.3:
            score -= 5
            penalties.append("OFF_HOURS(-5)")

        # === RANGING CEZASI ===
        if is_ranging:
            score -= 15
            penalties.append("RANGING_MARKET(-15)")

        # === TRIPLE TF ALIGNMENT ===
        if "HTF_CONFIRMATION" in components and "MTF_CONFIRMATION" in components:
            score += 3
            components.append("TRIPLE_TF_ALIGNMENT")

        # Normalize (0-100)
        max_possible = 130  # tüm bonuslar dahil teorik max
        score = max(0, score)
        confluence_score = min(100, round((score / max_possible) * 100, 1))

        analysis["confluence_score"] = confluence_score
        analysis["components"] = list(set(components))
        analysis["penalties"] = penalties

        return analysis

    # =================================================================
    #  BÖLÜM 18 — GÜVEN SKORU HESAPLAMA
    # =================================================================

    def _calculate_confidence(self, analysis):
        """
        Güven skoru (0-100).
        Confluence score + gate kalitesi + ceza sistemi.
        """
        base = analysis["confluence_score"]
        bonus = 0
        penalty = 0
        components = analysis.get("components", [])

        # Çoklu bileşen bonusu
        comp_count = len(components)
        if comp_count >= 6:
            bonus += 12
        elif comp_count >= 4:
            bonus += 8
        elif comp_count >= 3:
            bonus += 4

        # Gate bazlı bonuslar
        if "HTF_CONFIRMATION" in components:
            bonus += 5
        if "LIQUIDITY_SWEEP" in components:
            bonus += 5
        if "DISPLACEMENT" in components:
            bonus += 5
        if "FVG" in components:
            bonus += 3
        if "SWEEP_MSS_A_PLUS" in components:
            bonus += 10  # A+ setup → en güçlü sinyal
        if "KILLZONE_ACTIVE" in components:
            bonus += 3
        if "BREAKER_BLOCK" in components:
            bonus += 3
        if "TRIPLE_TF_ALIGNMENT" in components:
            bonus += 5

        # Cezalar
        if "DISPLACEMENT" not in components:
            penalty += 10
        if "ORDER_BLOCK" not in components and "FVG" not in components:
            penalty += 8
        if "DISCOUNT_ZONE" not in components and "PREMIUM_ZONE" not in components and "OTE" not in components:
            penalty += 5
        if analysis.get("htf_bias_block"):
            penalty += 15
        if analysis.get("is_ranging"):
            penalty += 10

        structure = analysis.get("structure", {})
        if structure.get("trend") in ["WEAKENING_BULL", "WEAKENING_BEAR"]:
            penalty += 5

        session = analysis.get("session", {})
        if session.get("quality", 1.0) <= 0.3:
            penalty += 5

        confidence = max(0, min(100, base + bonus - penalty))
        return round(confidence, 1)

    # =================================================================
    #  BÖLÜM 19 — SİNYAL ÜRETİMİ (Sequential Gate Protocol)
    # =================================================================

    def generate_signal(self, symbol, df, multi_tf_data=None):
        """
        ★ ANA SİNYAL ÜRETİMİ — Katı Sıralı ICT Protokolü.

        Her adım bir GATE'tir:
          Gate 1: HTF Bias → 4H trend yönü belirler
          Gate 2: Liquidity Sweep → Eski swing seviyesinin temizlenmesi
          Gate 3: Displacement + MSS → Tersine dönüş onayı
          Gate 4: FVG Entry Zone → Giriş bölgesi tespiti

        Tüm gate'ler geçerse → SIGNAL (FVG'ye limit emir)
        Kısmi gate'ler → WATCH (izlemeye al)
        Hiçbir gate geçmezse → None (sinyal yok)

        Returns: signal dict veya None
        """
        if df.empty or len(df) < 30:
            return None

        current_price = df["close"].iloc[-1]

        # ===== GATE 0: Ranging Market → Sinyal üretme =====
        if self.detect_ranging_market(df):
            return None

        # ===== GATE 1: HTF Bias (4H) → Yön tayini =====
        htf_result = self._analyze_htf_bias(multi_tf_data)
        if htf_result is None:
            return None  # HTF belirsiz → İŞLEM YOK
        bias = htf_result["bias"]  # "LONG" veya "SHORT"

        # LTF (15m) yapı analizi
        structure = self.detect_market_structure(df)

        # LTF trend HTF bias'a KARŞI mı? → Bekle (henüz dönmedi)
        if bias == "LONG" and structure["trend"] == "BEARISH":
            return None
        if bias == "SHORT" and structure["trend"] == "BULLISH":
            return None

        # ===== GATE 2: Liquidity Sweep → Stop hunt tespiti =====
        sweep = self._find_sweep_event(df, bias)
        if sweep is None:
            # Sweep yok → potansiyel WATCH sinyali kontrol et
            return self._build_watch_from_potential(symbol, df, multi_tf_data, htf_result, structure, bias)

        # ===== GATE 3: Displacement + MSS → Dönüş onayı =====
        confirmation = self._find_post_sweep_confirmation(df, sweep, bias)
        if confirmation is None:
            # Sweep var ama displacement yok → WATCH
            analysis = self.calculate_confluence(df, multi_tf_data)
            confidence = self._calculate_confidence(analysis)
            return self._build_signal_dict(
                symbol, bias, current_price, analysis, confidence,
                action="WATCH",
                watch_reason="Sweep tespit edildi, displacement bekleniyor"
            )

        # ===== GATE 4: Displacement FVG → Giriş bölgesi =====
        disp_idx = confirmation["displacement"]["index"]
        entry_fvg = self._find_displacement_fvg(df, disp_idx, bias)
        if entry_fvg is None:
            analysis = self.calculate_confluence(df, multi_tf_data)
            confidence = self._calculate_confidence(analysis)
            return self._build_signal_dict(
                symbol, bias, current_price, analysis, confidence,
                action="WATCH",
                watch_reason="Displacement sonrası FVG bekleniyor"
            )

        # ===== TÜM GATE'LER GEÇTİ — SİNYAL OLUŞTUR =====
        logger.info(f"🎯 {symbol}: Tüm ICT gate'leri geçti: HTF={htf_result['htf_trend']}, "
                    f"Sweep={sweep['sweep_type']}, Displacement+{'MSS' if confirmation['mss_confirmed'] else 'noMSS'}")

        # FVG'nin CE noktası (Consequent Encroachment = orta nokta) = ENTRY
        entry = (entry_fvg["high"] + entry_fvg["low"]) / 2

        # Yapısal SL
        sl = self._calc_structural_sl(df, sweep, bias, structure)
        if sl is None:
            return None

        # Draw on Liquidity TP
        tp = self._calc_opposing_liquidity_tp(df, multi_tf_data, entry, sl, bias, structure)
        if tp is None:
            return None

        # Seviye doğrulama
        if bias == "LONG":
            if sl >= entry or tp <= entry:
                logger.warning(f"❌ {symbol} LONG seviyeleri ters: E={entry} SL={sl} TP={tp}")
                return None
            risk = entry - sl
            reward = tp - entry
        else:
            if sl <= entry or tp >= entry:
                logger.warning(f"❌ {symbol} SHORT seviyeleri ters: E={entry} SL={sl} TP={tp}")
                return None
            risk = sl - entry
            reward = entry - tp

        if risk <= 0:
            return None

        rr_ratio = reward / risk
        if rr_ratio < 1.5:
            return None

        # SL mesafesi kontrolleri
        sl_distance_pct = risk / entry
        if sl_distance_pct < 0.003:
            return None  # SL çok yakın → volatilitede vurulur
        if sl_distance_pct > 0.06:
            return None  # SL çok uzak → risk çok yüksek

        # Entry modu: Fiyat FVG bölgesinde mi?
        if bias == "LONG":
            price_at_fvg = entry_fvg["low"] * 0.998 <= current_price <= entry_fvg["high"] * 1.002
        else:
            price_at_fvg = entry_fvg["low"] * 0.998 <= current_price <= entry_fvg["high"] * 1.002
        entry_mode = "MARKET" if price_at_fvg else "LIMIT"

        # Confluence ve confidence hesapla
        analysis = self.calculate_confluence(df, multi_tf_data)
        confluence_score = analysis["confluence_score"]
        confidence = self._calculate_confidence(analysis)

        # Minimum eşikler
        min_confluence = self.params.get("min_confluence_score", 70)
        min_confidence = self.params.get("min_confidence", 75)

        session = self.get_session_info()
        components = analysis.get("components", [])

        result = {
            "symbol": symbol,
            "direction": bias,
            "entry": round(entry, 8),
            "sl": round(sl, 8),
            "tp": round(tp, 8),
            "current_price": round(current_price, 8),
            "confluence_score": confluence_score,
            "confidence": confidence,
            "components": components,
            "penalties": analysis.get("penalties", []),
            "session": session.get("label", ""),
            "rr_ratio": round(rr_ratio, 2),
            "entry_type": f"FVG Limit ({entry_fvg['type']})" if entry_mode == "LIMIT" else f"FVG Market ({entry_fvg['type']})",
            "sl_type": "Yapısal Seviye (Sweep Invalidation)",
            "tp_type": self._get_tp_type(analysis, tp, bias),
            "entry_mode": entry_mode,
            "htf_bias": htf_result["htf_trend"],
            "sweep_level": sweep["swept_level"],
            "analysis": analysis
        }

        # Sinyal mi, izleme mi?
        if confluence_score >= min_confluence and confidence >= min_confidence:
            result["action"] = "SIGNAL"
            logger.info(
                f"🎯 SİNYAL: {symbol} {bias} | Entry: {entry:.8f} | SL: {sl:.8f} | TP: {tp:.8f} | "
                f"RR: {rr_ratio:.1f} | Score: {confluence_score} | Conf: {confidence}% | "
                f"Mode: {entry_mode} | Session: {session['label']}"
            )
        elif confluence_score >= min_confluence * 0.7:
            result["action"] = "WATCH"
            result["watch_reason"] = self._get_watch_reason(analysis)
            logger.info(
                f"👀 İZLEME: {symbol} {bias} | Score: {confluence_score} | "
                f"Conf: {confidence}% | Sebep: {result['watch_reason']}"
            )
        else:
            return None

        return result

    # =================================================================
    #  BÖLÜM 20 — YARDIMCI METODLAR
    # =================================================================

    def _build_watch_from_potential(self, symbol, df, multi_tf_data, htf_result, structure, bias):
        """
        Sweep henüz olmadığında potansiyel WATCH sinyali oluştur.
        Sadece yeterli potansiyel varsa (yüksek skor) döndürür.
        """
        analysis = self.calculate_confluence(df, multi_tf_data)
        confluence_score = analysis["confluence_score"]
        min_confluence = self.params.get("min_confluence_score", 70)

        # En az %60 potansiyel olmalı (sweep olmadan SIGNAL asla olmaz)
        if confluence_score < min_confluence * 0.6:
            return None

        confidence = self._calculate_confidence(analysis)
        current_price = df["close"].iloc[-1]

        return self._build_signal_dict(
            symbol, bias, current_price, analysis, confidence,
            action="WATCH",
            watch_reason="HTF bias uyumlu, likidite avı bekleniyor"
        )

    def _build_signal_dict(self, symbol, bias, current_price, analysis, confidence,
                           action="WATCH", watch_reason=""):
        """WATCH sinyalleri için ortak dict oluşturucu."""
        # Basit SL/TP tahmini (WATCH için yaklaşık)
        structure = analysis.get("structure", {})
        sl_pct = self.params.get("default_sl_pct", 0.015)
        tp_ratio = self.params.get("default_tp_ratio", 2.5)

        if bias == "LONG":
            sl = current_price * (1 - sl_pct)
            tp = current_price * (1 + sl_pct * tp_ratio)
        else:
            sl = current_price * (1 + sl_pct)
            tp = current_price * (1 - sl_pct * tp_ratio)

        risk = abs(current_price - sl)
        reward = abs(tp - current_price)
        rr_ratio = reward / risk if risk > 0 else 1.0

        session = self.get_session_info()

        result = {
            "symbol": symbol,
            "direction": bias,
            "entry": round(current_price, 8),
            "sl": round(sl, 8),
            "tp": round(tp, 8),
            "current_price": round(current_price, 8),
            "confluence_score": analysis.get("confluence_score", 0),
            "confidence": confidence,
            "components": analysis.get("components", []),
            "penalties": analysis.get("penalties", []),
            "session": session.get("label", ""),
            "rr_ratio": round(rr_ratio, 2),
            "entry_type": "Potansiyel (onay bekleniyor)",
            "sl_type": "Tahmini (onay sonrası kesinleşecek)",
            "tp_type": "Tahmini (onay sonrası kesinleşecek)",
            "entry_mode": "PENDING",
            "action": action,
            "watch_reason": watch_reason,
            "analysis": analysis
        }

        return result

    def _get_watch_reason(self, analysis):
        """İzleme sebebini açıkla (hangi gate eksik)."""
        reasons = []
        components = analysis.get("components", [])
        penalties = analysis.get("penalties", [])

        if "HTF_CONFIRMATION" not in components:
            reasons.append("HTF onayı bekleniyor")
        if "LIQUIDITY_SWEEP" not in components:
            reasons.append("Likidite avı bekleniyor")
        if "DISPLACEMENT" not in components:
            reasons.append("Displacement bekleniyor")
        if "FVG" not in components:
            reasons.append("FVG dolumu bekleniyor")
        if "MARKET_STRUCTURE" not in components:
            reasons.append("Yapı onayı bekleniyor")

        for p in penalties:
            if "OFF_HOURS" in p:
                reasons.append("Killzone dışı saat")
            elif "RANGING" in p:
                reasons.append("Yatay piyasa")

        if not reasons:
            reasons.append("Skor yetersiz, ek onay bekleniyor")

        return " | ".join(reasons[:3])

    def _get_tp_type(self, analysis, tp, direction):
        """TP seviyesinin ICT kaynağını belirle."""
        # HTF Draw on Liquidity?
        htf_liq = analysis.get("htf_liquidity", [])
        for liq in htf_liq:
            if direction == "LONG" and liq["type"] == "EQUAL_HIGHS" and not liq["swept"]:
                if abs(tp - liq["price"]) / tp < 0.005:
                    return "HTF Draw on Liquidity (4H Equal Highs)"
            elif direction == "SHORT" and liq["type"] == "EQUAL_LOWS" and not liq["swept"]:
                if abs(tp - liq["price"]) / tp < 0.005:
                    return "HTF Draw on Liquidity (4H Equal Lows)"

        # LTF liquidity?
        liq_levels = analysis.get("liquidity", [])
        for liq in liq_levels:
            if direction == "LONG" and liq["type"] == "EQUAL_HIGHS":
                if abs(tp - liq["price"]) / tp < 0.005:
                    return "Karşı Likidite (Equal Highs)"
            elif direction == "SHORT" and liq["type"] == "EQUAL_LOWS":
                if abs(tp - liq["price"]) / tp < 0.005:
                    return "Karşı Likidite (Equal Lows)"

        # Order Block?
        obs = analysis.get("order_blocks", [])
        for ob in obs:
            if direction == "LONG" and ob["type"] == "BEARISH_OB":
                if abs(tp - ob["low"]) / tp < 0.005:
                    return "Karşı Order Block (Bearish OB)"
            elif direction == "SHORT" and ob["type"] == "BULLISH_OB":
                if abs(tp - ob["high"]) / tp < 0.005:
                    return "Karşı Order Block (Bullish OB)"

        # Swing yapısı?
        structure = analysis.get("structure", {})
        if direction == "LONG":
            for sh in structure.get("swing_highs", []):
                if abs(tp - sh["price"]) / tp < 0.005:
                    return "Swing High Yapısal Hedef"
        else:
            for sl_p in structure.get("swing_lows", []):
                if abs(tp - sl_p["price"]) / tp < 0.005:
                    return "Swing Low Yapısal Hedef"

        return "Minimum R:R Hedefi"


# Global instance
ict_strategy = ICTStrategy()
