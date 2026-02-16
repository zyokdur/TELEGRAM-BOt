# =====================================================
# ICT Trading Bot - ICT Strateji Motoru
# =====================================================
# Michael J. Huddleston ICT Konseptleri implementasyonu:
# - Market Structure (BOS / CHoCH)
# - Order Blocks
# - Fair Value Gaps (FVG)
# - Liquidity Sweeps
# - Premium/Discount Zones
# - Displacement
# - Optimal Trade Entry (OTE)
# =====================================================

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
from config import ICT_PARAMS
from database import get_bot_param

logger = logging.getLogger("ICT-Bot.Strategy")


class ICTStrategy:
    """ICT Strateji Motoru - Tüm ICT konseptlerini analiz eder"""

    def __init__(self):
        self.params = self._load_params()

    def _load_params(self):
        """Veritabanından güncel parametreleri yükle, yoksa varsayılanları kullan"""
        params = {}
        for key, default_val in ICT_PARAMS.items():
            db_val = get_bot_param(key)
            params[key] = db_val if db_val is not None else default_val
        return params

    def reload_params(self):
        """Parametreleri yeniden yükle (optimizer güncellemesi sonrası)"""
        self.params = self._load_params()

    # =================== SESSION / KILLZONE ===================

    def get_session_info(self):
        """
        ICT Killzone (oturum) bilgisini hesapla.
        Kurumsal aktivite belirli saatlerde yoğunlaşır:
        - Asian Session:  00:00-08:00 UTC (düşük volatilite, likidite oluşumu)
        - London Killzone: 07:00-10:00 UTC (yüksek volatilite, ana harekeler)
        - NY Killzone:     12:00-15:00 UTC (yüksek volatilite, trend devamı)
        - London Close:    15:00-17:00 UTC (geri çekilmeler)
        - Off-hours:       17:00-00:00 UTC (düşük volatilite)
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

    # =================== RANGING MARKET TESPİTİ ===================

    def detect_ranging_market(self, df, lookback=20):
        """
        Yatay (ranging) piyasayı tespit et.
        Range-bound piyasalarda ICT sinyalleri düşük kalitelidir.
        ADX benzeri bir volatilite kontrolü uygular.
        Returns: True = ranging (sinyal üretme), False = trending (sinyal üret)
        """
        if len(df) < lookback:
            return False

        recent = df.tail(lookback)
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values

        # 1. ATR bazlı volatilite (ortalama range vs fiyat)
        avg_range = np.mean(highs - lows)
        avg_price = np.mean(closes)
        range_pct = avg_range / avg_price if avg_price > 0 else 0

        # 2. Net hareket / Toplam hareket oranı (efficiency ratio)
        net_move = abs(closes[-1] - closes[0])
        total_move = sum(abs(closes[i] - closes[i-1]) for i in range(1, len(closes)))
        efficiency = net_move / total_move if total_move > 0 else 0

        # 3. High-Low range genişliği kontrolü
        max_high = np.max(highs)
        min_low = np.min(lows)
        total_range_pct = (max_high - min_low) / avg_price if avg_price > 0 else 0

        # Ranging koşulları:
        # - Efficiency çok düşükse (fiyat ileri geri gidip geliyor)
        # - Toplam range dar ve volatilite düşükse
        is_ranging = (efficiency < 0.15 and total_range_pct < 0.02) or \
                     (efficiency < 0.10)

        if is_ranging:
            logger.debug(f"  📊 Ranging market tespit edildi: eff={efficiency:.3f}, range={total_range_pct:.4f}")

        return is_ranging

    # =================== MARKET STRUCTURE ===================

    def find_swing_points(self, df, lookback=None):
        """
        Swing High ve Swing Low noktalarını tespit et
        Swing High: lookback kadar önceki ve sonraki mumlardan yüksek
        Swing Low: lookback kadar önceki ve sonraki mumlardan düşük
        """
        if lookback is None:
            lookback = int(self.params["swing_lookback"])

        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        swing_highs = []
        swing_lows = []

        for i in range(lookback, n - lookback):
            # Swing High kontrolü
            is_swing_high = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs.append({
                    "index": i,
                    "price": highs[i],
                    "timestamp": df["timestamp"].iloc[i]
                })

            # Swing Low kontrolü
            is_swing_low = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append({
                    "index": i,
                    "price": lows[i],
                    "timestamp": df["timestamp"].iloc[i]
                })

        return swing_highs, swing_lows

    def detect_market_structure(self, df):
        """
        Piyasa yapısını analiz et:
        - Trend yönünü belirle
        - BOS (Break of Structure) tespit et
        - CHoCH (Change of Character) tespit et
        """
        swing_highs, swing_lows = self.find_swing_points(df)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {
                "trend": "NEUTRAL",
                "bos_events": [],
                "choch_events": [],
                "swing_highs": swing_highs,
                "swing_lows": swing_lows,
                "last_swing_high": None,
                "last_swing_low": None
            }

        bos_events = []
        choch_events = []
        current_trend = "NEUTRAL"
        min_displacement = self.params["bos_min_displacement"]

        # Son swing noktalarını analiz et
        all_swings = []
        for sh in swing_highs:
            all_swings.append({"type": "HIGH", **sh})
        for sl in swing_lows:
            all_swings.append({"type": "LOW", **sl})
        all_swings.sort(key=lambda x: x["index"])

        # Yapı kırılımlarını tespit et
        for i in range(2, len(all_swings)):
            current = all_swings[i]
            prev_same = None

            # Aynı türden bir önceki swing'i bul
            for j in range(i - 1, -1, -1):
                if all_swings[j]["type"] == current["type"]:
                    prev_same = all_swings[j]
                    break

            if prev_same is None:
                continue

            if current["type"] == "HIGH":
                if current["price"] > prev_same["price"]:
                    # Higher High - Yükseliş devamı
                    if current_trend == "BEARISH":
                        # CHoCH - Düşüşten yükselişe
                        displacement = (current["price"] - prev_same["price"]) / prev_same["price"]
                        if displacement > min_displacement:
                            choch_events.append({
                                "type": "BULLISH_CHOCH",
                                "index": current["index"],
                                "price": current["price"],
                                "prev_price": prev_same["price"],
                                "timestamp": current["timestamp"]
                            })
                            current_trend = "BULLISH"
                    else:
                        bos_events.append({
                            "type": "BULLISH_BOS",
                            "index": current["index"],
                            "price": current["price"],
                            "prev_price": prev_same["price"],
                            "timestamp": current["timestamp"]
                        })
                        current_trend = "BULLISH"
                else:
                    # Lower High
                    if current_trend == "BULLISH":
                        current_trend = "WEAKENING_BULL"

            elif current["type"] == "LOW":
                if current["price"] < prev_same["price"]:
                    # Lower Low - Düşüş devamı
                    if current_trend == "BULLISH":
                        displacement = (prev_same["price"] - current["price"]) / prev_same["price"]
                        if displacement > min_displacement:
                            choch_events.append({
                                "type": "BEARISH_CHOCH",
                                "index": current["index"],
                                "price": current["price"],
                                "prev_price": prev_same["price"],
                                "timestamp": current["timestamp"]
                            })
                            current_trend = "BEARISH"
                    else:
                        bos_events.append({
                            "type": "BEARISH_BOS",
                            "index": current["index"],
                            "price": current["price"],
                            "prev_price": prev_same["price"],
                            "timestamp": current["timestamp"]
                        })
                        current_trend = "BEARISH"
                else:
                    # Higher Low
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

    # =================== ORDER BLOCKS ===================

    def find_order_blocks(self, df, structure):
        """
        Order Block'ları tespit et:
        - Bullish OB: BOS öncesi son bearish mum
        - Bearish OB: BOS öncesi son bullish mum
        """
        order_blocks = []
        max_age = int(self.params["ob_max_age_candles"])
        min_body_ratio = self.params["ob_body_ratio_min"]
        current_idx = len(df) - 1

        bos_events = structure.get("bos_events", []) + structure.get("choch_events", [])

        for event in bos_events:
            event_idx = event["index"]

            # Çok eski OB'leri atla
            if current_idx - event_idx > max_age:
                continue

            if "BULLISH" in event["type"]:
                # BOS öncesi son bearish mumu bul
                for j in range(event_idx - 1, max(event_idx - 10, 0), -1):
                    if j >= len(df):
                        continue
                    candle = df.iloc[j]
                    body = abs(candle["close"] - candle["open"])
                    total_range = candle["high"] - candle["low"]

                    if total_range <= 0:
                        continue

                    body_ratio = body / total_range

                    # Bearish mum (close < open) ve yeterli gövde oranı
                    if candle["close"] < candle["open"] and body_ratio >= min_body_ratio:
                        order_blocks.append({
                            "type": "BULLISH_OB",
                            "index": j,
                            "high": candle["high"],
                            "low": candle["low"],
                            "open": candle["open"],
                            "close": candle["close"],
                            "timestamp": candle["timestamp"],
                            "mitigated": False,
                            "strength": body_ratio
                        })
                        break

            elif "BEARISH" in event["type"]:
                # BOS öncesi son bullish mumu bul
                for j in range(event_idx - 1, max(event_idx - 10, 0), -1):
                    if j >= len(df):
                        continue
                    candle = df.iloc[j]
                    body = abs(candle["close"] - candle["open"])
                    total_range = candle["high"] - candle["low"]

                    if total_range <= 0:
                        continue

                    body_ratio = body / total_range

                    if candle["close"] > candle["open"] and body_ratio >= min_body_ratio:
                        order_blocks.append({
                            "type": "BEARISH_OB",
                            "index": j,
                            "high": candle["high"],
                            "low": candle["low"],
                            "open": candle["open"],
                            "close": candle["close"],
                            "timestamp": candle["timestamp"],
                            "mitigated": False,
                            "strength": body_ratio
                        })
                        break

        # Mitigated OB kontrolü (fiyat OB bölgesinden geçmiş mi?)
        if order_blocks:
            last_price = df["close"].iloc[-1]
            for ob in order_blocks:
                if ob["type"] == "BULLISH_OB":
                    # Fiyat OB'nin altına düşmüşse mitigate olmuş
                    after_candles = df.iloc[ob["index"] + 1:]
                    if len(after_candles) > 0 and after_candles["low"].min() < ob["low"]:
                        ob["mitigated"] = True
                elif ob["type"] == "BEARISH_OB":
                    after_candles = df.iloc[ob["index"] + 1:]
                    if len(after_candles) > 0 and after_candles["high"].max() > ob["high"]:
                        ob["mitigated"] = True

        # Sadece henüz mitigate olmamış OB'leri döndür
        active_obs = [ob for ob in order_blocks if not ob["mitigated"]]
        return active_obs, order_blocks  # Hem aktif hem tüm OB'leri döndür

    # =================== BREAKER BLOCKS ===================

    def find_breaker_blocks(self, all_order_blocks, df):
        """
        Breaker Block tespiti:
        Mitigate olmuş bir OB, karşı yönde güçlü bir destek/direnç haline gelir.
        - Mitigated Bullish OB → Bearish Breaker (direnç)
        - Mitigated Bearish OB → Bullish Breaker (destek)
        Bu ICT'de yüksek olasılıklı trade setup'larından biridir.
        """
        breaker_blocks = []
        current_price = df["close"].iloc[-1]
        current_idx = len(df) - 1

        for ob in all_order_blocks:
            if not ob["mitigated"]:
                continue

            # Çok eski breaker'ları atla (max 40 mum)
            if current_idx - ob["index"] > 40:
                continue

            if ob["type"] == "BULLISH_OB":
                # Mitigated Bullish OB → Bearish Breaker (direnç olarak çalışır)
                # Fiyat bu bölgeye yaklaşırsa SHORT sinyali güçlenir
                if current_price >= ob["low"] * 0.998 and current_price <= ob["high"] * 1.005:
                    breaker_blocks.append({
                        "type": "BEARISH_BREAKER",
                        "high": ob["high"],
                        "low": ob["low"],
                        "index": ob["index"],
                        "timestamp": ob["timestamp"],
                        "original_ob": "BULLISH_OB"
                    })

            elif ob["type"] == "BEARISH_OB":
                # Mitigated Bearish OB → Bullish Breaker (destek olarak çalışır)
                if current_price >= ob["low"] * 0.995 and current_price <= ob["high"] * 1.002:
                    breaker_blocks.append({
                        "type": "BULLISH_BREAKER",
                        "high": ob["high"],
                        "low": ob["low"],
                        "index": ob["index"],
                        "timestamp": ob["timestamp"],
                        "original_ob": "BEARISH_OB"
                    })

        return breaker_blocks

    # =================== FAIR VALUE GAPS ===================

    def find_fvg(self, df):
        """
        Fair Value Gap (FVG) tespit et:
        - Bullish FVG: Candle[i-1].high < Candle[i+1].low (boşluk yukarıda)
        - Bearish FVG: Candle[i-1].low > Candle[i+1].high (boşluk aşağıda)
        """
        fvgs = []
        max_age = int(self.params["fvg_max_age_candles"])
        min_size_pct = self.params["fvg_min_size_pct"]
        n = len(df)
        current_idx = n - 1

        for i in range(1, n - 1):
            # Çok eski FVG'leri atla
            if current_idx - i > max_age:
                continue

            prev_candle = df.iloc[i - 1]
            curr_candle = df.iloc[i]
            next_candle = df.iloc[i + 1]

            mid_price = curr_candle["close"]
            if mid_price <= 0:
                continue

            # Bullish FVG
            if prev_candle["high"] < next_candle["low"]:
                gap_size = next_candle["low"] - prev_candle["high"]
                gap_pct = gap_size / mid_price

                if gap_pct >= min_size_pct:
                    fvg_high = next_candle["low"]
                    fvg_low = prev_candle["high"]

                    # FVG doldurulmuş mu kontrol et
                    filled = False
                    if i + 2 < n:
                        after = df.iloc[i + 2:]
                        if len(after) > 0 and after["low"].min() <= fvg_low:
                            filled = True

                    if not filled:
                        fvgs.append({
                            "type": "BULLISH_FVG",
                            "index": i,
                            "high": fvg_high,
                            "low": fvg_low,
                            "size_pct": round(gap_pct * 100, 4),
                            "timestamp": curr_candle["timestamp"],
                            "filled": False
                        })

            # Bearish FVG
            if prev_candle["low"] > next_candle["high"]:
                gap_size = prev_candle["low"] - next_candle["high"]
                gap_pct = gap_size / mid_price

                if gap_pct >= min_size_pct:
                    fvg_high = prev_candle["low"]
                    fvg_low = next_candle["high"]

                    filled = False
                    if i + 2 < n:
                        after = df.iloc[i + 2:]
                        if len(after) > 0 and after["high"].max() >= fvg_high:
                            filled = True

                    if not filled:
                        fvgs.append({
                            "type": "BEARISH_FVG",
                            "index": i,
                            "high": fvg_high,
                            "low": fvg_low,
                            "size_pct": round(gap_pct * 100, 4),
                            "timestamp": curr_candle["timestamp"],
                            "filled": False
                        })

        return fvgs

    # =================== LIQUIDITY ===================

    def find_liquidity_levels(self, df):
        """
        Likidite seviyelerini tespit et:
        - Equal Highs (eşit tepeler) -> Üstte likidite
        - Equal Lows (eşit dipler) -> Altta likidite
        """
        tolerance = self.params["liquidity_equal_tolerance"]
        min_touches = int(self.params["liquidity_min_touches"])
        swing_highs, swing_lows = self.find_swing_points(df)

        liquidity_levels = []

        # Equal Highs
        for i, sh in enumerate(swing_highs):
            touches = 1
            touched_indices = [sh["index"]]
            for j in range(i + 1, len(swing_highs)):
                if abs(swing_highs[j]["price"] - sh["price"]) / sh["price"] <= tolerance:
                    touches += 1
                    touched_indices.append(swing_highs[j]["index"])

            if touches >= min_touches:
                # Bu seviye zaten listeye eklenmemişse
                already_exists = False
                for ll in liquidity_levels:
                    if ll["type"] == "EQUAL_HIGHS" and abs(ll["price"] - sh["price"]) / sh["price"] <= tolerance:
                        already_exists = True
                        break

                if not already_exists:
                    # Sweep olmuş mu kontrol et
                    swept = False
                    max_idx = max(touched_indices)
                    if max_idx + 1 < len(df):
                        after_price = df.iloc[max_idx + 1:]["high"].max()
                        if after_price > sh["price"] * (1 + tolerance):
                            swept = True

                    liquidity_levels.append({
                        "type": "EQUAL_HIGHS",
                        "price": sh["price"],
                        "touches": touches,
                        "indices": touched_indices,
                        "swept": swept,
                        "side": "SELL"  # Üstte likidite = satıcı likiditesi
                    })

        # Equal Lows
        for i, sl in enumerate(swing_lows):
            touches = 1
            touched_indices = [sl["index"]]
            for j in range(i + 1, len(swing_lows)):
                if abs(swing_lows[j]["price"] - sl["price"]) / sl["price"] <= tolerance:
                    touches += 1
                    touched_indices.append(swing_lows[j]["index"])

            if touches >= min_touches:
                already_exists = False
                for ll in liquidity_levels:
                    if ll["type"] == "EQUAL_LOWS" and abs(ll["price"] - sl["price"]) / sl["price"] <= tolerance:
                        already_exists = True
                        break

                if not already_exists:
                    swept = False
                    max_idx = max(touched_indices)
                    if max_idx + 1 < len(df):
                        after_price = df.iloc[max_idx + 1:]["low"].min()
                        if after_price < sl["price"] * (1 - tolerance):
                            swept = True

                    liquidity_levels.append({
                        "type": "EQUAL_LOWS",
                        "price": sl["price"],
                        "touches": touches,
                        "indices": touched_indices,
                        "swept": swept,
                        "side": "BUY"
                    })

        return liquidity_levels

    # =================== DISPLACEMENT ===================

    def detect_displacement(self, df, lookback=10):
        """
        Displacement (güçlü momentum mumları) tespit et
        Büyük gövdeli, güçlü yönlü mumlar
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
                    "type": f"{direction}_DISPLACEMENT",
                    "index": i,
                    "body_ratio": round(body_ratio, 3),
                    "size_pct": round(size_pct * 100, 3),
                    "direction": direction,
                    "timestamp": candle["timestamp"]
                })

        return displacements

    # =================== PREMIUM / DISCOUNT ===================

    def calculate_premium_discount(self, df, structure):
        """
        Premium/Discount bölgelerini hesapla
        Son swing high ve swing low arasındaki 50% seviyesi (equilibrium)
        Premium: Üst yarı (satış bölgesi)
        Discount: Alt yarı (alış bölgesi)
        """
        last_high = structure.get("last_swing_high")
        last_low = structure.get("last_swing_low")

        if not last_high or not last_low:
            return None

        high_price = last_high["price"]
        low_price = last_low["price"]
        equilibrium = (high_price + low_price) / 2
        current_price = df["close"].iloc[-1]

        # OTE bölgesi (Fibonacci 0.618-0.786)
        fib_range = high_price - low_price
        ote_high = low_price + fib_range * 0.786
        ote_low = low_price + fib_range * 0.618

        zone = "PREMIUM" if current_price > equilibrium else "DISCOUNT"

        return {
            "high": high_price,
            "low": low_price,
            "equilibrium": equilibrium,
            "current_price": current_price,
            "zone": zone,
            "ote_high": ote_high,
            "ote_low": ote_low,
            "in_ote": ote_low <= current_price <= ote_high,
            "premium_level": round((current_price - low_price) / (high_price - low_price) * 100, 1)
                             if high_price != low_price else 50
        }

    # =================== CONFLUENCE SCORING ===================

    def calculate_confluence(self, df, multi_tf_data=None):
        """
        Tüm ICT konseptlerini analiz edip confluent skor hesapla.
        
        İYİLEŞTİRMELER:
        - Session/Killzone ağırlığı
        - Breaker Block tespiti
        - Recency weighting (yeni OB/FVG daha değerli)
        - Displacement zorunluluğu güçlendirildi
        - Ranging market cezası
        - HTF + MTF çift onay
        - WEAKENING trend cezası
        """
        analysis = {}
        components_triggered = []
        score = 0
        max_score = 0
        penalties = []

        current_price = df["close"].iloc[-1]
        current_idx = len(df) - 1

        # ===== RANGING MARKET KONTROLÜ =====
        is_ranging = self.detect_ranging_market(df)
        analysis["is_ranging"] = is_ranging

        # ===== SESSION / KILLZONE =====
        session_info = self.get_session_info()
        analysis["session"] = session_info

        # 1. Market Structure Analizi (25 puan)
        structure = self.detect_market_structure(df)
        analysis["structure"] = structure
        weight_structure = 25
        max_score += weight_structure

        if structure["trend"] in ["BULLISH", "BEARISH"]:
            score += weight_structure
            components_triggered.append("MARKET_STRUCTURE")
        elif structure["trend"] in ["WEAKENING_BULL", "WEAKENING_BEAR"]:
            # WEAKENING trendler daha düşük puan (tam onay yok)
            score += weight_structure * 0.3
            penalties.append("WEAKENING_TREND(-7)")

        # 2. Order Blocks (20 puan) + Recency Weighting
        active_obs, all_obs_raw = self.find_order_blocks(df, structure)
        analysis["order_blocks"] = active_obs
        analysis["all_order_blocks"] = all_obs_raw
        weight_ob = 20
        max_score += weight_ob

        relevant_obs = []
        for ob in active_obs:
            # Recency weighting: Son 10 mumda oluşan OB'ler daha değerli
            recency_factor = 1.0
            age = current_idx - ob["index"]
            if age <= 5:
                recency_factor = 1.0
            elif age <= 15:
                recency_factor = 0.8
            elif age <= 25:
                recency_factor = 0.6
            else:
                recency_factor = 0.4

            if ob["type"] == "BULLISH_OB" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                if ob["low"] <= current_price <= ob["high"] * 1.005:
                    relevant_obs.append(ob)
                    score += weight_ob * recency_factor
                    components_triggered.append("ORDER_BLOCK")
                    break
                elif current_price < ob["high"] * 1.02 and current_price > ob["low"] * 0.99:
                    score += weight_ob * 0.4 * recency_factor
            elif ob["type"] == "BEARISH_OB" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                if ob["low"] <= current_price <= ob["high"] * 1.005:
                    relevant_obs.append(ob)
                    score += weight_ob * recency_factor
                    components_triggered.append("ORDER_BLOCK")
                    break
                elif current_price > ob["low"] * 0.98 and current_price < ob["high"] * 1.01:
                    score += weight_ob * 0.4 * recency_factor

        analysis["relevant_obs"] = relevant_obs

        # 2b. Breaker Blocks (7 bonus puan)
        breaker_blocks = self.find_breaker_blocks(all_obs_raw, df)
        analysis["breaker_blocks"] = breaker_blocks

        for bb in breaker_blocks:
            if bb["type"] == "BULLISH_BREAKER" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                score += 7
                components_triggered.append("BREAKER_BLOCK")
                break
            elif bb["type"] == "BEARISH_BREAKER" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                score += 7
                components_triggered.append("BREAKER_BLOCK")
                break

        # 3. Fair Value Gaps (15 puan) + Recency Weighting
        fvgs = self.find_fvg(df)
        analysis["fvgs"] = fvgs
        weight_fvg = 15
        max_score += weight_fvg

        relevant_fvgs = []
        for fvg in fvgs:
            # Recency weighting
            fvg_age = current_idx - fvg["index"]
            fvg_recency = 1.0 if fvg_age <= 8 else (0.7 if fvg_age <= 15 else 0.4)

            if fvg["type"] == "BULLISH_FVG" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                if fvg["low"] * 0.998 <= current_price <= fvg["high"] * 1.002:
                    relevant_fvgs.append(fvg)
                    score += weight_fvg * fvg_recency
                    components_triggered.append("FVG")
                    break
            elif fvg["type"] == "BEARISH_FVG" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                if fvg["low"] * 0.998 <= current_price <= fvg["high"] * 1.002:
                    relevant_fvgs.append(fvg)
                    score += weight_fvg * fvg_recency
                    components_triggered.append("FVG")
                    break

        analysis["relevant_fvgs"] = relevant_fvgs

        # 4. Liquidity + Sweep-MSS A+ Setup (15 puan)
        liquidity = self.find_liquidity_levels(df)
        analysis["liquidity"] = liquidity
        weight_liq = 15
        max_score += weight_liq

        sweep_mss_detected = False
        for liq in liquidity:
            if liq["swept"]:
                if liq["type"] == "EQUAL_LOWS" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                    score += weight_liq
                    components_triggered.append("LIQUIDITY_SWEEP")

                    # A+ Setup: Sweep sonrası MSS (BOS/CHoCH) var mı?
                    sweep_idx = max(liq["indices"])
                    for bos in structure.get("bos_events", []) + structure.get("choch_events", []):
                        if bos["index"] > sweep_idx and "BULLISH" in bos["type"]:
                            sweep_mss_detected = True
                            break
                    break

                elif liq["type"] == "EQUAL_HIGHS" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                    score += weight_liq
                    components_triggered.append("LIQUIDITY_SWEEP")

                    sweep_idx = max(liq["indices"])
                    for bos in structure.get("bos_events", []) + structure.get("choch_events", []):
                        if bos["index"] > sweep_idx and "BEARISH" in bos["type"]:
                            sweep_mss_detected = True
                            break
                    break

        # Sweep + MSS = A+ Setup → ekstra 10 bonus
        if sweep_mss_detected:
            score += 10
            components_triggered.append("SWEEP_MSS_A_PLUS")
            logger.debug(f"  🅰️ A+ SETUP: Likidite sweep + MSS tespit edildi")
        analysis["sweep_mss"] = sweep_mss_detected

        # 5. Displacement (15 puan → önem artırıldı, ICT'de kritik onay)
        displacements = self.detect_displacement(df)
        analysis["displacements"] = displacements
        weight_disp = 15  # 10'dan 15'e çıkarıldı
        max_score += weight_disp

        has_displacement = False
        if displacements:
            last_disp = displacements[-1]
            if last_disp["direction"] == "BULLISH" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                score += weight_disp
                components_triggered.append("DISPLACEMENT")
                has_displacement = True
            elif last_disp["direction"] == "BEARISH" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                score += weight_disp
                components_triggered.append("DISPLACEMENT")
                has_displacement = True

        # Displacement yoksa ceza (kurumsal aktivite onayı eksik)
        if not has_displacement:
            penalties.append("NO_DISPLACEMENT(-8)")
            score -= 8

        # 6. Premium/Discount (15 puan)
        pd_zone = self.calculate_premium_discount(df, structure)
        analysis["premium_discount"] = pd_zone
        weight_pd = 15
        max_score += weight_pd

        if pd_zone:
            if pd_zone["zone"] == "DISCOUNT" and structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
                score += weight_pd * 0.7
                if pd_zone["in_ote"]:
                    score += weight_pd * 0.3
                    components_triggered.append("OTE")
                components_triggered.append("DISCOUNT_ZONE")
            elif pd_zone["zone"] == "PREMIUM" and structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
                score += weight_pd * 0.7
                if pd_zone["in_ote"]:
                    score += weight_pd * 0.3
                    components_triggered.append("OTE")
                components_triggered.append("PREMIUM_ZONE")

        # ===== HTF BIAS HARD FİLTRE + MTF ONAY =====
        htf_aligned = False
        mtf_aligned = False
        htf_bias_block = False   # HTF karşı yöndeyse sinyali tamamen engelle

        if multi_tf_data:
            # 4H (HTF) — "Bias" belirler, LTF buna uymalı
            if "4H" in multi_tf_data and not multi_tf_data["4H"].empty:
                htf_structure = self.detect_market_structure(multi_tf_data["4H"])
                htf_liquidity = self.find_liquidity_levels(multi_tf_data["4H"])
                analysis["htf_trend"] = htf_structure["trend"]
                analysis["htf_structure"] = htf_structure
                analysis["htf_liquidity"] = htf_liquidity

                if htf_structure["trend"] == structure["trend"]:
                    htf_aligned = True
                    score += 5
                    components_triggered.append("HTF_CONFIRMATION")
                elif htf_structure["trend"] in ["BULLISH", "BEARISH"] and \
                     structure["trend"] in ["BULLISH", "BEARISH"] and \
                     htf_structure["trend"] != structure["trend"]:
                    # 4H BEARISH iken LTF LONG → HARD BLOCK
                    # 4H BULLISH iken LTF SHORT → HARD BLOCK
                    htf_bias_block = True
                    score -= 15
                    penalties.append("HTF_BIAS_BLOCK(-15)")
                    logger.debug(f"  ⛔ HTF Bias Block: 4H={htf_structure['trend']} vs LTF={structure['trend']}")

            # 1H (MTF) onayı → +3 bonus
            if "1H" in multi_tf_data and not multi_tf_data["1H"].empty:
                mtf_structure = self.detect_market_structure(multi_tf_data["1H"])
                analysis["mtf_trend"] = mtf_structure["trend"]
                if mtf_structure["trend"] == structure["trend"]:
                    mtf_aligned = True
                    score += 3
                    components_triggered.append("MTF_CONFIRMATION")

        analysis["htf_bias_block"] = htf_bias_block

        # Triple timeframe alignment bonus
        if htf_aligned and mtf_aligned:
            score += 3
            components_triggered.append("TRIPLE_TF_ALIGNMENT")

        # ===== SESSION QUALITY BONUSU =====
        session_quality = session_info["quality"]
        if session_quality >= 0.8:
            score += 5
            components_triggered.append("KILLZONE_ACTIVE")
        elif session_quality <= 0.3:
            penalties.append("OFF_HOURS(-5)")
            score -= 5

        # ===== RANGING MARKET CEZASI =====
        if is_ranging:
            score -= 15
            penalties.append("RANGING_MARKET(-15)")

        # Normalize et (0-100), minimum 0
        score = max(0, score)
        confluence_score = min(100, round((score / max_score) * 100, 1)) if max_score > 0 else 0

        # Yön belirle
        direction = None
        if structure["trend"] in ["BULLISH", "WEAKENING_BEAR"]:
            direction = "LONG"
        elif structure["trend"] in ["BEARISH", "WEAKENING_BULL"]:
            direction = "SHORT"

        analysis["confluence_score"] = confluence_score
        analysis["direction"] = direction
        analysis["components"] = list(set(components_triggered))
        analysis["penalties"] = penalties
        analysis["current_price"] = current_price

        return analysis

    # =================== SİNYAL ÜRETİMİ ===================

    def generate_signal(self, symbol, df, multi_tf_data=None):
        """
        Analiz sonuçlarına göre sinyal üret veya izleme listesine al
        Returns:
            - signal dict (sinyal üretildiyse)
            - watch dict (izlemeye alınsın)
            - None (bir şey yok)
        """
        if df.empty or len(df) < 20:
            return None

        analysis = self.calculate_confluence(df, multi_tf_data)
        confluence_score = analysis["confluence_score"]
        direction = analysis["direction"]
        current_price = analysis["current_price"]
        min_confluence = self.params["min_confluence_score"]
        min_confidence = self.params["min_confidence"]

        if direction is None:
            return None

        # ===== HTF BIAS HARD FİLTRE =====
        # 4H trend karşı yöndeyse → işlem açma, boşuna SL yeme
        if analysis.get("htf_bias_block"):
            htf_trend = analysis.get("htf_trend", "?")
            logger.debug(f"  ⛔ {symbol} HTF Bias Block: 4H={htf_trend} vs sinyal={direction}")
            return None

        # Ranging market kontrolü → sinyal üretme
        if analysis.get("is_ranging"):
            logger.debug(f"  {symbol} ranging market - sinyal üretilmiyor")
            return None

        # Güven skoru hesapla (confluence + ek faktörler)
        confidence = self._calculate_confidence(analysis)

        # Entry, SL, TP hesapla
        entry, sl, tp = self._calculate_levels(analysis, df)

        if entry is None or sl is None or tp is None:
            return None

        # ===== YÖN DOĞRULAMASI =====
        # LONG: SL < Entry < TP olmalı
        # SHORT: TP < Entry < SL olmalı
        if direction == "LONG":
            if sl >= entry or tp <= entry:
                logger.warning(f"❌ {symbol} LONG seviyeleri ters: Entry={entry} SL={sl} TP={tp}")
                return None
            risk = entry - sl
            reward = tp - entry
        elif direction == "SHORT":
            if sl <= entry or tp >= entry:
                logger.warning(f"❌ {symbol} SHORT seviyeleri ters: Entry={entry} SL={sl} TP={tp}")
                return None
            risk = sl - entry
            reward = entry - tp
        else:
            return None

        if risk <= 0:
            return None

        # Minimum SL mesafesi kontrolü (%0.3 - çok yakın SL volatilitede vurulur)
        sl_distance_pct = risk / entry
        if sl_distance_pct < 0.003:
            logger.debug(f"  {symbol} SL çok yakın: %{sl_distance_pct*100:.2f} < %0.3")
            return None

        rr_ratio = reward / risk

        if rr_ratio < 1.5:
            return None  # Minimum 1.5 RR

        result = {
            "symbol": symbol,
            "direction": direction,
            "entry": round(entry, 8),
            "sl": round(sl, 8),
            "tp": round(tp, 8),
            "current_price": round(current_price, 8),
            "confluence_score": confluence_score,
            "confidence": confidence,
            "components": analysis["components"],
            "penalties": analysis.get("penalties", []),
            "session": analysis.get("session", {}).get("label", ""),
            "rr_ratio": round(rr_ratio, 2),
            "entry_type": self._get_entry_type(analysis, entry, current_price, direction),
            "sl_type": self._get_sl_type(analysis, sl, direction),
            "tp_type": self._get_tp_type(analysis, tp, direction),
            "analysis": analysis
        }

        # Sinyal mi, izleme mi?
        if confluence_score >= min_confluence and confidence >= min_confidence:
            result["action"] = "SIGNAL"
            session_label = analysis.get("session", {}).get("label", "")
            logger.info(f"🎯 SİNYAL: {symbol} {direction} | Entry: {entry} | SL: {sl} | TP: {tp} | "
                       f"Conf: {confidence}% | Score: {confluence_score} | Session: {session_label}")
        elif confluence_score >= min_confluence * 0.7:
            result["action"] = "WATCH"
            result["watch_reason"] = self._get_watch_reason(analysis)
            logger.info(f"👀 İZLEME: {symbol} {direction} | Score: {confluence_score} | "
                       f"Conf: {confidence}% | Sebep: {result['watch_reason']}")
        else:
            return None

        return result

    def _calculate_confidence(self, analysis):
        """
        Güven skoru hesapla (0-100)
        Confluence + ek kalite faktörleri + CEZA SİSTEMİ
        
        Eksik kritik bileşenler ceza alır:
        - Displacement yoksa: -10
        - OB veya FVG yoksa: -5
        - Uygun bölgede değilse: -5
        - Ranging market: -10
        """
        base = analysis["confluence_score"]
        bonus = 0
        penalty = 0

        components = analysis["components"]
        penalties = analysis.get("penalties", [])

        # === BONUSLAR ===
        # Birden fazla bileşen tetiklendiyse güven artar
        comp_count = len(components)
        if comp_count >= 5:
            bonus += 12
        elif comp_count >= 4:
            bonus += 8
        elif comp_count >= 3:
            bonus += 4

        # Güçlü trend varsa bonus
        structure = analysis["structure"]
        if structure["trend"] in ["BULLISH", "BEARISH"]:
            bonus += 5

        # Displacement varsa bonus
        if "DISPLACEMENT" in components:
            bonus += 5

        # HTF onayı varsa bonus
        if "HTF_CONFIRMATION" in components:
            bonus += 5

        # Triple TF alignment → ekstra bonus
        if "TRIPLE_TF_ALIGNMENT" in components:
            bonus += 5

        # Killzone aktifse bonus
        if "KILLZONE_ACTIVE" in components:
            bonus += 3

        # Breaker Block varsa bonus
        if "BREAKER_BLOCK" in components:
            bonus += 5

        # Sweep + MSS (A+ Setup) varsa → en güçlü sinyal
        if "SWEEP_MSS_A_PLUS" in components:
            bonus += 10

        # HTF Bias Block → ciddi ceza
        if analysis.get("htf_bias_block"):
            penalty += 15

        # === CEZALAR ===
        # Displacement yoksa → kurumsal aktivite onayı eksik
        if "DISPLACEMENT" not in components:
            penalty += 10

        # OB ve FVG ikisi de yoksa → giriş noktası belirsiz
        if "ORDER_BLOCK" not in components and "FVG" not in components:
            penalty += 8

        # Uygun bölgede değilse
        if "DISCOUNT_ZONE" not in components and "PREMIUM_ZONE" not in components and "OTE" not in components:
            penalty += 5

        # WEAKENING trend cezası
        if structure["trend"] in ["WEAKENING_BULL", "WEAKENING_BEAR"]:
            penalty += 5

        # Ranging market cezası
        if analysis.get("is_ranging"):
            penalty += 10

        # Off-hours cezası
        session = analysis.get("session", {})
        if session.get("quality", 1.0) <= 0.3:
            penalty += 5

        confidence = max(0, min(100, base + bonus - penalty))
        return round(confidence, 1)

    def _calculate_levels(self, analysis, df):
        """
        ICT'ye uygun Entry, Stop Loss ve Take Profit hesapla.

        ENTRY: OB seviyesi > FVG ortası > OTE bölgesi > Güncel fiyat
        SL:    OB gövdesinin altı/üstü > Likidite sweep mumu > Swing yapısı
        TP:    Karşı likidite havuzu > Karşı OB > Swing high/low yapısal hedef
        """
        current_price = analysis["current_price"]
        direction = analysis["direction"]
        structure = analysis["structure"]
        pd_zone = analysis.get("premium_discount")

        if direction == "LONG":
            entry = self._calc_long_entry(analysis, current_price, pd_zone)
            sl = self._calc_long_sl(analysis, df, entry)
            tp = self._calc_long_tp(analysis, df, entry, sl)
        elif direction == "SHORT":
            entry = self._calc_short_entry(analysis, current_price, pd_zone)
            sl = self._calc_short_sl(analysis, df, entry)
            tp = self._calc_short_tp(analysis, df, entry, sl)
        else:
            return None, None, None

        if entry is None or sl is None or tp is None:
            return None, None, None

        return entry, sl, tp

    # =================== ICT ENTRY HESAPLAMA ===================

    def _calc_long_entry(self, analysis, current_price, pd_zone):
        """
        LONG Entry - ICT öncelik sırası:
        1. Bullish OB'nin üst kenarı (fiyat OB'ye geri çekilecek)
        2. Bullish FVG ortası (boşluk dolumu)
        3. OTE bölgesi (Fib 0.705 - 0.618 ile 0.786 ortası)
        4. Güncel fiyat (yapısal onay varsa)
        """
        # 1. Aktif Bullish Order Block varsa → OB'nin üst kenarı (close seviyesi)
        if analysis.get("relevant_obs"):
            ob = analysis["relevant_obs"][0]
            if ob["type"] == "BULLISH_OB":
                # OB'nin open seviyesi (bearish mumun open'ı = OB'nin üst kenarı)
                ob_entry = ob["open"]
                # Fiyat OB'ye yeterince yakınsa bu entry'yi kullan
                if current_price <= ob_entry * 1.005:
                    return ob_entry

        # 2. Aktif Bullish FVG varsa → FVG'nin orta noktası
        if analysis.get("relevant_fvgs"):
            fvg = analysis["relevant_fvgs"][0]
            if fvg["type"] == "BULLISH_FVG":
                fvg_mid = (fvg["high"] + fvg["low"]) / 2
                if current_price <= fvg["high"] * 1.003:
                    return fvg_mid

        # 3. OTE bölgesi (Fibonacci 0.618-0.786 ortası)
        if pd_zone and pd_zone.get("in_ote"):
            ote_mid = (pd_zone["ote_high"] + pd_zone["ote_low"]) / 2
            return ote_mid

        # 4. Fiyat discount bölgesindeyse → current price kabul edilebilir
        if pd_zone and pd_zone["zone"] == "DISCOUNT":
            return current_price

        # 5. Güçlü displacement + yapısal onay varsa → current price
        return current_price

    def _calc_short_entry(self, analysis, current_price, pd_zone):
        """
        SHORT Entry - ICT öncelik sırası:
        1. Bearish OB'nin alt kenarı (fiyat OB'ye geri çekilecek)
        2. Bearish FVG ortası
        3. OTE bölgesi (üstten fib)
        4. Güncel fiyat
        """
        # 1. Aktif Bearish Order Block
        if analysis.get("relevant_obs"):
            ob = analysis["relevant_obs"][0]
            if ob["type"] == "BEARISH_OB":
                ob_entry = ob["open"]  # Bullish mumun open'ı = OB'nin alt kenarı
                if current_price >= ob_entry * 0.995:
                    return ob_entry

        # 2. Aktif Bearish FVG ortası
        if analysis.get("relevant_fvgs"):
            fvg = analysis["relevant_fvgs"][0]
            if fvg["type"] == "BEARISH_FVG":
                fvg_mid = (fvg["high"] + fvg["low"]) / 2
                if current_price >= fvg["low"] * 0.997:
                    return fvg_mid

        # 3. OTE bölgesi (premium taraftan)
        if pd_zone and pd_zone.get("in_ote"):
            ote_mid = (pd_zone["ote_high"] + pd_zone["ote_low"]) / 2
            return ote_mid

        # 4. Premium bölgesindeyse
        if pd_zone and pd_zone["zone"] == "PREMIUM":
            return current_price

        return current_price

    # =================== ICT STOP LOSS HESAPLAMA ===================

    def _calc_long_sl(self, analysis, df, entry):
        """
        LONG SL - ICT yapısal seviyeleri kullanır:
        1. Kullanılan OB'nin tam low seviyesinin altı (OB invalidation)
        2. Likidite sweep mumunun low'unun altı
        3. Son swing low'un altı (yapısal invalidation)
        Sabit yüzde KULLANILMAZ - her zaman yapısal seviye.
        """
        sl_candidates = []

        # 1. Bullish OB'nin low'u → OB invalidation seviyesi
        if analysis.get("relevant_obs"):
            ob = analysis["relevant_obs"][0]
            if ob["type"] == "BULLISH_OB":
                # OB'nin low'unun biraz altı (wick buffer)
                ob_sl = ob["low"] * 0.997
                sl_candidates.append(("OB_LOW", ob_sl))

        # 2. Likidite sweep yapılan mumun low'u
        liquidity = analysis.get("liquidity", [])
        for liq in liquidity:
            if liq["type"] == "EQUAL_LOWS" and liq["swept"]:
                # Sweep edilen seviyenin biraz altı
                sweep_sl = liq["price"] * 0.996
                sl_candidates.append(("LIQ_SWEEP", sweep_sl))

        # 3. Son swing low (yapısal invalidation)
        structure = analysis["structure"]
        if structure["last_swing_low"]:
            swing_sl = structure["last_swing_low"]["price"] * 0.997
            sl_candidates.append(("SWING_LOW", swing_sl))

        if not sl_candidates:
            return None

        # En yakın yapısal SL'yi seç (entry'ye en yakın ama altında olan)
        valid_sls = [(name, price) for name, price in sl_candidates if price < entry]
        if not valid_sls:
            return None

        # En yakın olanı seç (gereksiz büyük SL'den kaçın)
        best_sl = max(valid_sls, key=lambda x: x[1])

        # SL mesafesi entry'nin %8'inden fazlaysa, en yakın yapısal seviyeyi kullan
        sl_distance_pct = abs(entry - best_sl[1]) / entry
        if sl_distance_pct > 0.08:
            # Çok uzak, daha yakın bir yapısal seviye ara
            closer = [s for s in valid_sls if abs(entry - s[1]) / entry <= 0.04]
            if closer:
                best_sl = max(closer, key=lambda x: x[1])
            else:
                # Hiç yakın yapısal seviye yoksa None dön (sinyal üretme)
                return None

        logger.debug(f"  LONG SL: {best_sl[0]} @ {best_sl[1]:.8f}")
        return best_sl[1]

    def _calc_short_sl(self, analysis, df, entry):
        """
        SHORT SL - ICT yapısal seviyeleri:
        1. Bearish OB'nin high'ının üstü (OB invalidation)
        2. Likidite sweep mumunun high'ının üstü
        3. Son swing high'ın üstü
        """
        sl_candidates = []

        # 1. Bearish OB high
        if analysis.get("relevant_obs"):
            ob = analysis["relevant_obs"][0]
            if ob["type"] == "BEARISH_OB":
                ob_sl = ob["high"] * 1.003
                sl_candidates.append(("OB_HIGH", ob_sl))

        # 2. Likidite sweep
        liquidity = analysis.get("liquidity", [])
        for liq in liquidity:
            if liq["type"] == "EQUAL_HIGHS" and liq["swept"]:
                sweep_sl = liq["price"] * 1.004
                sl_candidates.append(("LIQ_SWEEP", sweep_sl))

        # 3. Son swing high
        structure = analysis["structure"]
        if structure["last_swing_high"]:
            swing_sl = structure["last_swing_high"]["price"] * 1.003
            sl_candidates.append(("SWING_HIGH", swing_sl))

        if not sl_candidates:
            return None

        valid_sls = [(name, price) for name, price in sl_candidates if price > entry]
        if not valid_sls:
            return None

        # En yakın olanı seç
        best_sl = min(valid_sls, key=lambda x: x[1])

        sl_distance_pct = abs(best_sl[1] - entry) / entry
        if sl_distance_pct > 0.08:
            closer = [s for s in valid_sls if abs(s[1] - entry) / entry <= 0.04]
            if closer:
                best_sl = min(closer, key=lambda x: x[1])
            else:
                return None

        logger.debug(f"  SHORT SL: {best_sl[0]} @ {best_sl[1]:.8f}")
        return best_sl[1]

    # =================== ICT TAKE PROFIT HESAPLAMA ===================

    def _calc_long_tp(self, analysis, df, entry, sl):
        """
        LONG TP - ICT Draw on Liquidity (karşı likiditeyi hedefle):
        1. HTF (4H) karşı likiditesi (en güçlü mıknatıs)
        2. LTF karşı taraf likiditesi (equal highs)
        3. Bearish Order Block seviyesi (karşı OB)
        4. Son swing high (yapısal direnç)
        5. Fallback: Minimum R:R
        """
        tp_candidates = []
        structure = analysis["structure"]

        # 0. HTF (4H) Draw on Liquidity → en güçlü hedef
        htf_liquidity = analysis.get("htf_liquidity", [])
        for liq in htf_liquidity:
            if liq["type"] == "EQUAL_HIGHS" and not liq["swept"]:
                if liq["price"] > entry:
                    tp_candidates.append(("HTF_DRAW_ON_LIQ", liq["price"] * 0.999))

        # 1. LTF Sell-side liquidity (equal highs) → ana hedef
        liquidity = analysis.get("liquidity", [])
        for liq in liquidity:
            if liq["type"] == "EQUAL_HIGHS" and not liq["swept"]:
                if liq["price"] > entry:
                    tp_candidates.append(("LIQUIDITY_HIGHS", liq["price"] * 0.999))

        # 2. Bearish Order Blocks (karşı OB) → fiyat burada tepki verir
        all_obs = analysis.get("order_blocks", [])
        for ob in all_obs:
            if ob["type"] == "BEARISH_OB" and ob["low"] > entry:
                tp_candidates.append(("OPPOSING_OB", ob["low"]))

        # 3. Son swing high → yapısal direnç
        if structure["last_swing_high"] and structure["last_swing_high"]["price"] > entry:
            tp_candidates.append(("SWING_HIGH", structure["last_swing_high"]["price"] * 0.998))

        # 4. Önceki swing high'lar
        for sh in structure.get("swing_highs", []):
            if sh["price"] > entry * 1.005:
                tp_candidates.append(("PREV_SWING_HIGH", sh["price"] * 0.998))

        if not tp_candidates:
            # Fallback: Yapısal hedef bulunamazsa minimum R:R ile hesapla
            if sl is not None and sl < entry:
                risk = entry - sl
                min_tp_ratio = self.params.get("default_tp_ratio", 2.5)
                return entry + (risk * min_tp_ratio)
            return None

        # En yakın mantıklı hedefi seç
        # Minimum 1.5 R:R sağlayan en yakın yapısal hedef
        risk = entry - sl if sl else entry * 0.015
        min_reward = risk * 1.5

        valid_tps = [(name, price) for name, price in tp_candidates
                     if (price - entry) >= min_reward]

        if valid_tps:
            # En yakın yapısal hedefi seç (muhafazakar yaklaşım)
            best_tp = min(valid_tps, key=lambda x: x[1])
            logger.debug(f"  LONG TP: {best_tp[0]} @ {best_tp[1]:.8f}")
            return best_tp[1]

        # Hiçbir yapısal hedef 1.5 RR sağlamıyorsa, en yüksek hedefi dene
        if tp_candidates:
            best_tp = max(tp_candidates, key=lambda x: x[1])
            if (best_tp[1] - entry) > risk * 1.0:  # En az 1:1
                return best_tp[1]

        # Gerçekten hiçbir hedef yoksa minimum R:R kullan
        min_tp_ratio = self.params.get("default_tp_ratio", 2.5)
        return entry + (risk * min_tp_ratio)

    def _calc_short_tp(self, analysis, df, entry, sl):
        """
        SHORT TP - ICT Draw on Liquidity (karşı likiditeyi hedefle):
        1. HTF (4H) karşı likiditesi (en güçlü mıknatıs)
        2. LTF karşı taraf likiditesi (equal lows)
        3. Bullish Order Block seviyesi (karşı OB)
        4. Son swing low (yapısal destek)
        5. Fallback: Minimum R:R
        """
        tp_candidates = []
        structure = analysis["structure"]

        # 0. HTF (4H) Draw on Liquidity → en güçlü hedef
        htf_liquidity = analysis.get("htf_liquidity", [])
        for liq in htf_liquidity:
            if liq["type"] == "EQUAL_LOWS" and not liq["swept"]:
                if liq["price"] < entry:
                    tp_candidates.append(("HTF_DRAW_ON_LIQ", liq["price"] * 1.001))

        # 1. LTF Buy-side liquidity (equal lows)
        liquidity = analysis.get("liquidity", [])
        for liq in liquidity:
            if liq["type"] == "EQUAL_LOWS" and not liq["swept"]:
                if liq["price"] < entry:
                    tp_candidates.append(("LIQUIDITY_LOWS", liq["price"] * 1.001))

        # 2. Bullish Order Blocks (karşı OB)
        all_obs = analysis.get("order_blocks", [])
        for ob in all_obs:
            if ob["type"] == "BULLISH_OB" and ob["high"] < entry:
                tp_candidates.append(("OPPOSING_OB", ob["high"]))

        # 3. Son swing low
        if structure["last_swing_low"] and structure["last_swing_low"]["price"] < entry:
            tp_candidates.append(("SWING_LOW", structure["last_swing_low"]["price"] * 1.002))

        # 4. Önceki swing low'lar
        for sl_point in structure.get("swing_lows", []):
            if sl_point["price"] < entry * 0.995:
                tp_candidates.append(("PREV_SWING_LOW", sl_point["price"] * 1.002))

        if not tp_candidates:
            if sl is not None and sl > entry:
                risk = sl - entry
                min_tp_ratio = self.params.get("default_tp_ratio", 2.5)
                return entry - (risk * min_tp_ratio)
            return None

        risk = sl - entry if sl else entry * 0.015
        min_reward = risk * 1.5

        valid_tps = [(name, price) for name, price in tp_candidates
                     if (entry - price) >= min_reward]

        if valid_tps:
            best_tp = max(valid_tps, key=lambda x: x[1])
            logger.debug(f"  SHORT TP: {best_tp[0]} @ {best_tp[1]:.8f}")
            return best_tp[1]

        if tp_candidates:
            best_tp = min(tp_candidates, key=lambda x: x[1])
            if (entry - best_tp[1]) > risk * 1.0:
                return best_tp[1]

        min_tp_ratio = self.params.get("default_tp_ratio", 2.5)
        return entry - (risk * min_tp_ratio)

    def _get_watch_reason(self, analysis):
        """İzleme sebebini açıkla"""
        reasons = []
        components = analysis["components"]
        penalties = analysis.get("penalties", [])

        if "MARKET_STRUCTURE" not in components:
            reasons.append("Yapı onayı bekleniyor")
        if "ORDER_BLOCK" not in components:
            reasons.append("OB temas bekleniyor")
        if "FVG" not in components:
            reasons.append("FVG dolumu bekleniyor")
        if "DISPLACEMENT" not in components:
            reasons.append("Displacement bekleniyor")
        if "HTF_CONFIRMATION" not in components and "MTF_CONFIRMATION" not in components:
            reasons.append("MTF/HTF onayı bekleniyor")

        # Ceza sebeplerini ekle
        for p in penalties:
            if "OFF_HOURS" in p:
                reasons.append("Killzone dışı saat")
            elif "RANGING" in p:
                reasons.append("Yatay piyasa")

        if not reasons:
            reasons.append("Güven skoru düşük, onay bekleniyor")

        return " | ".join(reasons[:3])

    def _get_entry_type(self, analysis, entry, current_price, direction):
        """Entry seviyesinin ICT kaynağını belirle"""
        if analysis.get("relevant_obs"):
            ob = analysis["relevant_obs"][0]
            if direction == "LONG" and ob["type"] == "BULLISH_OB":
                if abs(entry - ob["open"]) / entry < 0.003:
                    return "Order Block (OB üst kenar)"
            elif direction == "SHORT" and ob["type"] == "BEARISH_OB":
                if abs(entry - ob["open"]) / entry < 0.003:
                    return "Order Block (OB alt kenar)"

        if analysis.get("relevant_fvgs"):
            fvg = analysis["relevant_fvgs"][0]
            fvg_mid = (fvg["high"] + fvg["low"]) / 2
            if abs(entry - fvg_mid) / entry < 0.003:
                return "FVG Orta Nokta"

        pd_zone = analysis.get("premium_discount")
        if pd_zone and pd_zone.get("in_ote"):
            ote_mid = (pd_zone["ote_high"] + pd_zone["ote_low"]) / 2
            if abs(entry - ote_mid) / entry < 0.005:
                return "OTE Bölgesi (Fib 0.618-0.786)"

        if abs(entry - current_price) / entry < 0.001:
            if pd_zone:
                return f"Güncel Fiyat ({pd_zone['zone']} bölgesi)"
            return "Güncel Fiyat"

        return "Yapısal Seviye"

    def _get_sl_type(self, analysis, sl, direction):
        """SL seviyesinin ICT kaynağını belirle"""
        if analysis.get("relevant_obs"):
            ob = analysis["relevant_obs"][0]
            if direction == "LONG" and ob["type"] == "BULLISH_OB":
                if abs(sl - ob["low"] * 0.997) / sl < 0.005:
                    return "OB Invalidation (OB low altı)"
            elif direction == "SHORT" and ob["type"] == "BEARISH_OB":
                if abs(sl - ob["high"] * 1.003) / sl < 0.005:
                    return "OB Invalidation (OB high üstü)"

        structure = analysis["structure"]
        if direction == "LONG" and structure["last_swing_low"]:
            if abs(sl - structure["last_swing_low"]["price"] * 0.997) / sl < 0.005:
                return "Swing Low Yapısal Seviye"
        elif direction == "SHORT" and structure["last_swing_high"]:
            if abs(sl - structure["last_swing_high"]["price"] * 1.003) / sl < 0.005:
                return "Swing High Yapısal Seviye"

        return "Likidite Sweep Seviyesi"

    def _get_tp_type(self, analysis, tp, direction):
        """TP seviyesinin ICT kaynağını belirle"""
        liquidity = analysis.get("liquidity", [])
        for liq in liquidity:
            if direction == "LONG" and liq["type"] == "EQUAL_HIGHS" and not liq["swept"]:
                if abs(tp - liq["price"]) / tp < 0.005:
                    return "Karşı Likidite (Equal Highs)"
            elif direction == "SHORT" and liq["type"] == "EQUAL_LOWS" and not liq["swept"]:
                if abs(tp - liq["price"]) / tp < 0.005:
                    return "Karşı Likidite (Equal Lows)"

        all_obs = analysis.get("order_blocks", [])
        for ob in all_obs:
            if direction == "LONG" and ob["type"] == "BEARISH_OB":
                if abs(tp - ob["low"]) / tp < 0.005:
                    return "Karşı Order Block (Bearish OB)"
            elif direction == "SHORT" and ob["type"] == "BULLISH_OB":
                if abs(tp - ob["high"]) / tp < 0.005:
                    return "Karşı Order Block (Bullish OB)"

        structure = analysis["structure"]
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
