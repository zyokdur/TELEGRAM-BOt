# =====================================================
# ICT Forex/Commodities - TAM ICT UYUMLU Motor v2
# =====================================================
# Inner Circle Trader (ICT) metodolojisi ile
# Forex ve emtia piyasalarinda sinyal uretir.
#
# TAM ICT KAVRAMLARI:
#   1.  Market Structure (BOS / CHoCH / Swing H-L)
#   2.  Order Blocks (OB) + Mitigation tracking
#   3.  Breaker Blocks (basarisiz OB -> destek/direnc)
#   4.  Fair Value Gaps (FVG) + Consequent Encroachment
#   5.  Displacement (buyuk momentum mumlari)
#   6.  Liquidity Sweeps (EQH/EQL + stop hunt)
#   7.  Inducement (kucuk likidite kapanlari)
#   8.  Optimal Trade Entry (OTE) - Fib 0.618-0.786
#   9.  Premium / Discount Zones
#  10.  Kill Zones (London/NY/Asian sessions)
#  11.  ICT Silver Bullet windows
#  12.  Power of 3 / AMD pattern
#  13.  Judas Swing (fake move in Kill Zone)
#  14.  Daily Bias (HTF trend confirmation)
#  15.  Asian Range + London Breakout
#  16.  Smart Money Trap detection
# =====================================================

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("ICT-Bot.ForexICT")

# ── Enstruman haritasi ──
FOREX_INSTRUMENTS = {
    "EURUSD": {
        "yf_symbol": "EURUSD=X",
        "name": "EUR/USD",
        "category": "forex",
        "pip_size": 0.0001,
        "icon": "\u20ac",
        "desc": "Euro / ABD Dolari"
    },
    "GBPUSD": {
        "yf_symbol": "GBPUSD=X",
        "name": "GBP/USD",
        "category": "forex",
        "pip_size": 0.0001,
        "icon": "\u00a3",
        "desc": "Ingiliz Sterlini / ABD Dolari"
    },
    "USDJPY": {
        "yf_symbol": "USDJPY=X",
        "name": "USD/JPY",
        "category": "forex",
        "pip_size": 0.01,
        "icon": "\u00a5",
        "desc": "ABD Dolari / Japon Yeni"
    },
    "XAUUSD": {
        "yf_symbol": "GC=F",
        "name": "XAU/USD",
        "category": "commodity",
        "pip_size": 0.01,
        "icon": "\U0001f947",
        "desc": "Altin / ABD Dolari"
    },
    "XAGUSD": {
        "yf_symbol": "SI=F",
        "name": "XAG/USD",
        "category": "commodity",
        "pip_size": 0.001,
        "icon": "\U0001f948",
        "desc": "Gumus / ABD Dolari"
    },
    "USDCHF": {
        "yf_symbol": "USDCHF=X",
        "name": "USD/CHF",
        "category": "forex",
        "pip_size": 0.0001,
        "icon": "\u20a3",
        "desc": "ABD Dolari / Isvicre Frangi"
    },
}

TF_MAP = {
    "15m": {"interval": "15m", "period": "5d",  "label": "15 Dakika"},
    "1h":  {"interval": "1h",  "period": "30d", "label": "1 Saat"},
    "4h":  {"interval": "1h",  "period": "60d", "label": "4 Saat"},
    "1d":  {"interval": "1d",  "period": "6mo", "label": "Gunluk"},
}


class ForexICTEngine:
    """Tam ICT uyumlu Forex/Emtia analiz motoru"""

    def __init__(self):
        self._cache = {}
        self._cache_ttl = 30

    # ================================================================
    #  VERI CEKME
    # ================================================================

    def get_candles(self, instrument_key, timeframe="1h"):
        """yfinance'den mum verileri cek"""
        cache_key = f"fx_{instrument_key}_{timeframe}"
        now = datetime.now().timestamp()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return data

        inst = FOREX_INSTRUMENTS.get(instrument_key)
        if not inst:
            return pd.DataFrame()

        tf_cfg = TF_MAP.get(timeframe, TF_MAP["1h"])

        try:
            ticker = yf.Ticker(inst["yf_symbol"])

            if timeframe == "4h":
                raw = ticker.history(period=tf_cfg["period"], interval="1h", auto_adjust=True)
                if raw.empty:
                    return pd.DataFrame()
                raw = raw.reset_index()
                raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
                if "datetime" in raw.columns:
                    raw = raw.rename(columns={"datetime": "timestamp"})
                elif "date" in raw.columns:
                    raw = raw.rename(columns={"date": "timestamp"})
                raw["group"] = raw.index // 4
                df = raw.groupby("group").agg({
                    "timestamp": "first", "open": "first",
                    "high": "max", "low": "min",
                    "close": "last", "volume": "sum"
                }).reset_index(drop=True)
            else:
                raw = ticker.history(period=tf_cfg["period"], interval=tf_cfg["interval"], auto_adjust=True)
                if raw.empty:
                    return pd.DataFrame()
                raw = raw.reset_index()
                raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
                if "datetime" in raw.columns:
                    raw = raw.rename(columns={"datetime": "timestamp"})
                elif "date" in raw.columns:
                    raw = raw.rename(columns={"date": "timestamp"})
                df = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()

            df = df.dropna(subset=["close"]).reset_index(drop=True)
            self._cache[cache_key] = (now, df)
            return df
        except Exception as e:
            logger.error(f"Forex veri hatasi ({instrument_key}): {e}")
            return pd.DataFrame()

    def get_price(self, instrument_key):
        """Anlik fiyat bilgisi"""
        inst = FOREX_INSTRUMENTS.get(instrument_key)
        if not inst:
            return None
        try:
            ticker = yf.Ticker(inst["yf_symbol"])
            info = ticker.fast_info
            return {
                "last": round(float(info.get("lastPrice", 0) or info.get("previousClose", 0)), 5),
                "prev_close": round(float(info.get("previousClose", 0)), 5),
                "open": round(float(info.get("open", 0)), 5),
                "day_high": round(float(info.get("dayHigh", 0)), 5),
                "day_low": round(float(info.get("dayLow", 0)), 5),
            }
        except Exception as e:
            logger.error(f"Fiyat hatasi ({instrument_key}): {e}")
            return None

    # ================================================================
    #  1. MARKET STRUCTURE  (BOS / CHoCH / Swing)
    # ================================================================

    def detect_market_structure(self, df):
        """Swing High/Low, BOS, CHoCH tespiti"""
        if len(df) < 20:
            return {"trend": "NEUTRAL", "bos": [], "choch": None,
                    "swing_highs": [], "swing_lows": []}

        highs = df["high"].values
        lows = df["low"].values
        close = df["close"].values
        lookback = 5

        swing_highs = []
        swing_lows = []
        for i in range(lookback, len(df) - lookback):
            if highs[i] == max(highs[i - lookback:i + lookback + 1]):
                swing_highs.append({"idx": i, "price": float(highs[i])})
            if lows[i] == min(lows[i - lookback:i + lookback + 1]):
                swing_lows.append({"idx": i, "price": float(lows[i])})

        trend = "NEUTRAL"
        bos_list = []
        choch = None

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]["price"] > swing_highs[-2]["price"]
            hl = swing_lows[-1]["price"] > swing_lows[-2]["price"]
            lh = swing_highs[-1]["price"] < swing_highs[-2]["price"]
            ll = swing_lows[-1]["price"] < swing_lows[-2]["price"]

            if hh and hl:
                trend = "BULLISH"
            elif lh and ll:
                trend = "BEARISH"

            cur_price = close[-1]
            if swing_highs and cur_price > swing_highs[-1]["price"]:
                bos_list.append({"type": "BULLISH_BOS", "level": swing_highs[-1]["price"]})
            if swing_lows and cur_price < swing_lows[-1]["price"]:
                bos_list.append({"type": "BEARISH_BOS", "level": swing_lows[-1]["price"]})

            # CHoCH
            if len(swing_highs) >= 3 and len(swing_lows) >= 3:
                if (swing_highs[-3]["price"] < swing_highs[-2]["price"] and
                        swing_highs[-1]["price"] < swing_highs[-2]["price"]):
                    choch = {"type": "BEARISH_CHOCH", "level": swing_lows[-2]["price"],
                             "desc": "Yukselis trendinden dususe donus sinyali (CHoCH)"}
                if (swing_lows[-3]["price"] > swing_lows[-2]["price"] and
                        swing_lows[-1]["price"] > swing_lows[-2]["price"]):
                    choch = {"type": "BULLISH_CHOCH", "level": swing_highs[-2]["price"],
                             "desc": "Dusus trendinden yukselise donus sinyali (CHoCH)"}

        return {
            "trend": trend, "bos": bos_list, "choch": choch,
            "swing_highs": swing_highs[-4:] if swing_highs else [],
            "swing_lows": swing_lows[-4:] if swing_lows else [],
        }

    # ================================================================
    #  2. ORDER BLOCKS  (OB) + Mitigation
    # ================================================================

    def detect_order_blocks(self, df, cur_price=None):
        """OB tespiti + mitigate olup olmadigini izle"""
        if len(df) < 10:
            return []

        opens = df["open"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        blocks = []

        for i in range(2, len(df) - 2):
            candle_range = highs[i] - lows[i]
            if candle_range == 0:
                continue
            avg_range = np.mean([highs[j] - lows[j] for j in range(max(0, i - 10), i)])
            if avg_range == 0:
                continue
            next_move = abs(closes[min(i + 2, len(df) - 1)] - closes[i])

            if next_move > avg_range * 2:
                if closes[i] < opens[i] and closes[min(i + 2, len(df) - 1)] > closes[i]:
                    mitigated = False
                    if cur_price is not None:
                        for k in range(i + 3, len(df)):
                            if lows[k] <= closes[i]:
                                mitigated = True
                                break
                    blocks.append({
                        "type": "BULLISH_OB", "high": float(opens[i]),
                        "low": float(closes[i]), "idx": i,
                        "strength": round(next_move / avg_range, 1),
                        "mitigated": mitigated
                    })
                elif closes[i] > opens[i] and closes[min(i + 2, len(df) - 1)] < closes[i]:
                    mitigated = False
                    if cur_price is not None:
                        for k in range(i + 3, len(df)):
                            if highs[k] >= closes[i]:
                                mitigated = True
                                break
                    blocks.append({
                        "type": "BEARISH_OB", "high": float(closes[i]),
                        "low": float(opens[i]), "idx": i,
                        "strength": round(next_move / avg_range, 1),
                        "mitigated": mitigated
                    })

        blocks.sort(key=lambda x: x["strength"], reverse=True)
        return blocks[:8]

    # ================================================================
    #  3. BREAKER BLOCKS  (basarisiz OB -> S/R)
    # ================================================================

    def detect_breaker_blocks(self, df):
        """Basarisiz OB'leri Breaker Block olarak isle"""
        if len(df) < 15:
            return []

        obs = self.detect_order_blocks(df, cur_price=float(df["close"].iloc[-1]))
        breakers = []
        cur_price = float(df["close"].iloc[-1])

        for ob in obs:
            if ob["mitigated"]:
                mid = (ob["high"] + ob["low"]) / 2
                if ob["type"] == "BULLISH_OB":
                    if cur_price < mid:
                        breakers.append({
                            "type": "BEARISH_BREAKER",
                            "high": ob["high"], "low": ob["low"],
                            "desc": "Basarisiz Bullish OB -> Bearish Breaker (direnc)"
                        })
                elif ob["type"] == "BEARISH_OB":
                    if cur_price > mid:
                        breakers.append({
                            "type": "BULLISH_BREAKER",
                            "high": ob["high"], "low": ob["low"],
                            "desc": "Basarisiz Bearish OB -> Bullish Breaker (destek)"
                        })
        return breakers[:3]

    # ================================================================
    #  4. FVG + Consequent Encroachment (CE)
    # ================================================================

    def detect_fvg(self, df):
        """FVG tespiti + %50 seviyesine geri donus (CE) kontrolu"""
        if len(df) < 5:
            return []

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        gaps = []

        for i in range(2, len(df)):
            # Bullish FVG
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]
                ce_level = (lows[i] + highs[i - 2]) / 2  # %50 seviye
                filled = False
                ce_tested = False
                for k in range(i + 1, len(df)):
                    if lows[k] <= highs[i - 2]:
                        filled = True
                        break
                    if lows[k] <= ce_level:
                        ce_tested = True
                gaps.append({
                    "type": "BULLISH_FVG",
                    "top": float(lows[i]), "bottom": float(highs[i - 2]),
                    "ce_level": float(ce_level),
                    "size": float(gap_size), "idx": i,
                    "filled": filled, "ce_tested": ce_tested,
                })
            # Bearish FVG
            if highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]
                ce_level = (lows[i - 2] + highs[i]) / 2
                filled = False
                ce_tested = False
                for k in range(i + 1, len(df)):
                    if highs[k] >= lows[i - 2]:
                        filled = True
                        break
                    if highs[k] >= ce_level:
                        ce_tested = True
                gaps.append({
                    "type": "BEARISH_FVG",
                    "top": float(lows[i - 2]), "bottom": float(highs[i]),
                    "ce_level": float(ce_level),
                    "size": float(gap_size), "idx": i,
                    "filled": filled, "ce_tested": ce_tested,
                })

        return gaps[-12:]

    # ================================================================
    #  5. DISPLACEMENT  (buyuk momentum mumlari)
    # ================================================================

    def detect_displacement(self, df):
        """Buyuk govdeli momentum mumlari (displacement candles)"""
        if len(df) < 20:
            return []

        opens = df["open"].values
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        displacements = []

        avg_body = np.mean([abs(closes[i] - opens[i]) for i in range(max(0, len(df) - 20), len(df))])
        if avg_body == 0:
            return []

        for i in range(len(df) - 15, len(df)):
            if i < 0:
                continue
            body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0:
                continue
            body_ratio = body / candle_range

            if body > avg_body * 2.5 and body_ratio > 0.7:
                direction = "BULLISH" if closes[i] > opens[i] else "BEARISH"
                displacements.append({
                    "type": f"{direction}_DISPLACEMENT",
                    "idx": i,
                    "body_mult": round(body / avg_body, 1),
                    "body_ratio": round(body_ratio, 2),
                    "desc": f"{direction} Displacement - {round(body / avg_body, 1)}x ortalama govde"
                })

        return displacements[-5:]

    # ================================================================
    #  6. LIQUIDITY SWEEPS  (EQH/EQL + Stop Hunt)
    # ================================================================

    def detect_liquidity_sweeps(self, df):
        """Equal Highs/Lows uzerinde likidite avi"""
        if len(df) < 20:
            return []

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        tolerance = 0.001
        sweeps = []

        for i in range(5, len(df) - 1):
            # Sell-side sweep (EQH uzerine cikip donus)
            for j in range(max(0, i - 15), i - 3):
                if abs(highs[j] - highs[i - 1]) / max(highs[j], 0.0001) < tolerance:
                    if highs[i] > highs[j] * 1.001 and closes[i] < highs[j]:
                        sweeps.append({
                            "type": "SELL_SIDE_SWEEP",
                            "level": float(highs[j]),
                            "sweep_price": float(highs[i]),
                            "idx": i,
                            "desc": "Esit zirveler uzerinde likidite avi - dusus beklentisi"
                        })
                        break
            # Buy-side sweep (EQL altina inip donus)
            for j in range(max(0, i - 15), i - 3):
                if abs(lows[j] - lows[i - 1]) / max(lows[j], 0.0001) < tolerance:
                    if lows[i] < lows[j] * 0.999 and closes[i] > lows[j]:
                        sweeps.append({
                            "type": "BUY_SIDE_SWEEP",
                            "level": float(lows[j]),
                            "sweep_price": float(lows[i]),
                            "idx": i,
                            "desc": "Esit dipler altinda likidite avi - yukselis beklentisi"
                        })
                        break

        return sweeps[-5:]

    # ================================================================
    #  7. INDUCEMENT  (kucuk likidite kapanlari)
    # ================================================================

    def detect_inducement(self, df, swing_data):
        """
        Kucuk internal yapilarda likidite birikimi.
        Ana swing icinde olusmus minor high/low kirilmalari.
        """
        if len(df) < 20 or not swing_data.get("swing_highs") or not swing_data.get("swing_lows"):
            return []

        highs = df["high"].values
        lows = df["low"].values
        inducements = []

        last_sh = swing_data["swing_highs"][-1] if swing_data["swing_highs"] else None
        last_sl = swing_data["swing_lows"][-1] if swing_data["swing_lows"] else None

        if last_sh and last_sl:
            start_idx = min(last_sh["idx"], last_sl["idx"])
            end_idx = max(last_sh["idx"], last_sl["idx"])

            mini_highs = []
            mini_lows = []
            for i in range(start_idx + 1, min(end_idx, len(df) - 1)):
                if i >= 2 and i < len(df) - 2:
                    if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                        mini_highs.append({"idx": i, "price": float(highs[i])})
                    if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                        mini_lows.append({"idx": i, "price": float(lows[i])})

            for mh in mini_highs:
                for k in range(mh["idx"] + 1, min(mh["idx"] + 5, len(df))):
                    if highs[k] > mh["price"]:
                        inducements.append({
                            "type": "BULLISH_INDUCEMENT",
                            "level": mh["price"], "idx": k,
                            "desc": "Minor high kirildi - buy-side likidite toplandi"
                        })
                        break

            for ml in mini_lows:
                for k in range(ml["idx"] + 1, min(ml["idx"] + 5, len(df))):
                    if lows[k] < ml["price"]:
                        inducements.append({
                            "type": "BEARISH_INDUCEMENT",
                            "level": ml["price"], "idx": k,
                            "desc": "Minor low kirildi - sell-side likidite toplandi"
                        })
                        break

        return inducements[-3:]

    # ================================================================
    #  8. OTE  (Optimal Trade Entry - Fib 0.618-0.786)
    # ================================================================

    def calc_ote(self, df, ms):
        """Fibonacci 0.618-0.786 geri cekilme bolgesi"""
        if not ms["swing_highs"] or not ms["swing_lows"]:
            return None

        sh = ms["swing_highs"][-1]
        sl = ms["swing_lows"][-1]
        swing_range = sh["price"] - sl["price"]
        if swing_range <= 0:
            return None

        if ms["trend"] == "BULLISH":
            fib_618 = sh["price"] - swing_range * 0.618
            fib_786 = sh["price"] - swing_range * 0.786
            return {
                "direction": "LONG",
                "ote_top": round(fib_618, 5), "ote_bottom": round(fib_786, 5),
                "swing_high": sh["price"], "swing_low": sl["price"],
                "desc": f"OTE alim bolgesi: {round(fib_786,5)} - {round(fib_618,5)}"
            }
        elif ms["trend"] == "BEARISH":
            fib_618 = sl["price"] + swing_range * 0.618
            fib_786 = sl["price"] + swing_range * 0.786
            return {
                "direction": "SHORT",
                "ote_top": round(fib_786, 5), "ote_bottom": round(fib_618, 5),
                "swing_high": sh["price"], "swing_low": sl["price"],
                "desc": f"OTE satis bolgesi: {round(fib_618,5)} - {round(fib_786,5)}"
            }
        return None

    # ================================================================
    #  9. PREMIUM / DISCOUNT ZONES
    # ================================================================

    def calc_premium_discount(self, df):
        """Equilibrium ustu premium, alti discount"""
        if len(df) < 20:
            return {"zone": "NEUTRAL", "eq": 0, "range_high": 0, "range_low": 0}

        lookback = min(50, len(df))
        range_high = float(df["high"].iloc[-lookback:].max())
        range_low = float(df["low"].iloc[-lookback:].min())
        eq = (range_high + range_low) / 2
        cur_price = float(df["close"].iloc[-1])

        if cur_price > eq:
            pct = (cur_price - eq) / (range_high - eq) * 100 if range_high != eq else 50
            zone = "PREMIUM"
        else:
            pct = (eq - cur_price) / (eq - range_low) * 100 if eq != range_low else 50
            zone = "DISCOUNT"

        return {
            "zone": zone, "zone_pct": round(min(pct, 100), 1),
            "equilibrium": round(eq, 5),
            "range_high": round(range_high, 5),
            "range_low": round(range_low, 5),
            "current": round(cur_price, 5),
        }

    # ================================================================
    # 10. KILL ZONES
    # ================================================================

    def detect_kill_zones(self):
        """London / NY / Asian Kill Zone tespiti"""
        now = datetime.utcnow()
        hour = now.hour
        active = None
        zones = []

        kz_defs = [
            ("Asian",    0, 3,  "00:00-03:00 UTC"),
            ("London",   7, 10, "07:00-10:00 UTC"),
            ("New York", 12, 15, "12:00-15:00 UTC"),
        ]
        for name, start, end, hours_str in kz_defs:
            is_active = start <= hour < end
            if is_active:
                active = name.upper().replace(" ", "_")
            zones.append({"name": f"{name} Kill Zone", "active": is_active, "hours": hours_str})

        return {
            "active_zone": active, "zones": zones,
            "is_kill_zone": active is not None,
            "desc": (f"[AKTIF] {active} Kill Zone - yuksek volatilite beklenir"
                     if active else "Kill Zone disinda - dusuk volatilite"),
        }

    # ================================================================
    # 11. ICT SILVER BULLET
    # ================================================================

    def detect_silver_bullet(self):
        """
        ICT Silver Bullet pencereleri:
        - London SB: 03:00-04:00 EST (08:00-09:00 UTC)
        - NY AM SB:  10:00-11:00 EST (15:00-16:00 UTC)
        - NY PM SB:  14:00-15:00 EST (19:00-20:00 UTC)
        """
        now = datetime.utcnow()
        hour = now.hour
        minute = now.minute

        sb_windows = [
            {"name": "London Silver Bullet", "start_h": 8, "end_h": 9,
             "desc": "FVG giris firsati - London acilis"},
            {"name": "NY AM Silver Bullet", "start_h": 15, "end_h": 16,
             "desc": "FVG giris firsati - New York sabah"},
            {"name": "NY PM Silver Bullet", "start_h": 19, "end_h": 20,
             "desc": "FVG giris firsati - New York ogleden sonra"},
        ]

        active_sb = None
        for sb in sb_windows:
            sb["active"] = sb["start_h"] <= hour < sb["end_h"]
            if sb["active"]:
                active_sb = sb

        return {
            "windows": sb_windows,
            "active": active_sb,
            "is_active": active_sb is not None,
            "desc": (f"[AKTIF] {active_sb['name']} - {active_sb['desc']}"
                     if active_sb else "Silver Bullet penceresi disinda"),
        }

    # ================================================================
    # 12. POWER OF 3 / AMD  (Accumulation-Manipulation-Distribution)
    # ================================================================

    def detect_amd_pattern(self, df):
        """
        Son N mumda AMD patern arama:
        - Accumulation: dusuk volatiliteli daralma
        - Manipulation: ani spike (likidite avi)
        - Distribution: gercek yon hareketi
        """
        if len(df) < 20:
            return None

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        lookback = min(20, len(df) - 1)
        recent = df.iloc[-lookback:]

        # Accumulation: ilk %40 dusuk volatilite
        acc_end = int(lookback * 0.4)
        acc_range = highs[-lookback:-lookback + acc_end].max() - lows[-lookback:-lookback + acc_end].min()
        full_range = highs[-lookback:].max() - lows[-lookback:].min()

        if full_range == 0:
            return None

        acc_ratio = acc_range / full_range

        # Manipulation: ortadaki %30'da spike
        man_start = acc_end
        man_end = int(lookback * 0.7)
        if man_end <= man_start:
            return None

        man_high = highs[-lookback + man_start:-lookback + man_end].max() if man_start < man_end else 0
        man_low = lows[-lookback + man_start:-lookback + man_end].min() if man_start < man_end else 0

        # Distribution: son %30 gercek hareket
        dist_close = closes[-1]
        dist_open = closes[-lookback + man_end] if man_end < lookback else closes[-1]

        if acc_ratio < 0.4:  # Accumulation yeterince dar
            direction = "BULLISH" if dist_close > dist_open else "BEARISH"

            # Manipulation yonu: gercek yonun tersi olmali
            man_mid = (man_high + man_low) / 2
            acc_mid = (highs[-lookback:-lookback + acc_end].max() + lows[-lookback:-lookback + acc_end].min()) / 2

            if direction == "BULLISH" and man_low < lows[-lookback:-lookback + acc_end].min():
                return {
                    "pattern": "AMD_BULLISH",
                    "phase": "DISTRIBUTION",
                    "direction": "LONG",
                    "desc": "AMD Bullish: Birikme -> asagi manipulasyon -> yukari dagitim",
                    "acc_range": round(float(acc_range), 5),
                }
            elif direction == "BEARISH" and man_high > highs[-lookback:-lookback + acc_end].max():
                return {
                    "pattern": "AMD_BEARISH",
                    "phase": "DISTRIBUTION",
                    "direction": "SHORT",
                    "desc": "AMD Bearish: Birikme -> yukari manipulasyon -> asagi dagitim",
                    "acc_range": round(float(acc_range), 5),
                }
        return None

    # ================================================================
    # 13. JUDAS SWING
    # ================================================================

    def detect_judas_swing(self, df):
        """Kill Zone'da ilk hareketin tersi yonde sinyal"""
        if len(df) < 10:
            return None

        kill = self.detect_kill_zones()
        if not kill["is_kill_zone"]:
            return None

        # Son 5 mumun ilk 2'si vs son 3'u
        opens = df["open"].values
        closes = df["close"].values

        first_move = closes[-5] - opens[-5]
        last_move = closes[-1] - closes[-3]

        if first_move > 0 and last_move < 0 and abs(last_move) > abs(first_move) * 0.8:
            return {
                "type": "BEARISH_JUDAS",
                "desc": f"Judas Swing: {kill['active_zone']} acilisinda yukari sahte hareket, gercek yon asagi",
                "kill_zone": kill["active_zone"]
            }
        elif first_move < 0 and last_move > 0 and abs(last_move) > abs(first_move) * 0.8:
            return {
                "type": "BULLISH_JUDAS",
                "desc": f"Judas Swing: {kill['active_zone']} acilisinda asagi sahte hareket, gercek yon yukari",
                "kill_zone": kill["active_zone"]
            }
        return None

    # ================================================================
    # 14. DAILY BIAS  (HTF yon teyidi)
    # ================================================================

    def calc_daily_bias(self, instrument_key):
        """Gunluk TF'den trend yonu (HTF bias)"""
        df_daily = self.get_candles(instrument_key, "1d")
        if df_daily.empty or len(df_daily) < 20:
            return {"bias": "NEUTRAL", "desc": "Yetersiz gunluk veri"}

        ms = self.detect_market_structure(df_daily)
        pd_zone = self.calc_premium_discount(df_daily)

        bias = "NEUTRAL"
        desc = "Gunluk bias belirsiz"

        if ms["trend"] == "BULLISH":
            bias = "BULLISH"
            desc = "Gunluk trend yukselis - LONG oncelikli"
            if pd_zone["zone"] == "PREMIUM" and pd_zone["zone_pct"] > 70:
                desc += " (DIKKAT: Premium bolgede, geri cekilme riski)"
        elif ms["trend"] == "BEARISH":
            bias = "BEARISH"
            desc = "Gunluk trend dusus - SHORT oncelikli"
            if pd_zone["zone"] == "DISCOUNT" and pd_zone["zone_pct"] > 70:
                desc += " (DIKKAT: Discount bolgede, toparlanma riski)"

        if ms["choch"]:
            if ms["choch"]["type"] == "BULLISH_CHOCH":
                desc += " | CHoCH: Yukselise donus sinyali!"
            elif ms["choch"]["type"] == "BEARISH_CHOCH":
                desc += " | CHoCH: Dususe donus sinyali!"

        return {"bias": bias, "desc": desc, "trend": ms["trend"],
                "zone": pd_zone["zone"], "zone_pct": pd_zone["zone_pct"]}

    # ================================================================
    # 15. ASIAN RANGE + LONDON BREAKOUT
    # ================================================================

    def detect_asian_range_breakout(self, df):
        """Asian session range'i + London kirilim tespiti"""
        if len(df) < 10:
            return None

        # Basitlesilmis: son 20 mumun ilk 8'i Asian, sonraki London
        if len(df) < 20:
            return None

        asian_highs = df["high"].iloc[-20:-12].values
        asian_lows = df["low"].iloc[-20:-12].values
        if len(asian_highs) == 0:
            return None

        asian_high = float(np.max(asian_highs))
        asian_low = float(np.min(asian_lows))
        asian_mid = (asian_high + asian_low) / 2

        cur_price = float(df["close"].iloc[-1])

        breakout = None
        if cur_price > asian_high:
            breakout = {
                "type": "BULLISH_BREAKOUT",
                "asian_high": round(asian_high, 5),
                "asian_low": round(asian_low, 5),
                "desc": f"Asian Range yukari kirildi ({round(asian_high,5)})"
            }
        elif cur_price < asian_low:
            breakout = {
                "type": "BEARISH_BREAKOUT",
                "asian_high": round(asian_high, 5),
                "asian_low": round(asian_low, 5),
                "desc": f"Asian Range asagi kirildi ({round(asian_low,5)})"
            }
        else:
            breakout = {
                "type": "INSIDE_RANGE",
                "asian_high": round(asian_high, 5),
                "asian_low": round(asian_low, 5),
                "desc": f"Fiyat Asian Range icinde ({round(asian_low,5)} - {round(asian_high,5)})"
            }

        return breakout

    # ================================================================
    # 16. SMART MONEY TRAP
    # ================================================================

    def detect_smart_money_trap(self, df, ms):
        """
        Retail trader'larin tuzaga dusuruldugu noktalar:
        - Breakout sonrasi hizli geri donus
        - Stop hunt + reversal
        """
        if len(df) < 10:
            return None

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        # Son 5 mumda: bir yonde kirilim (wick) + ters kapnis
        for i in range(len(df) - 3, len(df)):
            if i < 2:
                continue
            wick_up = highs[i] - max(closes[i], df["open"].iloc[i])
            wick_down = min(closes[i], df["open"].iloc[i]) - lows[i]
            body = abs(closes[i] - df["open"].iloc[i])
            candle_range = highs[i] - lows[i]

            if candle_range == 0:
                continue

            # Yukari trap: uzun ust fitil + ayissi kapnis
            if wick_up > body * 2 and wick_up > candle_range * 0.6 and closes[i] < df["open"].iloc[i]:
                return {
                    "type": "BULL_TRAP",
                    "idx": i,
                    "desc": "Smart Money Trap: Yukari sahte kirilim + geri donus - SHORT sinyali",
                    "trap_level": float(highs[i])
                }
            # Asagi trap: uzun alt fitil + bogaci kapnis
            if wick_down > body * 2 and wick_down > candle_range * 0.6 and closes[i] > df["open"].iloc[i]:
                return {
                    "type": "BEAR_TRAP",
                    "idx": i,
                    "desc": "Smart Money Trap: Asagi sahte kirilim + geri donus - LONG sinyali",
                    "trap_level": float(lows[i])
                }
        return None

    # ================================================================
    #  TEKNIK GOSTERGELER (RSI, EMA, ATR)
    # ================================================================

    def calc_indicators(self, df):
        """Destekleyici teknik gostergeler"""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss_s = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss_s.ewm(alpha=1 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # EMA
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else pd.Series([np.nan] * len(close))

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, min_periods=14).mean()

        return {
            "rsi": float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50,
            "ema20": float(ema20.iloc[-1]),
            "ema50": float(ema50.iloc[-1]),
            "ema200": float(ema200.iloc[-1]) if not np.isnan(ema200.iloc[-1]) else None,
            "atr": float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0,
            "atr_pct": round(float(atr.iloc[-1]) / float(close.iloc[-1]) * 100, 3) if not np.isnan(atr.iloc[-1]) else 0,
        }

    # ================================================================
    #  ANA SINYAL URETICISI  (tam ICT uyumlu confluence)
    # ================================================================

    def generate_signal(self, instrument_key, timeframe="1h"):
        """Tum ICT bilesenleri birlestir -> LONG/SHORT/WAIT"""
        df = self.get_candles(instrument_key, timeframe)
        if df.empty or len(df) < 30:
            return {"error": "Yetersiz veri"}

        inst = FOREX_INSTRUMENTS[instrument_key]
        cur_price = float(df["close"].iloc[-1])

        # Tum ICT analiz
        ms = self.detect_market_structure(df)
        obs = self.detect_order_blocks(df, cur_price)
        breakers = self.detect_breaker_blocks(df)
        fvgs = self.detect_fvg(df)
        displacements = self.detect_displacement(df)
        sweeps = self.detect_liquidity_sweeps(df)
        inducements = self.detect_inducement(df, ms)
        ote = self.calc_ote(df, ms)
        pd_zone = self.calc_premium_discount(df)
        kill = self.detect_kill_zones()
        silver_bullet = self.detect_silver_bullet()
        amd = self.detect_amd_pattern(df)
        judas = self.detect_judas_swing(df)
        daily_bias = self.calc_daily_bias(instrument_key)
        asian_bo = self.detect_asian_range_breakout(df)
        smt = self.detect_smart_money_trap(df, ms)
        indicators = self.calc_indicators(df)

        # ── CONFLUENCE SKOR SISTEMI ──
        bull_score = 0
        bear_score = 0
        reasons_bull = []
        reasons_bear = []
        confluence_count = {"bull": 0, "bear": 0}

        # 1. Market Structure (30 puan)
        if ms["trend"] == "BULLISH":
            bull_score += 30
            confluence_count["bull"] += 1
            reasons_bull.append("Piyasa yapisi yukselis trendinde (HH + HL)")
        elif ms["trend"] == "BEARISH":
            bear_score += 30
            confluence_count["bear"] += 1
            reasons_bear.append("Piyasa yapisi dusus trendinde (LH + LL)")

        for bos in ms["bos"]:
            if bos["type"] == "BULLISH_BOS":
                bull_score += 10
                confluence_count["bull"] += 1
                reasons_bull.append(f"Bullish BOS @ {round(bos['level'], 5)}")
            elif bos["type"] == "BEARISH_BOS":
                bear_score += 10
                confluence_count["bear"] += 1
                reasons_bear.append(f"Bearish BOS @ {round(bos['level'], 5)}")

        if ms["choch"]:
            if ms["choch"]["type"] == "BULLISH_CHOCH":
                bull_score += 15
                confluence_count["bull"] += 1
                reasons_bull.append("Bullish CHoCH - trend donusu!")
            elif ms["choch"]["type"] == "BEARISH_CHOCH":
                bear_score += 15
                confluence_count["bear"] += 1
                reasons_bear.append("Bearish CHoCH - trend donusu!")

        # 2. Order Blocks (20 puan) - sadece mitigate olmamislar
        active_obs = [ob for ob in obs if not ob["mitigated"]]
        bull_ob = [ob for ob in active_obs
                   if ob["type"] == "BULLISH_OB" and ob["low"] <= cur_price <= ob["high"]]
        bear_ob = [ob for ob in active_obs
                   if ob["type"] == "BEARISH_OB" and ob["low"] <= cur_price <= ob["high"]]

        if bull_ob:
            bull_score += 20
            confluence_count["bull"] += 1
            reasons_bull.append(f"Aktif Bullish OB icinde (guc: {bull_ob[0]['strength']}x)")
        if bear_ob:
            bear_score += 20
            confluence_count["bear"] += 1
            reasons_bear.append(f"Aktif Bearish OB icinde (guc: {bear_ob[0]['strength']}x)")

        # 3. Breaker Blocks (10 puan)
        for bb in breakers:
            if bb["type"] == "BULLISH_BREAKER":
                bull_score += 10
                confluence_count["bull"] += 1
                reasons_bull.append("Bullish Breaker Block destegi")
            elif bb["type"] == "BEARISH_BREAKER":
                bear_score += 10
                confluence_count["bear"] += 1
                reasons_bear.append("Bearish Breaker Block direnci")

        # 4. FVG (15 puan) - doldurulmamis, CE test edilmis olanlar daha guclu
        active_fvgs = [f for f in fvgs if not f["filled"] and f["idx"] >= len(df) - 15]
        bull_fvg = [f for f in active_fvgs if f["type"] == "BULLISH_FVG"]
        bear_fvg = [f for f in active_fvgs if f["type"] == "BEARISH_FVG"]
        ce_bull = [f for f in bull_fvg if f["ce_tested"]]
        ce_bear = [f for f in bear_fvg if f["ce_tested"]]

        if bull_fvg:
            pts = 15 if ce_bull else 10
            bull_score += pts
            confluence_count["bull"] += 1
            ce_txt = " (CE test edildi!)" if ce_bull else ""
            reasons_bull.append(f"{len(bull_fvg)} Bullish FVG{ce_txt}")
        if bear_fvg:
            pts = 15 if ce_bear else 10
            bear_score += pts
            confluence_count["bear"] += 1
            ce_txt = " (CE test edildi!)" if ce_bear else ""
            reasons_bear.append(f"{len(bear_fvg)} Bearish FVG{ce_txt}")

        # 5. Displacement (10 puan)
        recent_disp = [d for d in displacements if d["idx"] >= len(df) - 5]
        for d in recent_disp:
            if "BULLISH" in d["type"]:
                bull_score += 10
                confluence_count["bull"] += 1
                reasons_bull.append(f"Bullish Displacement ({d['body_mult']}x)")
            else:
                bear_score += 10
                confluence_count["bear"] += 1
                reasons_bear.append(f"Bearish Displacement ({d['body_mult']}x)")

        # 6. Liquidity Sweeps (15 puan)
        recent_sweeps = [s for s in sweeps if s["idx"] >= len(df) - 5]
        for sw in recent_sweeps:
            if sw["type"] == "BUY_SIDE_SWEEP":
                bull_score += 15
                confluence_count["bull"] += 1
                reasons_bull.append("Buy-side likidite avi - donus beklentisi")
            elif sw["type"] == "SELL_SIDE_SWEEP":
                bear_score += 15
                confluence_count["bear"] += 1
                reasons_bear.append("Sell-side likidite avi - dusus beklentisi")

        # 7. Inducement (5 puan)
        for ind in inducements:
            if ind["type"] == "BULLISH_INDUCEMENT":
                bull_score += 5
                reasons_bull.append("Bullish Inducement - likidite toplandi")
            elif ind["type"] == "BEARISH_INDUCEMENT":
                bear_score += 5
                reasons_bear.append("Bearish Inducement - likidite toplandi")

        # 8. OTE (10 puan)
        if ote:
            if ote["direction"] == "LONG" and ote["ote_bottom"] <= cur_price <= ote["ote_top"]:
                bull_score += 10
                confluence_count["bull"] += 1
                reasons_bull.append("Fiyat OTE alim bolgesinde (Fib 0.618-0.786)")
            elif ote["direction"] == "SHORT" and ote["ote_bottom"] <= cur_price <= ote["ote_top"]:
                bear_score += 10
                confluence_count["bear"] += 1
                reasons_bear.append("Fiyat OTE satis bolgesinde (Fib 0.618-0.786)")

        # 9. Premium/Discount (10 puan)
        if pd_zone["zone"] == "DISCOUNT" and pd_zone["zone_pct"] > 60:
            bull_score += 10
            confluence_count["bull"] += 1
            reasons_bull.append(f"Discount bolgesi (%{pd_zone['zone_pct']})")
        elif pd_zone["zone"] == "PREMIUM" and pd_zone["zone_pct"] > 60:
            bear_score += 10
            confluence_count["bear"] += 1
            reasons_bear.append(f"Premium bolgesi (%{pd_zone['zone_pct']})")

        # 10. Kill Zone bonus (5 puan)
        if kill["is_kill_zone"]:
            if bull_score > bear_score:
                bull_score += 5
                reasons_bull.append(f"{kill['active_zone']} Kill Zone aktif")
            elif bear_score > bull_score:
                bear_score += 5
                reasons_bear.append(f"{kill['active_zone']} Kill Zone aktif")

        # 11. Silver Bullet (5 puan)
        if silver_bullet["is_active"]:
            if bull_fvg or bear_fvg:
                if bull_score > bear_score:
                    bull_score += 5
                    reasons_bull.append(f"Silver Bullet + FVG confluence")
                else:
                    bear_score += 5
                    reasons_bear.append(f"Silver Bullet + FVG confluence")

        # 12. AMD Pattern (10 puan)
        if amd:
            if amd["direction"] == "LONG":
                bull_score += 10
                confluence_count["bull"] += 1
                reasons_bull.append(f"AMD Bullish pattern (Manipulation -> Distribution)")
            elif amd["direction"] == "SHORT":
                bear_score += 10
                confluence_count["bear"] += 1
                reasons_bear.append(f"AMD Bearish pattern (Manipulation -> Distribution)")

        # 13. Judas Swing (10 puan)
        if judas:
            if judas["type"] == "BULLISH_JUDAS":
                bull_score += 10
                confluence_count["bull"] += 1
                reasons_bull.append(f"Judas Swing: sahte dusus -> gercek yukselis")
            elif judas["type"] == "BEARISH_JUDAS":
                bear_score += 10
                confluence_count["bear"] += 1
                reasons_bear.append(f"Judas Swing: sahte yukselis -> gercek dusus")

        # 14. Daily Bias (15 puan)
        if daily_bias["bias"] == "BULLISH":
            bull_score += 15
            confluence_count["bull"] += 1
            reasons_bull.append(f"Gunluk Bias: YUKSELIS ({daily_bias['desc']})")
        elif daily_bias["bias"] == "BEARISH":
            bear_score += 15
            confluence_count["bear"] += 1
            reasons_bear.append(f"Gunluk Bias: DUSUS ({daily_bias['desc']})")

        # 15. Asian Range Breakout (5 puan)
        if asian_bo and asian_bo["type"] != "INSIDE_RANGE":
            if asian_bo["type"] == "BULLISH_BREAKOUT":
                bull_score += 5
                reasons_bull.append(f"Asian Range yukari kirildi")
            elif asian_bo["type"] == "BEARISH_BREAKOUT":
                bear_score += 5
                reasons_bear.append(f"Asian Range asagi kirildi")

        # 16. Smart Money Trap (15 puan)
        if smt:
            if smt["type"] == "BEAR_TRAP":
                bull_score += 15
                confluence_count["bull"] += 1
                reasons_bull.append(f"Smart Money Trap: LONG ({smt['desc']})")
            elif smt["type"] == "BULL_TRAP":
                bear_score += 15
                confluence_count["bear"] += 1
                reasons_bear.append(f"Smart Money Trap: SHORT ({smt['desc']})")

        # RSI konfirmasyon (5 puan)
        if indicators["rsi"] < 35:
            bull_score += 5
            reasons_bull.append(f"RSI asiri satim ({indicators['rsi']:.1f})")
        elif indicators["rsi"] > 65:
            bear_score += 5
            reasons_bear.append(f"RSI asiri alim ({indicators['rsi']:.1f})")

        # ── SINYAL KARARI ──
        net_score = bull_score - bear_score
        max_conf = max(confluence_count["bull"], confluence_count["bear"])

        # Confluence sayisi + skor birlikte degerlendirilir
        if net_score >= 50 and confluence_count["bull"] >= 4:
            signal = "STRONG_LONG"
            label = "GUCLU ALIS"
            desc = "ICT tam confluence: Market Structure + OB + FVG + Displacement uyumlu"
        elif net_score >= 25 and confluence_count["bull"] >= 3:
            signal = "LONG"
            label = "ALIS"
            desc = "ICT gostergeleri alis yonunu destekliyor"
        elif net_score <= -50 and confluence_count["bear"] >= 4:
            signal = "STRONG_SHORT"
            label = "GUCLU SATIS"
            desc = "ICT tam confluence: Market Structure + OB + FVG + Displacement uyumlu"
        elif net_score <= -25 and confluence_count["bear"] >= 3:
            signal = "SHORT"
            label = "SATIS"
            desc = "ICT gostergeleri satis yonunu destekliyor"
        else:
            signal = "WAIT"
            label = "BEKLE"
            desc = "Net bir ICT confluence yok. Daha iyi setup icin bekleyin."

        # Kill zone notu
        if kill["is_kill_zone"]:
            desc += f" | {kill['active_zone']} Kill Zone aktif."
        if silver_bullet["is_active"]:
            desc += f" | {silver_bullet['active']['name']} penceresi acik."

        # SL / TP hesapla
        atr = indicators["atr"]
        sl_tp = None
        if signal in ("STRONG_LONG", "LONG"):
            sl = round(cur_price - atr * 1.5, 5)
            tp1 = round(cur_price + atr * 2.0, 5)
            tp2 = round(cur_price + atr * 3.5, 5)
            sl_tp = {"sl": sl, "tp1": tp1, "tp2": tp2, "direction": "LONG"}
        elif signal in ("STRONG_SHORT", "SHORT"):
            sl = round(cur_price + atr * 1.5, 5)
            tp1 = round(cur_price - atr * 2.0, 5)
            tp2 = round(cur_price - atr * 3.5, 5)
            sl_tp = {"sl": sl, "tp1": tp1, "tp2": tp2, "direction": "SHORT"}

        return {
            "instrument": instrument_key,
            "name": inst["name"],
            "category": inst["category"],
            "icon": inst["icon"],
            "desc": inst["desc"],
            "price": cur_price,
            "timeframe": timeframe,
            "signal": signal,
            "label": label,
            "description": desc,
            "net_score": net_score,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "confluence_bull": confluence_count["bull"],
            "confluence_bear": confluence_count["bear"],
            "reasons_bull": reasons_bull,
            "reasons_bear": reasons_bear,
            "sl_tp": sl_tp,
            # ICT detaylari
            "market_structure": ms,
            "order_blocks": [ob for ob in obs if not ob["mitigated"]][:5],
            "breaker_blocks": breakers,
            "fvg": {
                "bull": len(bull_fvg), "bear": len(bear_fvg),
                "ce_bull": len(ce_bull), "ce_bear": len(ce_bear),
                "active": active_fvgs[:6],
            },
            "displacement": displacements,
            "liquidity_sweeps": sweeps,
            "inducement": inducements,
            "ote": ote,
            "premium_discount": pd_zone,
            "kill_zones": kill,
            "silver_bullet": silver_bullet,
            "amd": amd,
            "judas": judas,
            "daily_bias": daily_bias,
            "asian_breakout": asian_bo,
            "smart_money_trap": smt,
            "indicators": indicators,
            "timestamp": datetime.now().isoformat(),
        }

    def scan_all(self, timeframe="1h"):
        """Tum enstrumanlari tara"""
        results = []
        for key in FOREX_INSTRUMENTS:
            try:
                sig = self.generate_signal(key, timeframe)
                if "error" not in sig:
                    results.append(sig)
            except Exception as e:
                logger.error(f"Forex tarama hatasi ({key}): {e}")
        return results


# Singleton
forex_ict = ForexICTEngine()
