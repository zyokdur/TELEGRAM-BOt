# =====================================================
# ICT Trading Bot - OKX Gerçek Zamanlı Veri Modülü
# =====================================================
# OKX Public API üzerinden gerçek piyasa verileri çeker.
# 24 saatlik hacmi 5M USDT üzerindeki coinleri dinamik filtreler.
# Hiçbir demo/mock/test verisi kullanılmaz.
# =====================================================

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from config import OKX_API_V5, MIN_VOLUME_USDT, MAX_COINS_TO_SCAN, VOLUME_REFRESH_INTERVAL

logger = logging.getLogger("ICT-Bot.DataFetcher")


class OKXDataFetcher:
    """OKX Public API'den gerçek zamanlı veri çeken sınıf"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "ICT-Trading-Bot/1.0"
        })
        self._cache = {}
        self._cache_ttl = 15  # saniye
        self._active_coins = []           # Dinamik coin listesi
        self._coins_last_refresh = 0      # Son yenileme zamanı
        self._coin_volumes = {}           # Coin -> hacim bilgisi

    def _make_request(self, endpoint, params=None):
        """API isteği gönder"""
        url = f"{OKX_API_V5}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == "0":
                return data.get("data", [])
            else:
                logger.warning(f"OKX API hatası: {data.get('msg', 'Bilinmeyen hata')}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"OKX API bağlantı hatası: {e}")
            return []

    def get_candles(self, symbol, timeframe="15m", limit=100):
        """
        Mum verilerini çek
        Returns: DataFrame [timestamp, open, high, low, close, volume]
        """
        cache_key = f"candles_{symbol}_{timeframe}_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        params = {
            "instId": symbol,
            "bar": timeframe,
            "limit": str(limit)
        }
        data = self._make_request("/market/candles", params)

        if not data:
            return pd.DataFrame()

        # OKX formatı: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close",
            "volume", "volCcy", "volCcyQuote", "confirm"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._set_cache(cache_key, df)
        return df

    def get_ticker(self, symbol):
        """Anlık fiyat bilgisi"""
        cache_key = f"ticker_{symbol}"
        cached = self._get_cached(cache_key, ttl=5)
        if cached is not None:
            return cached

        params = {"instId": symbol}
        data = self._make_request("/market/ticker", params)

        if data:
            ticker = {
                "symbol": data[0].get("instId"),
                "last": float(data[0].get("last", 0)),
                "bid": float(data[0].get("bidPx", 0)),
                "ask": float(data[0].get("askPx", 0)),
                "high24h": float(data[0].get("high24h", 0)),
                "low24h": float(data[0].get("low24h", 0)),
                "vol24h": float(data[0].get("vol24h", 0)),
                "change24h": float(data[0].get("last", 0)) - float(data[0].get("open24h", 0)),
                "changePct24h": ((float(data[0].get("last", 0)) - float(data[0].get("open24h", 1))) /
                                 float(data[0].get("open24h", 1))) * 100
                                 if float(data[0].get("open24h", 0)) > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            self._set_cache(cache_key, ticker, ttl=5)
            return ticker

        return None

    def get_all_tickers(self, inst_type="SPOT"):
        """Tüm USDT çiftlerinin gerçek zamanlı fiyat ve hacim verilerini çek"""
        cache_key = f"all_tickers_{inst_type}"
        cached = self._get_cached(cache_key, ttl=10)
        if cached is not None:
            return cached

        params = {"instType": inst_type}
        data = self._make_request("/market/tickers", params)

        tickers = {}
        for item in data:
            symbol = item.get("instId", "")
            if not symbol.endswith("-USDT"):
                continue

            last_price = float(item.get("last", 0))
            vol_coin = float(item.get("vol24h", 0))           # Coin cinsinden hacim
            vol_ccy = float(item.get("volCcy24h", 0))         # USDT cinsinden hacim
            open_24h = float(item.get("open24h", 0))

            # volCcy24h yoksa fiyat × adet ile hesapla
            volume_usdt = vol_ccy if vol_ccy > 0 else (vol_coin * last_price)

            change_pct = 0
            if open_24h > 0:
                change_pct = ((last_price - open_24h) / open_24h) * 100

            tickers[symbol] = {
                "symbol": symbol,
                "last": last_price,
                "vol24h": vol_coin,
                "vol24h_usdt": volume_usdt,
                "changePct24h": round(change_pct, 2)
            }

        self._set_cache(cache_key, tickers, ttl=10)
        return tickers

    # =================== DİNAMİK COİN LİSTESİ ===================

    def get_high_volume_coins(self, force_refresh=False):
        """
        OKX'ten 24 saatlik hacmi MIN_VOLUME_USDT üzerindeki USDT çiftlerini çek.
        Sonuçlar hacme göre sıralanır. Her VOLUME_REFRESH_INTERVAL saniyede yenilenir.
        Sabit/hardcoded liste YOKTUR - tamamen gerçek zamanlı.
        """
        now = time.time()
        if not force_refresh and self._active_coins and (now - self._coins_last_refresh) < VOLUME_REFRESH_INTERVAL:
            return self._active_coins

        logger.info(f"📊 OKX'ten yüksek hacimli coinler çekiliyor (min ${MIN_VOLUME_USDT:,.0f})...")

        tickers = self.get_all_tickers("SPOT")
        if not tickers:
            logger.warning("OKX ticker verisi alınamadı!")
            return self._active_coins if self._active_coins else []

        # Hacme göre filtrele ve sırala
        qualified = []
        for symbol, data in tickers.items():
            vol_usdt = data.get("vol24h_usdt", 0)
            if vol_usdt >= MIN_VOLUME_USDT:
                qualified.append({
                    "symbol": symbol,
                    "volume_usdt": vol_usdt,
                    "last_price": data["last"],
                    "change_pct": data["changePct24h"]
                })

        # Hacme göre büyükten küçüğe sırala
        qualified.sort(key=lambda x: x["volume_usdt"], reverse=True)

        # Maksimum coin sayısını uygula
        qualified = qualified[:MAX_COINS_TO_SCAN]

        self._active_coins = [c["symbol"] for c in qualified]
        self._coin_volumes = {c["symbol"]: c for c in qualified}
        self._coins_last_refresh = now

        logger.info(f"✅ {len(self._active_coins)} coin bulundu (24h hacim ≥ ${MIN_VOLUME_USDT:,.0f})")
        if self._active_coins:
            top3 = self._active_coins[:3]
            top3_info = [f"{s} (${self._coin_volumes[s]['volume_usdt']:,.0f})" for s in top3]
            logger.info(f"   En yüksek hacimli: {', '.join(top3_info)}")

        return self._active_coins

    def get_coin_volume_info(self, symbol):
        """Belirli bir coinin hacim bilgisini döndür"""
        return self._coin_volumes.get(symbol, None)

    def get_all_coin_volumes(self):
        """Tüm aktif coinlerin hacim bilgilerini döndür"""
        return self._coin_volumes

    def get_multi_timeframe_data(self, symbol):
        """
        Birden fazla zaman diliminde veri çek
        HTF (4H)  -> Yapı analizi + HTF Bias Gate
        MTF (1H)  -> Sinyal onayı + MTF trend kontrolü
        LTF (15m) -> Giriş noktası + Sweep/Displacement/FVG tespiti
        5m        -> Watchlist onay akışı (5 dakikalık mum takibi)
        """
        data = {}
        
        # 4 saatlik - HTF Bias (yapı analizi)
        data["4H"] = self.get_candles(symbol, "4H", 100)
        time.sleep(0.1)  # Rate limit

        # 1 saatlik - MTF (sinyal onayı)
        data["1H"] = self.get_candles(symbol, "1H", 100)
        time.sleep(0.1)

        # 15 dakikalık - LTF (giriş noktası)
        data["15m"] = self.get_candles(symbol, "15m", 100)
        time.sleep(0.1)

        # 5 dakikalık - Watchlist onay akışı
        data["5m"] = self.get_candles(symbol, "5m", 120)

        return data

    def get_orderbook(self, symbol, depth=20):
        """Order book verisi (destek/direnç seviyeleri için)"""
        params = {
            "instId": symbol,
            "sz": str(depth)
        }
        data = self._make_request("/market/books", params)

        if data:
            book = {
                "asks": [[float(x[0]), float(x[1])] for x in data[0].get("asks", [])],
                "bids": [[float(x[0]), float(x[1])] for x in data[0].get("bids", [])],
                "timestamp": datetime.now().isoformat()
            }
            return book
        return None

    # =================== CACHE ===================

    def _get_cached(self, key, ttl=None):
        if ttl is None:
            ttl = self._cache_ttl
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["time"] < ttl:
                return entry["data"]
        return None

    def _set_cache(self, key, data, ttl=None):
        self._cache[key] = {
            "data": data,
            "time": time.time()
        }

    def clear_cache(self):
        self._cache.clear()


# Global instance
data_fetcher = OKXDataFetcher()
