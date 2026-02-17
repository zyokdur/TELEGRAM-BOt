# =====================================================
# ICT Trading Bot - Ana Flask Uygulaması
# =====================================================

import eventlet
eventlet.monkey_patch()

import logging
import time
import json
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from config import (
    HOST, PORT, DEBUG,
    SCAN_INTERVAL_SECONDS, TRADE_CHECK_INTERVAL,
    OPTIMIZER_CONFIG, ICT_PARAMS, MIN_VOLUME_USDT
)
from database import (
    init_db, get_active_signals, get_signal_history,
    get_watching_items, get_optimization_logs,
    get_performance_summary, update_signal_status,
    get_bot_param, get_recently_expired
)
from data_fetcher import data_fetcher
from ict_strategy import ict_strategy
from trade_manager import trade_manager
from self_optimizer import self_optimizer
from market_regime import market_regime
from forex_ict import forex_ict, FOREX_INSTRUMENTS

# =================== LOGGING ===================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ICT-Bot")

# =================== FLASK APP ===================

app = Flask(__name__)
app.config["SECRET_KEY"] = "ict-bot-secret-2024"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Bot durumu
bot_state = {
    "running": False,
    "last_scan": None,
    "last_trade_check": None,
    "last_optimization": None,
    "scan_count": 0,
    "symbols_scanned": 0,
    "errors": []
}

scan_lock = threading.Lock()


# =================== ARKA PLAN GÖREVLERİ ===================

def scan_markets():
    """OKX'ten gerçek zamanlı 5M+ hacimli coinleri tara ve sinyal üret"""
    if not bot_state["running"]:
        return

    if not scan_lock.acquire(blocking=False):
        return

    try:
        logger.info("🔍 Piyasa taraması başlıyor...")
        bot_state["last_scan"] = datetime.now().isoformat()
        bot_state["scan_count"] += 1
        symbols_scanned = 0
        new_signals = []

        # OKX'ten hacimli coinleri gerçek zamanlı çek
        active_coins = data_fetcher.get_high_volume_coins()
        bot_state["active_coin_count"] = len(active_coins)

        if not active_coins:
            logger.warning("OKX'ten yüksek hacimli coin bulunamadı, bağlantı kontrol edin")
            return

        # ── Piyasa rejimi analizi (sadece bilgi amaçlı, ICT'yi filtrelemez) ──
        try:
            regime_result = market_regime.analyze_market(active_coins)
            regime = regime_result["regime"]
            bot_state["current_regime"] = regime
            bot_state["btc_bias"] = regime_result["btc_bias"]
            bot_state["long_candidates"] = len(regime_result["long_candidates"])
            bot_state["short_candidates"] = len(regime_result["short_candidates"])
            socketio.emit("regime_update", market_regime.get_regime_summary())
        except Exception as e:
            logger.warning(f"Rejim analizi hatası (tarama devam eder): {e}")
            regime = bot_state.get("current_regime", "UNKNOWN")

        # ── Tüm coinleri ICT ile tara (rejim filtresi yok) ──
        for symbol in active_coins:
            if market_regime._is_btc(symbol):
                continue  # BTC referans, sinyale gerek yok

            try:
                # Gerçek zamanlı çoklu zaman dilimi verisi çek
                multi_tf = data_fetcher.get_multi_timeframe_data(symbol)
                ltf_data = multi_tf.get("15m")

                if ltf_data is None or ltf_data.empty:
                    continue

                # ICT strateji analizi — tüm yönler serbest
                result = ict_strategy.generate_signal(symbol, ltf_data, multi_tf)

                if result:
                    trade_result = trade_manager.process_signal(result)
                    if trade_result:
                        trade_result["regime"] = regime
                        new_signals.append(trade_result)
                        socketio.emit("new_signal", trade_result)

                symbols_scanned += 1
                time.sleep(0.15)  # Rate limit

            except Exception as e:
                logger.error(f"Hata ({symbol}): {e}")
                bot_state["errors"].append({
                    "time": datetime.now().isoformat(),
                    "symbol": symbol,
                    "error": str(e)
                })
                bot_state["errors"] = bot_state["errors"][-20:]

        bot_state["symbols_scanned"] = symbols_scanned
        logger.info(f"✅ Tarama tamamlandı: {symbols_scanned} coin, {len(new_signals)} sinyal | Rejim: {regime}")

        # Dashboard güncelle
        socketio.emit("scan_complete", {
            "symbols_scanned": symbols_scanned,
            "new_signals": len(new_signals),
            "regime": regime,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Tarama hatası: {e}")
    finally:
        scan_lock.release()


def check_trades():
    """Açık işlemleri kontrol et"""
    if not bot_state["running"]:
        return

    try:
        results = trade_manager.check_open_trades()
        bot_state["last_trade_check"] = datetime.now().isoformat()

        # Kapanan işlemleri bildir
        for r in results:
            if r["status"] in ["WON", "LOST"]:
                socketio.emit("trade_closed", r)

        # İzleme listesini kontrol et
        promoted = trade_manager.check_watchlist(ict_strategy)
        for p in promoted:
            socketio.emit("watch_promoted", p)

        # Dashboard güncelle
        socketio.emit("trades_updated", {
            "active_results": results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"İşlem kontrol hatası: {e}")


def run_optimizer():
    """Otomatik optimizasyonu çalıştır"""
    if not bot_state["running"]:
        return

    try:
        # ICT Optimizer
        result = self_optimizer.run_optimization()
        bot_state["last_optimization"] = datetime.now().isoformat()

        if result["changes"]:
            ict_strategy.reload_params()
            socketio.emit("optimization_done", result)
            logger.info(f"🧠 ICT Optimizasyon: {len(result['changes'])} değişiklik")

    except Exception as e:
        logger.error(f"Optimizasyon hatası: {e}")


# Scheduler - her start/stop döngüsünde yeniden oluşturulur
scheduler = None

def create_scheduler():
    """Yeni scheduler oluştur (shutdown sonrası yeniden kullanılamaz)"""
    global scheduler
    scheduler = BackgroundScheduler()
    return scheduler



# =================== API ROUTES ===================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    """Health check — Render uyku engelleme ve durum kontrolü"""
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})


@app.route("/api/status")
def api_status():
    """Bot durumu"""
    active_coins = data_fetcher.get_high_volume_coins()
    return jsonify({
        "running": bot_state["running"],
        "last_scan": bot_state["last_scan"],
        "last_trade_check": bot_state["last_trade_check"],
        "last_optimization": bot_state["last_optimization"],
        "scan_count": bot_state["scan_count"],
        "symbols_scanned": bot_state["symbols_scanned"],
        "watchlist_count": len(active_coins),
        "min_volume": MIN_VOLUME_USDT,
        "current_regime": bot_state.get("current_regime", "UNKNOWN"),
        "btc_bias": bot_state.get("btc_bias", "UNKNOWN"),
        "long_candidates": bot_state.get("long_candidates", 0),
        "short_candidates": bot_state.get("short_candidates", 0),
        "server_time": datetime.now().isoformat()
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    """Botu başlat"""
    if bot_state["running"]:
        return jsonify({"status": "already_running"})

    bot_state["running"] = True

    # Yeni scheduler oluştur ve görevleri ekle
    create_scheduler()
    scheduler.add_job(scan_markets, "interval", seconds=SCAN_INTERVAL_SECONDS,
                     id="scan_markets", replace_existing=True)
    scheduler.add_job(check_trades, "interval", seconds=TRADE_CHECK_INTERVAL,
                     id="check_trades", replace_existing=True)
    scheduler.add_job(run_optimizer, "interval",
                     minutes=OPTIMIZER_CONFIG["optimization_interval_minutes"],
                     id="run_optimizer", replace_existing=True)
    scheduler.start()

    # İlk taramayı hemen yap
    threading.Thread(target=scan_markets, daemon=True).start()

    logger.info("🚀 Bot başlatıldı!")
    socketio.emit("bot_status", {"running": True})

    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Botu durdur"""
    bot_state["running"] = False

    global scheduler
    if scheduler and scheduler.running:
        try:
            scheduler.remove_all_jobs()
            scheduler.shutdown(wait=False)
        except Exception:
            pass
    scheduler = None

    logger.info("🛑 Bot durduruldu!")
    socketio.emit("bot_status", {"running": False})

    return jsonify({"status": "stopped"})


@app.route("/api/signals/active")
def api_active_signals():
    """Aktif sinyaller"""
    signals = get_active_signals()
    # Her sinyal için güncel fiyat ekle
    for s in signals:
        ticker = data_fetcher.get_ticker(s["symbol"])
        if ticker:
            s["current_price"] = ticker["last"]
            entry = s["entry_price"]
            if s["direction"] == "LONG":
                s["unrealized_pnl"] = round(((ticker["last"] - entry) / entry) * 100, 2)
            else:
                s["unrealized_pnl"] = round(((entry - ticker["last"]) / entry) * 100, 2)
        else:
            s["current_price"] = None
            s["unrealized_pnl"] = 0
    return jsonify(signals)


@app.route("/api/signals/history")
def api_signal_history():
    """Sinyal geçmişi"""
    limit = request.args.get("limit", 50, type=int)
    history = get_signal_history(limit)
    return jsonify(history)


@app.route("/api/watchlist")
def api_watchlist():
    """İzleme listesi + son expired"""
    items = get_watching_items()
    return jsonify(items)


@app.route("/api/watchlist/expired")
def api_watchlist_expired():
    """Son 30 dakikada expire edilen öğeler (neden bilgisiyle)"""
    minutes = request.args.get("minutes", 30, type=int)
    items = get_recently_expired(minutes)
    return jsonify(items)


@app.route("/api/performance")
def api_performance():
    """Performans istatistikleri"""
    stats = get_performance_summary()
    return jsonify(stats)


@app.route("/api/optimization/logs")
def api_optimization_logs():
    """Optimizasyon logları"""
    limit = request.args.get("limit", 30, type=int)
    logs = get_optimization_logs(limit)
    return jsonify(logs)


@app.route("/api/optimization/summary")
def api_optimization_summary():
    """Optimizasyon özeti"""
    summary = self_optimizer.get_optimization_summary()
    return jsonify(summary)


@app.route("/api/optimization/run", methods=["POST"])
def api_run_optimization():
    """Manuel optimizasyon tetikle"""
    result = self_optimizer.run_optimization()
    if result["changes"]:
        ict_strategy.reload_params()
    return jsonify(result)


@app.route("/api/signal/<int:signal_id>/cancel", methods=["POST"])
def api_cancel_signal(signal_id):
    """Sinyali iptal et"""
    update_signal_status(signal_id, "CANCELLED")
    return jsonify({"status": "cancelled", "signal_id": signal_id})


@app.route("/api/analyze/<symbol>")
def api_analyze_symbol(symbol):
    """Tek bir coini analiz et"""
    try:
        multi_tf = data_fetcher.get_multi_timeframe_data(symbol)
        ltf_data = multi_tf.get("15m")

        if ltf_data is None or ltf_data.empty:
            return jsonify({"error": "Veri alınamadı"}), 400

        analysis = ict_strategy.calculate_confluence(ltf_data, multi_tf)

        # Timestamp'leri string'e çevir
        def serialize(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        return jsonify(json.loads(json.dumps(analysis, default=serialize)))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/coin-detail/<symbol>")
def api_coin_detail(symbol):
    """
    Gelişmiş coin detay popup: çoklu TF teknik analiz.
    RSI, Stochastic RSI, MACD, Bollinger Bands, ADX, ATR,
    OBV, Volume, FVG, Support/Resistance, Diverjans, Order Book,
    trend yapısı ve ağırlıklı güven skoru ile genel yorum.
    """
    import numpy as np

    # ── TEMEL HESAPLAMA FONKSİYONLARI ──

    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _macd(series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _stoch_rsi(series, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """Stochastic RSI — RSI'nin RSI'si, daha hassas aşırı alım/satım"""
        rsi = _rsi(series, rsi_period)
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        stoch = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100
        k = stoch.rolling(window=k_period).mean()
        d = k.rolling(window=d_period).mean()
        return k, d

    def _bollinger_bands(series, period=20, std_dev=2):
        """Bollinger Bands — volatilite ve fiyat pozisyonu"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    def _adx(df, period=14):
        """ADX — Trend gücü ölçümü (0-100)"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan))

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()
        return adx, plus_di, minus_di

    def _atr(df, period=14):
        """ATR — Volatilite ölçümü"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, min_periods=period).mean()

    def _obv(df):
        """OBV — On Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    def _find_support_resistance(df, lookback=50):
        """Pivot tabanlı destek/direnç seviyeleri"""
        if len(df) < lookback:
            lookback = len(df)
        recent = df.iloc[-lookback:]
        supports = []
        resistances = []
        for i in range(2, len(recent) - 2):
            # Pivot Low (destek)
            if (recent["low"].iloc[i] < recent["low"].iloc[i-1] and
                recent["low"].iloc[i] < recent["low"].iloc[i-2] and
                recent["low"].iloc[i] < recent["low"].iloc[i+1] and
                recent["low"].iloc[i] < recent["low"].iloc[i+2]):
                supports.append(recent["low"].iloc[i])
            # Pivot High (direnç)
            if (recent["high"].iloc[i] > recent["high"].iloc[i-1] and
                recent["high"].iloc[i] > recent["high"].iloc[i-2] and
                recent["high"].iloc[i] > recent["high"].iloc[i+1] and
                recent["high"].iloc[i] > recent["high"].iloc[i+2]):
                resistances.append(recent["high"].iloc[i])
        return supports, resistances

    def _detect_divergence(price_series, indicator_series, lookback=20):
        """RSI/MACD diverjans tespiti"""
        if len(price_series) < lookback or len(indicator_series) < lookback:
            return None

        price = price_series.iloc[-lookback:]
        ind = indicator_series.iloc[-lookback:]

        # Son 2 swing low/high bul
        price_lows = []
        price_highs = []
        for i in range(2, len(price) - 2):
            if price.iloc[i] < price.iloc[i-1] and price.iloc[i] < price.iloc[i+1]:
                price_lows.append((i, price.iloc[i], ind.iloc[i]))
            if price.iloc[i] > price.iloc[i-1] and price.iloc[i] > price.iloc[i+1]:
                price_highs.append((i, price.iloc[i], ind.iloc[i]))

        # Bullish divergence: Fiyat düşük dip, RSI yüksek dip
        if len(price_lows) >= 2:
            last = price_lows[-1]
            prev = price_lows[-2]
            if last[1] < prev[1] and last[2] > prev[2]:
                return {
                    "type": "BULLISH",
                    "label": "Boğa Diverjansı",
                    "desc": "Fiyat düşük dip yaparken gösterge yüksek dip yapıyor — gizli alım gücü, dönüş sinyali.",
                    "color": "green"
                }

        # Bearish divergence: Fiyat yüksek tepe, RSI düşük tepe
        if len(price_highs) >= 2:
            last = price_highs[-1]
            prev = price_highs[-2]
            if last[1] > prev[1] and last[2] < prev[2]:
                return {
                    "type": "BEARISH",
                    "label": "Ayı Diverjansı",
                    "desc": "Fiyat yüksek tepe yaparken gösterge düşük tepe yapıyor — gizli satış baskısı, dönüş sinyali.",
                    "color": "red"
                }

        return {"type": "NONE", "label": "Diverjans yok", "desc": "Fiyat ve göstergeler uyumlu hareket ediyor.", "color": "gray"}

    # ── YORUM FONKSİYONLARI ──

    def _interpret_rsi(val):
        if val is None or np.isnan(val):
            return {"value": None, "label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        val = round(val, 2)
        if val >= 80:
            return {"value": val, "label": "Aşırı Alım (Güçlü)", "signal": "BEARISH", "color": "red",
                    "desc": f"RSI {val} — Çok güçlü aşırı alım. Fiyat sürdürülemez seviyede gerilmiş. Hacim düşüşü ve mum formasyonuyla birlikte geri çekilme olasılığı çok yüksek. Yeni LONG açmayın."}
        elif val >= 70:
            return {"value": val, "label": "Aşırı Alım", "signal": "BEARISH", "color": "red",
                    "desc": f"RSI {val} — Aşırı alım bölgesi. Güçlü trendlerde RSI 70+ kalabilir ancak momentum zayıflarsa düzeltme kaçınılmaz. Kısa vade dikkat."}
        elif val >= 60:
            return {"value": val, "label": "Boğa Momentumu", "signal": "BULLISH", "color": "green",
                    "desc": f"RSI {val} — Alıcılar güçlü pozisyonda. Fiyat-hacim uyumuna bakarak trendin sağlığını doğrulayın."}
        elif val >= 45:
            return {"value": val, "label": "Nötr Bölge", "signal": "NEUTRAL", "color": "gray",
                    "desc": f"RSI {val} — Denge bölgesi, piyasa kararsız. Tek başına RSI yön vermez, diğer göstergelerle birlikte değerlendirin."}
        elif val >= 30:
            return {"value": val, "label": "Ayı Momentumu", "signal": "BEARISH", "color": "orange",
                    "desc": f"RSI {val} — Satıcılar baskın. Düşüş trendi aktif, karşı yönde işlem riskli. Destek seviyelerini izleyin."}
        elif val >= 20:
            return {"value": val, "label": "Aşırı Satım", "signal": "BULLISH", "color": "green",
                    "desc": f"RSI {val} — Aşırı satım bölgesi. Destek seviyesiyle birleşirse alım fırsatı olabilir. Tek başına yeterli değil, hacim onayı şart."}
        else:
            return {"value": val, "label": "Aşırı Satım (Güçlü)", "signal": "BULLISH", "color": "green",
                    "desc": f"RSI {val} — Çok güçlü aşırı satım. Teknik tepki beklenir ancak düşüş devam edebilir. Hacim ve diverjans onayı olmadan körlemesine LONG açmayın."}

    def _interpret_stoch_rsi(k_val, d_val):
        if k_val is None or np.isnan(k_val):
            return {"k": None, "d": None, "label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        k_r = round(k_val, 2)
        d_r = round(d_val, 2) if d_val is not None and not np.isnan(d_val) else None
        result = {"k": k_r, "d": d_r}

        if k_r >= 80 and d_r and k_r < d_r:
            result.update({"label": "Aşırı Alım + Çapraz Aşağı", "signal": "BEARISH", "color": "red",
                          "desc": f"StochRSI K:{k_r} D:{d_r} — K çizgisi D'yi 80 üzerinde aşağı kesti. Kısa vadeli dönüş olasılığı yüksek. RSI ve hacimle birlikte değerlendirin."})
        elif k_r >= 80:
            result.update({"label": "Aşırı Alım Bölgesi", "signal": "NEUTRAL", "color": "orange",
                          "desc": f"StochRSI K:{k_r} — Aşırı alım bölgesinde. Tek başına satış sinyali değil, çapraz aşağı kesimi bekleyin."})
        elif k_r <= 20 and d_r and k_r > d_r:
            result.update({"label": "Aşırı Satım + Çapraz Yukarı", "signal": "BULLISH", "color": "green",
                          "desc": f"StochRSI K:{k_r} D:{d_r} — K çizgisi D'yi 20 altında yukarı kesti. Teknik olarak güçlü alım sinyali, ancak trend yönüne karşıysa dikkat."})
        elif k_r <= 20:
            result.update({"label": "Aşırı Satım Bölgesi", "signal": "NEUTRAL", "color": "green",
                          "desc": f"StochRSI K:{k_r} — Aşırı satım. Çapraz yukarı bekleniyor. Hareket başlayana kadar sinyal net değil."})
        elif k_r > 55 and d_r and k_r > d_r:
            result.update({"label": "Boğa Momentumu", "signal": "BULLISH", "color": "green",
                          "desc": f"StochRSI K:{k_r} > D:{d_r} — Kısa vadeli momentum alıcı lehine, devam sinyali."})
        elif k_r < 45 and d_r and k_r < d_r:
            result.update({"label": "Ayı Momentumu", "signal": "BEARISH", "color": "orange",
                          "desc": f"StochRSI K:{k_r} < D:{d_r} — Kısa vadeli momentum satıcı lehine."})
        else:
            result.update({"label": "Nötr", "signal": "NEUTRAL", "color": "gray",
                          "desc": f"StochRSI K:{k_r} — Kararsız bölge. Net sinyal için aşırı bölgelerden çapraz bekleyin."})
        return result

    def _interpret_macd(macd_val, signal_val, hist_val, prev_hist=None):
        if macd_val is None or np.isnan(macd_val):
            return {"macd": None, "signal": None, "histogram": None,
                    "label": "Veri yok", "signal_type": "NEUTRAL", "color": "gray"}
        result = {
            "macd": round(macd_val, 6), "signal": round(signal_val, 6),
            "histogram": round(hist_val, 6)
        }
        if hist_val > 0 and (prev_hist is not None and prev_hist <= 0):
            result.update({"label": "Boğa Kesişimi ↑", "signal_type": "BULLISH", "color": "green",
                          "desc": "MACD histogram pozitife döndü — taze alım sinyali. En güvenilir MACD sinyallerinden biri. Hacim artışıyla desteklenirse güçlü."})
        elif hist_val < 0 and (prev_hist is not None and prev_hist >= 0):
            result.update({"label": "Ayı Kesişimi ↓", "signal_type": "BEARISH", "color": "red",
                          "desc": "MACD histogram negatife döndü — taze satım sinyali. Trend dönüşü veya düzeltme başlangıcı."})
        elif hist_val > 0:
            if prev_hist is not None and hist_val > prev_hist:
                result.update({"label": "Boğa Güçleniyor ↗", "signal_type": "BULLISH", "color": "green",
                              "desc": "MACD histogram pozitif ve büyüyor — momentum sağlıklı artıyor. Mevcut LONG pozisyon korunabilir."})
            else:
                result.update({"label": "Boğa Zayıflıyor ↘", "signal_type": "WEAKENING_BULL", "color": "orange",
                              "desc": "MACD histogram pozitif ama daralmaya başladı — yükseliş hız kesiyor. Yeni giriş için erken, çıkış planı hazırlayın."})
        elif hist_val < 0:
            if prev_hist is not None and hist_val < prev_hist:
                result.update({"label": "Ayı Güçleniyor ↘", "signal_type": "BEARISH", "color": "red",
                              "desc": "MACD histogram negatif ve büyüyor — düşüş ivmeleniyor. Kısa vadeli destek kırılabilir."})
            else:
                result.update({"label": "Ayı Zayıflıyor ↗", "signal_type": "WEAKENING_BEAR", "color": "orange",
                              "desc": "MACD histogram daralmaya başladı — satış baskısı azalıyor. Sabırlı ol, pozitife geçiş onayı bekle."})
        else:
            result.update({"label": "Nötr ─", "signal_type": "NEUTRAL", "color": "gray",
                          "desc": "MACD sıfır çizgisinde — yön kararı yaklaşıyor. İlk hareketi bekleyin."})
        return result

    def _interpret_bb(close_val, upper, middle, lower, bb_width, prev_width=None):
        """Bollinger Bands yorumu"""
        if close_val is None or np.isnan(close_val):
            return {"label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        pct_b = ((close_val - lower) / (upper - lower)) * 100 if (upper - lower) > 0 else 50
        result = {
            "upper": round(upper, 8), "middle": round(middle, 8), "lower": round(lower, 8),
            "width": round(bb_width, 4), "pct_b": round(pct_b, 1)
        }

        # Squeeze tespiti
        is_squeeze = bb_width < 0.02  # Sıkışma (coin'e göre normalize edilmiş)
        if prev_width and bb_width > prev_width * 1.5:
            result["squeeze_status"] = "PATLAMA"
            result["squeeze_desc"] = f"Bantlar genişliyor (genişlik: {bb_width:.4f} → önceki: {prev_width:.4f}). Sıkışma patlaması sinyali — güçlü yönlü hareket başladı. Kırılım yönünde pozisyon almak için diğer göstergelerle onaylayın."
        elif is_squeeze:
            result["squeeze_status"] = "SIKILIK"
            result["squeeze_desc"] = f"Bant genişliği çok dar ({bb_width:.4f}). Patlama öncesi sıkışma — büyük bir hareket kapıda. Kırılım yönünü tahmin etmeyin, kırılım sonrası girin."
        else:
            result["squeeze_status"] = "NORMAL"
            result["squeeze_desc"] = f"Bantlar normal genişlikte ({bb_width:.4f}). Olağan volatilite — trend takip stratejileri uygulanabilir."

        if pct_b >= 95:
            result.update({"label": "Üst Bant Üzerinde", "signal": "BEARISH", "color": "red",
                          "desc": f"%B: {pct_b:.0f} — Fiyat üst bandın üzerine taştı. Aşırı alım bölgesi, geri çekilme olasılığı yüksek. Ancak güçlü trendlerde bant üzerinde yürüyüş ('walking the band') olabilir — ADX'e bakın."})
        elif pct_b >= 80:
            result.update({"label": "Üst Banda Yakın", "signal": "NEUTRAL", "color": "orange",
                          "desc": f"%B: {pct_b:.0f} — Fiyat üst bant bölgesinde. Trend güçlüyse (ADX>25) devam edebilir, trend zayıfsa geri çekilme riski var. Tek başına sinyal olarak yeterli değil."})
        elif pct_b <= 5:
            result.update({"label": "Alt Bant Altında", "signal": "BULLISH", "color": "green",
                          "desc": f"%B: {pct_b:.0f} — Fiyat alt bandın altına düştü. Aşırı satım bölgesi — tepki yükselişi gelebilir. Ama düşüş trendi güçlüyse bant altında yürüyüş de olabilir."})
        elif pct_b <= 20:
            result.update({"label": "Alt Banda Yakın", "signal": "NEUTRAL", "color": "green",
                          "desc": f"%B: {pct_b:.0f} — Fiyat alt bant bölgesinde. Potansiyel destek alanı ama körlemesine alım yapmayın — RSI ve hacim onayı gerekli."})
        else:
            result.update({"label": "Orta Bölge", "signal": "NEUTRAL", "color": "gray",
                          "desc": f"%B: {pct_b:.0f} — Fiyat bantların ortasında, BB'den anlamlı sinyal yok. Diğer göstergelere bakın."})
        return result

    def _interpret_adx(adx_val, plus_di, minus_di):
        """ADX yorumu — trend gücü ve yönü"""
        if adx_val is None or np.isnan(adx_val):
            return {"adx": None, "label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        adx_r = round(adx_val, 1)
        pdi_r = round(plus_di, 1)
        mdi_r = round(minus_di, 1)
        result = {"adx": adx_r, "plus_di": pdi_r, "minus_di": mdi_r}

        # DI yönü — fark büyüklüğü de önemli
        di_diff = abs(pdi_r - mdi_r)
        if pdi_r > mdi_r:
            di_dir = "BULLISH"
            di_label = f"+DI ({pdi_r}) > -DI ({mdi_r}), fark: {di_diff:.1f}"
        else:
            di_dir = "BEARISH"
            di_label = f"-DI ({mdi_r}) > +DI ({pdi_r}), fark: {di_diff:.1f}"
        result["di_direction"] = di_dir
        result["di_label"] = di_label

        # DI farkı çok düşükse (< 5) yön güvenilir değil
        di_weak = di_diff < 5

        if adx_r >= 50:
            result.update({"label": f"Çok Güçlü Trend ({adx_r})", "signal": di_dir, "color": "green" if di_dir == "BULLISH" else "red",
                          "desc": f"ADX {adx_r} — Çok güçlü trend aktif. {di_label}. Trende KARŞI pozisyon almak çok riskli. Trend yönünde geri çekilmeleri fırsat olarak değerlendirin."})
        elif adx_r >= 25:
            if di_weak:
                result.update({"label": f"Trend Var, Yön Belirsiz ({adx_r})", "signal": "NEUTRAL", "color": "orange",
                              "desc": f"ADX {adx_r} — Trend gücü yeterli AMA +DI ve -DI çok yakın (fark: {di_diff:.1f}). Yön netleşene kadar bekleyin."})
            else:
                result.update({"label": f"Güçlü Trend ({adx_r})", "signal": di_dir, "color": "green" if di_dir == "BULLISH" else "red",
                              "desc": f"ADX {adx_r} — Belirgin trend var. {di_label}. Trend yönünde pozisyon alınabilir, trende karşı gidenler zarar eder."})
        elif adx_r >= 20:
            result.update({"label": f"Gelişen Trend ({adx_r})", "signal": "NEUTRAL", "color": "orange",
                          "desc": f"ADX {adx_r} — Trend oluşmaya başlıyor ama henüz olgunlaşmadı. {di_label}. DI kesişimi ve ADX>25 onayını bekleyin."})
        else:
            result.update({"label": f"Trendsiz/Yatay ({adx_r})", "signal": "NEUTRAL", "color": "gray",
                          "desc": f"ADX {adx_r} — Piyasada güçlü trend yok, yatay hareket. Trend takip stratejileri çalışmaz — range (destek-direnç arası) alım-satım stratejisi uygulayın."})
        return result

    def _interpret_atr(atr_val, close_val):
        """ATR yorumu — volatilite yüzdesi ve pozisyon boyutlandırma"""
        if atr_val is None or np.isnan(atr_val) or close_val == 0:
            return {"atr": None, "label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        atr_pct = (atr_val / close_val) * 100
        suggested_sl = round(atr_val * 1.5, 8)  # 1.5x ATR stop-loss önerisi
        result = {"atr": round(atr_val, 8), "atr_pct": round(atr_pct, 2), "suggested_sl_distance": suggested_sl}
        if atr_pct >= 5:
            result.update({"label": f"Çok Yüksek Volatilite (%{atr_pct:.1f})", "signal": "HIGH", "color": "red",
                          "desc": f"ATR fiyatın %{atr_pct:.1f}'i — Çok yüksek volatilite! Pozisyon boyutunu normalin %50'sine düşürün. SL en az 1.5x ATR ({suggested_sl}) uzaklıkta olmalı. Likidasyon riski yüksek."})
        elif atr_pct >= 3:
            result.update({"label": f"Yüksek Volatilite (%{atr_pct:.1f})", "signal": "HIGH", "color": "orange",
                          "desc": f"ATR fiyatın %{atr_pct:.1f}'i — Yüksek volatilite, normal SL tetiklenebilir. Geniş SL kullanın (önerilen: {suggested_sl}). Kaldıraç düşük tutun."})
        elif atr_pct >= 1:
            result.update({"label": f"Normal Volatilite (%{atr_pct:.1f})", "signal": "NORMAL", "color": "gray",
                          "desc": f"ATR fiyatın %{atr_pct:.1f}'i — Normal piyasa koşulları. Standart pozisyon boyutu ve SL ({suggested_sl}) uygulanabilir."})
        else:
            result.update({"label": f"Düşük Volatilite (%{atr_pct:.1f})", "signal": "LOW", "color": "blue",
                          "desc": f"ATR fiyatın %{atr_pct:.1f}'i — Düşük volatilite, sıkışma patlaması yaklaşıyor olabilir. Dar SL ile kırılım pozisyonu planlanabilir."})
        return result

    def _interpret_obv(obv_series, price_series):
        """OBV yorumu — akıllı para akışı ve fiyat-hacim uyumsuzluğu"""
        if len(obv_series) < 10:
            return {"label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        obv_sma = obv_series.rolling(10).mean()
        obv_now = obv_series.iloc[-1]
        obv_sma_now = obv_sma.iloc[-1]
        obv_5_ago = obv_series.iloc[-5] if len(obv_series) >= 5 else obv_now
        obv_10_ago = obv_series.iloc[-10] if len(obv_series) >= 10 else obv_now
        price_5_ago = price_series.iloc[-5] if len(price_series) >= 5 else price_series.iloc[-1]
        price_now = price_series.iloc[-1]

        obv_trend = "UP" if obv_now > obv_5_ago else "DOWN"
        obv_long_trend = "UP" if obv_now > obv_10_ago else "DOWN"
        price_trend = "UP" if price_now > price_5_ago else "DOWN"
        price_change_pct = round(((price_now - price_5_ago) / price_5_ago) * 100, 2) if price_5_ago > 0 else 0

        result = {
            "obv_trend": obv_trend,
            "obv_long_trend": obv_long_trend,
            "above_sma": bool(obv_now > obv_sma_now),
        }

        # Diverjans kontrolü — en güvenilir OBV sinyali
        if price_trend == "UP" and obv_trend == "DOWN":
            result.update({"label": "Ayı Diverjansı (OBV)", "signal": "BEARISH", "color": "red",
                          "desc": f"Fiyat %{price_change_pct:+.1f} yükselirken OBV düşüyor. Akıllı para yükselişe katılmıyor, ralli'ye satıyor. Bu bir gizli satış sinyali — yükseliş sürdürülebilir değil."})
        elif price_trend == "DOWN" and obv_trend == "UP":
            result.update({"label": "Boğa Diverjansı (OBV)", "signal": "BULLISH", "color": "green",
                          "desc": f"Fiyat %{price_change_pct:+.1f} düşerken OBV yükseliyor. Akıllı para düşüşte sessizce biriktiriyor. Bu bir gizli alım sinyali — dönüş yakın olabilir."})
        elif obv_now > obv_sma_now and obv_trend == "UP":
            result.update({"label": "Güçlü Alım Akışı", "signal": "BULLISH", "color": "green",
                          "desc": f"OBV ortalamanın üzerinde ve yükseliyor. Hacim fiyatı destekliyor — sağlıklı bir yükseliş. Para akışı boğa yönünde."})
        elif obv_now < obv_sma_now and obv_trend == "DOWN":
            result.update({"label": "Güçlü Satış Akışı", "signal": "BEARISH", "color": "red",
                          "desc": f"OBV ortalamanın altında ve düşüyor. Sürekli satış baskısı var — para bu coin'den çıkıyor. Long pozisyonlardan kaçının."})
        elif obv_now > obv_sma_now and obv_trend == "DOWN":
            result.update({"label": "Zayıflayan Alım", "signal": "NEUTRAL", "color": "orange",
                          "desc": f"OBV hala ortalamanın üzerinde ama düşüyor. Alım baskısı zayıflıyor — yükselişin sonu yaklaşıyor olabilir."})
        elif obv_now < obv_sma_now and obv_trend == "UP":
            result.update({"label": "Toparlanma Sinyali", "signal": "NEUTRAL", "color": "orange",
                          "desc": f"OBV ortalamanın altında ama yükseliyor. Satış baskısı azalıyor — henüz alım sinyali değil ama izlemeye değer."})
        else:
            result.update({"label": "Nötr Hacim Akışı", "signal": "NEUTRAL", "color": "gray",
                          "desc": "OBV dengede — belirgin bir para akışı yok. Büyük oyuncularda henüz net bir pozisyonlanma görülmüyor."})
        return result

    def _analyze_volume(df):
        if df.empty or "volume" not in df.columns or len(df) < 20:
            return {"label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        vol = df["volume"]
        current_vol = vol.iloc[-1]
        avg_vol_20 = vol.iloc[-20:].mean()
        avg_vol_5 = vol.iloc[-5:].mean()
        ratio = round(current_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0
        trend_ratio = round(avg_vol_5 / avg_vol_20, 2) if avg_vol_20 > 0 else 1.0
        price_change = ((df["close"].iloc[-1] - df["close"].iloc[-5]) / df["close"].iloc[-5] * 100) if len(df) >= 5 else 0

        result = {"current": round(current_vol, 2), "avg_20": round(avg_vol_20, 2),
                  "ratio": ratio, "trend_ratio": trend_ratio}

        if ratio >= 2.5:
            result.update({"label": "Çok Yüksek Hacim", "signal": "HIGH", "color": "green",
                          "desc": f"Hacim ortalamanın {ratio}x katında — güçlü kurumsal hareket sinyali."})
        elif ratio >= 1.5:
            result.update({"label": "Yüksek Hacim", "signal": "HIGH", "color": "green",
                          "desc": f"Hacim normalin %{round((ratio-1)*100)} üzerinde — artan piyasa ilgisi."})
        elif ratio >= 0.8:
            result.update({"label": "Normal Hacim", "signal": "NEUTRAL", "color": "gray",
                          "desc": "Hacim ortalama seviyede — olağan piyasa aktivitesi."})
        else:
            result.update({"label": "Düşük Hacim", "signal": "LOW", "color": "orange",
                          "desc": f"Hacim ortalamanın %{round(ratio*100)}'i — zayıf ilgi, fake-out riski yüksek."})

        if trend_ratio >= 1.3:
            result["trend"] = "ARTIYOR"
            result["trend_desc"] = "Son 5 mum hacmi yükseliyor — momentum artıyor."
        elif trend_ratio <= 0.7:
            result["trend"] = "AZALIYOR"
            result["trend_desc"] = "Son 5 mum hacmi düşüyor — momentum zayıflıyor."
        else:
            result["trend"] = "STABİL"
            result["trend_desc"] = "Hacim dengeli — belirgin bir değişiklik yok."

        if price_change > 0 and ratio >= 1.5:
            result["price_vol_harmony"] = "Fiyat ↑ + Yüksek hacim = SAĞLIKLI YÜKSELİŞ ✓"
        elif price_change < 0 and ratio >= 1.5:
            result["price_vol_harmony"] = "Fiyat ↓ + Yüksek hacim = GÜÇLÜ SATIŞ BASKISI ✗"
        elif price_change > 0 and ratio < 0.8:
            result["price_vol_harmony"] = "Fiyat ↑ + Düşük hacim = ZAYIF RALLY ⚠ (dikkat!)"
        elif price_change < 0 and ratio < 0.8:
            result["price_vol_harmony"] = "Fiyat ↓ + Düşük hacim = İlgi kaybı, yatay beklentisi"
        else:
            result["price_vol_harmony"] = "Fiyat-hacim uyumu nötr"
        return result

    def _check_fvg(df):
        """ICT Fair Value Gap (FVG) analizi — kurumsal likidite boşlukları"""
        fvgs = {"bullish": [], "bearish": []}
        if len(df) < 3:
            return {"has_fvg": False, "label": "Veri yetersiz", "signal": "NEUTRAL", "color": "gray"}
        n = len(df)
        search_start = max(0, n - 30)  # Son 30 mum
        current_price = df["close"].iloc[-1]

        for i in range(search_start + 1, n - 1):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            next_ = df.iloc[i + 1]

            if prev["high"] < next_["low"]:
                gap = next_["low"] - prev["high"]
                gap_pct = (gap / curr["close"]) * 100
                if gap_pct >= 0.05:
                    filled = False
                    if i + 2 < n:
                        if df.iloc[i + 2:]["low"].min() <= prev["high"]:
                            filled = True
                    fvgs["bullish"].append({
                        "index": i, "gap_pct": round(gap_pct, 3),
                        "high": round(next_["low"], 8), "low": round(prev["high"], 8),
                        "filled": filled, "distance_bars": n - 1 - i
                    })

            if prev["low"] > next_["high"]:
                gap = prev["low"] - next_["high"]
                gap_pct = (gap / curr["close"]) * 100
                if gap_pct >= 0.05:
                    filled = False
                    if i + 2 < n:
                        if df.iloc[i + 2:]["high"].max() >= prev["low"]:
                            filled = True
                    fvgs["bearish"].append({
                        "index": i, "gap_pct": round(gap_pct, 3),
                        "high": round(prev["low"], 8), "low": round(next_["high"], 8),
                        "filled": filled, "distance_bars": n - 1 - i
                    })

        unfilled_bull = [f for f in fvgs["bullish"] if not f["filled"]]
        unfilled_bear = [f for f in fvgs["bearish"] if not f["filled"]]
        total_unfilled = len(unfilled_bull) + len(unfilled_bear)

        # En yakın FVG'leri bul
        nearest_bull = min(unfilled_bull, key=lambda x: x["distance_bars"]) if unfilled_bull else None
        nearest_bear = min(unfilled_bear, key=lambda x: x["distance_bars"]) if unfilled_bear else None

        result = {
            "has_fvg": total_unfilled > 0,
            "bullish_count": len(fvgs["bullish"]),
            "bearish_count": len(fvgs["bearish"]),
            "unfilled_bullish": len(unfilled_bull),
            "unfilled_bearish": len(unfilled_bear),
            "nearest_bull_price": nearest_bull["low"] if nearest_bull else None,
            "nearest_bear_price": nearest_bear["high"] if nearest_bear else None,
        }

        # FVG'ler fiyata yakınlık kontrolü
        bull_near_price = nearest_bull and abs(nearest_bull["low"] - current_price) / current_price * 100 < 1.5
        bear_near_price = nearest_bear and abs(nearest_bear["high"] - current_price) / current_price * 100 < 1.5

        if len(unfilled_bull) > len(unfilled_bear):
            proximity_note = ""
            if bull_near_price:
                proximity_note = f" En yakın boğa FVG ({nearest_bull['low']}) fiyata çok yakın — fiyat bu bölgeye çekilebilir (alım fırsatı)."
            result.update({"label": f"{len(unfilled_bull)} Boğa FVG", "signal": "BULLISH", "color": "green",
                          "desc": f"{len(unfilled_bull)} doldurulmamış boğa FVG tespit edildi. ICT teorisine göre fiyat bu boşlukları doldurmaya eğilimlidir.{proximity_note} FVG bölgelerinde limit emir konabilir."})
        elif len(unfilled_bear) > len(unfilled_bull):
            proximity_note = ""
            if bear_near_price:
                proximity_note = f" En yakın ayı FVG ({nearest_bear['high']}) fiyata çok yakın — fiyat bu bölgeye yükselebilir (satış bölgesi)."
            result.update({"label": f"{len(unfilled_bear)} Ayı FVG", "signal": "BEARISH", "color": "red",
                          "desc": f"{len(unfilled_bear)} doldurulmamış ayı FVG tespit edildi. Fiyat yukarı doğru bu boşlukları doldurup sonra dönebilir.{proximity_note} FVG bölgelerinde SHORT planlanabilir."})
        elif total_unfilled > 0:
            result.update({"label": f"{total_unfilled} FVG (karışık)", "signal": "NEUTRAL", "color": "orange",
                          "desc": f"Hem boğa ({len(unfilled_bull)}) hem ayı ({len(unfilled_bear)}) FVG mevcut — yön belirsiz. FVG'ler birbirini nötralize ediyor, diğer göstergelere bakın."})
        else:
            result.update({"label": "FVG Yok", "signal": "NEUTRAL", "color": "gray",
                          "desc": "Doldurulmamış FVG bulunamadı — tüm boşluklar kapanmış. Piyasa dengelenmiş durumda, yeni impuls hareketi bekleyin."})
        return result

    def _interpret_sr(supports, resistances, current_price):
        """Destek/direnç yorumu — pozisyon planlama için kritik seviyeler"""
        result = {"supports": [], "resistances": [], "nearest_support": None, "nearest_resistance": None}

        if supports:
            unique_s = sorted(set([round(s, 8) for s in supports if s < current_price]), reverse=True)[:3]
            result["supports"] = unique_s
            if unique_s:
                result["nearest_support"] = unique_s[0]
                dist_pct = ((current_price - unique_s[0]) / current_price) * 100
                result["support_dist_pct"] = round(dist_pct, 2)

        if resistances:
            unique_r = sorted(set([round(r, 8) for r in resistances if r > current_price]))[:3]
            result["resistances"] = unique_r
            if unique_r:
                result["nearest_resistance"] = unique_r[0]
                dist_pct = ((unique_r[0] - current_price) / current_price) * 100
                result["resistance_dist_pct"] = round(dist_pct, 2)

        # Konum analizi
        if result["nearest_support"] and result["nearest_resistance"]:
            s_dist = result.get("support_dist_pct", 100)
            r_dist = result.get("resistance_dist_pct", 100)
            total = s_dist + r_dist
            position = (s_dist / total * 100) if total > 0 else 50
            result["position_pct"] = round(position, 1)
            rr_ratio = round(r_dist / s_dist, 2) if s_dist > 0 else 0
            result["risk_reward"] = rr_ratio

            if s_dist < 0.3:
                result.update({"label": "Destek Üzerinde", "signal": "BULLISH", "color": "green",
                              "desc": f"Fiyat en yakın desteğe çok yakın (%{s_dist:.2f}). Burada tutunursa R/R: {rr_ratio:.1f}x ile alım fırsatı. SL destek altına konmalı."})
            elif r_dist < 0.3:
                result.update({"label": "Direnç Altında", "signal": "BEARISH", "color": "red",
                              "desc": f"Fiyat en yakın dirence çok yakın (%{r_dist:.2f}). Kırılamazsa geri çekilir. Burada yeni LONG açmak riskli — kırılım onayı bekleyin."})
            elif position < 25:
                result.update({"label": "Desteğe Yakın", "signal": "BULLISH", "color": "green",
                              "desc": f"Fiyat destek bölgesine yakın (destek: %{s_dist:.1f}, direnç: %{r_dist:.1f}). R/R: {rr_ratio:.1f}x — {'iyi alım bölgesi' if rr_ratio >= 2 else 'R/R oranı düşük, dikkatli olun'}."})
            elif position > 75:
                result.update({"label": "Direce Yakın", "signal": "BEARISH", "color": "orange",
                              "desc": f"Fiyat direnç bölgesine yakın (direnç: %{r_dist:.1f}, destek: %{s_dist:.1f}). R/R kötü — burada LONG açmak riskli. Direnç kırılırsa farklı hikaye."})
            else:
                result.update({"label": "Orta Bölge", "signal": "NEUTRAL", "color": "gray",
                              "desc": f"Fiyat destek (%{s_dist:.1f}) ve direnç (%{r_dist:.1f}) arasında ortada. R/R: {rr_ratio:.1f}x. Net alım/satım bölgesi değil — kenar seviyelere yaklaşana kadar bekleyin."})
        elif result["nearest_support"]:
            s_dist = result.get("support_dist_pct", 100)
            result.update({"label": f"Destek: {result['nearest_support']}", "signal": "NEUTRAL", "color": "gray",
                          "desc": f"Üst direnç tespit edilemedi. En yakın destek %{s_dist:.1f} aşağıda. Yeni zirve bölgesinde veya konsolidasyonda — kırılım yönünü bekleyin."})
        elif result["nearest_resistance"]:
            r_dist = result.get("resistance_dist_pct", 100)
            result.update({"label": f"Direnç: {result['nearest_resistance']}", "signal": "NEUTRAL", "color": "gray",
                          "desc": f"Alt destek tespit edilemedi. En yakın direnç %{r_dist:.1f} yukarıda. Düşüş sürecinde yeni dip aranıyor — LONG için acele etmeyin."})
        else:
            result.update({"label": "S/R Tespit Edilemedi", "signal": "NEUTRAL", "color": "gray",
                          "desc": "Yeterli pivot noktası bulunamadı. Veri yetersiz veya çok yatay hareket — bu göstergeden sinyal türetilemiyor."})
        return result

    # ── ANA ANALİZ FONKSİYONU ──

    def _analyze_tf(df, tf_label):
        """Tek TF için 10 gösterge ile gelişmiş teknik analiz"""
        if df is None or df.empty or len(df) < 30:
            return {
                "timeframe": tf_label, "error": "Yetersiz veri",
                "rsi": {"value": None, "label": "Veri yok", "signal": "NEUTRAL"},
                "stoch_rsi": {"label": "Veri yok", "signal": "NEUTRAL"},
                "macd": {"label": "Veri yok", "signal_type": "NEUTRAL"},
                "bollinger": {"label": "Veri yok", "signal": "NEUTRAL"},
                "adx": {"label": "Veri yok", "signal": "NEUTRAL"},
                "atr": {"label": "Veri yok", "signal": "NEUTRAL"},
                "obv": {"label": "Veri yok", "signal": "NEUTRAL"},
                "volume": {"label": "Veri yok", "signal": "NEUTRAL"},
                "fvg": {"label": "Veri yok", "signal": "NEUTRAL"},
                "support_resistance": {"label": "Veri yok", "signal": "NEUTRAL"},
                "divergence": {"label": "Veri yok", "type": "NONE"},
                "trend": "UNKNOWN", "verdict": "VERİ YOK",
                "confidence": 0
            }

        close = df["close"]
        current_price = close.iloc[-1]

        # 1. RSI
        rsi_series = _rsi(close, 14)
        rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else None
        rsi_result = _interpret_rsi(rsi_val)

        # 2. Stochastic RSI
        stoch_k, stoch_d = _stoch_rsi(close)
        k_val = stoch_k.iloc[-1] if not stoch_k.empty and not np.isnan(stoch_k.iloc[-1]) else None
        d_val = stoch_d.iloc[-1] if not stoch_d.empty and not np.isnan(stoch_d.iloc[-1]) else None
        stoch_result = _interpret_stoch_rsi(k_val, d_val)

        # 3. MACD
        macd_line, signal_line, histogram = _macd(close)
        macd_val = macd_line.iloc[-1] if not macd_line.empty else None
        sig_val = signal_line.iloc[-1] if not signal_line.empty else None
        hist_val = histogram.iloc[-1] if not histogram.empty else None
        prev_hist = histogram.iloc[-2] if len(histogram) >= 2 else None
        macd_result = _interpret_macd(macd_val, sig_val, hist_val, prev_hist)

        # 4. Bollinger Bands
        bb_upper, bb_middle, bb_lower = _bollinger_bands(close)
        bb_width_series = ((bb_upper - bb_lower) / bb_middle).dropna()
        bb_width = bb_width_series.iloc[-1] if not bb_width_series.empty else 0
        prev_bb_width = bb_width_series.iloc[-5] if len(bb_width_series) >= 5 else None
        bb_result = _interpret_bb(current_price,
                                  bb_upper.iloc[-1], bb_middle.iloc[-1], bb_lower.iloc[-1],
                                  bb_width, prev_bb_width)

        # 5. ADX
        adx_series, plus_di, minus_di = _adx(df)
        adx_val = adx_series.iloc[-1] if not adx_series.empty and not np.isnan(adx_series.iloc[-1]) else None
        pdi_val = plus_di.iloc[-1] if not plus_di.empty else None
        mdi_val = minus_di.iloc[-1] if not minus_di.empty else None
        adx_result = _interpret_adx(adx_val, pdi_val, mdi_val)

        # 6. ATR
        atr_series = _atr(df)
        atr_val = atr_series.iloc[-1] if not atr_series.empty else None
        atr_result = _interpret_atr(atr_val, current_price)

        # 7. OBV
        obv_series = _obv(df)
        obv_result = _interpret_obv(obv_series, close)

        # 8. Volume
        vol_result = _analyze_volume(df)

        # 9. FVG
        fvg_result = _check_fvg(df)

        # 10. Destek/Direnç
        supports, resistances = _find_support_resistance(df)
        sr_result = _interpret_sr(supports, resistances, current_price)

        # 11. Diverjans (RSI + fiyat)
        div_result = _detect_divergence(close, rsi_series, lookback=25)
        if div_result is None:
            div_result = {"type": "NONE", "label": "Veri yetersiz", "desc": "", "color": "gray"}

        # ── EMA Trend Yapısı ──
        ema_8 = close.ewm(span=8, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean() if len(close) >= 50 else None
        ema_200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else None

        ema8_val = ema_8.iloc[-1]
        ema21_val = ema_21.iloc[-1]
        ema50_val = ema_50.iloc[-1] if ema_50 is not None else None
        ema200_val = ema_200.iloc[-1] if ema_200 is not None else None

        # EMA sıralaması (golden/death cross)
        ema_order_bull = ema8_val > ema21_val and (ema50_val is None or ema21_val > ema50_val)
        ema_order_bear = ema8_val < ema21_val and (ema50_val is None or ema21_val < ema50_val)

        trend_signals = []
        if ema8_val > ema21_val:
            trend_signals.append("BULL")
        else:
            trend_signals.append("BEAR")

        if ema50_val is not None:
            if current_price > ema50_val:
                trend_signals.append("ABOVE_50")
            else:
                trend_signals.append("BELOW_50")

        if "BULL" in trend_signals and "ABOVE_50" in trend_signals:
            trend = "BULLISH"
            trend_label = "Güçlü Yükseliş Trendi"
            trend_desc = "EMA8 > EMA21 ve fiyat EMA50 üzerinde — yapısal yükseliş."
        elif "BEAR" in trend_signals and "BELOW_50" in trend_signals:
            trend = "BEARISH"
            trend_label = "Güçlü Düşüş Trendi"
            trend_desc = "EMA8 < EMA21 ve fiyat EMA50 altında — yapısal düşüş."
        elif "BULL" in trend_signals:
            trend = "WEAKENING_BEAR"
            trend_label = "Zayıflayan Düşüş"
            trend_desc = "EMA8 > EMA21 ama fiyat EMA50 altında — erken dönüş sinyali."
        else:
            trend = "WEAKENING_BULL"
            trend_label = "Zayıflayan Yükseliş"
            trend_desc = "EMA8 < EMA21 ama fiyat EMA50 üzerinde — momentum kaybolıyor."

        if ema_order_bull:
            trend_desc += " EMA'lar boğa sıralamasında (8>21>50)."
        elif ema_order_bear:
            trend_desc += " EMA'lar ayı sıralamasında (8<21<50)."

        # ── AĞIRLIKLI GÜVEN SKORU (0-100) ──
        # Her gösterge ağırlıklı puan verir
        weights = {
            "trend": 20,      # %20 — trend en önemli
            "adx": 15,        # %15 — trend gücü
            "macd": 15,       # %15 — momentum
            "rsi": 10,        # %10
            "stoch_rsi": 8,   # %8
            "volume": 10,     # %10
            "obv": 7,         # %7
            "bollinger": 5,   # %5
            "fvg": 5,         # %5
            "divergence": 5,  # %5
        }

        bull_score = 0
        bear_score = 0
        indicator_scores = {}

        # Trend skoru
        if trend == "BULLISH":
            bull_score += weights["trend"]
            indicator_scores["trend"] = {"direction": "BULL", "score": weights["trend"]}
        elif trend == "BEARISH":
            bear_score += weights["trend"]
            indicator_scores["trend"] = {"direction": "BEAR", "score": weights["trend"]}
        elif trend == "WEAKENING_BEAR":
            bull_score += weights["trend"] * 0.4
            indicator_scores["trend"] = {"direction": "BULL", "score": round(weights["trend"] * 0.4, 1)}
        elif trend == "WEAKENING_BULL":
            bear_score += weights["trend"] * 0.4
            indicator_scores["trend"] = {"direction": "BEAR", "score": round(weights["trend"] * 0.4, 1)}

        # ADX skoru
        if adx_result.get("signal") == "BULLISH":
            s = weights["adx"] * min(adx_val / 50, 1.0) if adx_val else 0
            bull_score += s
            indicator_scores["adx"] = {"direction": "BULL", "score": round(s, 1)}
        elif adx_result.get("signal") == "BEARISH":
            s = weights["adx"] * min(adx_val / 50, 1.0) if adx_val else 0
            bear_score += s
            indicator_scores["adx"] = {"direction": "BEAR", "score": round(s, 1)}

        # MACD skoru
        if macd_result.get("signal_type") == "BULLISH":
            bull_score += weights["macd"]
            indicator_scores["macd"] = {"direction": "BULL", "score": weights["macd"]}
        elif macd_result.get("signal_type") == "BEARISH":
            bear_score += weights["macd"]
            indicator_scores["macd"] = {"direction": "BEAR", "score": weights["macd"]}
        elif macd_result.get("signal_type") == "WEAKENING_BULL":
            bull_score += weights["macd"] * 0.3
            indicator_scores["macd"] = {"direction": "BULL", "score": round(weights["macd"] * 0.3, 1)}
        elif macd_result.get("signal_type") == "WEAKENING_BEAR":
            bear_score += weights["macd"] * 0.3
            indicator_scores["macd"] = {"direction": "BEAR", "score": round(weights["macd"] * 0.3, 1)}

        # RSI skoru
        if rsi_result.get("signal") == "BULLISH":
            bull_score += weights["rsi"]
            indicator_scores["rsi"] = {"direction": "BULL", "score": weights["rsi"]}
        elif rsi_result.get("signal") == "BEARISH":
            bear_score += weights["rsi"]
            indicator_scores["rsi"] = {"direction": "BEAR", "score": weights["rsi"]}

        # StochRSI skoru
        if stoch_result.get("signal") == "BULLISH":
            bull_score += weights["stoch_rsi"]
            indicator_scores["stoch_rsi"] = {"direction": "BULL", "score": weights["stoch_rsi"]}
        elif stoch_result.get("signal") == "BEARISH":
            bear_score += weights["stoch_rsi"]
            indicator_scores["stoch_rsi"] = {"direction": "BEAR", "score": weights["stoch_rsi"]}

        # Volume skoru (yönle birlikte)
        if vol_result.get("signal") == "HIGH":
            if trend in ("BULLISH", "WEAKENING_BEAR"):
                bull_score += weights["volume"]
                indicator_scores["volume"] = {"direction": "BULL", "score": weights["volume"]}
            else:
                bear_score += weights["volume"]
                indicator_scores["volume"] = {"direction": "BEAR", "score": weights["volume"]}

        # OBV skoru
        if obv_result.get("signal") == "BULLISH":
            bull_score += weights["obv"]
            indicator_scores["obv"] = {"direction": "BULL", "score": weights["obv"]}
        elif obv_result.get("signal") == "BEARISH":
            bear_score += weights["obv"]
            indicator_scores["obv"] = {"direction": "BEAR", "score": weights["obv"]}

        # Bollinger skoru
        if bb_result.get("signal") == "BULLISH":
            bull_score += weights["bollinger"]
            indicator_scores["bollinger"] = {"direction": "BULL", "score": weights["bollinger"]}
        elif bb_result.get("signal") == "BEARISH":
            bear_score += weights["bollinger"]
            indicator_scores["bollinger"] = {"direction": "BEAR", "score": weights["bollinger"]}

        # FVG skoru
        if fvg_result.get("signal") == "BULLISH":
            bull_score += weights["fvg"]
            indicator_scores["fvg"] = {"direction": "BULL", "score": weights["fvg"]}
        elif fvg_result.get("signal") == "BEARISH":
            bear_score += weights["fvg"]
            indicator_scores["fvg"] = {"direction": "BEAR", "score": weights["fvg"]}

        # Diverjans skoru — contrarian sinyal
        if div_result.get("type") == "BULLISH":
            bull_score += weights["divergence"]
            indicator_scores["divergence"] = {"direction": "BULL", "score": weights["divergence"]}
        elif div_result.get("type") == "BEARISH":
            bear_score += weights["divergence"]
            indicator_scores["divergence"] = {"direction": "BEAR", "score": weights["divergence"]}

        total_possible = sum(weights.values())  # 100
        confidence = round(max(bull_score, bear_score), 1)
        net_score = round(bull_score - bear_score, 1)

        # Verdict belirleme
        if net_score >= 40:
            verdict = "STRONG_BULLISH"
            verdict_label = "GÜÇLÜ BOĞA"
            verdict_color = "green"
        elif net_score >= 20:
            verdict = "BULLISH"
            verdict_label = "BOĞA"
            verdict_color = "green"
        elif net_score >= 8:
            verdict = "LEANING_BULLISH"
            verdict_label = "HAFİF BOĞA"
            verdict_color = "lightgreen"
        elif net_score <= -40:
            verdict = "STRONG_BEARISH"
            verdict_label = "GÜÇLÜ AYI"
            verdict_color = "red"
        elif net_score <= -20:
            verdict = "BEARISH"
            verdict_label = "AYI"
            verdict_color = "red"
        elif net_score <= -8:
            verdict = "LEANING_BEARISH"
            verdict_label = "HAFİF AYI"
            verdict_color = "orange"
        else:
            verdict = "NEUTRAL"
            verdict_label = "NÖTR"
            verdict_color = "gray"

        return {
            "timeframe": tf_label,
            "rsi": rsi_result,
            "stoch_rsi": stoch_result,
            "macd": macd_result,
            "bollinger": bb_result,
            "adx": adx_result,
            "atr": atr_result,
            "obv": obv_result,
            "volume": vol_result,
            "fvg": fvg_result,
            "support_resistance": sr_result,
            "divergence": div_result,
            "trend": trend,
            "trend_label": trend_label,
            "trend_desc": trend_desc,
            "ema": {
                "ema8": round(ema8_val, 8),
                "ema21": round(ema21_val, 8),
                "ema50": round(ema50_val, 8) if ema50_val else None,
                "ema200": round(ema200_val, 8) if ema200_val else None,
                "order": "BULL" if ema_order_bull else ("BEAR" if ema_order_bear else "MIXED")
            },
            "verdict": verdict,
            "verdict_label": verdict_label,
            "verdict_color": verdict_color,
            "bull_score": round(bull_score, 1),
            "bear_score": round(bear_score, 1),
            "net_score": net_score,
            "confidence": confidence,
            "indicator_scores": indicator_scores
        }

    try:
        import pandas as pd
        multi_tf = data_fetcher.get_multi_timeframe_data(symbol)

        # Her TF için gelişmiş analiz
        tf_results = {}
        for tf_key, tf_label in [("15m", "15 Dakika"), ("1H", "1 Saat"), ("4H", "4 Saat")]:
            df_tf = multi_tf.get(tf_key)
            tf_results[tf_key] = _analyze_tf(df_tf, tf_label)

        # Ticker bilgisi
        ticker = data_fetcher.get_ticker(symbol)
        price_info = {}
        if ticker:
            price_info = {
                "last": ticker["last"],
                "high24h": ticker.get("high24h", 0),
                "low24h": ticker.get("low24h", 0),
                "change24h": ticker.get("changePct24h", 0),
                "vol24h": ticker.get("vol24h", 0)
            }

        # Order Book analizi
        orderbook_result = {"label": "Veri yok", "signal": "NEUTRAL", "color": "gray"}
        try:
            book = data_fetcher.get_orderbook(symbol, depth=20)
            if book:
                total_bids = sum(b[1] for b in book["bids"])
                total_asks = sum(a[1] for a in book["asks"])
                imbalance = (total_bids / (total_bids + total_asks)) * 100 if (total_bids + total_asks) > 0 else 50

                # Büyük duvarlar
                avg_bid = total_bids / max(len(book["bids"]), 1)
                avg_ask = total_asks / max(len(book["asks"]), 1)
                bid_walls = sum(1 for b in book["bids"] if b[1] > avg_bid * 3)
                ask_walls = sum(1 for a in book["asks"] if a[1] > avg_ask * 3)

                orderbook_result = {
                    "bid_total": round(total_bids, 2),
                    "ask_total": round(total_asks, 2),
                    "imbalance": round(imbalance, 1),
                    "bid_walls": bid_walls,
                    "ask_walls": ask_walls,
                    "spread_pct": round(((book["asks"][0][0] - book["bids"][0][0]) / book["bids"][0][0]) * 100, 4) if book["bids"] and book["asks"] else 0,
                }

                if imbalance >= 65:
                    orderbook_result.update({"label": f"Güçlü Alım Baskısı (%{imbalance:.0f})", "signal": "BULLISH", "color": "green",
                                            "desc": f"Alım emirleri %{imbalance:.0f} ile baskın — alıcılar fiyatı yukarı itiyor."})
                elif imbalance >= 55:
                    orderbook_result.update({"label": f"Hafif Alım Baskısı (%{imbalance:.0f})", "signal": "BULLISH", "color": "lightgreen",
                                            "desc": f"Alım tarafı hafif baskın (%{imbalance:.0f}) — kısa vadeli destek mevcut."})
                elif imbalance <= 35:
                    orderbook_result.update({"label": f"Güçlü Satış Baskısı (%{imbalance:.0f})", "signal": "BEARISH", "color": "red",
                                            "desc": f"Satış emirleri %{100-imbalance:.0f} ile baskın — satıcılar fiyatı aşağı çekiyor."})
                elif imbalance <= 45:
                    orderbook_result.update({"label": f"Hafif Satış Baskısı (%{imbalance:.0f})", "signal": "BEARISH", "color": "orange",
                                            "desc": f"Satış tarafı hafif baskın (%{100-imbalance:.0f}) — kısa vadeli baskı var."})
                else:
                    orderbook_result.update({"label": f"Dengeli (%{imbalance:.0f})", "signal": "NEUTRAL", "color": "gray",
                                            "desc": "Alım-satım emirleri dengede — belirleyici bir taraf yok."})

                if bid_walls > 0:
                    orderbook_result["desc"] += f" | {bid_walls} büyük alım duvarı tespit edildi."
                if ask_walls > 0:
                    orderbook_result["desc"] += f" | {ask_walls} büyük satış duvarı tespit edildi."
        except Exception:
            pass

        # ── GELİŞMİŞ GENEL YORUM (Ağırlıklı TF Kombinasyonu) ──
        # 4H: %50, 1H: %30, 15m: %20 ağırlık
        tf_weights = {"4H": 0.50, "1H": 0.30, "15m": 0.20}
        total_bull = 0
        total_bear = 0

        for tf_key, weight in tf_weights.items():
            tf = tf_results[tf_key]
            total_bull += tf.get("bull_score", 0) * weight
            total_bear += tf.get("bear_score", 0) * weight

        overall_net = round(total_bull - total_bear, 1)
        overall_confidence = round(max(total_bull, total_bear), 1)

        # TF uyum kontrolü (tüm TF'ler aynı yönde = ekstra güven)
        tf_verdicts = [tf_results[k].get("verdict", "NEUTRAL") for k in ["15m", "1H", "4H"]]
        all_bull = all(v in ("STRONG_BULLISH", "BULLISH", "LEANING_BULLISH") for v in tf_verdicts)
        all_bear = all(v in ("STRONG_BEARISH", "BEARISH", "LEANING_BEARISH") for v in tf_verdicts)

        # TF çelişki kontrolü — 4H ve 15m zıt yönde ise güveni düşür
        v_4h = tf_results["4H"].get("verdict", "NEUTRAL")
        v_15m = tf_results["15m"].get("verdict", "NEUTRAL")
        bull_set = {"STRONG_BULLISH", "BULLISH", "LEANING_BULLISH"}
        bear_set = {"STRONG_BEARISH", "BEARISH", "LEANING_BEARISH"}
        tf_conflict = (v_4h in bull_set and v_15m in bear_set) or (v_4h in bear_set and v_15m in bull_set)

        # Orderbook ekstra puan (azaltıldı: max ±2)
        if orderbook_result.get("signal") == "BULLISH":
            overall_net += 2
        elif orderbook_result.get("signal") == "BEARISH":
            overall_net -= 2

        confluence_bonus = ""
        if all_bull and not tf_conflict:
            overall_net += 8
            confluence_bonus = " Tüm zaman dilimleri boğa yönünde uyumlu."
        elif all_bear and not tf_conflict:
            overall_net -= 8
            confluence_bonus = " Tüm zaman dilimleri ayı yönünde uyumlu."
        elif tf_conflict:
            overall_net *= 0.6  # TF çelişkisi varsa güveni %40 azalt
            overall_net = round(overall_net, 1)
            confluence_bonus = " ⚠ 4H ve 15m zıt sinyaller veriyor — yön netleşene kadar temkinli olun."

        # Overall verdict — eşikler yükseltildi (false signal azaltmak için)
        if overall_net >= 30:
            overall = "STRONG_BULLISH"
            overall_label = "GÜÇLÜ BOĞA"
            overall_emoji = "🟢🟢"
            overall_desc = f"Çoklu gösterge ve zaman dilimi güçlü yükseliş sinyali veriyor (skor: +{overall_net}).{confluence_bonus} Geri çekilmelerde LONG pozisyon değerlendirilebilir. Risk yönetimini ihmal etmeyin."
        elif overall_net >= 15:
            overall = "BULLISH"
            overall_label = "BOĞA"
            overall_emoji = "🟢"
            overall_desc = f"Göstergeler yükseliş yönünde ağırlıklı (skor: +{overall_net}).{confluence_bonus} Yükseliş eğilimi var ancak mutlaka 4H trend onayı kontrol edin."
        elif overall_net >= 6:
            overall = "LEANING_BULLISH"
            overall_label = "HAFİF BOĞA"
            overall_emoji = "🟡"
            overall_desc = f"Hafif boğa eğilimi (skor: +{overall_net}). Sinyal güçlü değil — tek başına pozisyon almak için yetersiz. 4H kapanışını ve hacim onayını bekleyin."
        elif overall_net <= -30:
            overall = "STRONG_BEARISH"
            overall_label = "GÜÇLÜ AYI"
            overall_emoji = "🔴🔴"
            overall_desc = f"Çoklu gösterge ve zaman dilimi güçlü düşüş sinyali veriyor (skor: {overall_net}).{confluence_bonus} Yükselişlerde SHORT düşünülebilir. SL mutlaka kullanın."
        elif overall_net <= -15:
            overall = "BEARISH"
            overall_label = "AYI"
            overall_emoji = "🔴"
            overall_desc = f"Göstergeler düşüş yönünde ağırlıklı (skor: {overall_net}).{confluence_bonus} Düşüş trendi aktif. LONG pozisyonlardan kaçının."
        elif overall_net <= -6:
            overall = "LEANING_BEARISH"
            overall_label = "HAFİF AYI"
            overall_emoji = "🟠"
            overall_desc = f"Hafif ayı eğilimi (skor: {overall_net}). Sinyal güçlü değil — kesin yön için 4H trend ve hacim onayı bekleyin."
        else:
            overall = "NEUTRAL"
            overall_label = "NÖTR"
            overall_emoji = "⚪"
            overall_desc = f"Göstergeler karışık veya zayıf sinyal veriyor (skor: {overall_net}). Bu coin şu an net yön vermiyor — pozisyon almak yerine izlemeye alın."

        # Ek uyarılar
        warnings = []
        atr_4h = tf_results["4H"].get("atr", {})
        if atr_4h.get("signal") == "HIGH":
            warnings.append("⚠ Yüksek volatilite — pozisyon boyutunu küçültün, geniş SL kullanın.")
        if tf_conflict:
            warnings.append("⚠ 4H ve 15m zaman dilimleri zıt sinyal veriyor — güvenilirlik düşük.")
        if any(tf_results[k].get("divergence", {}).get("type") in ("BULLISH", "BEARISH") for k in ["1H", "4H"]):
            div_tfs = [k for k in ["1H", "4H"] if tf_results[k].get("divergence", {}).get("type") in ("BULLISH", "BEARISH")]
            div_types = [tf_results[k]["divergence"]["type"] for k in div_tfs]
            warnings.append(f"⚠ {', '.join(div_tfs)}'de {'/'.join(div_types)} diverjansı — mevcut trend zayıflıyor olabilir!")
        if orderbook_result.get("bid_walls", 0) >= 2:
            warnings.append("🛡 Güçlü alım duvarları — aşağı yönlü destek güçlü.")
        if orderbook_result.get("ask_walls", 0) >= 2:
            warnings.append("🧱 Güçlü satış duvarları — yukarı yönlü direnç var.")
        
        # Düşük güven uyarısı
        if abs(overall_net) < 10:
            warnings.append("ℹ Skor düşük — yüksek güvenli sinyal için daha fazla gösterge uyumu gerekli.")

        # ── AI TRADİNG SENARYO MOTORU ──
        def _generate_trading_scenario(tf_results, price_info, orderbook_result, overall_net, overall, warnings):
            """
            Tüm TF verilerini ve teknik analizi birleştirerek
            detaylı long/short trading senaryosu üretir.
            """
            current_price = price_info.get("last", 0)
            if not current_price:
                return None

            # ─── Tüm TF'lerden veri topla ─── 
            tf_4h = tf_results.get("4H", {})
            tf_1h = tf_results.get("1H", {})
            tf_15m = tf_results.get("15m", {})

            # Trend bilgileri
            trend_4h = tf_4h.get("trend", "UNKNOWN")
            trend_1h = tf_1h.get("trend", "UNKNOWN")
            trend_15m = tf_15m.get("trend", "UNKNOWN")

            # EMA değerleri (4H ana referans)
            ema_4h = tf_4h.get("ema", {})
            ema_1h = tf_1h.get("ema", {})
            ema_15m = tf_15m.get("ema", {})

            ema8_4h = ema_4h.get("ema8")
            ema21_4h = ema_4h.get("ema21")
            ema50_4h = ema_4h.get("ema50")
            ema200_4h = ema_4h.get("ema200")

            # Destek/Direnç (4H ve 1H)
            sr_4h = tf_4h.get("support_resistance", {})
            sr_1h = tf_1h.get("support_resistance", {})
            sr_15m = tf_15m.get("support_resistance", {})

            support_4h = sr_4h.get("nearest_support")
            resistance_4h = sr_4h.get("nearest_resistance")
            support_1h = sr_1h.get("nearest_support")
            resistance_1h = sr_1h.get("nearest_resistance")
            support_15m = sr_15m.get("nearest_support")
            resistance_15m = sr_15m.get("nearest_resistance")

            # ATR (stop loss hesabı için)
            atr_4h = tf_4h.get("atr", {})
            atr_1h = tf_1h.get("atr", {})
            atr_15m = tf_15m.get("atr", {})
            atr_val_4h = atr_4h.get("atr", 0)
            atr_val_1h = atr_1h.get("atr", 0)
            atr_val_15m = atr_15m.get("atr", 0)
            atr_pct_4h = atr_4h.get("atr_pct", 0)

            # Bollinger
            bb_4h = tf_4h.get("bollinger", {})
            bb_1h = tf_1h.get("bollinger", {})
            bb_upper_4h = bb_4h.get("upper", 0)
            bb_lower_4h = bb_4h.get("lower", 0)
            bb_middle_4h = bb_4h.get("middle", 0)
            bb_squeeze_4h = bb_4h.get("squeeze_status", "")

            # RSI değerleri
            rsi_4h = tf_4h.get("rsi", {}).get("value", 50)
            rsi_1h = tf_1h.get("rsi", {}).get("value", 50)
            rsi_15m = tf_15m.get("rsi", {}).get("value", 50)

            # MACD
            macd_4h = tf_4h.get("macd", {})
            macd_1h = tf_1h.get("macd", {})

            # ADX
            adx_4h = tf_4h.get("adx", {})
            adx_val = adx_4h.get("adx", 0)

            # Volume
            vol_15m = tf_15m.get("volume", {})
            vol_ratio = vol_15m.get("ratio", 1)

            # FVG
            fvg_15m = tf_15m.get("fvg", {})
            fvg_1h = tf_1h.get("fvg", {})

            # Diverjans
            div_4h = tf_4h.get("divergence", {}).get("type", "NONE")
            div_1h = tf_1h.get("divergence", {}).get("type", "NONE")

            # Orderbook  
            ob_signal = orderbook_result.get("signal", "NEUTRAL")
            ob_imbalance = orderbook_result.get("imbalance", 50)

            # ─── Verdict'leri belirle ───
            v_4h = tf_4h.get("verdict", "NEUTRAL")
            v_1h = tf_1h.get("verdict", "NEUTRAL")
            v_15m = tf_15m.get("verdict", "NEUTRAL")

            bull_verdicts = {"STRONG_BULLISH", "BULLISH", "LEANING_BULLISH"}
            bear_verdicts = {"STRONG_BEARISH", "BEARISH", "LEANING_BEARISH"}

            is_bull_4h = v_4h in bull_verdicts
            is_bear_4h = v_4h in bear_verdicts
            is_bull_1h = v_1h in bull_verdicts
            is_bear_1h = v_1h in bear_verdicts
            all_bull = v_4h in bull_verdicts and v_1h in bull_verdicts and v_15m in bull_verdicts
            all_bear = v_4h in bear_verdicts and v_1h in bear_verdicts and v_15m in bear_verdicts

            # ─── Fiyat formatı ─── 
            def fmt(val):
                if val is None or val == 0:
                    return "N/A"
                if val >= 1:
                    return f"{val:.4f}"
                elif val >= 0.001:
                    return f"{val:.6f}"
                else:
                    return f"{val:.8f}"

            # ─── Anahtar seviyeler ───
            key_levels = []
            if ema50_4h:
                key_levels.append(("4H EMA50", ema50_4h))
            if ema200_4h:
                key_levels.append(("4H EMA200", ema200_4h))
            if bb_upper_4h:
                key_levels.append(("BB Üst", bb_upper_4h))
            if bb_lower_4h:
                key_levels.append(("BB Alt", bb_lower_4h))
            if bb_middle_4h:
                key_levels.append(("BB Orta", bb_middle_4h))
            if support_4h:
                key_levels.append(("4H Destek", support_4h))
            if resistance_4h:
                key_levels.append(("4H Direnç", resistance_4h))
            if support_1h:
                key_levels.append(("1H Destek", support_1h))
            if resistance_1h:
                key_levels.append(("1H Direnç", resistance_1h))

            # ─── LONG SENARYO ─── 
            long_scenario = {"quality": 0, "sections": []}

            if is_bull_4h:
                long_scenario["quality"] += 35
            elif v_4h == "NEUTRAL":
                long_scenario["quality"] += 10
            if is_bull_1h:
                long_scenario["quality"] += 25
            if v_15m in bull_verdicts:
                long_scenario["quality"] += 15
            if ob_signal == "BULLISH":
                long_scenario["quality"] += 8
            if div_4h == "BULLISH" or div_1h == "BULLISH":
                long_scenario["quality"] += 12
            if all_bull:
                long_scenario["quality"] += 5

            # Piyasa bağlamı
            ctx_lines = []
            ctx_lines.append(f"4H Trend: {'Yükseliş ✅' if trend_4h == 'BULLISH' else 'Düşüş ❌' if trend_4h == 'BEARISH' else 'Zayıflıyor ⚠'}")
            ctx_lines.append(f"1H Trend: {'Yükseliş ✅' if trend_1h == 'BULLISH' else 'Düşüş ❌' if trend_1h == 'BEARISH' else 'Zayıflıyor ⚠'}")
            ctx_lines.append(f"15m Trend: {'Yükseliş ✅' if trend_15m == 'BULLISH' else 'Düşüş ❌' if trend_15m == 'BEARISH' else 'Zayıflıyor ⚠'}")
            if rsi_4h:
                ctx_lines.append(f"RSI: 4H={rsi_4h:.0f} | 1H={rsi_1h:.0f} | 15m={rsi_15m:.0f}")
            if adx_val:
                adx_text = "Güçlü" if adx_val > 25 else "Zayıf"
                ctx_lines.append(f"Trend Gücü (ADX): {adx_val:.0f} — {adx_text}")
            long_scenario["sections"].append({"title": "📊 Piyasa Bağlamı", "lines": ctx_lines})

            # Giriş koşulları
            entry_lines = []
            if is_bull_4h and is_bull_1h:
                # İdeal senaryo: 4H+1H uyumlu
                if support_1h and current_price > support_1h:
                    pullback_zone = support_1h * 1.005  # %0.5 üstü
                    entry_lines.append(f"🎯 İdeal Giriş: Fiyat {fmt(support_1h)} - {fmt(pullback_zone)} destek bölgesine geri çekildiğinde")
                    entry_lines.append(f"Bu bölgede 15m mumun kapanışını bekleyin (en az 2 mum yeşil kapansın)")
                if ema21_4h and current_price > ema21_4h:
                    entry_lines.append(f"Alternatif: 4H EMA21 ({fmt(ema21_4h)}) testinde tepki alımda giriş")
                if not entry_lines:
                    entry_lines.append(f"Mevcut fiyat ({fmt(current_price)}) seviyesinde 15m'de yeşil mum onayı ile giriş değerlendirilebilir")
            elif is_bull_4h and not is_bull_1h:
                entry_lines.append(f"⏳ 4H boğa ama 1H henüz onaylamamış — 1H'de EMA8>EMA21 geçişini bekleyin")
                if ema_1h.get("ema21"):
                    entry_lines.append(f"1H EMA21: {fmt(ema_1h['ema21'])} — fiyat bunun üzerine kapanmalı")
                entry_lines.append(f"Erken giriş: 15m'de art arda 3 yeşil mum kapanışı + artan hacim ile giriş denenebilir")
            elif not is_bull_4h:
                entry_lines.append(f"⚠ 4H trend henüz boğa değil — yüksek risk")
                if support_4h:
                    entry_lines.append(f"Sadece {fmt(support_4h)} güçlü 4H desteğinde tepki alım denenebilir")
                entry_lines.append(f"En az 2 adet 15m mum bu seviyede kapanmalı (wick rejection)")
                if div_4h == "BULLISH" or div_1h == "BULLISH":
                    entry_lines.append(f"✨ Boğa diverjansı tespit edildi — erken dönüş sinyali, dikkatli izleyin")

            # Bollinger/FVG ekstra
            if bb_squeeze_4h == "DARALIYOR":
                entry_lines.append(f"🔥 Bollinger sıkışması var — patlama aşağı veya yukarı olabilir, yön onayı bekleyin")
            if fvg_1h.get("signal") == "BULLISH" and fvg_1h.get("unfilled_bullish", 0) > 0:
                entry_lines.append(f"📐 1H'de doldurulmamış Boğa FVG var — bu bölge likidite çeker, geri çekilmede giriş noktası")

            long_scenario["sections"].append({"title": "🟢 Giriş Koşulları", "lines": entry_lines})

            # Stop Loss
            sl_lines = []
            if support_1h:
                sl_price = support_1h * 0.995  # Destek altı %0.5
                sl_pct = abs((current_price - sl_price) / current_price * 100)
                sl_lines.append(f"Agresif SL: {fmt(sl_price)} (1H destek altı, -%{sl_pct:.1f})")
            if support_4h and support_4h < current_price:
                sl_price_safe = support_4h * 0.99  # 4H destek altı %1
                sl_pct_safe = abs((current_price - sl_price_safe) / current_price * 100)
                sl_lines.append(f"Güvenli SL: {fmt(sl_price_safe)} (4H destek altı, -%{sl_pct_safe:.1f})")
            if atr_val_1h:
                sl_atr = current_price - (atr_val_1h * 1.5)
                sl_pct_atr = abs((current_price - sl_atr) / current_price * 100)
                sl_lines.append(f"ATR Bazlı SL: {fmt(sl_atr)} (1.5x ATR, -%{sl_pct_atr:.1f})")
            if not sl_lines:
                sl_lines.append(f"ATR bazlı SL önerilir: Mevcut fiyattan 1.5-2x ATR altı")
            long_scenario["sections"].append({"title": "🛑 Stop Loss", "lines": sl_lines})

            # Target (TP)
            tp_lines = []
            if resistance_1h:
                tp_pct = abs((resistance_1h - current_price) / current_price * 100)
                tp_lines.append(f"TP1: {fmt(resistance_1h)} (1H direnç, +%{tp_pct:.1f})")
            if resistance_4h and resistance_4h != resistance_1h:
                tp_pct2 = abs((resistance_4h - current_price) / current_price * 100)
                tp_lines.append(f"TP2: {fmt(resistance_4h)} (4H direnç, +%{tp_pct2:.1f})")
            if bb_upper_4h and bb_upper_4h > current_price:
                tp_pct3 = abs((bb_upper_4h - current_price) / current_price * 100)
                tp_lines.append(f"TP3: {fmt(bb_upper_4h)} (BB üst bant, +%{tp_pct3:.1f})")
            if ema200_4h and ema200_4h > current_price * 1.02:
                tp_pct4 = abs((ema200_4h - current_price) / current_price * 100)
                tp_lines.append(f"Uzun Vadeli: {fmt(ema200_4h)} (4H EMA200, +%{tp_pct4:.1f})")
            if not tp_lines:
                if atr_val_1h:
                    tp_auto = current_price + (atr_val_1h * 3)
                    tp_lines.append(f"TP: {fmt(tp_auto)} (3x ATR hedef)")
            long_scenario["sections"].append({"title": "🎯 Hedef (Take Profit)", "lines": tp_lines})

            # R:R hesabı
            rr_lines = []
            best_sl = None
            best_tp = None
            if support_1h:
                best_sl = support_1h * 0.995
            elif atr_val_1h:
                best_sl = current_price - (atr_val_1h * 1.5)
            if resistance_1h:
                best_tp = resistance_1h
            elif resistance_4h:
                best_tp = resistance_4h

            if best_sl and best_tp and best_sl < current_price < best_tp:
                risk = current_price - best_sl
                reward = best_tp - current_price
                rr = reward / risk if risk > 0 else 0
                rr_lines.append(f"Risk: {fmt(risk)} ({abs(risk/current_price*100):.1f}%) | Ödül: {fmt(reward)} ({abs(reward/current_price*100):.1f}%)")
                rr_lines.append(f"Risk:Ödül = 1:{rr:.1f} {'✅ Uygun' if rr >= 2 else '⚠ Düşük (min 1:2 önerilir)'}")
            long_scenario["sections"].append({"title": "📐 Risk/Ödül", "lines": rr_lines})

            # ─── SHORT SENARYO ─── 
            short_scenario = {"quality": 0, "sections": []}

            if is_bear_4h:
                short_scenario["quality"] += 35
            elif v_4h == "NEUTRAL":
                short_scenario["quality"] += 10
            if is_bear_1h:
                short_scenario["quality"] += 25
            if v_15m in bear_verdicts:
                short_scenario["quality"] += 15
            if ob_signal == "BEARISH":
                short_scenario["quality"] += 8
            if div_4h == "BEARISH" or div_1h == "BEARISH":
                short_scenario["quality"] += 12
            if all_bear:
                short_scenario["quality"] += 5

            # Short giriş
            s_entry = []
            if is_bear_4h and is_bear_1h:
                if resistance_1h and current_price < resistance_1h:
                    pullback_zone = resistance_1h * 0.995
                    s_entry.append(f"🎯 İdeal Giriş: Fiyat {fmt(pullback_zone)} - {fmt(resistance_1h)} direnç bölgesine çekildiğinde")
                    s_entry.append(f"Bu bölgede 15m mumun kapanışını bekleyin (en az 2 mum kırmızı kapansın)")
                if ema21_4h and current_price < ema21_4h:
                    s_entry.append(f"Alternatif: 4H EMA21 ({fmt(ema21_4h)}) ret sinyalinde short giriş")
                if not s_entry:
                    s_entry.append(f"Mevcut fiyat ({fmt(current_price)}) seviyesinde 15m'de kırmızı mum onayı ile short değerlendirilebilir")
            elif is_bear_4h and not is_bear_1h:
                s_entry.append(f"⏳ 4H ayı ama 1H henüz onaylamamış — 1H'de EMA8<EMA21 geçişini bekleyin")
                if ema_1h.get("ema21"):
                    s_entry.append(f"1H EMA21: {fmt(ema_1h['ema21'])} — fiyat bunun altına kapanmalı")
                s_entry.append(f"Erken giriş: 15m'de art arda 3 kırmızı mum + artan hacim ile short denenebilir")
            elif not is_bear_4h:
                s_entry.append(f"⚠ 4H trend henüz ayı değil — yüksek risk")
                if resistance_4h:
                    s_entry.append(f"Sadece {fmt(resistance_4h)} güçlü 4H dirençte ret satış denenebilir")
                s_entry.append(f"En az 2 adet 15m mum bu seviyede kapanmalı (üst wick rejection)")
                if div_4h == "BEARISH" or div_1h == "BEARISH":
                    s_entry.append(f"✨ Ayı diverjansı tespit edildi — trend dönüşünün erken sinyali")

            if bb_squeeze_4h == "DARALIYOR":
                s_entry.append(f"🔥 Bollinger sıkışması — kırılım bekleyin, erken girmeyin")
            if fvg_1h.get("signal") == "BEARISH" and fvg_1h.get("unfilled_bearish", 0) > 0:
                s_entry.append(f"📐 1H'de doldurulmamış Ayı FVG — yükselişte short giriş noktası")

            short_scenario["sections"].append({"title": "🔴 Giriş Koşulları", "lines": s_entry})

            # Short SL
            s_sl = []
            if resistance_1h:
                sl_price = resistance_1h * 1.005
                sl_pct = abs((sl_price - current_price) / current_price * 100)
                s_sl.append(f"Agresif SL: {fmt(sl_price)} (1H direnç üstü, +%{sl_pct:.1f})")
            if resistance_4h and resistance_4h > current_price:
                sl_price_safe = resistance_4h * 1.01
                sl_pct_safe = abs((sl_price_safe - current_price) / current_price * 100)
                s_sl.append(f"Güvenli SL: {fmt(sl_price_safe)} (4H direnç üstü, +%{sl_pct_safe:.1f})")
            if atr_val_1h:
                sl_atr = current_price + (atr_val_1h * 1.5)
                sl_pct_atr = abs((sl_atr - current_price) / current_price * 100)
                s_sl.append(f"ATR Bazlı SL: {fmt(sl_atr)} (1.5x ATR, +%{sl_pct_atr:.1f})")
            if not s_sl:
                s_sl.append(f"ATR bazlı SL önerilir: Mevcut fiyattan 1.5-2x ATR üstü")
            short_scenario["sections"].append({"title": "🛑 Stop Loss", "lines": s_sl})

            # Short TP
            s_tp = []
            if support_1h:
                tp_pct = abs((current_price - support_1h) / current_price * 100)
                s_tp.append(f"TP1: {fmt(support_1h)} (1H destek, +%{tp_pct:.1f})")
            if support_4h and support_4h != support_1h:
                tp_pct2 = abs((current_price - support_4h) / current_price * 100)
                s_tp.append(f"TP2: {fmt(support_4h)} (4H destek, +%{tp_pct2:.1f})")
            if bb_lower_4h and bb_lower_4h < current_price:
                tp_pct3 = abs((current_price - bb_lower_4h) / current_price * 100)
                s_tp.append(f"TP3: {fmt(bb_lower_4h)} (BB alt bant, +%{tp_pct3:.1f})")
            if not s_tp:
                if atr_val_1h:
                    tp_auto = current_price - (atr_val_1h * 3)
                    s_tp.append(f"TP: {fmt(tp_auto)} (3x ATR hedef)")
            short_scenario["sections"].append({"title": "🎯 Hedef (Take Profit)", "lines": s_tp})

            # Short R:R
            s_rr = []
            best_sl_s = None
            best_tp_s = None
            if resistance_1h:
                best_sl_s = resistance_1h * 1.005
            elif atr_val_1h:
                best_sl_s = current_price + (atr_val_1h * 1.5)
            if support_1h:
                best_tp_s = support_1h
            elif support_4h:
                best_tp_s = support_4h

            if best_sl_s and best_tp_s and best_tp_s < current_price < best_sl_s:
                risk = best_sl_s - current_price
                reward = current_price - best_tp_s
                rr = reward / risk if risk > 0 else 0
                s_rr.append(f"Risk: {fmt(risk)} ({abs(risk/current_price*100):.1f}%) | Ödül: {fmt(reward)} ({abs(reward/current_price*100):.1f}%)")
                s_rr.append(f"Risk:Ödül = 1:{rr:.1f} {'✅ Uygun' if rr >= 2 else '⚠ Düşük (min 1:2 önerilir)'}")
            short_scenario["sections"].append({"title": "📐 Risk/Ödül", "lines": s_rr})

            # ─── GENEL STRATEJİ ÖNERİSİ ─── 
            strategy_lines = []

            # Ana yön belirleme
            if overall_net >= 30:
                strategy_lines.append("🟢 GÜÇLÜ LONG ÖNCELİKLİ — Tüm göstergeler yükseliş destekliyor.")
                strategy_lines.append("Geri çekilmeleri alım fırsatı olarak değerlendirin.")
                recommended = "LONG"
            elif overall_net >= 15:
                strategy_lines.append("🟢 LONG ÖNCELİKLİ — Yükseliş trendi aktif.")
                strategy_lines.append("Short riskli. Sadece önemli dirençlerde kısa vadeli short denenebilir.")
                recommended = "LONG"
            elif overall_net >= 6:
                strategy_lines.append("🟡 HAFİF LONG EĞİLİMLİ — Sinyal güçlü değil.")
                strategy_lines.append("Pozisyon almak için ek onay (hacim artışı, mum kapanışı) bekleyin.")
                recommended = "LONG_CAUTIOUS"
            elif overall_net <= -30:
                strategy_lines.append("🔴 GÜÇLÜ SHORT ÖNCELİKLİ — Tüm göstergeler düşüş destekliyor.")
                strategy_lines.append("Yükselişleri satış fırsatı olarak değerlendirin.")
                recommended = "SHORT"
            elif overall_net <= -15:
                strategy_lines.append("🔴 SHORT ÖNCELİKLİ — Düşüş trendi aktif.")
                strategy_lines.append("Long riskli. Sadece güçlü desteklerde tepki alım denenebilir.")
                recommended = "SHORT"
            elif overall_net <= -6:
                strategy_lines.append("🟠 HAFİF SHORT EĞİLİMLİ — Sinyal güçlü değil.")
                strategy_lines.append("Net kırılım olmadan short girmeyin, 4H mum kapanışı bekleyin.")
                recommended = "SHORT_CAUTIOUS"
            else:
                strategy_lines.append("⚪ NÖTR — Piyasa yön vermiyor.")
                strategy_lines.append("Pozisyon almak riskli. Kenarda kalıp net sinyal bekleyin.")
                recommended = "WAIT"

            # Önemli uyarılar
            if atr_pct_4h and atr_pct_4h > 5:
                strategy_lines.append(f"⚡ Yüksek volatilite (%{atr_pct_4h:.1f}) — Pozisyon boyutunu %50 küçültün.")
            if rsi_4h and rsi_4h > 75:
                strategy_lines.append(f"⚠ 4H RSI aşırı alım ({rsi_4h:.0f}) — Long'da dikkatli olun, geri çekilme yakın.")
            elif rsi_4h and rsi_4h < 25:
                strategy_lines.append(f"⚠ 4H RSI aşırı satım ({rsi_4h:.0f}) — Short'da dikkatli olun, bouncing yakın.")

            if vol_ratio and vol_ratio < 0.5:
                strategy_lines.append("📉 Düşük hacim — Breakout'lar güvenilmez, tuzak olabilir.")
            elif vol_ratio and vol_ratio > 2:
                strategy_lines.append("📈 Yüksek hacim — Hareket güçlü, trend devam edebilir.")

            if bb_squeeze_4h == "DARALIYOR":
                strategy_lines.append("🔥 4H Bollinger sıkışması — Büyük bir hareket yaklaşıyor, yön belli olana kadar bekleyin.")

            # Bekleme stratejisi detayı
            wait_lines = []
            if recommended == "LONG" or recommended == "LONG_CAUTIOUS":
                if support_1h:
                    wait_lines.append(f"📍 Beklenen geri çekilme bölgesi: {fmt(support_1h)} civarı")
                wait_lines.append("✅ Giriş onayı: 15m'de en az 2 yeşil mum kapanışı + MACD histogram pozitife dönmeli")
                wait_lines.append("✅ Hacim onayı: Son mumların hacmi 20-periyot ortalamasının üzerinde olmalı")
                if ema_15m.get("ema8") and ema_15m.get("ema21"):
                    wait_lines.append(f"✅ EMA onayı: 15m EMA8 ({fmt(ema_15m['ema8'])}) > EMA21 ({fmt(ema_15m['ema21'])}) kalmalı")
            elif recommended == "SHORT" or recommended == "SHORT_CAUTIOUS":
                if resistance_1h:
                    wait_lines.append(f"📍 Beklenen çekilme bölgesi: {fmt(resistance_1h)} civarı")
                wait_lines.append("✅ Giriş onayı: 15m'de en az 2 kırmızı mum kapanışı + MACD histogram negatife dönmeli")
                wait_lines.append("✅ Hacim onayı: Son mumların hacmi 20-periyot ortalamasının üzerinde olmalı")
                if ema_15m.get("ema8") and ema_15m.get("ema21"):
                    wait_lines.append(f"✅ EMA onayı: 15m EMA8 ({fmt(ema_15m['ema8'])}) < EMA21 ({fmt(ema_15m['ema21'])}) kalmalı")
            else:
                wait_lines.append("⏸ Şu an pozisyon almak riskli — aşağıdaki seviyelerden birinin kırılmasını bekleyin:")
                if resistance_1h:
                    wait_lines.append(f"  Yukarı kırılım: {fmt(resistance_1h)} üzeri kapanış → LONG sinyali")
                if support_1h:
                    wait_lines.append(f"  Aşağı kırılım: {fmt(support_1h)} altı kapanış → SHORT sinyali")

            # Anahtar seviyeler tablosu
            levels = []
            if resistance_4h:
                levels.append({"name": "4H Direnç", "price": resistance_4h, "type": "resistance"})
            if resistance_1h:
                levels.append({"name": "1H Direnç", "price": resistance_1h, "type": "resistance"})
            if bb_upper_4h and bb_upper_4h > current_price:
                levels.append({"name": "BB Üst", "price": bb_upper_4h, "type": "resistance"})
            if ema50_4h and ema50_4h > current_price:
                levels.append({"name": "4H EMA50", "price": ema50_4h, "type": "resistance"})
            if ema50_4h and ema50_4h < current_price:
                levels.append({"name": "4H EMA50", "price": ema50_4h, "type": "support"})
            if support_1h:
                levels.append({"name": "1H Destek", "price": support_1h, "type": "support"})
            if support_4h:
                levels.append({"name": "4H Destek", "price": support_4h, "type": "support"})
            if bb_lower_4h and bb_lower_4h < current_price:
                levels.append({"name": "BB Alt", "price": bb_lower_4h, "type": "support"})

            # Seviyeleri sırala (yüksekten düşüğe)
            levels.sort(key=lambda x: x["price"], reverse=True)

            return {
                "recommended": recommended,
                "long": long_scenario,
                "short": short_scenario,
                "strategy": strategy_lines,
                "wait_conditions": wait_lines,
                "key_levels": [{"name": l["name"], "price": fmt(l["price"]), "price_raw": l["price"], "type": l["type"]} for l in levels],
                "current_price": current_price,
                "current_price_fmt": fmt(current_price)
            }

        # Senaryo üret
        scenario = _generate_trading_scenario(tf_results, price_info, orderbook_result, overall_net, overall, warnings)

        response = {
            "symbol": symbol,
            "price": price_info,
            "timeframes": tf_results,
            "orderbook": orderbook_result,
            "overall": {
                "verdict": overall,
                "label": f"{overall_emoji} {overall_label}",
                "description": overall_desc,
                "net_score": overall_net,
                "bull_total": round(total_bull, 1),
                "bear_total": round(total_bear, 1),
                "confidence": overall_confidence,
                "warnings": warnings,
                "tf_confluence": "ALL_BULL" if all_bull else ("ALL_BEAR" if all_bear else "MIXED")
            },
            "scenario": scenario,
            "timestamp": datetime.now().isoformat()
        }

        def serialize(obj):
            if hasattr(obj, "isoformat"):
                return obj.isoformat()
            elif hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        return jsonify(json.loads(json.dumps(response, default=serialize)))

    except Exception as e:
        logger.error(f"Coin detay hatası ({symbol}): {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/ticker/<symbol>")
def api_ticker(symbol):
    """Anlık fiyat"""
    ticker = data_fetcher.get_ticker(symbol)
    if ticker:
        return jsonify(ticker)
    return jsonify({"error": "Fiyat alınamadı"}), 400


@app.route("/api/backtest/<symbol>")
def api_backtest(symbol):
    """
    Strateji backtest: Geçmiş mum verilerinde sinyalleri simüle ederek 
    win rate, PnL, R:R ve trade detayları döndürür.
    """
    import numpy as np
    import pandas as pd

    tf = request.args.get("tf", "1H")
    limit = min(int(request.args.get("limit", 300)), 300)
    min_score = int(request.args.get("min_score", 20))

    try:
        df = data_fetcher.get_candles(symbol, timeframe=tf, limit=limit)
        if df is None or df.empty or len(df) < 50:
            return jsonify({"error": "Yetersiz veri — en az 50 mum gerekli"}), 400

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ── GÖSTERGE HESAPLAMA ──
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        # Stochastic RSI
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        stoch = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)) * 100
        stoch_k = stoch.rolling(window=3).mean()
        stoch_d = stoch_k.rolling(window=3).mean()

        # EMA'lar
        ema8 = close.ewm(span=8, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean() if len(close) >= 50 else pd.Series([np.nan]*len(close), index=close.index)

        # Bollinger Bands
        bb_sma = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_sma + (bb_std * 2)
        bb_lower = bb_sma - (bb_std * 2)

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.ewm(alpha=1/14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr_series.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14).mean() / atr_series.replace(0, np.nan))
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=1/14, min_periods=14).mean()

        # Volume ratio
        vol_sma = df["volume"].rolling(window=20).mean()
        vol_ratio = df["volume"] / vol_sma.replace(0, np.nan)

        # ── SINYAL TARAMA (OPTİMİZE v3) ──
        # Prensipler: Az ama kaliteli giriş, trailing yok, geniş R:R
        trades = []
        in_trade = False
        trade_entry = None

        for i in range(50, len(df) - 1):
            if in_trade:
                # Basit SL/TP — trailing yok (kazançlıları kesmesin)
                next_high = high.iloc[i]
                next_low = low.iloc[i]

                if trade_entry["direction"] == "LONG":
                    if next_low <= trade_entry["sl"]:
                        pnl = ((trade_entry["sl"] - trade_entry["price"]) / trade_entry["price"]) * 100
                        trade_entry["exit"] = trade_entry["sl"]
                        trade_entry["pnl"] = round(pnl, 3)
                        trade_entry["result"] = "LOSS"
                        trade_entry["exit_idx"] = i
                        trades.append(trade_entry)
                        in_trade = False
                    elif next_high >= trade_entry["tp"]:
                        pnl = ((trade_entry["tp"] - trade_entry["price"]) / trade_entry["price"]) * 100
                        trade_entry["exit"] = trade_entry["tp"]
                        trade_entry["pnl"] = round(pnl, 3)
                        trade_entry["result"] = "WIN"
                        trade_entry["exit_idx"] = i
                        trades.append(trade_entry)
                        in_trade = False

                elif trade_entry["direction"] == "SHORT":
                    if next_high >= trade_entry["sl"]:
                        pnl = ((trade_entry["price"] - trade_entry["sl"]) / trade_entry["price"]) * 100
                        trade_entry["exit"] = trade_entry["sl"]
                        trade_entry["pnl"] = round(pnl, 3)
                        trade_entry["result"] = "LOSS"
                        trade_entry["exit_idx"] = i
                        trades.append(trade_entry)
                        in_trade = False
                    elif next_low <= trade_entry["tp"]:
                        pnl = ((trade_entry["price"] - trade_entry["tp"]) / trade_entry["price"]) * 100
                        trade_entry["exit"] = trade_entry["tp"]
                        trade_entry["pnl"] = round(pnl, 3)
                        trade_entry["result"] = "WIN"
                        trade_entry["exit_idx"] = i
                        trades.append(trade_entry)
                        in_trade = False
                continue

            # ── SKOR HESAPLAMA ──
            cur_close = close.iloc[i]
            cur_open = df["open"].iloc[i]
            cur_rsi = rsi.iloc[i] if not np.isnan(rsi.iloc[i]) else 50
            cur_stoch_k = stoch_k.iloc[i] if not np.isnan(stoch_k.iloc[i]) else 50
            cur_stoch_d = stoch_d.iloc[i] if not np.isnan(stoch_d.iloc[i]) else 50
            cur_macd = macd_line.iloc[i] if not np.isnan(macd_line.iloc[i]) else 0
            cur_signal = signal_line.iloc[i] if not np.isnan(signal_line.iloc[i]) else 0
            cur_hist = histogram.iloc[i] if not np.isnan(histogram.iloc[i]) else 0
            prev_hist = histogram.iloc[i-1] if i > 0 and not np.isnan(histogram.iloc[i-1]) else 0
            cur_ema8 = ema8.iloc[i]
            cur_ema21 = ema21.iloc[i]
            cur_ema50 = ema50.iloc[i] if not np.isnan(ema50.iloc[i]) else None
            cur_adx = adx.iloc[i] if not np.isnan(adx.iloc[i]) else 0
            cur_pdi = plus_di.iloc[i] if not np.isnan(plus_di.iloc[i]) else 0
            cur_mdi = minus_di.iloc[i] if not np.isnan(minus_di.iloc[i]) else 0
            cur_atr = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else 0
            cur_vol_ratio = vol_ratio.iloc[i] if not np.isnan(vol_ratio.iloc[i]) else 1
            cur_bb_upper = bb_upper.iloc[i] if not np.isnan(bb_upper.iloc[i]) else cur_close * 1.02
            cur_bb_lower = bb_lower.iloc[i] if not np.isnan(bb_lower.iloc[i]) else cur_close * 0.98

            bull_score = 0
            bear_score = 0
            bull_confirms = 0
            bear_confirms = 0

            # 1) Trend — EMA hizalama (20 puan, 1 onay)
            if cur_ema8 > cur_ema21:
                bull_confirms += 1
                if cur_ema50 and cur_close > cur_ema50:
                    bull_score += 20
                else:
                    bull_score += 8
            else:
                bear_confirms += 1
                if cur_ema50 and cur_close < cur_ema50:
                    bear_score += 20
                else:
                    bear_score += 8

            # 2) ADX Yünü — +DI vs -DI (15 puan, 1 onay)
            if cur_adx > 20:
                adx_s = 15 * min(cur_adx / 50, 1.0)
                if cur_pdi > cur_mdi:
                    bull_score += adx_s
                    bull_confirms += 1
                else:
                    bear_score += adx_s
                    bear_confirms += 1

            # 3) MACD (15 puan, 1 onay)
            if cur_macd > cur_signal:
                bull_confirms += 1
                if cur_hist > prev_hist:
                    bull_score += 15
                else:
                    bull_score += 5
            else:
                bear_confirms += 1
                if cur_hist < prev_hist:
                    bear_score += 15
                else:
                    bear_score += 5

            # 4) RSI (10 puan, 1 onay)
            if cur_rsi > 55:
                bull_score += 10
                bull_confirms += 1
            elif cur_rsi < 45:
                bear_score += 10
                bear_confirms += 1

            # 5) StochRSI (8 puan, 1 onay)
            if cur_stoch_k > 50 and cur_stoch_k > cur_stoch_d:
                bull_score += 8
                bull_confirms += 1
            elif cur_stoch_k < 50 and cur_stoch_k < cur_stoch_d:
                bear_score += 8
                bear_confirms += 1

            # Volume (10 puan — onay sayılmaz, güç katkısı)
            if cur_vol_ratio > 1.2:
                if cur_ema8 > cur_ema21:
                    bull_score += 10
                else:
                    bear_score += 10

            # Bollinger (5 puan — onay sayılmaz, aşırı bölge katkısı)
            if cur_close <= cur_bb_lower and cur_rsi < 35:
                bull_score += 5
            elif cur_close >= cur_bb_upper and cur_rsi > 65:
                bear_score += 5

            net_score = round(bull_score - bear_score, 1)

            # ── KALİTE FİLTRELERİ ──
            # 1. Minimum skor eşiği
            if abs(net_score) < min_score:
                continue

            direction = "LONG" if net_score > 0 else "SHORT"
            confirms = bull_confirms if direction == "LONG" else bear_confirms

            # 2. En az 4/5 gösterge aynı yönde olmalı
            if confirms < 4:
                continue

            # 3. EMA yayılma filtresi — EMA8 ve EMA21 çok yakınsa (piyasa kararsız)
            ema_spread = abs(cur_ema8 - cur_ema21) / cur_close
            if ema_spread < 0.0015:
                continue

            # 4. Mum yönü filtresi — sinyal yönünde kapanmış mum gerekli
            candle_bullish = cur_close > cur_open
            if direction == "LONG" and not candle_bullish:
                continue
            if direction == "SHORT" and candle_bullish:
                continue

            entry_price = close.iloc[i + 1]

            # Sabit SL/TP: 1.5 ATR SL, 1:2.5 R:R
            sl_distance = cur_atr * 1.5 if cur_atr > 0 else entry_price * 0.015
            tp_distance = sl_distance * 2.5

            if direction == "LONG":
                sl = entry_price - sl_distance
                tp = entry_price + tp_distance
            else:
                sl = entry_price + sl_distance
                tp = entry_price - tp_distance

            in_trade = True
            trade_entry = {
                "direction": direction,
                "price": entry_price,
                "sl": sl,
                "tp": tp,
                "score": net_score,
                "entry_idx": i + 1,
                "atr": cur_atr
            }

        # Açık kalan trade'i son fiyattan kapat
        if in_trade and trade_entry:
            last_price = close.iloc[-1]
            if trade_entry["direction"] == "LONG":
                pnl = ((last_price - trade_entry["price"]) / trade_entry["price"]) * 100
            else:
                pnl = ((trade_entry["price"] - last_price) / trade_entry["price"]) * 100
            trade_entry["exit"] = last_price
            trade_entry["pnl"] = round(pnl, 3)
            trade_entry["result"] = "WIN" if pnl > 0 else "LOSS"
            trade_entry["exit_idx"] = len(df) - 1
            trades.append(trade_entry)

        # ── SONUÇ HESAPLAMA ──
        wins = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)

        win_pnls = [t["pnl"] for t in trades if t["result"] == "WIN"]
        loss_pnls = [t["pnl"] for t in trades if t["result"] == "LOSS"]
        avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
        avg_loss = abs(sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 1
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

        best_trade = max((t["pnl"] for t in trades), default=0)
        worst_trade = min((t["pnl"] for t in trades), default=0)

        # Equity curve
        equity_curve = [t["pnl"] for t in trades]

        # Fiyat format
        def fmt_bt(val):
            if val >= 1:
                return f"{val:.4f}"
            elif val >= 0.001:
                return f"{val:.6f}"
            else:
                return f"{val:.8f}"

        # Trades listesi
        trades_output = [{
            "direction": t["direction"],
            "entry_price": fmt_bt(t["price"]),
            "exit_price": fmt_bt(t["exit"]),
            "sl_price": fmt_bt(t["sl"]),
            "tp_price": fmt_bt(t["tp"]),
            "result": t["result"],
            "pnl": t["pnl"],
            "score": t["score"]
        } for t in trades]

        # Strateji analizi
        analysis = []
        if total_trades == 0:
            analysis.append("ℹ Bu ayarlarla hiç sinyal üretilmedi. Min skor eşiğini düşürmeyi veya daha uzun periyot seçmeyi deneyin.")
        else:
            analysis.append(f"📊 Toplam {total_trades} işlem simüle edildi ({tf} zaman diliminde, {limit} mum)")

            if win_rate >= 60:
                analysis.append(f"✅ Kazanma oranı %{win_rate:.0f} — strateji bu coin için başarılı görünüyor")
            elif win_rate >= 45:
                analysis.append(f"⚠ Kazanma oranı %{win_rate:.0f} — ortalama performans, R:R oranı önemli")
            else:
                analysis.append(f"❌ Kazanma oranı %{win_rate:.0f} — strateji bu coin için zayıf")

            if avg_rr >= 2:
                analysis.append(f"✅ Ortalama R:R 1:{avg_rr:.1f} — iyi risk/ödül dengesı")
            elif avg_rr >= 1:
                analysis.append(f"⚠ Ortalama R:R 1:{avg_rr:.1f} — kabul edilebilir ama geliştirilebilir")
            else:
                analysis.append(f"❌ Ortalama R:R 1:{avg_rr:.1f} — kötü risk/ödül, SL/TP ayarı gözden geçirilmeli")

            if total_pnl > 0:
                analysis.append(f"💰 Toplam PnL: +%{total_pnl:.2f} — kârlı strateji")
            else:
                analysis.append(f"📉 Toplam PnL: %{total_pnl:.2f} — zararda, strateji bu markete uygun olmayabilir")

            long_trades = [t for t in trades if t["direction"] == "LONG"]
            short_trades = [t for t in trades if t["direction"] == "SHORT"]
            long_wr = (sum(1 for t in long_trades if t["result"] == "WIN") / len(long_trades) * 100) if long_trades else 0
            short_wr = (sum(1 for t in short_trades if t["result"] == "WIN") / len(short_trades) * 100) if short_trades else 0

            if long_trades:
                long_pnl = sum(t["pnl"] for t in long_trades)
                analysis.append(f"📈 LONG: {len(long_trades)} işlem, %{long_wr:.0f} başarı, PnL: {'+' if long_pnl>=0 else ''}{long_pnl:.2f}%")
            if short_trades:
                short_pnl = sum(t["pnl"] for t in short_trades)
                analysis.append(f"📉 SHORT: {len(short_trades)} işlem, %{short_wr:.0f} başarı, PnL: {'+' if short_pnl>=0 else ''}{short_pnl:.2f}%")

            # Yüksek skorlu işlemlerin performansı
            high_score_trades = [t for t in trades if abs(t["score"]) >= 30]
            if high_score_trades:
                hs_wr = sum(1 for t in high_score_trades if t["result"] == "WIN") / len(high_score_trades) * 100
                hs_pnl = sum(t["pnl"] for t in high_score_trades)
                analysis.append(f"🎯 Yüksek skorlu (30+) işlemler: {len(high_score_trades)} adet, %{hs_wr:.0f} başarı, PnL: {'+' if hs_pnl>=0 else ''}{hs_pnl:.2f}%")

            # Max drawdown
            cumulative = 0
            peak = 0
            max_dd = 0
            for t in trades:
                cumulative += t["pnl"]
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
            if max_dd > 0:
                analysis.append(f"📊 Maksimum düşüş (drawdown): %{max_dd:.2f}")

            # Ardışık kayıp
            max_losing_streak = 0
            current_streak = 0
            for t in trades:
                if t["result"] == "LOSS":
                    current_streak += 1
                    max_losing_streak = max(max_losing_streak, current_streak)
                else:
                    current_streak = 0
            if max_losing_streak >= 3:
                analysis.append(f"⚠ En uzun kayıp serisi: {max_losing_streak} ardışık kayıp — duygusal kontrol önemli")

        return jsonify({
            "symbol": symbol,
            "timeframe": tf,
            "candles": limit,
            "min_score": min_score,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_rr": round(avg_rr, 1),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "equity_curve": equity_curve,
            "trades": trades_output,
            "analysis": analysis
        })

    except Exception as e:
        logger.error(f"Backtest hatası ({symbol}): {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/params")
def api_params():
    """Güncel bot parametreleri"""
    params = {}
    for key, default_val in ICT_PARAMS.items():
        current = get_bot_param(key)
        params[key] = {
            "current": current if current is not None else default_val,
            "default": default_val
        }
    return jsonify(params)


@app.route("/api/coins")
def api_coins():
    """OKX'ten 5M+ hacimli aktif coin listesi"""
    coins = data_fetcher.get_high_volume_coins(force_refresh=True)
    volumes = data_fetcher.get_all_coin_volumes()
    result = []
    for symbol in coins:
        info = volumes.get(symbol, {})
        result.append({
            "symbol": symbol,
            "volume_usdt": info.get("volume_usdt", 0),
            "last_price": info.get("last_price", 0),
            "change_pct": info.get("change_pct", 0)
        })
    return jsonify({
        "min_volume": MIN_VOLUME_USDT,
        "total_coins": len(result),
        "coins": result
    })



# =================== REGIME API ===================

@app.route("/api/regime")
def api_regime():
    """Piyasa rejimi detayları — BTC trend, BTC.D, USDT flow, RS sıralaması"""
    summary = market_regime.get_regime_summary()
    return jsonify(summary)


@app.route("/api/regime/rankings")
def api_regime_rankings():
    """Tüm coinlerin rölatif güç sıralaması"""
    cached = market_regime.get_cached_regime()
    if not cached:
        return jsonify([])
    return jsonify(cached.get("rs_rankings", []))


# =================== FOREX / EMTİA ICT API ===================

@app.route("/api/forex/instruments")
def api_forex_instruments():
    """Desteklenen forex/emtia enstrümanları"""
    instruments = []
    for key, inst in FOREX_INSTRUMENTS.items():
        instruments.append({
            "key": key,
            "name": inst["name"],
            "category": inst["category"],
            "icon": inst["icon"],
            "desc": inst["desc"],
        })
    return jsonify(instruments)


@app.route("/api/forex/scan")
def api_forex_scan():
    """Tüm forex enstrümanlarını ICT ile tara"""
    tf = request.args.get("tf", "1h")
    try:
        results = forex_ict.scan_all(timeframe=tf)
        return jsonify({"results": results, "timeframe": tf, "count": len(results)})
    except Exception as e:
        logger.error(f"Forex tarama hatası: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forex/signal/<instrument>")
def api_forex_signal(instrument):
    """Tek enstrüman ICT sinyal analizi"""
    tf = request.args.get("tf", "1h")
    instrument = instrument.upper()
    if instrument not in FOREX_INSTRUMENTS:
        return jsonify({"error": f"Bilinmeyen enstrüman: {instrument}"}), 400
    try:
        result = forex_ict.generate_signal(instrument, timeframe=tf)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Forex sinyal hatası ({instrument}): {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/forex/kill-zones")
def api_forex_kill_zones():
    """Aktif Kill Zone bilgisi"""
    return jsonify(forex_ict.detect_kill_zones())


# =================== WEBSOCKET ===================

@socketio.on("connect")
def handle_connect():
    logger.info("WebSocket client bağlandı")
    socketio.emit("bot_status", {"running": bot_state["running"]})


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("WebSocket client ayrıldı")


@socketio.on("request_update")
def handle_request_update():
    """Frontend'den anlık güncelleme isteği"""
    stats = get_performance_summary()
    signals = get_active_signals()
    watching = get_watching_items()

    socketio.emit("full_update", {
        "stats": stats,
        "active_signals": signals,
        "watching": watching,
        "bot_state": bot_state,
        "timestamp": datetime.now().isoformat()
    })


# =================== BAŞLATMA ===================

import os

# Render/Gunicorn ile çalışırken de DB'yi başlat
init_db()
logger.info("ICT Trading Bot v1.0 - Veritabanı hazır")

# =================== SELF-PING (Render uyku engelleme) ===================

def self_ping():
    """Render free tier'da uyumayı engelle — her 10 dakikada bir health endpoint'i çağır"""
    try:
        render_url = os.environ.get("RENDER_EXTERNAL_URL")
        if render_url:
            import requests
            resp = requests.get(f"{render_url}/api/health", timeout=10)
            logger.debug(f"Self-ping OK: {resp.status_code}")
    except Exception as e:
        logger.debug(f"Self-ping hata (önemsiz): {e}")


# Render'da otomatik başlat (gunicorn ile)
if os.environ.get("RENDER"):
    # Render ortamında botu otomatik başlat
    import atexit
    def auto_start_bot():
        """Gunicorn worker başladığında botu otomatik başlat"""
        if not bot_state["running"]:
            bot_state["running"] = True
            create_scheduler()
            scheduler.add_job(scan_markets, "interval", seconds=SCAN_INTERVAL_SECONDS,
                             id="scan_markets", replace_existing=True)
            scheduler.add_job(check_trades, "interval", seconds=TRADE_CHECK_INTERVAL,
                             id="check_trades", replace_existing=True)
            scheduler.add_job(run_optimizer, "interval",
                             minutes=OPTIMIZER_CONFIG["optimization_interval_minutes"],
                             id="run_optimizer", replace_existing=True)
            # Self-ping: her 10 dakikada Render'ı uyanık tut
            scheduler.add_job(self_ping, "interval", minutes=10,
                             id="self_ping", replace_existing=True)
            scheduler.start()
            threading.Thread(target=scan_markets, daemon=True).start()
            logger.info("🚀 Bot Render'da otomatik başlatıldı! (Self-ping aktif)")

    # İlk request'te değil, uygulama başlarken çalıştır
    auto_start_timer = threading.Timer(5.0, auto_start_bot)
    auto_start_timer.daemon = True
    auto_start_timer.start()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  ICT Trading Bot v1.0 - GERÇEK VERİ")
    logger.info("  OKX Public API | Dinamik Coin Filtresi")
    logger.info("=" * 60)
    logger.info(f"  Min 24h Hacim: ${MIN_VOLUME_USDT:,.0f} USDT")
    logger.info(f"  Tarama aralığı: {SCAN_INTERVAL_SECONDS}s")
    logger.info(f"  Web arayüz: http://localhost:{PORT}")
    logger.info("=" * 60)

    coins = data_fetcher.get_high_volume_coins(force_refresh=True)
    logger.info(f"  Başlangıç: {len(coins)} coin 5M+ hacimle tespit edildi")
    logger.info("=" * 60)

    socketio.run(app, host=HOST, port=PORT, debug=DEBUG)
