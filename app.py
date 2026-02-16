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

        # OKX'ten 5M+ hacimli coinleri gerçek zamanlı çek
        active_coins = data_fetcher.get_high_volume_coins()
        bot_state["active_coin_count"] = len(active_coins)

        if not active_coins:
            logger.warning("OKX'ten yüksek hacimli coin bulunamadı, bağlantı kontrol edin")
            return

        for symbol in active_coins:
            try:
                # Gerçek zamanlı çoklu zaman dilimi verisi çek
                multi_tf = data_fetcher.get_multi_timeframe_data(symbol)
                ltf_data = multi_tf.get("15m")

                if ltf_data is None or ltf_data.empty:
                    continue

                # ICT strateji analizi ve sinyal üretimi
                result = ict_strategy.generate_signal(symbol, ltf_data, multi_tf)

                if result:
                    trade_result = trade_manager.process_signal(result)
                    if trade_result:
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
        logger.info(f"✅ Tarama tamamlandı: {symbols_scanned} coin, {len(new_signals)} yeni sinyal")

        # Dashboard güncelle
        socketio.emit("scan_complete", {
            "symbols_scanned": symbols_scanned,
            "new_signals": len(new_signals),
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
        result = self_optimizer.run_optimization()
        bot_state["last_optimization"] = datetime.now().isoformat()

        if result["changes"]:
            # Strateji parametrelerini yenile
            ict_strategy.reload_params()

            socketio.emit("optimization_done", result)
            logger.info(f"🧠 Optimizasyon: {len(result['changes'])} değişiklik yapıldı")

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


@app.route("/api/ticker/<symbol>")
def api_ticker(symbol):
    """Anlık fiyat"""
    ticker = data_fetcher.get_ticker(symbol)
    if ticker:
        return jsonify(ticker)
    return jsonify({"error": "Fiyat alınamadı"}), 400


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
