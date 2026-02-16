# =====================================================
# ICT Trading Bot - Veritabanı İşlemleri
# =====================================================

import sqlite3
import json
import threading
from datetime import datetime
from config import DB_PATH

_local = threading.local()


def get_db():
    """Thread-safe veritabanı bağlantısı"""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Veritabanı tablolarını oluştur"""
    conn = get_db()
    cursor = conn.cursor()

    # Sinyaller tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,           -- 'LONG' veya 'SHORT'
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            confidence REAL NOT NULL,          -- 0-100 arası güven skoru
            confluence_score REAL NOT NULL,    -- 0-100 arası confluent skor
            status TEXT DEFAULT 'WAITING',     -- WAITING, ACTIVE, WON, LOST, CANCELLED, WATCHING
            components TEXT,                   -- JSON: hangi ICT bileşenleri tetikledi
            timeframe TEXT,
            entry_time TEXT,
            close_time TEXT,
            close_price REAL,
            pnl_pct REAL,                     -- Kar/zarar yüzdesi
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # İzleme listesi (sabırlı mod)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            potential_entry REAL,
            potential_sl REAL,
            potential_tp REAL,
            watch_reason TEXT,                -- Neden izleniyor
            candles_watched INTEGER DEFAULT 0,
            confirmation_count INTEGER DEFAULT 0,
            last_5m_candle_ts TEXT,
            max_watch_candles INTEGER DEFAULT 3,
            initial_score REAL,
            current_score REAL,
            status TEXT DEFAULT 'WATCHING',   -- WATCHING, PROMOTED, EXPIRED
            components TEXT,                  -- JSON
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Eski veritabanlarında eksik kolonları güvenli şekilde ekle
    existing_cols = {
        row["name"] for row in conn.execute("PRAGMA table_info(watchlist)").fetchall()
    }
    if "confirmation_count" not in existing_cols:
        conn.execute("ALTER TABLE watchlist ADD COLUMN confirmation_count INTEGER DEFAULT 0")
    if "last_5m_candle_ts" not in existing_cols:
        conn.execute("ALTER TABLE watchlist ADD COLUMN last_5m_candle_ts TEXT")
    if "expire_reason" not in existing_cols:
        conn.execute("ALTER TABLE watchlist ADD COLUMN expire_reason TEXT")

    # Optimizasyon logları
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS optimization_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            param_name TEXT NOT NULL,
            old_value REAL NOT NULL,
            new_value REAL NOT NULL,
            reason TEXT,
            win_rate_before REAL,
            win_rate_after REAL,
            total_trades_analyzed INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Bot parametreleri (güncel değerler)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bot_params (
            param_name TEXT PRIMARY KEY,
            param_value REAL NOT NULL,
            default_value REAL NOT NULL,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Performans istatistikleri
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_signals INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            total_pnl_pct REAL DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            avg_confluence REAL DEFAULT 0,
            best_component TEXT,              -- En başarılı ICT bileşeni
            worst_component TEXT,             -- En kötü ICT bileşeni
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()


# =================== SİNYAL İŞLEMLERİ ===================

def add_signal(symbol, direction, entry_price, stop_loss, take_profit,
               confidence, confluence_score, components, timeframe, status="WAITING", notes=""):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO signals (symbol, direction, entry_price, stop_loss, take_profit,
                           confidence, confluence_score, components, timeframe, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, direction, entry_price, stop_loss, take_profit,
          confidence, confluence_score, json.dumps(components), timeframe, status, notes))
    conn.commit()
    return cursor.lastrowid


def update_signal_status(signal_id, status, close_price=None, pnl_pct=None):
    conn = get_db()
    now = datetime.now().isoformat()
    if close_price is not None:
        conn.execute("""
            UPDATE signals SET status=?, close_price=?, pnl_pct=?, close_time=?, updated_at=?
            WHERE id=?
        """, (status, close_price, pnl_pct, now, now, signal_id))
    else:
        conn.execute("""
            UPDATE signals SET status=?, updated_at=? WHERE id=?
        """, (status, now, signal_id))
    conn.commit()


def activate_signal(signal_id):
    conn = get_db()
    now = datetime.now().isoformat()
    conn.execute("""
        UPDATE signals SET status='ACTIVE', entry_time=?, updated_at=? WHERE id=?
    """, (now, now, signal_id))
    conn.commit()


def get_active_signals():
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM signals WHERE status IN ('ACTIVE', 'WAITING') ORDER BY created_at DESC
    """).fetchall()
    return [dict(row) for row in rows]


def get_signal_history(limit=50):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM signals ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_completed_signals(limit=100):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM signals WHERE status IN ('WON', 'LOST') ORDER BY close_time DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_active_trade_count():
    conn = get_db()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM signals WHERE status = 'ACTIVE'
    """).fetchone()
    return row["cnt"] if row else 0


# =================== İZLEME LİSTESİ ===================

def add_to_watchlist(symbol, direction, potential_entry, potential_sl, potential_tp,
                     watch_reason, initial_score, components, max_watch=3):
    conn = get_db()
    cursor = conn.cursor()
    # Aynı sembol ve yönde zaten izleniyor mu?
    existing = conn.execute("""
        SELECT id FROM watchlist WHERE symbol=? AND direction=? AND status='WATCHING'
    """, (symbol, direction)).fetchone()
    if existing:
        return existing["id"]
    # Son 15 dk içinde expire edilmişse tekrar ekleme (cooldown)
    recent_expired = conn.execute("""
        SELECT id FROM watchlist
        WHERE symbol=? AND direction=? AND status='EXPIRED'
          AND updated_at > datetime('now', '-15 minutes')
    """, (symbol, direction)).fetchone()
    if recent_expired:
        return None
    cursor.execute("""
        INSERT INTO watchlist (symbol, direction, potential_entry, potential_sl, potential_tp,
                         watch_reason, candles_watched, confirmation_count, last_5m_candle_ts,
                         initial_score, current_score, components, max_watch_candles)
          VALUES (?, ?, ?, ?, ?, ?, 0, 0, NULL, ?, ?, ?, ?)
    """, (symbol, direction, potential_entry, potential_sl, potential_tp,
            watch_reason, initial_score, initial_score, json.dumps(components), max_watch))
    conn.commit()
    return cursor.lastrowid


def update_watchlist_item(item_id, candles_watched, current_score,
                         confirmation_count=None, last_5m_candle_ts=None, status="WATCHING"):
    conn = get_db()
    now = datetime.now().isoformat()
    if confirmation_count is None and last_5m_candle_ts is None:
        conn.execute("""
            UPDATE watchlist SET candles_watched=?, current_score=?, status=?, updated_at=?
            WHERE id=?
        """, (candles_watched, current_score, status, now, item_id))
    else:
        conn.execute("""
            UPDATE watchlist
            SET candles_watched=?, current_score=?, confirmation_count=?,
                last_5m_candle_ts=?, status=?, updated_at=?
            WHERE id=?
        """, (candles_watched, current_score,
              confirmation_count if confirmation_count is not None else 0,
              last_5m_candle_ts, status, now, item_id))
    conn.commit()


def get_watching_items():
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM watchlist WHERE status='WATCHING' ORDER BY current_score DESC
    """).fetchall()
    return [dict(row) for row in rows]


def promote_watchlist_item(item_id):
    conn = get_db()
    now = datetime.now().isoformat()
    conn.execute("""
        UPDATE watchlist SET status='PROMOTED', updated_at=? WHERE id=?
    """, (now, item_id))
    conn.commit()


def expire_watchlist_item(item_id, reason=None):
    conn = get_db()
    now = datetime.now().isoformat()
    conn.execute("""
        UPDATE watchlist SET status='EXPIRED', expire_reason=?, updated_at=? WHERE id=?
    """, (reason, now, item_id))
    conn.commit()


def get_recently_expired(minutes=30):
    """Son N dakikada expire edilen watchlist öğeleri"""
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM watchlist
        WHERE status='EXPIRED' AND updated_at > datetime('now', ? || ' minutes')
        ORDER BY updated_at DESC
        LIMIT 20
    """, (str(-minutes),)).fetchall()
    return [dict(row) for row in rows]


# =================== OPTİMİZASYON ===================

def add_optimization_log(param_name, old_value, new_value, reason,
                         win_rate_before, win_rate_after, total_trades):
    conn = get_db()
    conn.execute("""
        INSERT INTO optimization_logs (param_name, old_value, new_value, reason,
                                      win_rate_before, win_rate_after, total_trades_analyzed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (param_name, old_value, new_value, reason, win_rate_before, win_rate_after, total_trades))
    conn.commit()


def get_optimization_logs(limit=30):
    conn = get_db()
    rows = conn.execute("""
        SELECT * FROM optimization_logs ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(row) for row in rows]


# =================== BOT PARAMETRELERİ ===================

def save_bot_param(param_name, param_value, default_value=None):
    conn = get_db()
    now = datetime.now().isoformat()
    if default_value is None:
        default_value = param_value
    conn.execute("""
        INSERT OR REPLACE INTO bot_params (param_name, param_value, default_value, last_updated)
        VALUES (?, ?, ?, ?)
    """, (param_name, param_value, default_value, now))
    conn.commit()


def get_bot_param(param_name, default=None):
    conn = get_db()
    row = conn.execute("""
        SELECT param_value FROM bot_params WHERE param_name=?
    """, (param_name,)).fetchone()
    return row["param_value"] if row else default


def get_all_bot_params():
    conn = get_db()
    rows = conn.execute("SELECT * FROM bot_params").fetchall()
    return {row["param_name"]: row["param_value"] for row in rows}


# =================== İSTATİSTİKLER ===================

def get_performance_summary():
    conn = get_db()
    stats = {}

    # Toplam işlem
    row = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE status IN ('WON','LOST')").fetchone()
    stats["total_trades"] = row["cnt"] if row else 0

    # Kazananlar
    row = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE status='WON'").fetchone()
    stats["winning_trades"] = row["cnt"] if row else 0

    # Kaybedenler
    row = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE status='LOST'").fetchone()
    stats["losing_trades"] = row["cnt"] if row else 0

    # Win rate
    if stats["total_trades"] > 0:
        stats["win_rate"] = round(stats["winning_trades"] / stats["total_trades"] * 100, 1)
    else:
        stats["win_rate"] = 0

    # Toplam PnL
    row = conn.execute("SELECT COALESCE(SUM(pnl_pct), 0) as total FROM signals WHERE status IN ('WON','LOST')").fetchone()
    stats["total_pnl"] = round(row["total"], 2) if row else 0

    # Ortalama güven skoru
    row = conn.execute("SELECT AVG(confidence) as avg_conf FROM signals WHERE status IN ('WON','LOST')").fetchone()
    stats["avg_confidence"] = round(row["avg_conf"], 1) if row and row["avg_conf"] else 0

    # Aktif sinyaller
    row = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE status='ACTIVE'").fetchone()
    stats["active_trades"] = row["cnt"] if row else 0

    # Bekleyen sinyaller
    row = conn.execute("SELECT COUNT(*) as cnt FROM signals WHERE status='WAITING'").fetchone()
    stats["waiting_signals"] = row["cnt"] if row else 0

    # İzlenen coinler
    row = conn.execute("SELECT COUNT(*) as cnt FROM watchlist WHERE status='WATCHING'").fetchone()
    stats["watching_count"] = row["cnt"] if row else 0

    # Ortalama RR
    row = conn.execute("""
        SELECT AVG(ABS(pnl_pct)) as avg_pnl FROM signals WHERE status='WON'
    """).fetchone()
    avg_win = row["avg_pnl"] if row and row["avg_pnl"] else 0

    row = conn.execute("""
        SELECT AVG(ABS(pnl_pct)) as avg_pnl FROM signals WHERE status='LOST'
    """).fetchone()
    avg_loss = row["avg_pnl"] if row and row["avg_pnl"] else 1

    stats["avg_rr"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0

    # Bileşen bazlı performans
    stats["component_performance"] = get_component_performance()

    return stats


def get_component_performance():
    """Her ICT bileşeninin başarı oranını hesapla"""
    conn = get_db()
    rows = conn.execute("""
        SELECT components, status, notes, pnl_pct FROM signals
        WHERE status IN ('WON', 'LOST') AND components IS NOT NULL
    """).fetchall()

    comp_stats = {}
    for row in rows:
        try:
            components = json.loads(row["components"])
        except (json.JSONDecodeError, TypeError):
            continue
        for comp in components:
            if comp not in comp_stats:
                comp_stats[comp] = {"wins": 0, "losses": 0, "total": 0, "pnl_sum": 0.0}
            comp_stats[comp]["total"] += 1
            pnl = row["pnl_pct"] if row["pnl_pct"] else 0
            comp_stats[comp]["pnl_sum"] += pnl
            if row["status"] == "WON":
                comp_stats[comp]["wins"] += 1
            else:
                comp_stats[comp]["losses"] += 1

    # Win rate + ortalama PnL hesapla
    for comp in comp_stats:
        total = comp_stats[comp]["total"]
        if total > 0:
            comp_stats[comp]["win_rate"] = round(comp_stats[comp]["wins"] / total * 100, 1)
            comp_stats[comp]["avg_pnl"] = round(comp_stats[comp]["pnl_sum"] / total, 3)
        else:
            comp_stats[comp]["win_rate"] = 0
            comp_stats[comp]["avg_pnl"] = 0

    return comp_stats


def get_loss_analysis(limit=30):
    """
    Kaybeden işlemlerin detaylı analizini çıkar.
    Bot buradan öğrenecek: neden kaybettik?
    """
    conn = get_db()
    rows = conn.execute("""
        SELECT symbol, direction, components, notes, pnl_pct, confidence,
               confluence_score, created_at, close_time
        FROM signals
        WHERE status = 'LOST'
        ORDER BY close_time DESC LIMIT ?
    """, (limit,)).fetchall()

    analysis = {
        "total_losses": len(rows),
        "avg_loss_pct": 0,
        "common_components": {},       # Kayıpta en çok hangi bileşenler vardı
        "missing_components": {},      # Kayıpta en çok hangi bileşenler EKSİKTİ
        "low_confidence_losses": 0,    # Düşük güvenle girip kaybeden
        "lesson_summary": []
    }

    if not rows:
        return analysis

    all_possible = [
        "MARKET_STRUCTURE", "ORDER_BLOCK", "FVG", "DISPLACEMENT",
        "LIQUIDITY_SWEEP", "OTE", "HTF_CONFIRMATION", "KILLZONE_ACTIVE",
        "DISCOUNT_ZONE", "PREMIUM_ZONE", "BREAKER_BLOCK"
    ]
    total_pnl = 0

    for row in rows:
        total_pnl += abs(row["pnl_pct"]) if row["pnl_pct"] else 0

        if row["confidence"] and row["confidence"] < 65:
            analysis["low_confidence_losses"] += 1

        try:
            comps = json.loads(row["components"]) if row["components"] else []
        except (json.JSONDecodeError, TypeError):
            comps = []

        for c in comps:
            analysis["common_components"][c] = analysis["common_components"].get(c, 0) + 1

        for possible in all_possible:
            if possible not in comps:
                analysis["missing_components"][possible] = \
                    analysis["missing_components"].get(possible, 0) + 1

    analysis["avg_loss_pct"] = round(total_pnl / len(rows), 3) if rows else 0

    # Ders çıkar
    if analysis["low_confidence_losses"] > len(rows) * 0.4:
        analysis["lesson_summary"].append(
            "Kayıpların çoğu düşük güvenle açılmış — min_confidence yükseltilmeli"
        )

    # En çok eksik olan bileşeni bul
    if analysis["missing_components"]:
        worst_missing = max(analysis["missing_components"], key=analysis["missing_components"].get)
        miss_pct = analysis["missing_components"][worst_missing] / len(rows)
        if miss_pct > 0.6:
            analysis["lesson_summary"].append(
                f"Kayıpların %{miss_pct*100:.0f}'inde {worst_missing} eksikti — bu bileşen zorunlu tutulabilir"
            )

    # En çok bulunan ama yine kaybeden bileşeni bul
    if analysis["common_components"]:
        worst_present = max(analysis["common_components"], key=analysis["common_components"].get)
        present_pct = analysis["common_components"][worst_present] / len(rows)
        if present_pct > 0.6:
            analysis["lesson_summary"].append(
                f"{worst_present} bileşeni kayıpların %{present_pct*100:.0f}'inde vardı — "
                f"tek başına güvenilir değil, ek onay gerekli"
            )

    return analysis


# Uygulama başlatıldığında DB'yi initialize et
init_db()
