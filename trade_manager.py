# =====================================================
# ICT Trading Bot - Trade Yönetim Modülü
# =====================================================
# Açık işlemlerin SL/TP takibi, pozisyon yönetimi
# =====================================================

import logging
from datetime import datetime
from data_fetcher import data_fetcher
from database import (
    get_active_signals, update_signal_status, activate_signal,
    get_active_trade_count, add_signal, add_to_watchlist,
    get_watching_items, update_watchlist_item, promote_watchlist_item,
    expire_watchlist_item, get_signal_history, get_bot_param
)
from config import ICT_PARAMS

logger = logging.getLogger("ICT-Bot.TradeManager")


class TradeManager:
    """Açık işlemlerin yönetimi ve takibi"""

    def __init__(self):
        # Breakeven/Trailing SL takibi: {signal_id: {"breakeven_moved": bool, "trailing_sl": float}}
        self._trade_state = {}

    def _param(self, name):
        """Optimizer ile güncellenen parametreleri DB'den oku, yoksa varsayılanı kullan."""
        return get_bot_param(name, ICT_PARAMS[name])

    def process_signal(self, signal_result):
        """
        Strateji motorundan gelen sonucu işle
        - SIGNAL -> İşleme al
        - WATCH -> İzleme listesine ekle
        """
        if signal_result is None:
            return None

        action = signal_result.get("action")

        if action == "SIGNAL":
            return self._open_trade(signal_result)
        elif action == "WATCH":
            return self._add_to_watch(signal_result)

        return None

    def _open_trade(self, signal):
        """Yeni işlem aç"""
        # Max eşzamanlı işlem kontrolü
        max_concurrent = int(self._param("max_concurrent_trades"))
        active_count = get_active_trade_count()
        if active_count >= max_concurrent:
            logger.warning(f"Maksimum eşzamanlı işlem limitine ulaşıldı ({max_concurrent})")
            return {"status": "REJECTED", "reason": "Maksimum işlem limiti"}

        # Aynı coinde aktif işlem var mı?
        active_signals = get_active_signals()
        for s in active_signals:
            if s["symbol"] == signal["symbol"] and s["status"] == "ACTIVE":
                logger.info(f"{signal['symbol']} için zaten aktif işlem var, atlanıyor.")
                return {"status": "REJECTED", "reason": "Aktif işlem mevcut"}

        # Son 15 dakikada aynı coinde işlem yapılmış mı? (cooldown)
        from datetime import datetime, timedelta
        recent_history = get_signal_history(30)
        cooldown_minutes = int(self._param("signal_cooldown_minutes"))
        now = datetime.now()
        for s in recent_history:
            if s["symbol"] == signal["symbol"]:
                created = s.get("created_at", "")
                if created:
                    try:
                        created_dt = datetime.fromisoformat(created)
                        if (now - created_dt).total_seconds() < cooldown_minutes * 60:
                            logger.info(f"{signal['symbol']} için {cooldown_minutes}dk cooldown aktif, atlanıyor.")
                            return {"status": "REJECTED", "reason": f"{cooldown_minutes}dk cooldown"}
                    except Exception:
                        pass

        # İşleme al
        signal_id = add_signal(
            symbol=signal["symbol"],
            direction=signal["direction"],
            entry_price=signal["entry"],
            stop_loss=signal["sl"],
            take_profit=signal["tp"],
            confidence=signal["confidence"],
            confluence_score=signal["confluence_score"],
            components=signal["components"],
            timeframe="15m",
            status="ACTIVE",
            notes=f"RR: {signal['rr_ratio']} | Bileşenler: {', '.join(signal['components'])}"
        )

        activate_signal(signal_id)

        logger.info(f"✅ İŞLEM AÇILDI: #{signal_id} {signal['symbol']} {signal['direction']} | "
                    f"Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']}")

        return {
            "status": "OPENED",
            "signal_id": signal_id,
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "entry": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"]
        }

    def _add_to_watch(self, signal):
        """İzleme listesine ekle"""
        watch_candles = int(self._param("patience_watch_candles"))
        watch_id = add_to_watchlist(
            symbol=signal["symbol"],
            direction=signal["direction"],
            potential_entry=signal["entry"],
            potential_sl=signal["sl"],
            potential_tp=signal["tp"],
            watch_reason=signal.get("watch_reason", "Onay bekleniyor"),
            initial_score=signal["confluence_score"],
            components=signal["components"],
            max_watch=watch_candles
        )

        logger.info(f"👁️ İZLEMEYE ALINDI: {signal['symbol']} {signal['direction']} | "
                    f"Score: {signal['confluence_score']}%")

        return {
            "status": "WATCHING",
            "watch_id": watch_id,
            "symbol": signal["symbol"]
        }

    def check_open_trades(self):
        """
        Açık işlemlerin SL/TP durumunu kontrol et.
        İYİLEŞTİRMELER:
        - Breakeven: Fiyat TP'nin %50'sine ulaşınca SL'yi entry'ye taşı
        - Trailing SL: Fiyat TP'nin %75'ine ulaşınca SL'yi kârın %50'sinde tut
        """
        active_signals = get_active_signals()
        results = []

        for signal in active_signals:
            if signal["status"] != "ACTIVE":
                continue

            symbol = signal["symbol"]
            ticker = data_fetcher.get_ticker(symbol)

            if not ticker:
                continue

            current_price = ticker["last"]
            entry_price = signal["entry_price"]
            stop_loss = signal["stop_loss"]
            take_profit = signal["take_profit"]
            direction = signal["direction"]
            signal_id = signal["id"]

            result = {
                "signal_id": signal_id,
                "symbol": symbol,
                "direction": direction,
                "current_price": current_price,
                "entry_price": entry_price,
                "status": "ACTIVE"
            }

            # Seviye doğrulama (ters SL/TP eski sinyalleri temizle)
            if direction == "LONG" and (stop_loss >= entry_price or take_profit <= entry_price):
                logger.warning(f"⚠️ #{signal_id} {symbol} LONG ters seviyeler - iptal ediliyor")
                update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
                self._trade_state.pop(signal_id, None)
                result["status"] = "CANCELLED"
                results.append(result)
                continue
            elif direction == "SHORT" and (stop_loss <= entry_price or take_profit >= entry_price):
                logger.warning(f"⚠️ #{signal_id} {symbol} SHORT ters seviyeler - iptal ediliyor")
                update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
                self._trade_state.pop(signal_id, None)
                result["status"] = "CANCELLED"
                results.append(result)
                continue

            # ===== BREAKEVEN / TRAILING SL HESAPLA =====
            state = self._trade_state.get(signal_id, {"breakeven_moved": False, "trailing_sl": None})
            effective_sl = stop_loss

            if direction == "LONG":
                total_distance = take_profit - entry_price
                current_progress = current_price - entry_price

                if total_distance > 0 and current_progress > 0:
                    progress_pct = current_progress / total_distance

                    # %75'e ulaştıysa → Trailing SL (kârın %50'sini koru)
                    if progress_pct >= 0.75:
                        trailing = entry_price + (current_progress * 0.50)
                        if state["trailing_sl"] is None or trailing > state["trailing_sl"]:
                            state["trailing_sl"] = trailing
                            effective_sl = max(effective_sl, trailing)
                            if not state.get("trailing_logged"):
                                logger.info(f"📈 #{signal_id} {symbol} TRAILING SL: {trailing:.6f} (kâr koruma)")
                                state["trailing_logged"] = True

                    # %50'ye ulaştıysa → Breakeven (SL'yi entry'ye taşı)
                    elif progress_pct >= 0.50 and not state["breakeven_moved"]:
                        state["breakeven_moved"] = True
                        effective_sl = entry_price * 1.001  # Küçük buffer
                        logger.info(f"🔒 #{signal_id} {symbol} BREAKEVEN: SL → {effective_sl:.6f}")

                    if state["trailing_sl"]:
                        effective_sl = max(effective_sl, state["trailing_sl"])

                # TP kontrolü
                if current_price >= take_profit:
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    update_signal_status(signal_id, "WON", close_price=current_price, pnl_pct=pnl_pct)
                    self._trade_state.pop(signal_id, None)
                    result["status"] = "WON"
                    result["pnl_pct"] = round(pnl_pct, 2)
                    logger.info(f"🏆 KAZANDIK: #{signal_id} {symbol} LONG | PnL: +{pnl_pct:.2f}%")

                # SL kontrolü (effective_sl kullan)
                elif current_price <= effective_sl:
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    sl_type = "TRAILING_SL" if state.get("trailing_sl") else ("BREAKEVEN" if state["breakeven_moved"] else "SL")
                    status = "WON" if pnl_pct > 0 else "LOST"
                    update_signal_status(signal_id, status, close_price=current_price, pnl_pct=pnl_pct)
                    self._trade_state.pop(signal_id, None)
                    result["status"] = status
                    result["pnl_pct"] = round(pnl_pct, 2)
                    emoji = "🏆" if pnl_pct > 0 else "❌"
                    logger.info(f"{emoji} {sl_type}: #{signal_id} {symbol} LONG | PnL: {pnl_pct:+.2f}%")

                else:
                    unrealized_pnl = ((current_price - entry_price) / entry_price) * 100
                    result["unrealized_pnl"] = round(unrealized_pnl, 2)
                    if state["breakeven_moved"] or state.get("trailing_sl"):
                        result["effective_sl"] = round(effective_sl, 8)

            elif direction == "SHORT":
                total_distance = entry_price - take_profit
                current_progress = entry_price - current_price

                if total_distance > 0 and current_progress > 0:
                    progress_pct = current_progress / total_distance

                    # %75'e ulaştıysa → Trailing SL
                    if progress_pct >= 0.75:
                        trailing = entry_price - (current_progress * 0.50)
                        if state["trailing_sl"] is None or trailing < state["trailing_sl"]:
                            state["trailing_sl"] = trailing
                            effective_sl = min(effective_sl, trailing)
                            if not state.get("trailing_logged"):
                                logger.info(f"📉 #{signal_id} {symbol} TRAILING SL: {trailing:.6f} (kâr koruma)")
                                state["trailing_logged"] = True

                    # %50'ye ulaştıysa → Breakeven
                    elif progress_pct >= 0.50 and not state["breakeven_moved"]:
                        state["breakeven_moved"] = True
                        effective_sl = entry_price * 0.999
                        logger.info(f"🔒 #{signal_id} {symbol} BREAKEVEN: SL → {effective_sl:.6f}")

                    if state["trailing_sl"]:
                        effective_sl = min(effective_sl, state["trailing_sl"])

                # TP kontrolü
                if current_price <= take_profit:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    update_signal_status(signal_id, "WON", close_price=current_price, pnl_pct=pnl_pct)
                    self._trade_state.pop(signal_id, None)
                    result["status"] = "WON"
                    result["pnl_pct"] = round(pnl_pct, 2)
                    logger.info(f"🏆 KAZANDIK: #{signal_id} {symbol} SHORT | PnL: +{pnl_pct:.2f}%")

                # SL kontrolü (effective_sl kullan)
                elif current_price >= effective_sl:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    sl_type = "TRAILING_SL" if state.get("trailing_sl") else ("BREAKEVEN" if state["breakeven_moved"] else "SL")
                    status = "WON" if pnl_pct > 0 else "LOST"
                    update_signal_status(signal_id, status, close_price=current_price, pnl_pct=pnl_pct)
                    self._trade_state.pop(signal_id, None)
                    result["status"] = status
                    result["pnl_pct"] = round(pnl_pct, 2)
                    emoji = "🏆" if pnl_pct > 0 else "❌"
                    logger.info(f"{emoji} {sl_type}: #{signal_id} {symbol} SHORT | PnL: {pnl_pct:+.2f}%")

                else:
                    unrealized_pnl = ((entry_price - current_price) / entry_price) * 100
                    result["unrealized_pnl"] = round(unrealized_pnl, 2)
                    if state["breakeven_moved"] or state.get("trailing_sl"):
                        result["effective_sl"] = round(effective_sl, 8)

            # State'i kaydet
            self._trade_state[signal_id] = state
            results.append(result)

        return results

    def check_watchlist(self, strategy_engine):
        """
        İzleme listesindeki coinleri kontrol et
        - Mum sayısı yeterliyse ve skor yükseldiyse sinyal üret
        - Skor düştüyse veya süre dolduysa listeden çıkar
        """
        watching_items = get_watching_items()
        promoted = []

        for item in watching_items:
            symbol = item["symbol"]
            candles_watched = item["candles_watched"] + 1
            max_watch = item["max_watch_candles"]

            # Yeni veri çek ve tekrar analiz et (ilk tarama ile aynı şekilde multi-timeframe)
            multi_tf = data_fetcher.get_multi_timeframe_data(symbol)
            df = multi_tf.get("15m") if multi_tf else None
            if df is None or df.empty:
                continue

            analysis = strategy_engine.calculate_confluence(df, multi_tf)
            new_score = analysis["confluence_score"]
            min_confluence = strategy_engine.params["min_confluence_score"]
            min_confidence = strategy_engine.params["min_confidence"]

            confidence = strategy_engine._calculate_confidence(analysis)

            if new_score >= min_confluence and confidence >= min_confidence:
                # Sinyal olgunlaştı mı gerçekten kontrol et (multi-timeframe ile)
                signal_result = strategy_engine.generate_signal(symbol, df, multi_tf)
                if signal_result and signal_result["action"] == "SIGNAL":
                    promote_watchlist_item(item["id"])
                    trade_result = self._open_trade(signal_result)
                    promoted.append({
                        "symbol": symbol,
                        "action": "PROMOTED",
                        "trade_result": trade_result
                    })
                    logger.info(f"⬆️ İZLEMEDEN SİNYALE: {symbol} | "
                              f"Score: {item['initial_score']} -> {new_score}")
                    continue

            if candles_watched >= max_watch:
                # Süre doldu ve yeterli skor yok - expire et
                expire_watchlist_item(item["id"])
                logger.info(f"⏰ İZLEME SÜRESİ DOLDU: {symbol} | Son Score: {new_score}")
                continue

            # Score düştüyse expire et
            if new_score < item["initial_score"] * 0.5:
                expire_watchlist_item(item["id"])
                logger.info(f"📉 İZLEME SKOR DÜŞTÜ: {symbol} | {item['initial_score']} -> {new_score}")
                continue

            # Güncelle ve beklemeye devam
            update_watchlist_item(item["id"], candles_watched, new_score)

        return promoted


# Global instance
trade_manager = TradeManager()
