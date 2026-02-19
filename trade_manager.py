# =====================================================
# ICT Trading Bot - Trade Yönetim Modülü v2.0
# (Smart Money Concepts - Limit Emir Protokolü)
# =====================================================
#
# DEĞİŞİKLİKLER (v2.0):
#   1. LIMIT EMİR KONSEPTİ:
#      Strateji motoru FVG entry bölgesinde sinyal üretir.
#      Fiyat FVG'de değilse → WAITING (Limit emir bekliyor)
#      Fiyat FVG'ye ulaştığında → ACTIVE (Emir gerçekleşti)
#
#   2. EMİR ZAMAN AŞIMI:
#      Limit emirler LIMIT_ORDER_EXPIRY_HOURS saat içinde
#      gerçekleşmezse otomatik iptal edilir.
#
#   3. YAPISAL SEVIYELER:
#      SL her zaman yapısal seviyede (% tabanlı yedek yok).
#      TP her zaman karşı likidite hedefinde.
#
#   4. BREAKEVEN / TRAILING SL:
#      TP'nin %50'sinde → SL entry'ye taşınır
#      TP'nin %75'inde → SL kârın %50'sini korur
# =====================================================

import logging
from datetime import datetime, timedelta
from data_fetcher import data_fetcher
from database import (
    get_active_signals, update_signal_status, activate_signal,
    get_active_trade_count, add_signal, add_to_watchlist,
    get_watching_items, update_watchlist_item, promote_watchlist_item,
    expire_watchlist_item, get_signal_history, get_bot_param
)
from config import (
    ICT_PARAMS,
    WATCH_CONFIRM_TIMEFRAME,
    WATCH_CONFIRM_CANDLES,
    WATCH_REQUIRED_CONFIRMATIONS
)

logger = logging.getLogger("ICT-Bot.TradeManager")

# Limit emir zaman aşımı (config'de yoksa varsayılan)
LIMIT_ORDER_EXPIRY_HOURS = 6


class TradeManager:
    """
    Açık işlemlerin yönetimi ve takibi.
    
    Akış:
      process_signal → _add_to_watch (zorunlu 5m onay)
      check_watchlist → _open_trade (onay gelirse)
      _open_trade → WAITING (limit) veya ACTIVE (market)
      check_open_trades → WAITING→ACTIVE + SL/TP takibi
    """

    def __init__(self):
        # Breakeven / Trailing SL takibi
        # {signal_id: {"breakeven_moved": bool, "trailing_sl": float}}
        self._trade_state = {}

    def _param(self, name):
        """Optimizer ile güncellenen parametreleri DB'den oku, yoksa config varsayılanı kullan."""
        return get_bot_param(name, ICT_PARAMS.get(name))

    def process_signal(self, signal_result):
        """
        Strateji motorundan gelen sonucu işle.
        
        A+ / A tier SIGNAL → doğrudan işlem aç (sweep + displacement geçmiş)
        B tier SIGNAL ve WATCH → izleme listesine al, 5m onay bekle
        """
        if signal_result is None:
            return None

        action = signal_result.get("action")
        quality_tier = signal_result.get("quality_tier", "?")

        if action == "SIGNAL" and quality_tier in ("A+", "A"):
            # A-tier sinyal: Tüm gate'ler geçmiş, doğrudan işlem aç
            logger.info(f"🎯 {signal_result['symbol']} Tier-{quality_tier} SIGNAL → doğrudan işlem")
            return self._open_trade(signal_result)
        elif action in ["SIGNAL", "WATCH"]:
            if action == "SIGNAL":
                signal_result = dict(signal_result)
                signal_result["watch_reason"] = f"Tier-{quality_tier} onay bekleniyor, 5m doğrulama"
            return self._add_to_watch(signal_result)

        return None

    def _open_trade(self, signal):
        """
        Yeni işlem aç.
        
        entry_mode:
          "MARKET" → Fiyat şu an FVG bölgesinde → hemen ACTIVE
          "LIMIT"  → Fiyat FVG bölgesinin dışında → WAITING (limit emir bekliyor)
          "PENDING"→ Potansiyel sinyal → MARKET gibi işle
        """
        # Max eşzamanlı işlem kontrolü
        max_concurrent = int(self._param("max_concurrent_trades"))
        active_count = get_active_trade_count()
        if active_count >= max_concurrent:
            logger.warning(f"⛔ Max eşzamanlı işlem limitine ulaşıldı ({max_concurrent})")
            return {"status": "REJECTED", "reason": "Maksimum işlem limiti"}

        # Aynı coinde aktif işlem var mı?
        active_signals = get_active_signals()
        for s in active_signals:
            if s["symbol"] == signal["symbol"] and s["status"] in ("ACTIVE", "WAITING"):
                logger.info(f"⏭️ {signal['symbol']} için zaten aktif/bekleyen işlem var, atlanıyor.")
                return {"status": "REJECTED", "reason": "Aktif/bekleyen işlem mevcut"}

        # Cooldown kontrolü: Sadece KAPANMIŞ işlemler (WON/LOST/CANCELLED) için
        # Watchlist expire ve bekleyen sinyaller cooldown'a dahil DEĞİL
        recent_history = get_signal_history(30)
        cooldown_minutes = int(self._param("signal_cooldown_minutes"))
        now = datetime.now()
        for s in recent_history:
            if s["symbol"] == signal["symbol"]:
                # Sadece gerçekten kapanmış işlemler cooldown oluşturur
                if s.get("status") not in ("WON", "LOST", "CANCELLED"):
                    continue
                close_time = s.get("close_time") or s.get("created_at", "")
                if close_time:
                    try:
                        close_dt = datetime.fromisoformat(close_time)
                        if (now - close_dt).total_seconds() < cooldown_minutes * 60:
                            logger.info(f"⏳ {signal['symbol']} için {cooldown_minutes}dk cooldown aktif ({s['status']}).")
                            return {"status": "REJECTED", "reason": f"{cooldown_minutes}dk cooldown"}
                    except Exception:
                        pass

        # Entry modu belirleme
        entry_mode = signal.get("entry_mode", "MARKET")
        if entry_mode == "PENDING":
            entry_mode = "MARKET"

        # LIMIT ise status=WAITING, MARKET ise status=ACTIVE
        initial_status = "WAITING" if entry_mode == "LIMIT" else "ACTIVE"

        # Giriş sebeplerini kaydet (optimizer öğrensin)
        quality_tier = signal.get("quality_tier", "?")

        # B-tier risk yönetimi: Pozisyon büyüklüğü tavsiyesi
        position_note = ""
        if quality_tier == "B":
            position_note = " | ⚠️ B-TIER: %50 pozisyon önerilir (sweep yok)"
        elif quality_tier == "A":
            position_note = " | A-TIER: %75 pozisyon (MSS yok)"
        # A+ = tam pozisyon (varsayılan)

        components = signal.get("components", [])
        entry_reasons = (
            f"Tier: {quality_tier} | "
            f"Mode: {entry_mode} | "
            f"RR: {signal.get('rr_ratio', '?')} | "
            f"Score: {signal.get('confluence_score', 0)} | "
            f"Conf: {signal.get('confidence', 0)}% | "
            f"HTF: {signal.get('htf_bias', '?')} | "
            f"Session: {signal.get('session', '')} | "
            f"Entry: {signal.get('entry_type', '?')} | "
            f"SL: {signal.get('sl_type', '?')} | "
            f"TP: {signal.get('tp_type', '?')} | "
            f"Bileşenler: {', '.join(components)} | "
            f"Cezalar: {', '.join(signal.get('penalties', []))}"
            f"{position_note}"
        )

        signal_id = add_signal(
            symbol=signal["symbol"],
            direction=signal["direction"],
            entry_price=signal["entry"],
            stop_loss=signal["sl"],
            take_profit=signal["tp"],
            confidence=signal.get("confidence", 0),
            confluence_score=signal.get("confluence_score", 0),
            components=components,
            timeframe="15m",
            status=initial_status,
            notes=entry_reasons,
            entry_mode=entry_mode,
            htf_bias=signal.get("htf_bias"),
            rr_ratio=signal.get("rr_ratio")
        )

        if initial_status == "ACTIVE":
            activate_signal(signal_id)
            logger.info(
                f"✅ İŞLEM AÇILDI (MARKET): #{signal_id} {signal['symbol']} {signal['direction']} | "
                f"Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} | "
                f"RR: {signal.get('rr_ratio', '?')}"
            )
        else:
            logger.info(
                f"⏳ LİMİT EMİR KURULDU: #{signal_id} {signal['symbol']} {signal['direction']} | "
                f"FVG Entry: {signal['entry']} | SL: {signal['sl']} | TP: {signal['tp']} | "
                f"RR: {signal.get('rr_ratio', '?')} | "
                f"Beklenecek max: {LIMIT_ORDER_EXPIRY_HOURS} saat"
            )

        return {
            "status": "OPENED" if initial_status == "ACTIVE" else "LIMIT_PLACED",
            "signal_id": signal_id,
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "entry": signal["entry"],
            "sl": signal["sl"],
            "tp": signal["tp"],
            "entry_mode": entry_mode
        }

    def _add_to_watch(self, signal):
        """İzleme listesine ekle (5m onay akışı)."""
        # Çok düşük skorlu sinyalleri izlemeye bile alma (flip-flop engel)
        if signal.get("confluence_score", 0) < 25:
            logger.debug(f"⏭️ {signal['symbol']} skor çok düşük ({signal['confluence_score']}), izlemeye alınmadı")
            return None

        # ===== DUPLICATE KORUMASI =====
        # Aynı coinde aktif/bekleyen trade varsa izlemeye almayı engelle
        active_signals = get_active_signals()
        for s in active_signals:
            if s["symbol"] == signal["symbol"] and s["status"] in ("ACTIVE", "WAITING"):
                logger.debug(f"⏭️ {signal['symbol']} zaten aktif/bekleyen işlemde, izlemeye alınmadı")
                return None

        # Aynı coinde zaten izleme varsa (herhangi bir yönde) tekrar ekleme
        # Hem aynı yön hem ters yön koruması: Aynı coin için çift sinyal engeli
        watching_items = get_watching_items()
        for w in watching_items:
            if w["symbol"] == signal["symbol"]:
                if w["direction"] == signal["direction"]:
                    logger.debug(f"⏭️ {signal['symbol']} {signal['direction']} zaten izleme listesinde, atlanıyor")
                else:
                    logger.debug(f"⏭️ {signal['symbol']} ters yön ({w['direction']}) izlemede, {signal['direction']} atlanıyor")
                return None

        watch_candles = WATCH_CONFIRM_CANDLES
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

        if watch_id is None:
            logger.debug(f"⏳ {signal['symbol']} son 15 dk içinde expire edildi, cooldown bekleniyor")
            return None

        logger.info(
            f"👁️ İZLEMEYE ALINDI: {signal['symbol']} {signal['direction']} | "
            f"Score: {signal['confluence_score']}% | Mode: {signal.get('entry_mode', '?')}"
        )

        return {
            "status": "WATCHING",
            "watch_id": watch_id,
            "symbol": signal["symbol"],
            "direction": signal["direction"]
        }

    def check_open_trades(self):
        """
        Açık ve bekleyen işlemleri kontrol et.

        İki aşamalı kontrol:
        1. WAITING sinyaller → Fiyat FVG entry'ye ulaştı mı? → ACTIVE'e geç
           (Zaman aşımı: LIMIT_ORDER_EXPIRY_HOURS saat)
        2. ACTIVE sinyaller → SL/TP takibi + Breakeven/Trailing SL
        """
        active_signals = get_active_signals()
        results = []

        for signal in active_signals:
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
            status = signal["status"]

            # ===== WAITING (Limit Emir Bekliyor) =====
            if status == "WAITING":
                result = self._check_waiting_signal(
                    signal, current_price, entry_price, stop_loss,
                    direction, signal_id
                )
                if result:
                    results.append(result)
                continue

            # ===== ACTIVE (İşlem Açık — SL/TP Takibi) =====
            if status == "ACTIVE":
                result = self._check_active_signal(
                    signal, current_price, entry_price, stop_loss,
                    take_profit, direction, signal_id
                )
                if result:
                    results.append(result)

        return results

    def _check_waiting_signal(self, signal, current_price, entry_price,
                               stop_loss, direction, signal_id):
        """
        WAITING (limit emir) sinyalini kontrol et.
        
        Fiyat FVG entry seviyesine geldi mi?
        LONG: current_price <= entry_price (fiyat FVG'ye indi)
        SHORT: current_price >= entry_price (fiyat FVG'ye çıktı)
        
        Zaman aşımı kontrolü de burada yapılır.
        """
        symbol = signal["symbol"]

        # Zaman aşımı kontrolü
        created_at = signal.get("created_at", "")
        if created_at:
            try:
                created_dt = datetime.fromisoformat(created_at)
                elapsed_hours = (datetime.now() - created_dt).total_seconds() / 3600
                if elapsed_hours > LIMIT_ORDER_EXPIRY_HOURS:
                    update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
                    logger.info(
                        f"⏰ LİMİT EMİR ZAMAN AŞIMI: #{signal_id} {symbol} | "
                        f"{elapsed_hours:.1f} saat geçti (max {LIMIT_ORDER_EXPIRY_HOURS}h)"
                    )
                    return {
                        "signal_id": signal_id, "symbol": symbol,
                        "direction": direction, "status": "CANCELLED",
                        "reason": "Limit emir zaman aşımı"
                    }
            except Exception:
                pass

        # Fiyat SL'ye ulaştıysa → emir gerçekleşmeden iptal
        if direction == "LONG" and current_price <= stop_loss:
            update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
            logger.info(f"❌ LİMİT EMİR İPTAL: #{signal_id} {symbol} | Fiyat SL'ye ulaştı (entry'siz)")
            return {
                "signal_id": signal_id, "symbol": symbol,
                "direction": direction, "status": "CANCELLED",
                "reason": "Fiyat SL seviyesine ulaştı (limit emir gerçekleşmeden)"
            }
        elif direction == "SHORT" and current_price >= stop_loss:
            update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
            logger.info(f"❌ LİMİT EMİR İPTAL: #{signal_id} {symbol} | Fiyat SL'ye ulaştı (entry'siz)")
            return {
                "signal_id": signal_id, "symbol": symbol,
                "direction": direction, "status": "CANCELLED",
                "reason": "Fiyat SL seviyesine ulaştı (limit emir gerçekleşmeden)"
            }

        # Fiyat FVG entry seviyesine ulaştı mı?
        entry_buffer = entry_price * 0.001  # %0.1 buffer

        if direction == "LONG":
            # LONG: Fiyat FVG entry bölgesine veya altına indi
            if current_price <= entry_price + entry_buffer:
                activate_signal(signal_id)
                logger.info(
                    f"🎯 LİMİT EMİR GERÇEKLEŞTİ: #{signal_id} {symbol} LONG | "
                    f"Hedef: {entry_price:.8f} | Gerçekleşen: {current_price:.8f}"
                )
                return {
                    "signal_id": signal_id, "symbol": symbol,
                    "direction": direction, "status": "ACTIVATED",
                    "current_price": current_price
                }
        elif direction == "SHORT":
            # SHORT: Fiyat FVG entry bölgesine veya üstüne çıktı
            if current_price >= entry_price - entry_buffer:
                activate_signal(signal_id)
                logger.info(
                    f"🎯 LİMİT EMİR GERÇEKLEŞTİ: #{signal_id} {symbol} SHORT | "
                    f"Hedef: {entry_price:.8f} | Gerçekleşen: {current_price:.8f}"
                )
                return {
                    "signal_id": signal_id, "symbol": symbol,
                    "direction": direction, "status": "ACTIVATED",
                    "current_price": current_price
                }

        return None  # Hâlâ bekliyor

    def _check_active_signal(self, signal, current_price, entry_price,
                              stop_loss, take_profit, direction, signal_id):
        """
        ACTIVE sinyalin SL/TP takibini yap.
        
        Breakeven ve Trailing SL yönetimi:
        - TP'nin %50'sine ulaştıysa → SL'yi entry'ye taşı (breakeven)
        - TP'nin %75'ine ulaştıysa → SL'yi kârın %50'sinde tut (trailing)
        """
        symbol = signal["symbol"]
        result = {
            "signal_id": signal_id, "symbol": symbol,
            "direction": direction, "current_price": current_price,
            "entry_price": entry_price, "status": "ACTIVE"
        }

        # Seviye doğrulama (ters SL/TP eski sinyalleri temizle)
        if direction == "LONG" and (stop_loss >= entry_price or take_profit <= entry_price):
            logger.warning(f"⚠️ #{signal_id} {symbol} LONG ters seviyeler - iptal")
            update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
            self._trade_state.pop(signal_id, None)
            result["status"] = "CANCELLED"
            return result
        elif direction == "SHORT" and (stop_loss <= entry_price or take_profit >= entry_price):
            logger.warning(f"⚠️ #{signal_id} {symbol} SHORT ters seviyeler - iptal")
            update_signal_status(signal_id, "CANCELLED", close_price=current_price, pnl_pct=0)
            self._trade_state.pop(signal_id, None)
            result["status"] = "CANCELLED"
            return result

        # Breakeven / Trailing SL hesaplama
        state = self._trade_state.get(signal_id, {"breakeven_moved": False, "trailing_sl": None})
        effective_sl = stop_loss

        if direction == "LONG":
            effective_sl = self._manage_long_sl(
                signal_id, symbol, entry_price, current_price,
                stop_loss, take_profit, state, effective_sl
            )
            # TP kontrolü
            if current_price >= take_profit:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                update_signal_status(signal_id, "WON", close_price=current_price, pnl_pct=pnl_pct)
                self._trade_state.pop(signal_id, None)
                result["status"] = "WON"
                result["pnl_pct"] = round(pnl_pct, 2)
                logger.info(f"🏆 KAZANDIK: #{signal_id} {symbol} LONG | PnL: +{pnl_pct:.2f}%")
            # SL kontrolü
            elif current_price <= effective_sl:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                sl_type = self._get_sl_close_type(state)
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
            effective_sl = self._manage_short_sl(
                signal_id, symbol, entry_price, current_price,
                stop_loss, take_profit, state, effective_sl
            )
            # TP kontrolü
            if current_price <= take_profit:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                update_signal_status(signal_id, "WON", close_price=current_price, pnl_pct=pnl_pct)
                self._trade_state.pop(signal_id, None)
                result["status"] = "WON"
                result["pnl_pct"] = round(pnl_pct, 2)
                logger.info(f"🏆 KAZANDIK: #{signal_id} {symbol} SHORT | PnL: +{pnl_pct:.2f}%")
            # SL kontrolü
            elif current_price >= effective_sl:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                sl_type = self._get_sl_close_type(state)
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

        # State kaydet
        self._trade_state[signal_id] = state
        return result

    def _manage_long_sl(self, signal_id, symbol, entry_price, current_price,
                         stop_loss, take_profit, state, effective_sl):
        """LONG pozisyon için Breakeven ve Trailing SL yönetimi."""
        total_distance = take_profit - entry_price
        current_progress = current_price - entry_price

        if total_distance > 0 and current_progress > 0:
            progress_pct = current_progress / total_distance

            # %75 → Trailing SL (kârın %50'sini koru)
            if progress_pct >= 0.75:
                trailing = entry_price + (current_progress * 0.50)
                if state["trailing_sl"] is None or trailing > state["trailing_sl"]:
                    state["trailing_sl"] = trailing
                    effective_sl = max(effective_sl, trailing)
                    if not state.get("trailing_logged"):
                        logger.info(f"📈 #{signal_id} {symbol} TRAILING SL: {trailing:.6f}")
                        state["trailing_logged"] = True

            # %50 → Breakeven (SL'yi entry+buffer'a taşı)
            elif progress_pct >= 0.50 and not state["breakeven_moved"]:
                state["breakeven_moved"] = True
                effective_sl = entry_price * 1.001
                logger.info(f"🔒 #{signal_id} {symbol} BREAKEVEN: SL → {effective_sl:.6f}")

            if state["trailing_sl"]:
                effective_sl = max(effective_sl, state["trailing_sl"])

        return effective_sl

    def _manage_short_sl(self, signal_id, symbol, entry_price, current_price,
                          stop_loss, take_profit, state, effective_sl):
        """SHORT pozisyon için Breakeven ve Trailing SL yönetimi."""
        total_distance = entry_price - take_profit
        current_progress = entry_price - current_price

        if total_distance > 0 and current_progress > 0:
            progress_pct = current_progress / total_distance

            # %75 → Trailing SL
            if progress_pct >= 0.75:
                trailing = entry_price - (current_progress * 0.50)
                if state["trailing_sl"] is None or trailing < state["trailing_sl"]:
                    state["trailing_sl"] = trailing
                    effective_sl = min(effective_sl, trailing)
                    if not state.get("trailing_logged"):
                        logger.info(f"📉 #{signal_id} {symbol} TRAILING SL: {trailing:.6f}")
                        state["trailing_logged"] = True

            # %50 → Breakeven
            elif progress_pct >= 0.50 and not state["breakeven_moved"]:
                state["breakeven_moved"] = True
                effective_sl = entry_price * 0.999
                logger.info(f"🔒 #{signal_id} {symbol} BREAKEVEN: SL → {effective_sl:.6f}")

            if state["trailing_sl"]:
                effective_sl = min(effective_sl, state["trailing_sl"])

        return effective_sl

    def _get_sl_close_type(self, state):
        """SL kapanış tipini belirle."""
        if state.get("trailing_sl"):
            return "TRAILING_SL"
        elif state.get("breakeven_moved"):
            return "BREAKEVEN"
        return "STRUCTURAL_SL"

    def check_watchlist(self, strategy_engine):
        """
        İzleme listesindeki coinleri kontrol et.

        5m kapanan mumları tek tek izler:
        - 3 mum içinde yeterli onay toplanırsa işleme alır
        - Yetersiz onayda expire eder
        - Score çok düşerse erken expire eder
        """
        watching_items = get_watching_items()
        promoted = []
        min_confluence = strategy_engine.params["min_confluence_score"]

        for item in watching_items:
            symbol = item["symbol"]
            candles_watched = int(item.get("candles_watched", 0))
            confirmation_count = int(item.get("confirmation_count", 0))
            max_watch = item["max_watch_candles"]
            required_confirmations = min(max_watch, WATCH_REQUIRED_CONFIRMATIONS)
            expected_direction = item["direction"]

            # 5m verisi + HTF/MTF verisi çek
            watch_df = data_fetcher.get_candles(symbol, WATCH_CONFIRM_TIMEFRAME, 120)
            multi_tf = data_fetcher.get_multi_timeframe_data(symbol)
            if watch_df is None or watch_df.empty or multi_tf is None:
                continue

            # Sadece yeni kapanan 5m mum geldiğinde sayaç artır
            last_candle_ts = watch_df["timestamp"].iloc[-1].isoformat()
            if item.get("last_5m_candle_ts") == last_candle_ts:
                continue

            candles_watched += 1

            # === 5m Onay Analizi (basitleştirilmiş) ===
            # 5m veride tam ICT analizi yapmak yerine, yön uyumu ve fiyat
            # hareketi kontrol edilir. Çünkü ICT yapıları (FVG, OB vb.)
            # 15m'de tespit edilmiştir — 5m'de aynı yapıları aramak anlamsız.
            
            # 1) Yön kontrolü: 5m yapısal trend
            structure_5m = strategy_engine.detect_market_structure(watch_df)
            trend_5m = structure_5m.get("trend", "NEUTRAL")
            
            if expected_direction == "LONG":
                direction_ok = trend_5m in ["BULLISH", "WEAKENING_BEAR", "NEUTRAL"]
            else:
                direction_ok = trend_5m in ["BEARISH", "WEAKENING_BULL", "NEUTRAL"]
            
            # 2) Ranging kontrolü
            market_ok = not strategy_engine.detect_ranging_market(watch_df)
            
            # 3) Fiyat hareketi kontrolü: Son mum yöne uygun mu?
            last_candle = watch_df.iloc[-1]
            if expected_direction == "LONG":
                price_ok = last_candle["close"] >= last_candle["open"]  # Yeşil mum
            else:
                price_ok = last_candle["close"] <= last_candle["open"]  # Kırmızı mum
            
            # 4) Potansiyel entry hâlâ geçerli mi?
            potential_entry = item.get("potential_entry", 0)
            potential_sl = item.get("potential_sl", 0)
            current_5m_price = last_candle["close"]
            
            if expected_direction == "LONG":
                level_ok = current_5m_price > potential_sl if potential_sl > 0 else True
            else:
                level_ok = current_5m_price < potential_sl if potential_sl > 0 else True
            
            # Basit skor: Başlangıç skorunun yönsel korunması
            new_score = item["initial_score"]  # Değişmediğini varsay
            if not direction_ok:
                new_score *= 0.3
            if not market_ok:
                new_score *= 0.5
            
            # Onay kriterleri (basitleştirilmiş)
            candle_confirmed = all([
                direction_ok,
                market_ok or price_ok,  # İkisinden biri yeterli
                level_ok,
            ])
            if candle_confirmed:
                confirmation_count += 1

            logger.debug(
                f"  👁️ {symbol} mum #{candles_watched}: "
                f"yön={'✓' if direction_ok else '✗'}({trend_5m}) "
                f"market={'✓' if market_ok else '✗'} "
                f"price={'✓' if price_ok else '✗'} "
                f"level={'✓' if level_ok else '✗'} "
                f"onay={confirmation_count}/{candles_watched}"
            )

            # Mum onayı tamamlandıysa nihai karar ver
            if candles_watched >= max_watch:
                if confirmation_count >= required_confirmations:
                    # 15m verisiyle sinyal üret (5m değil!)
                    multi_15m_df = data_fetcher.get_candles(symbol, "15m", 120)
                    if multi_15m_df is not None and not multi_15m_df.empty:
                        signal_result = strategy_engine.generate_signal(symbol, multi_15m_df, multi_tf)
                    else:
                        signal_result = None
                    
                    if signal_result and signal_result.get("action") in ("SIGNAL", "WATCH"):
                        promote_watchlist_item(item["id"])
                        # WATCH bile olsa, 5m onaydan geçtiği için işleme al
                        if signal_result.get("action") == "WATCH":
                            signal_result = dict(signal_result)
                            signal_result["action"] = "SIGNAL"
                        trade_result = self._open_trade(signal_result)
                        promoted.append({
                            "symbol": symbol,
                            "action": "PROMOTED",
                            "trade_result": trade_result
                        })
                        logger.info(
                            f"⬆️ İZLEMEDEN SİNYALE: {symbol} | "
                            f"Onay: {confirmation_count}/{candles_watched} | "
                            f"Mode: {signal_result.get('entry_mode', '?')}"
                        )
                        continue
                
                expire_watchlist_item(
                    item["id"],
                    reason=f"Mum onay yetersiz ({confirmation_count}/{candles_watched})"
                )
                logger.info(
                    f"⏰ İZLEME BİTTİ: {symbol} | "
                    f"Onay: {confirmation_count}/{candles_watched} | Son Score: {new_score:.1f}"
                )
                continue

            # SL seviyesi ihlal edildiyse erken expire
            if not level_ok:
                expire_watchlist_item(
                    item["id"],
                    reason=f"SL seviyesi ihlal edildi ({symbol})"
                )
                logger.info(f"📉 İZLEME SL İHLAL: {symbol} | Yön tersine döndü")
                continue

            # Güncelle ve beklemeye devam
            update_watchlist_item(
                item["id"],
                candles_watched,
                new_score,
                confirmation_count=confirmation_count,
                last_5m_candle_ts=last_candle_ts,
                status="WATCHING"
            )

        return promoted


# Global instance
trade_manager = TradeManager()
