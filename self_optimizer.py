# =====================================================
# ICT Trading Bot - Otomatik Optimizasyon Modülü
# =====================================================
# Kazanma/kaybetme havuzunu analiz ederek parametreleri
# otomatik günceller ve sürekli öğrenir.
# =====================================================

import logging
import json
from datetime import datetime
from database import (
    get_completed_signals, get_performance_summary,
    get_component_performance, save_bot_param, get_bot_param,
    add_optimization_log, get_all_bot_params, get_loss_analysis
)
from config import ICT_PARAMS, OPTIMIZER_CONFIG

logger = logging.getLogger("ICT-Bot.Optimizer")


class SelfOptimizer:
    """
    Otomatik öğrenen optimizer.
    Her optimizasyon döngüsünde:
    1. Tamamlanan işlemleri analiz et
    2. Bileşen bazlı performansı değerlendir
    3. Zayıf parametreleri tespit et
    4. Kontrollü şekilde güncelle
    5. Log tut
    """

    def __init__(self):
        self.learning_rate = OPTIMIZER_CONFIG["learning_rate"]
        self.max_change = OPTIMIZER_CONFIG["max_param_change_pct"]
        self.min_trades = OPTIMIZER_CONFIG["min_trades_for_optimization"]
        self.target_win_rate = OPTIMIZER_CONFIG["win_rate_target"]
        self.optimization_history = []

    def run_optimization(self):
        """Ana optimizasyon döngüsü"""
        logger.info("🔄 Optimizasyon döngüsü başlatılıyor...")

        stats = get_performance_summary()
        total_trades = stats["total_trades"]

        if total_trades < self.min_trades:
            logger.info(f"Yeterli işlem yok ({total_trades}/{self.min_trades}), optimizasyon atlanıyor.")
            return {
                "status": "SKIPPED",
                "reason": f"Minimum {self.min_trades} işlem gerekli, şu an: {total_trades}",
                "changes": []
            }

        changes = []

        # 1. Win Rate bazlı güven eşiği ayarlama
        wr_change = self._optimize_confidence_threshold(stats)
        if wr_change:
            changes.append(wr_change)

        # 2. Bileşen bazlı ağırlık ayarlama
        comp_changes = self._optimize_component_weights(stats)
        changes.extend(comp_changes)

        # 3. Risk yönetimi parametreleri
        risk_changes = self._optimize_risk_params(stats)
        changes.extend(risk_changes)

        # 4. Sabırlı mod ayarları
        patience_change = self._optimize_patience(stats)
        if patience_change:
            changes.append(patience_change)

        # 5. Kayıp analizi → derin öğrenme (neden kaybettik?)
        loss_changes = self._learn_from_losses()
        changes.extend(loss_changes)

        if changes:
            logger.info(f"✅ Optimizasyon tamamlandı: {len(changes)} parametre güncellendi")
        else:
            logger.info("ℹ️ Optimizasyon: Değişiklik gerekli değil")

        return {
            "status": "COMPLETED",
            "total_trades_analyzed": total_trades,
            "win_rate": stats["win_rate"],
            "changes": changes
        }

    def _optimize_confidence_threshold(self, stats):
        """
        Win rate'e göre minimum güven eşiğini ayarla:
        - Win rate düşükse -> eşiği yükselt (daha seçici ol)
        - Win rate hedefin üstündeyse -> eşiği biraz düşür (daha fazla fırsat)
        """
        current_wr = stats["win_rate"] / 100
        current_threshold = get_bot_param("min_confidence", ICT_PARAMS["min_confidence"])

        if current_wr < self.target_win_rate * 0.85:
            # Win rate çok düşük, eşiği yükselt
            adjustment = self.learning_rate * (self.target_win_rate - current_wr) * 100
            new_threshold = min(90, current_threshold + adjustment)
            reason = f"Win rate düşük ({stats['win_rate']}%), güven eşiği yükseltiliyor"

        elif current_wr > self.target_win_rate * 1.15:
            # Win rate çok yüksek, biraz daha fazla sinyal üretilebilir
            adjustment = self.learning_rate * 5
            new_threshold = max(55, current_threshold - adjustment)
            reason = f"Win rate yüksek ({stats['win_rate']}%), güven eşiği düşürülüyor"

        else:
            return None

        new_threshold = round(new_threshold, 1)
        if abs(new_threshold - current_threshold) < 0.5:
            return None

        # Uygula ve logla
        save_bot_param("min_confidence", new_threshold, ICT_PARAMS["min_confidence"])
        add_optimization_log(
            "min_confidence", current_threshold, new_threshold, reason,
            stats["win_rate"], stats["win_rate"], stats["total_trades"]
        )

        logger.info(f"📊 Güven eşiği: {current_threshold} -> {new_threshold} ({reason})")

        return {
            "param": "min_confidence",
            "old": current_threshold,
            "new": new_threshold,
            "reason": reason
        }

    def _optimize_component_weights(self, stats):
        """
        Bileşen bazlı performans analizi:
        - Başarılı bileşenlerin ağırlığını artır (dolaylı olarak)
        - Başarısız bileşenlerin etkisini azalt
        """
        changes = []
        comp_perf = stats.get("component_performance", {})

        if not comp_perf:
            return changes

        # Her bileşen için confluence score eşiklerini ayarla
        param_mapping = {
            "ORDER_BLOCK": "ob_body_ratio_min",
            "FVG": "fvg_min_size_pct",
            "LIQUIDITY_SWEEP": "liquidity_equal_tolerance",
            "DISPLACEMENT": "displacement_min_body_ratio"
        }

        for comp_name, param_name in param_mapping.items():
            if comp_name not in comp_perf:
                continue

            comp = comp_perf[comp_name]
            if comp["total"] < 5:  # Yeterli veri yok
                continue

            win_rate = comp["win_rate"] / 100
            current_val = get_bot_param(param_name, ICT_PARAMS[param_name])
            new_val = current_val

            if win_rate < 0.4:
                # Bu bileşen kötü performans gösteriyor - daha seçici ol
                adjustment = current_val * self.learning_rate
                new_val = current_val + adjustment
                reason = f"{comp_name} düşük WR ({comp['win_rate']}%), daha seçici"

            elif win_rate > 0.75:
                # Çok iyi performans - biraz gevşet
                adjustment = current_val * self.learning_rate * 0.5
                new_val = max(current_val * 0.5, current_val - adjustment)
                reason = f"{comp_name} yüksek WR ({comp['win_rate']}%), biraz gevşetiliyor"

            else:
                continue

            # Max değişim sınırı
            max_change_abs = current_val * self.max_change
            if abs(new_val - current_val) > max_change_abs:
                new_val = current_val + (max_change_abs if new_val > current_val else -max_change_abs)

            new_val = round(new_val, 6)
            if abs(new_val - current_val) < current_val * 0.01:
                continue

            save_bot_param(param_name, new_val, ICT_PARAMS[param_name])
            add_optimization_log(
                param_name, current_val, new_val, reason,
                stats["win_rate"], stats["win_rate"], stats["total_trades"]
            )

            changes.append({
                "param": param_name,
                "old": current_val,
                "new": new_val,
                "reason": reason
            })

            logger.info(f"📊 {param_name}: {current_val} -> {new_val} ({reason})")

        return changes

    def _optimize_risk_params(self, stats):
        """
        Risk yönetimi parametrelerini optimize et:
        - Ortalama RR oranına göre TP/SL ayarla
        - Kaybeden işlemlerdeki SL mesafesini analiz et
        """
        changes = []

        completed = get_completed_signals(50)
        if len(completed) < self.min_trades:
            return changes

        # Kazanan/kaybeden analizi
        winners = [s for s in completed if s["status"] == "WON"]
        losers = [s for s in completed if s["status"] == "LOST"]

        if not losers:
            return changes

        # Ortalama kayıp yüzdesi
        avg_loss = sum(abs(s["pnl_pct"]) for s in losers) / len(losers) if losers else 0
        avg_win = sum(abs(s["pnl_pct"]) for s in winners) / len(winners) if winners else 0

        current_sl = get_bot_param("default_sl_pct", ICT_PARAMS["default_sl_pct"])
        current_tp_ratio = get_bot_param("default_tp_ratio", ICT_PARAMS["default_tp_ratio"])

        # SL çok sık tetikleniyorsa (kayıp oranı yüksek) ve ortalama kayıp artıyorsa
        loss_rate = len(losers) / len(completed) if completed else 0

        if loss_rate > 0.55 and avg_loss > current_sl * 100 * 0.9:
            # SL'yi biraz genişlet (market noise'ı azalt)
            new_sl = min(current_sl * 1.1, 0.03)  # Max %3
            new_sl = round(new_sl, 4)

            if abs(new_sl - current_sl) > 0.0005:
                save_bot_param("default_sl_pct", new_sl, ICT_PARAMS["default_sl_pct"])
                reason = f"Kayıp oranı yüksek ({loss_rate:.0%}), SL genişletiliyor"
                add_optimization_log(
                    "default_sl_pct", current_sl, new_sl, reason,
                    stats["win_rate"], stats["win_rate"], stats["total_trades"]
                )
                changes.append({
                    "param": "default_sl_pct",
                    "old": current_sl,
                    "new": new_sl,
                    "reason": reason
                })

        # RR oranı ayarlama
        if avg_win > 0 and avg_loss > 0:
            actual_rr = avg_win / avg_loss
            if actual_rr < 1.5 and current_tp_ratio < 3.5:
                new_tp_ratio = min(current_tp_ratio + 0.2, 4.0)
                new_tp_ratio = round(new_tp_ratio, 1)

                save_bot_param("default_tp_ratio", new_tp_ratio, ICT_PARAMS["default_tp_ratio"])
                reason = f"Gerçek RR düşük ({actual_rr:.1f}), TP oranı artırılıyor"
                add_optimization_log(
                    "default_tp_ratio", current_tp_ratio, new_tp_ratio, reason,
                    stats["win_rate"], stats["win_rate"], stats["total_trades"]
                )
                changes.append({
                    "param": "default_tp_ratio",
                    "old": current_tp_ratio,
                    "new": new_tp_ratio,
                    "reason": reason
                })

        return changes

    def _optimize_patience(self, stats):
        """
        Sabırlı mod optimizasyonu:
        - Watch'tan promote edilen sinyallerin başarısını analiz et
        - Bekleme süresini ayarla
        """
        current_watch = get_bot_param("patience_watch_candles", ICT_PARAMS["patience_watch_candles"])
        completed = get_completed_signals(50)

        if len(completed) < self.min_trades:
            return None

        # Düşük güvenle açılan işlemlerin sonuçlarını analiz et
        low_conf_trades = [s for s in completed if s["confidence"] and s["confidence"] < 70]
        high_conf_trades = [s for s in completed if s["confidence"] and s["confidence"] >= 70]

        if not low_conf_trades or not high_conf_trades:
            return None

        low_wr = sum(1 for s in low_conf_trades if s["status"] == "WON") / len(low_conf_trades)
        high_wr = sum(1 for s in high_conf_trades if s["status"] == "WON") / len(high_conf_trades)

        # Düşük güvenli işlemler çok başarısızsa, daha sabırlı ol
        if low_wr < 0.35 and current_watch < 5:
            new_watch = min(current_watch + 1, 5)
            reason = f"Düşük güvenli WR: {low_wr:.0%}, bekleme artırılıyor"

            save_bot_param("patience_watch_candles", new_watch, ICT_PARAMS["patience_watch_candles"])
            add_optimization_log(
                "patience_watch_candles", current_watch, new_watch, reason,
                stats["win_rate"], stats["win_rate"], stats["total_trades"]
            )

            return {
                "param": "patience_watch_candles",
                "old": current_watch,
                "new": new_watch,
                "reason": reason
            }

        return None

    def get_optimization_summary(self):
        """Optimizasyon özetini döndür"""
        stats = get_performance_summary()
        all_params = get_all_bot_params()
        loss_info = get_loss_analysis(30)

        # Varsayılandan değişen parametreleri bul
        changed_params = {}
        for key, default_val in ICT_PARAMS.items():
            current_val = all_params.get(key, default_val)
            if isinstance(current_val, (int, float)) and isinstance(default_val, (int, float)):
                if abs(current_val - default_val) > 0.0001:
                    changed_params[key] = {
                        "default": default_val,
                        "current": current_val,
                        "change_pct": round(((current_val - default_val) / default_val) * 100, 1)
                                      if default_val != 0 else 0
                    }

        return {
            "total_optimizations": len(changed_params),
            "current_win_rate": stats["win_rate"],
            "target_win_rate": self.target_win_rate * 100,
            "changed_params": changed_params,
            "performance": stats,
            "loss_lessons": loss_info.get("lesson_summary", []),
            "last_check": datetime.now().isoformat()
        }


    def _learn_from_losses(self):
        """
        Kayıp analizi yaparak otomatik ders çıkar.
        Neden kaybettik? Hangi bileşen eksikti? Hangi bileşen yanılttı?
        """
        changes = []
        loss_info = get_loss_analysis(30)

        if loss_info["total_losses"] < 5:
            return changes

        stats = get_performance_summary()

        # 1. Düşük güvenle girilen kayıplar çoğunluksa → min_confidence artır
        if loss_info["total_losses"] > 0:
            low_conf_ratio = loss_info["low_confidence_losses"] / loss_info["total_losses"]
            if low_conf_ratio > 0.4:
                current = get_bot_param("min_confidence", ICT_PARAMS["min_confidence"])
                # Küçük adımlarla artır (agresif değil, ideal)
                new_val = min(85, current + self.learning_rate * 15)
                new_val = round(new_val, 1)
                if new_val - current >= 1.0:
                    save_bot_param("min_confidence", new_val, ICT_PARAMS["min_confidence"])
                    reason = (f"Kayıpların %{low_conf_ratio*100:.0f}'i düşük güvenli — "
                             f"eşik {current} → {new_val}")
                    add_optimization_log("min_confidence", current, new_val, reason,
                                        stats["win_rate"], stats["win_rate"], stats["total_trades"])
                    changes.append({"param": "min_confidence", "old": current,
                                   "new": new_val, "reason": reason})
                    logger.info(f"🧠 DERS: {reason}")

        # 2. En çok eksik olan bileşeni kontrol et → confluence eşiğini ayarla
        missing = loss_info.get("missing_components", {})
        total_losses = loss_info["total_losses"]

        # Displacement kayıplarda çok eksikse → displacement cezasını artır
        disp_missing = missing.get("DISPLACEMENT", 0)
        if total_losses > 0 and disp_missing / total_losses > 0.6:
            current = get_bot_param("displacement_min_body_ratio",
                                   ICT_PARAMS["displacement_min_body_ratio"])
            # Displacement parametresini sıkılaştırmak yerine, confluence eşiğini hafif artır
            current_conf = get_bot_param("min_confluence_score", ICT_PARAMS["min_confluence_score"])
            new_conf = min(80, current_conf + 1.0)
            if new_conf > current_conf:
                save_bot_param("min_confluence_score", new_conf, ICT_PARAMS["min_confluence_score"])
                reason = (f"Kayıpların %{disp_missing/total_losses*100:.0f}'inde DISPLACEMENT eksik — "
                         f"confluence {current_conf} → {new_conf}")
                add_optimization_log("min_confluence_score", current_conf, new_conf, reason,
                                    stats["win_rate"], stats["win_rate"], stats["total_trades"])
                changes.append({"param": "min_confluence_score", "old": current_conf,
                               "new": new_conf, "reason": reason})
                logger.info(f"🧠 DERS: {reason}")

        # 3. HTF onaysız kayıplar çoksa → HTF uyumsuzluk cezasını artır (dolaylı: eşik)
        htf_missing = missing.get("HTF_CONFIRMATION", 0)
        if total_losses > 0 and htf_missing / total_losses > 0.65:
            reason = (f"Kayıpların %{htf_missing/total_losses*100:.0f}'inde HTF onayı yoktu — "
                     f"HTF uyumu kritik")
            logger.info(f"🧠 NOT: {reason}")
            # Bu bilgiyi lesson olarak sakla, agresif parametre değişikliği yapma

        # 4. Ortalama kayıp büyükse → SL mesafesini kontrol et
        if loss_info["avg_loss_pct"] > 2.0:
            current_sl = get_bot_param("default_sl_pct", ICT_PARAMS["default_sl_pct"])
            # SL çok geniş olabilir, daralt
            new_sl = max(0.008, current_sl * 0.92)
            new_sl = round(new_sl, 4)
            if abs(new_sl - current_sl) > 0.001:
                save_bot_param("default_sl_pct", new_sl, ICT_PARAMS["default_sl_pct"])
                reason = (f"Ortalama kayıp %{loss_info['avg_loss_pct']:.1f} çok yüksek — "
                         f"SL {current_sl} → {new_sl}")
                add_optimization_log("default_sl_pct", current_sl, new_sl, reason,
                                    stats["win_rate"], stats["win_rate"], stats["total_trades"])
                changes.append({"param": "default_sl_pct", "old": current_sl,
                               "new": new_sl, "reason": reason})
                logger.info(f"🧠 DERS: {reason}")

        # Ders özetini logla
        for lesson in loss_info.get("lesson_summary", []):
            logger.info(f"📝 Optimizer Ders: {lesson}")

        return changes


# Global instance
self_optimizer = SelfOptimizer()
