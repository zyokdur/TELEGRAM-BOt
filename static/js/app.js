// =====================================================
// ICT Trading Bot v2.0 - Frontend JavaScript
// =====================================================

const socket = io();
let botRunning = false;
let updateInterval = null;

// =================== WEBSOCKET ===================

socket.on("connect", () => {
    updateConnectionStatus(true);
    socket.emit("request_update");
});

socket.on("disconnect", () => {
    updateConnectionStatus(false);
});

socket.on("bot_status", (data) => {
    botRunning = data.running;
    updateBotButton();
});

socket.on("new_signal", (data) => {
    showToast(
        `ICT Sinyal: ${data.symbol} ${data.direction}`,
        data.status === "OPENED" ? "success" : "info"
    );
    loadActiveSignals();
    loadPerformance();
});

socket.on("trade_closed", (data) => {
    const isWon = data.status === "WON";
    showToast(
        `ICT ${data.symbol} ${isWon ? "KAZANDI" : "KAYBETTİ"} | PnL: ${data.pnl_pct}%`,
        isWon ? "success" : "error"
    );
    loadActiveSignals();
    loadHistory();
    loadPerformance();
});

socket.on("watch_promoted", (data) => {
    showToast(`${data.symbol} izlemeden sinyale yükseltildi!`, "info");
    loadActiveSignals();
    loadWatchlist();
});

socket.on("scan_complete", (data) => {
    document.getElementById("lastScanTime").innerHTML =
        `<i class="fas fa-satellite-dish"></i> Son: ${formatTime(data.timestamp)}`;
    document.getElementById("scanCount").innerHTML =
        `<i class="fas fa-hashtag"></i> ${data.symbols_scanned} coin`;
});

socket.on("regime_update", (data) => {
    updateRegimeTopbar(data);
});

socket.on("optimization_done", (data) => {
    if (data.changes && data.changes.length > 0) {
        showToast(
            `ICT Optimizasyon: ${data.changes.length} parametre güncellendi`,
            "warning"
        );
        loadOptimization();
    }
});

socket.on("full_update", (data) => {
    if (data.stats) updateStats(data.stats);
});

socket.on("trades_updated", () => {
    loadActiveSignals();
});

// =================== INIT ===================

document.addEventListener("DOMContentLoaded", () => {
    loadAllData();
    updateInterval = setInterval(() => {
        if (botRunning) {
            loadActiveSignals();
            loadPerformance();
        }
        updateServerTime();
    }, 10000);
    updateServerTime();
    setInterval(updateServerTime, 1000);

    // Restore sidebar state
    const collapsed = localStorage.getItem("sidebarCollapsed") === "true";
    if (collapsed) {
        document.getElementById("sidebar").classList.add("collapsed");
    }
});

function loadAllData() {
    loadActiveSignals();
    loadCoins();
    loadWatchlist();
    loadHistory();
    loadPerformance();
    loadOptimization();
    loadBotStatus();
    loadRegime();
}

// =================== SIDEBAR ===================

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const isMobile = window.innerWidth <= 768;

    if (isMobile) {
        sidebar.classList.toggle("mobile-open");
    } else {
        sidebar.classList.toggle("collapsed");
        localStorage.setItem("sidebarCollapsed", sidebar.classList.contains("collapsed"));
    }
}

// Close sidebar on mobile when clicking outside
document.addEventListener("click", (e) => {
    if (window.innerWidth <= 768) {
        const sidebar = document.getElementById("sidebar");
        const overlay = document.getElementById("sidebarOverlay");
        if (sidebar.classList.contains("mobile-open") && !sidebar.contains(e.target) && !e.target.closest(".mobile-menu-btn")) {
            sidebar.classList.remove("mobile-open");
        }
    }
});

// =================== API CALLS ===================

async function apiFetch(url) {
    try {
        const res = await fetch(url);
        return await res.json();
    } catch (e) {
        console.error(`API Hatası (${url}):`, e);
        return null;
    }
}

async function apiPost(url) {
    try {
        const res = await fetch(url, { method: "POST" });
        return await res.json();
    } catch (e) {
        console.error(`API POST Hatası (${url}):`, e);
        return null;
    }
}

// =================== BOT CONTROL ===================

async function toggleBot() {
    const btn = document.getElementById("btnToggleBot");
    btn.disabled = true;

    if (botRunning) {
        const result = await apiPost("/api/stop");
        if (result) {
            botRunning = false;
            showToast("Bot durduruldu", "info");
        }
    } else {
        const result = await apiPost("/api/start");
        if (result) {
            botRunning = true;
            showToast("Bot başlatıldı! Piyasa taranıyor...", "success");
        }
    }

    updateBotButton();
    btn.disabled = false;
}

async function loadBotStatus() {
    const data = await apiFetch("/api/status");
    if (data) {
        botRunning = data.running;
        updateBotButton();
        if (data.last_scan) {
            document.getElementById("lastScanTime").innerHTML =
                `<i class="fas fa-satellite-dish"></i> Son: ${formatTime(data.last_scan)}`;
        }
    }
}

function updateBotButton() {
    const btn = document.getElementById("btnToggleBot");
    if (botRunning) {
        btn.className = "bot-control-btn running";
        btn.innerHTML = '<span class="bot-btn-dot"></span><span class="bot-btn-text">Durdur</span>';
    } else {
        btn.className = "bot-control-btn stopped";
        btn.innerHTML = '<span class="bot-btn-dot"></span><span class="bot-btn-text">Başlat</span>';
    }
}

function updateConnectionStatus(connected) {
    const el = document.getElementById("connectionIndicator");
    if (connected) {
        el.innerHTML = '<span class="conn-dot connected"></span><span class="conn-text">Bağlı</span>';
    } else {
        el.innerHTML = '<span class="conn-dot disconnected"></span><span class="conn-text">Bağlantı kesildi</span>';
    }
}

// =================== TAB NAVIGATION ===================

const pageTitles = {
    signals: { icon: "fas fa-signal", text: "ICT Sinyalleri" },
    coins: { icon: "fas fa-coins", text: "Taranan Coinler" },
    watchlist: { icon: "fas fa-eye", text: "İzleme Listesi" },
    regime: { icon: "fas fa-globe", text: "Piyasa Rejimi" },
    history: { icon: "fas fa-clock-rotate-left", text: "Geçmiş İşlemler" },
    performance: { icon: "fas fa-chart-pie", text: "Performans" },
    optimizer: { icon: "fas fa-brain", text: "Optimizasyon" },
    backtest: { icon: "fas fa-flask", text: "Strateji Backtest" },
    forex: { icon: "fas fa-chart-line", text: "Forex & Altın ICT" },
    coindetail: { icon: "fas fa-magnifying-glass-chart", text: "Coin Detay Analizi" }
};

function switchTab(tabName) {
    // Nav items
    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    // Tab contents
    document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.toggle("active", content.id === `tab-${tabName}`);
    });

    // Update page title
    const titleInfo = pageTitles[tabName] || { icon: "fas fa-circle", text: tabName };
    document.getElementById("pageTitle").innerHTML = `<i class="${titleInfo.icon}"></i><span>${titleInfo.text}</span>`;

    // Coin detay sekmesinde stats-row gizle (daha fazla alan)
    const statsRow = document.getElementById("statsRow");
    if (statsRow) {
        statsRow.classList.toggle("hidden", tabName === "coindetail" || tabName === "backtest" || tabName === "forex");
    }

    // Auto-refresh: coindetail'den çıkınca durdur
    if (tabName !== "coindetail" && autoRefreshActive) {
        stopAutoRefresh();
    }

    // Close sidebar on mobile
    if (window.innerWidth <= 768) {
        document.getElementById("sidebar").classList.remove("mobile-open");
    }

    // Reload data
    switch (tabName) {
        case "signals": loadActiveSignals(); break;
        case "coins": loadCoins(); break;
        case "watchlist": loadWatchlist(); break;
        case "regime": loadRegime(); break;
        case "history": loadHistory(); break;
        case "optimizer": loadOptimization(); break;
        case "performance": loadPerformance(); break;
        case "backtest": loadBacktestCoins(); break;
        case "forex": loadForexKillZones(); break;
    }
}

// =================== ACTIVE SIGNALS ===================

async function loadActiveSignals() {
    const data = await apiFetch("/api/signals/active");
    if (!data) return;

    const tbody = document.getElementById("activeSignalsTable");
    const stats = await apiFetch("/api/performance");
    if (stats) updateStats(stats);

    // Update nav badge
    const badge = document.getElementById("navBadgeSignals");
    if (data.length > 0) {
        badge.textContent = data.length;
    } else {
        badge.textContent = "";
    }

    if (data.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row"><td colspan="11">
                <div class="empty-state">
                    <div class="empty-icon"><i class="fas fa-inbox"></i></div>
                    <p class="empty-title">Henüz aktif sinyal yok</p>
                    <p class="empty-desc">Bot başlatıldığında sinyaller burada görünecek</p>
                </div>
            </td></tr>`;
        return;
    }

    tbody.innerHTML = data.map(s => {
        const pnl = s.unrealized_pnl || 0;
        const pnlClass = pnl >= 0 ? "pnl-positive" : "pnl-negative";
        const pnlText = pnl >= 0 ? `+${pnl}%` : `${pnl}%`;
        const dirBadge = s.direction === "LONG"
            ? '<span class="badge badge-long"><i class="fas fa-arrow-up"></i> LONG</span>'
            : '<span class="badge badge-short"><i class="fas fa-arrow-down"></i> SHORT</span>';
        const statusBadge = s.status === "ACTIVE"
            ? '<span class="badge badge-active"><i class="fas fa-bolt"></i> Aktif</span>'
            : '<span class="badge badge-waiting"><i class="fas fa-clock"></i> Bekliyor</span>';
        const confidence = s.confidence || 0;
        const confClass = confidence >= 75 ? "high" : confidence >= 55 ? "medium" : "low";

        const coinParts = s.symbol.split("-");
        const coinDisplay = `<div class="coin-symbol"><span class="coin-name">${coinParts[0]}</span><span class="coin-pair">/${coinParts[1]}</span></div>`;

        return `<tr>
            <td style="color:var(--text-muted)">#${s.id}</td>
            <td>${coinDisplay}</td>
            <td>${dirBadge}</td>
            <td>${formatPrice(s.entry_price)}</td>
            <td>${s.current_price ? formatPrice(s.current_price) : "--"}</td>
            <td style="color:var(--red)">${formatPrice(s.stop_loss)}</td>
            <td style="color:var(--green)">${formatPrice(s.take_profit)}</td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-track">
                        <div class="confidence-fill ${confClass}" style="width:${confidence}%"></div>
                    </div>
                    <span style="font-size:11px;color:var(--text-secondary)">${confidence}%</span>
                </div>
            </td>
            <td><span class="${pnlClass}">${pnlText}</span></td>
            <td>${statusBadge}</td>
            <td>
                <button class="btn-cancel" onclick="cancelSignal(${s.id})">
                    <i class="fas fa-times"></i> İptal
                </button>
            </td>
        </tr>`;
    }).join("");
}

async function cancelSignal(signalId) {
    if (!confirm("Bu sinyali iptal etmek istediğinize emin misiniz?")) return;
    await apiPost(`/api/signal/${signalId}/cancel`);
    showToast("Sinyal iptal edildi", "info");
    loadActiveSignals();
}

// =================== COINS ===================

let allCoinsData = [];

async function loadCoins() {
    const data = await apiFetch("/api/coins");
    if (!data) return;

    allCoinsData = data.coins || [];
    document.getElementById("activeCoinCount").textContent = data.total_coins || 0;

    // Nav badge
    const badge = document.getElementById("navBadgeCoins");
    if (allCoinsData.length > 0) {
        badge.textContent = allCoinsData.length;
    } else {
        badge.textContent = "";
    }

    renderCoinsTable(allCoinsData);
}

function filterCoins() {
    const query = document.getElementById("coinSearchInput").value.toUpperCase().trim();
    if (!query) {
        renderCoinsTable(allCoinsData);
        return;
    }
    const filtered = allCoinsData.filter(c => c.symbol.toUpperCase().includes(query));
    renderCoinsTable(filtered);
}

function renderCoinsTable(coins) {
    const tbody = document.getElementById("coinsTable");

    if (!coins || coins.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row"><td colspan="6">
                <div class="empty-state">
                    <div class="empty-icon"><i class="fas fa-satellite-dish"></i></div>
                    <p class="empty-title">Coin bulunamadı</p>
                    <p class="empty-desc">OKX'ten $5M+ hacimli coinler yükleniyor...</p>
                </div>
            </td></tr>`;
        return;
    }

    tbody.innerHTML = coins.map((coin, index) => {
        const parts = coin.symbol.split("-");
        const coinName = parts[0] || coin.symbol;
        const pairName = parts[1] || "USDT";
        const coinDisplay = `<div class="coin-symbol"><span class="coin-name">${coinName}</span><span class="coin-pair">/${pairName}</span></div>`;

        const price = coin.last_price || 0;
        const change = coin.change_pct || 0;
        const changeClass = change >= 0 ? "pnl-positive" : "pnl-negative";
        const changeIcon = change >= 0 ? "fa-caret-up" : "fa-caret-down";
        const changeText = change >= 0 ? `+${change.toFixed(2)}%` : `${change.toFixed(2)}%`;

        const volume = coin.volume_usdt || 0;
        const volText = volume >= 1_000_000_000
            ? `$${(volume / 1_000_000_000).toFixed(2)}B`
            : volume >= 1_000_000
            ? `$${(volume / 1_000_000).toFixed(2)}M`
            : `$${volume.toLocaleString("tr-TR")}`;

        const volBarPct = Math.min(100, (volume / (coins[0]?.volume_usdt || 1)) * 100);

        return `<tr class="coin-row-clickable" ondblclick="openCoinDetail('${coin.symbol}')" title="Çift tıkla: Detaylı Teknik Analiz">
            <td style="color:var(--text-muted);font-size:11px">${index + 1}</td>
            <td>${coinDisplay}</td>
            <td style="font-weight:600">${formatPrice(price)}</td>
            <td><span class="${changeClass}"><i class="fas ${changeIcon}" style="font-size:12px;margin-right:3px"></i>${changeText}</span></td>
            <td>
                <div class="volume-cell">
                    <span class="vol-text">${volText}</span>
                    <div class="vol-track">
                        <div class="vol-fill" style="width:${volBarPct}%"></div>
                    </div>
                </div>
            </td>
            <td>
                <button class="btn-analyze" onclick="event.stopPropagation(); openCoinDetail('${coin.symbol}')">
                    <i class="fas fa-magnifying-glass-chart"></i> Detay
                </button>
            </td>
        </tr>`;
    }).join("");
}

// analyzeSymbol replaced by openCoinDetail modal

// =================== WATCHLIST ===================

async function loadWatchlist() {
    const ictData = await apiFetch("/api/watchlist");
    const allItems = (ictData || []).map(i => ({ ...i, _strategy: "ICT" }));

    const tbody = document.getElementById("watchlistTable");

    // Nav badge
    const badge = document.getElementById("navBadgeWatch");
    badge.textContent = allItems.length > 0 ? allItems.length : "";

    if (allItems.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row"><td colspan="9">
                <div class="empty-state">
                    <div class="empty-icon"><i class="fas fa-binoculars"></i></div>
                    <p class="empty-title">İzleme listesi boş</p>
                    <p class="empty-desc">Potansiyel sinyaller burada bekletilir</p>
                </div>
            </td></tr>`;
        return;
    }

    tbody.innerHTML = allItems.map(item => {
        const stratBadge = '<span class="badge badge-ict"><i class="fas fa-signal"></i> ICT</span>';

        const dirBadge = item.direction === "LONG"
            ? '<span class="badge badge-long"><i class="fas fa-arrow-up"></i> LONG</span>'
            : '<span class="badge badge-short"><i class="fas fa-arrow-down"></i> SHORT</span>';

        const progressPct = Math.round((item.candles_watched / item.max_watch_candles) * 100);
        const scoreChange = item.current_score - item.initial_score;
        const scoreChangeText = scoreChange >= 0
            ? `<span class="pnl-positive">+${scoreChange.toFixed(1)}</span>`
            : `<span class="pnl-negative">${scoreChange.toFixed(1)}</span>`;

        const coinParts = item.symbol.split("-");
        const coinDisplay = `<div class="coin-symbol"><span class="coin-name">${coinParts[0]}</span><span class="coin-pair">/${coinParts[1]}</span></div>`;

        return `<tr>
            <td>${stratBadge}</td>
            <td>${coinDisplay}</td>
            <td>${dirBadge}</td>
            <td>${formatPrice(item.potential_entry)}</td>
            <td>${item.initial_score ? item.initial_score.toFixed(1) : "--"}%</td>
            <td>${item.current_score ? item.current_score.toFixed(1) : "--"}% ${scoreChangeText}</td>
            <td>
                <div class="confidence-bar">
                    <div class="confidence-track" style="width:60px">
                        <div class="confidence-fill medium" style="width:${progressPct}%"></div>
                    </div>
                    <span style="font-size:11px;color:var(--text-muted)">${item.candles_watched}/${item.max_watch_candles}</span>
                </div>
            </td>
            <td style="font-size:11px;color:var(--text-secondary);font-family:var(--font-main);max-width:200px">${item.watch_reason || "--"}</td>
            <td><span class="badge badge-watching"><i class="fas fa-eye"></i> İzleniyor</span></td>
        </tr>`;
    }).join("");

    loadExpired();
}

async function loadExpired() {
    const data = await apiFetch("/api/watchlist/expired?minutes=30");
    const section = document.getElementById("expiredSection");
    const list = document.getElementById("expiredList");

    if (!data || data.length === 0) {
        section.style.display = "none";
        return;
    }

    section.style.display = "block";
    list.innerHTML = data.map(item => {
        const coin = item.symbol.split("-")[0];
        const dir = item.direction === "LONG" ? "↑" : "↓";
        const reason = item.expire_reason || "Bilinmiyor";
        const time = item.updated_at ? new Date(item.updated_at).toLocaleTimeString("tr-TR", {hour:"2-digit", minute:"2-digit"}) : "";
        return `<div class="expired-chip">
            <span class="chip-coin">${coin} ${dir}</span>
            <span class="chip-reason">${reason}</span>
            <span class="chip-time">${time}</span>
        </div>`;
    }).join("");
}

// =================== HISTORY ===================

async function loadHistory() {
    const data = await apiFetch("/api/signals/history?limit=50");
    if (!data) return;

    const tbody = document.getElementById("historyTable");
    const completed = data.filter(s => s.status === "WON" || s.status === "LOST");
    const wins = completed.filter(s => s.status === "WON").length;
    const losses = completed.filter(s => s.status === "LOST").length;

    document.getElementById("histWins").textContent = wins;
    document.getElementById("histLosses").textContent = losses;

    if (data.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row"><td colspan="11">
                <div class="empty-state">
                    <div class="empty-icon"><i class="fas fa-archive"></i></div>
                    <p class="empty-title">Geçmiş işlem yok</p>
                    <p class="empty-desc">Tamamlanan işlemler burada görünecek</p>
                </div>
            </td></tr>`;
        return;
    }

    tbody.innerHTML = data.map(s => {
        const dirBadge = s.direction === "LONG"
            ? '<span class="badge badge-long"><i class="fas fa-arrow-up"></i> LONG</span>'
            : '<span class="badge badge-short"><i class="fas fa-arrow-down"></i> SHORT</span>';

        let statusBadge = "";
        switch (s.status) {
            case "WON":
                statusBadge = '<span class="badge badge-won"><i class="fas fa-trophy"></i> Kazandı</span>';
                break;
            case "LOST":
                statusBadge = '<span class="badge badge-lost"><i class="fas fa-times"></i> Kaybetti</span>';
                break;
            case "CANCELLED":
                statusBadge = '<span class="badge badge-cancelled"><i class="fas fa-ban"></i> İptal</span>';
                break;
            case "ACTIVE":
                statusBadge = '<span class="badge badge-active"><i class="fas fa-bolt"></i> Aktif</span>';
                break;
            default:
                statusBadge = `<span class="badge badge-waiting">${s.status}</span>`;
        }

        const pnl = s.pnl_pct || 0;
        const pnlClass = pnl >= 0 ? "pnl-positive" : "pnl-negative";
        const pnlText = pnl !== 0 ? (pnl >= 0 ? `+${pnl.toFixed(2)}%` : `${pnl.toFixed(2)}%`) : "--";

        const coinParts = s.symbol.split("-");
        const coinDisplay = `<div class="coin-symbol"><span class="coin-name">${coinParts[0]}</span><span class="coin-pair">/${coinParts[1]}</span></div>`;

        return `<tr>
            <td style="color:var(--text-muted)">#${s.id}</td>
            <td>${coinDisplay}</td>
            <td>${dirBadge}</td>
            <td>${formatPrice(s.entry_price)}</td>
            <td>${s.close_price ? formatPrice(s.close_price) : "--"}</td>
            <td style="color:var(--red)">${formatPrice(s.stop_loss)}</td>
            <td style="color:var(--green)">${formatPrice(s.take_profit)}</td>
            <td><span class="${pnlClass}">${pnlText}</span></td>
            <td>${s.confidence ? s.confidence.toFixed(0) + "%" : "--"}</td>
            <td>${statusBadge}</td>
            <td style="font-size:11px;color:var(--text-muted);font-family:var(--font-main)">${formatDate(s.created_at)}</td>
        </tr>`;
    }).join("");
}

// =================== OPTIMIZATION ===================

async function loadOptimization() {
    const summary = await apiFetch("/api/optimization/summary");
    if (summary) {
        document.getElementById("optCurrentWR").textContent = `%${summary.current_win_rate || 0}`;
        document.getElementById("optTargetWR").textContent = `%${summary.target_win_rate || 60}`;
        document.getElementById("optChangedParams").textContent = summary.total_optimizations || 0;

        const paramsGrid = document.getElementById("changedParamsGrid");
        const changed = summary.changed_params || {};
        const keys = Object.keys(changed);

        if (keys.length === 0) {
            paramsGrid.innerHTML = '<div class="empty-state small"><p class="empty-title">Henüz parametre değişikliği yok</p></div>';
        } else {
            paramsGrid.innerHTML = keys.map(key => {
                const p = changed[key];
                return `<div class="param-card">
                    <div class="param-name">${key}</div>
                    <div class="param-values">
                        <span class="param-old">${formatParamValue(p.default)}</span>
                        <span class="param-arrow"><i class="fas fa-arrow-right"></i></span>
                        <span class="param-new">${formatParamValue(p.current)}</span>
                    </div>
                    <div class="param-change">Değişim: ${p.change_pct > 0 ? "+" : ""}${p.change_pct}%</div>
                </div>`;
            }).join("");
        }
    }

    const logs = await apiFetch("/api/optimization/logs?limit=20");
    const logContainer = document.getElementById("optLogContainer");

    if (!logs || logs.length === 0) {
        logContainer.innerHTML = '<div class="empty-state small"><p class="empty-title">Günlük boş</p></div>';
        return;
    }

    logContainer.innerHTML = logs.map(log => {
        return `<div class="opt-log-item">
            <div class="opt-log-icon"><i class="fas fa-gear"></i></div>
            <div class="opt-log-content">
                <div class="opt-log-title">${log.param_name}</div>
                <div class="opt-log-detail">
                    ${formatParamValue(log.old_value)} → ${formatParamValue(log.new_value)} | ${log.reason || ""}
                </div>
            </div>
            <div class="opt-log-time">${formatDate(log.created_at)}</div>
        </div>`;
    }).join("");
}

async function runOptimization() {
    showToast("Optimizasyon çalıştırılıyor...", "info");
    const result = await apiPost("/api/optimization/run");
    if (result) {
        if (result.changes && result.changes.length > 0) {
            showToast(`${result.changes.length} parametre güncellendi!`, "success");
        } else {
            showToast(result.reason || "Değişiklik gerekli değil", "info");
        }
        loadOptimization();
    }
}

// =================== PERFORMANCE ===================

async function loadPerformance() {
    const stats = await apiFetch("/api/performance");
    if (!stats) return;

    updateStats(stats);

    const grid = document.getElementById("performanceGrid");
    const compPerf = stats.component_performance || {};
    const keys = Object.keys(compPerf);

    if (keys.length === 0) {
        grid.innerHTML = `<div class="empty-state">
            <div class="empty-icon"><i class="fas fa-chart-pie"></i></div>
            <p class="empty-title">Yeterli veri bekleniyor</p>
            <p class="empty-desc">İşlem tamamlandıkça bileşen performansları görünecek</p>
        </div>`;
    } else {
        renderPerfGrid(grid, compPerf, "ict");
    }

}

function renderPerfGrid(grid, compPerf, type) {
    const keys = Object.keys(compPerf);

    if (keys.length === 0) {
        grid.innerHTML = `<div class="empty-state">
            <div class="empty-icon"><i class="fas fa-chart-pie"></i></div>
            <p class="empty-title">Yeterli veri bekleniyor</p>
        </div>`;
        return;
    }

    const compNames = {
        "MARKET_STRUCTURE": "Piyasa Yapısı",
        "ORDER_BLOCK": "Order Block",
        "FVG": "Fair Value Gap",
        "LIQUIDITY_SWEEP": "Likidite Taraması",
        "DISPLACEMENT": "Displacement",
        "OTE": "Optimal Trade Entry",
        "DISCOUNT_ZONE": "İndirim Bölgesi",
        "PREMIUM_ZONE": "Prim Bölgesi",
        "HTF_CONFIRMATION": "HTF Onayı",
    };

    const compIcons = {
        "MARKET_STRUCTURE": "fas fa-project-diagram",
        "ORDER_BLOCK": "fas fa-cube",
        "FVG": "fas fa-expand",
        "LIQUIDITY_SWEEP": "fas fa-water",
        "DISPLACEMENT": "fas fa-bolt",
        "OTE": "fas fa-bullseye",
        "DISCOUNT_ZONE": "fas fa-tag",
        "PREMIUM_ZONE": "fas fa-gem",
        "HTF_CONFIRMATION": "fas fa-check-double",
    };

    const accentColor = "var(--blue)";

    grid.innerHTML = keys.map(key => {
        const comp = compPerf[key];
        const wr = comp.win_rate || 0;
        const wrClass = wr >= 60 ? "good" : wr >= 45 ? "medium" : "bad";
        const barColor = wr >= 60 ? "var(--green)" : wr >= 45 ? "var(--yellow)" : "var(--red)";
        const icon = compIcons[key] || "fas fa-circle";
        const name = compNames[key] || key;

        return `<div class="perf-card">
            <div class="perf-card-header">
                <span class="perf-comp-name"><i class="${icon}" style="margin-right:6px;color:${accentColor}"></i>${name}</span>
                <span class="perf-wr ${wrClass}">%${wr}</span>
            </div>
            <div class="perf-bar">
                <div class="perf-bar-fill" style="width:${wr}%;background:${barColor}"></div>
            </div>
            <div class="perf-stats">
                <span class="perf-stat-win"><i class="fas fa-check"></i> ${comp.wins} Win</span>
                <span class="perf-stat-loss"><i class="fas fa-times"></i> ${comp.losses} Loss</span>
                <span>Total: ${comp.total}</span>
            </div>
        </div>`;
    }).join("");
}

// =================== MARKET REGIME ===================

const REGIME_META = {
    RISK_ON:       { label: "Risk-On", emoji: "🟢", color: "var(--green)", bg: "rgba(34, 197, 94, 0.12)", border: "rgba(34, 197, 94, 0.3)" },
    ALT_SEASON:    { label: "Alt Season", emoji: "🚀", color: "var(--purple)", bg: "rgba(168, 85, 247, 0.12)", border: "rgba(168, 85, 247, 0.3)" },
    RISK_OFF:      { label: "Risk-Off", emoji: "🔴", color: "var(--red)", bg: "rgba(239, 68, 68, 0.12)", border: "rgba(239, 68, 68, 0.3)" },
    CAPITULATION:  { label: "Kapitülasyon", emoji: "☠️", color: "#f97316", bg: "rgba(249, 115, 22, 0.12)", border: "rgba(249, 115, 22, 0.3)" },
    NEUTRAL:       { label: "Nötr", emoji: "⚪", color: "var(--text-secondary)", bg: "rgba(148, 163, 184, 0.1)", border: "rgba(148, 163, 184, 0.2)" },
    UNKNOWN:       { label: "Bilinmiyor", emoji: "❓", color: "var(--text-muted)", bg: "rgba(100,100,100,0.1)", border: "rgba(100,100,100,0.2)" }
};

const FLOW_LABELS = {
    INFLOW: { text: "Para Girişi", icon: "fa-arrow-trend-up", color: "var(--green)" },
    OUTFLOW: { text: "Para Çıkışı", icon: "fa-arrow-trend-down", color: "var(--red)" },
    PANIC_SELL: { text: "Panik Satış", icon: "fa-triangle-exclamation", color: "#f97316" },
    NEUTRAL: { text: "Nötr", icon: "fa-minus", color: "var(--text-secondary)" }
};

const BTCD_LABELS = {
    RISING: { text: "Yükseliş (BTC Güçlü)", color: "#f59e0b" },
    FALLING: { text: "Düşüş (Altlar Güçlü)", color: "var(--green)" },
    NEUTRAL: { text: "Stabil", color: "var(--text-secondary)" }
};

function updateRegimeTopbar(data) {
    const meta = REGIME_META[data.regime] || REGIME_META.UNKNOWN;
    const dot = document.getElementById("regimeDot");
    const label = document.getElementById("regimeLabel");
    const bias = document.getElementById("regimeBias");

    dot.style.background = meta.color;
    dot.style.boxShadow = `0 0 8px ${meta.color}`;
    label.textContent = `${meta.emoji} ${meta.label}`;
    label.style.color = meta.color;

    const btcBias = data.btc_bias || "NEUTRAL";
    if (btcBias === "LONG") {
        bias.innerHTML = `<i class="fas fa-arrow-up"></i> BTC`;
        bias.style.color = "var(--green)";
    } else if (btcBias === "SHORT") {
        bias.innerHTML = `<i class="fas fa-arrow-down"></i> BTC`;
        bias.style.color = "var(--red)";
    } else {
        bias.innerHTML = `<i class="fas fa-minus"></i> BTC`;
        bias.style.color = "var(--text-muted)";
    }
}

async function loadRegime() {
    const data = await apiFetch("/api/regime");
    if (!data) return;

    updateRegimeTopbar(data);

    // Rejim ana kartı
    const meta = REGIME_META[data.regime] || REGIME_META.UNKNOWN;
    const mainIcon = document.getElementById("regimeMainIcon");
    mainIcon.style.background = meta.bg;
    mainIcon.style.color = meta.color;
    mainIcon.style.border = `1px solid ${meta.border}`;
    document.getElementById("regimeMainValue").textContent = `${meta.emoji} ${meta.label}`;
    document.getElementById("regimeMainValue").style.color = meta.color;

    // BTC Trend
    const btcDetails = data.btc_details || {};
    const btcBias = btcDetails.bias || "NEUTRAL";
    const btcEl = document.getElementById("regimeBtcTrend");
    if (btcBias === "LONG") {
        btcEl.innerHTML = `<i class="fas fa-arrow-up"></i> Yükseliyor (${btcDetails.strength || ""})`;
        btcEl.style.color = "var(--green)";
    } else if (btcBias === "SHORT") {
        btcEl.innerHTML = `<i class="fas fa-arrow-down"></i> Düşüyor (${btcDetails.strength || ""})`;
        btcEl.style.color = "var(--red)";
    } else {
        btcEl.innerHTML = `<i class="fas fa-minus"></i> Yatay`;
        btcEl.style.color = "var(--text-muted)";
    }

    // BTC.D
    const btcD = data.btc_dominance || {};
    const btcdMeta = BTCD_LABELS[btcD.direction] || BTCD_LABELS.NEUTRAL;
    const btcDEl = document.getElementById("regimeBtcD");
    btcDEl.textContent = btcdMeta.text;
    btcDEl.style.color = btcdMeta.color;

    // Para Akışı
    const flow = data.usdt_flow || {};
    const flowMeta = FLOW_LABELS[flow.direction] || FLOW_LABELS.NEUTRAL;
    const flowEl = document.getElementById("regimeFlow");
    flowEl.innerHTML = `<i class="fas ${flowMeta.icon}"></i> ${flowMeta.text}`;
    flowEl.style.color = flowMeta.color;

    // Fırsat listesi
    const longList = (data.long_candidates || []).map(s => s.split("-")[0]);
    const shortList = (data.short_candidates || []).map(s => s.split("-")[0]);
    document.getElementById("regimeLongList").textContent = longList.length > 0 ? longList.join(", ") : "Yok";
    document.getElementById("regimeShortList").textContent = shortList.length > 0 ? shortList.join(", ") : "Yok";

    // RS Sıralaması tablosu
    const rankings = data.rs_rankings || [];
    const tbody = document.getElementById("rsRankingsTable");

    if (rankings.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="7">
            <div class="empty-state">
                <div class="empty-icon"><i class="fas fa-globe"></i></div>
                <p class="empty-title">Rejim verisi bekleniyor</p>
                <p class="empty-desc">Bot çalıştığında piyasa analizi burada görünecek</p>
            </div>
        </td></tr>`;
        return;
    }

    tbody.innerHTML = rankings.map((coin, idx) => {
        const parts = coin.symbol.split("-");
        const coinName = `<div class="coin-symbol"><span class="coin-name">${parts[0]}</span><span class="coin-pair">/${parts[1]}</span></div>`;

        const rs = coin.rs_score;
        const rsClass = rs > 1 ? "pnl-positive" : rs < -1 ? "pnl-negative" : "";
        const rsBar = Math.min(Math.abs(rs) * 20, 100);
        const rsColor = rs > 0 ? "var(--green)" : "var(--red)";

        const chg = coin.price_change_1h || 0;
        const chgClass = chg >= 0 ? "pnl-positive" : "pnl-negative";
        const chgText = chg >= 0 ? `+${chg.toFixed(2)}%` : `${chg.toFixed(2)}%`;

        const vol = coin.vol_ratio || 0;
        const volColor = vol >= 1.5 ? "var(--green)" : vol >= 0.8 ? "var(--text-primary)" : "var(--red)";

        const strs = coin.short_term_rs || 0;
        const strsClass = strs > 0 ? "pnl-positive" : strs < 0 ? "pnl-negative" : "";

        // Durum badge
        const isLong = longList.includes(parts[0]);
        const isShort = shortList.includes(parts[0]);
        let statusBadge = '<span class="badge" style="background:rgba(100,100,100,0.2);color:var(--text-muted)">Nötr</span>';
        if (isLong) statusBadge = '<span class="badge badge-long"><i class="fas fa-arrow-up"></i> LONG Aday</span>';
        if (isShort) statusBadge = '<span class="badge badge-short"><i class="fas fa-arrow-down"></i> SHORT Aday</span>';

        return `<tr>
            <td style="color:var(--text-muted)">${idx + 1}</td>
            <td>${coinName}</td>
            <td>
                <div style="display:flex;align-items:center;gap:8px">
                    <div style="width:60px;height:6px;border-radius:3px;background:rgba(255,255,255,0.06);overflow:hidden">
                        <div style="width:${rsBar}%;height:100%;border-radius:3px;background:${rsColor}"></div>
                    </div>
                    <span class="${rsClass}" style="font-weight:700;font-family:'JetBrains Mono',monospace">${rs > 0 ? "+" : ""}${rs.toFixed(2)}</span>
                </div>
            </td>
            <td><span class="${chgClass}">${chgText}</span></td>
            <td style="color:${volColor};font-family:'JetBrains Mono',monospace">${vol.toFixed(2)}x</td>
            <td><span class="${strsClass}">${strs > 0 ? "+" : ""}${strs.toFixed(2)}</span></td>
            <td>${statusBadge}</td>
        </tr>`;
    }).join("");
}

// =================== STATS UPDATE ===================

function updateStats(stats) {
    document.getElementById("statWinRate").textContent = `%${stats.win_rate || 0}`;
    document.getElementById("statTotalTrades").textContent = stats.total_trades || 0;
    document.getElementById("statActiveTrades").textContent = stats.active_trades || 0;

    const pnl = stats.total_pnl || 0;
    const pnlEl = document.getElementById("statTotalPnl");
    pnlEl.textContent = `${pnl >= 0 ? "+" : ""}${pnl}%`;
    pnlEl.style.color = pnl >= 0 ? "var(--green)" : "var(--red)";

    document.getElementById("statWatching").textContent = stats.watching_count || 0;
    document.getElementById("statAvgRR").textContent = stats.avg_rr || 0;
}

// =================== TOAST NOTIFICATIONS ===================

function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;

    const icons = {
        success: "fas fa-check-circle",
        error: "fas fa-circle-exclamation",
        info: "fas fa-circle-info",
        warning: "fas fa-triangle-exclamation"
    };

    toast.innerHTML = `<i class="${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);

    toast.addEventListener("click", () => removeToast(toast));
    setTimeout(() => removeToast(toast), 5000);
}

function removeToast(toast) {
    toast.style.animation = "toastSlideOut 0.3s ease";
    setTimeout(() => toast.remove(), 300);
}

// =================== HELPERS ===================

function formatPrice(price) {
    if (!price && price !== 0) return "--";
    price = parseFloat(price);
    if (price >= 1000) return price.toFixed(2);
    if (price >= 1) return price.toFixed(4);
    if (price >= 0.01) return price.toFixed(6);
    return price.toFixed(8);
}

function formatTime(isoString) {
    if (!isoString) return "--";
    const d = new Date(isoString);
    return d.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function formatDate(isoString) {
    if (!isoString) return "--";
    const d = new Date(isoString);
    return d.toLocaleDateString("tr-TR", {
        day: "2-digit", month: "2-digit", year: "numeric",
        hour: "2-digit", minute: "2-digit"
    });
}

function formatParamValue(val) {
    if (val === null || val === undefined) return "--";
    if (Number.isInteger(val)) return val.toString();
    return parseFloat(val).toFixed(4);
}

function updateServerTime() {
    const now = new Date();
    document.getElementById("serverTime").textContent =
        now.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// =================== COIN DETAIL TAB ===================

let coinDetailData = null;
let currentModalTf = "15m";

async function openCoinDetail(symbol) {
    // Switch to coindetail tab
    switchTab('coindetail');

    // Reset state
    document.getElementById("modalLoading").style.display = "flex";
    document.getElementById("tfSection").style.display = "none";
    document.getElementById("orderbookSection").style.display = "none";
    document.getElementById("verdictScoreBar").style.display = "none";
    document.getElementById("verdictWarnings").innerHTML = "";
    document.getElementById("scenarioSection").style.display = "none";

    const parts = symbol.split("-");
    document.getElementById("modalCoinName").textContent = `${parts[0]}/${parts[1] || "USDT"}`;
    document.getElementById("modalCoinPrice").textContent = "Yükleniyor...";

    const verdictBar = document.getElementById("modalOverallVerdict");
    verdictBar.className = "detail-verdict-card";
    document.getElementById("verdictLabel").textContent = "Gelişmiş analiz hesaplanıyor...";
    document.getElementById("verdictDesc").textContent = "";

    // Update page title with coin name
    document.getElementById("pageTitle").innerHTML = `<i class="fas fa-magnifying-glass-chart"></i><span>${parts[0]} Detay Analizi</span>`;

    // Fetch data
    const data = await apiFetch(`/api/coin-detail/${symbol}`);
    document.getElementById("modalLoading").style.display = "none";

    if (!data || data.error) {
        document.getElementById("verdictLabel").textContent = "Analiz başarısız";
        document.getElementById("verdictDesc").textContent = data?.error || "Veri alınamadı";
        return;
    }

    coinDetailData = data;
    coinDetailData._symbol = symbol;
    currentModalTf = "15m";

    // Price info
    const price = data.price || {};
    const change = price.change24h || 0;
    const changeStr = change >= 0 ? `+${change.toFixed(2)}%` : `${change.toFixed(2)}%`;
    const changeColor = change >= 0 ? "var(--green)" : "var(--red)";
    document.getElementById("modalCoinPrice").innerHTML =
        `${formatPrice(price.last)} <span style="color:${changeColor};font-size:12px;font-weight:600;margin-left:8px">${changeStr}</span>` +
        `<span style="color:var(--text-muted);margin-left:10px;font-size:11px">24h H: ${formatPrice(price.high24h)} / L: ${formatPrice(price.low24h)}</span>`;

    // Overall verdict
    const ov = data.overall || {};
    const verdictType = ov.verdict || "NEUTRAL";
    verdictBar.className = `detail-verdict-card verdict-${verdictType}`;
    document.getElementById("verdictLabel").textContent = `🤖 ${ov.label || "Bilinmiyor"}`;
    document.getElementById("verdictDesc").textContent = ov.description || "";

    // Score bar
    const scoreBar = document.getElementById("verdictScoreBar");
    scoreBar.style.display = "block";
    const bullTotal = ov.bull_total || 0;
    const bearTotal = ov.bear_total || 0;
    const maxScore = 50;
    document.getElementById("scoreFillBull").style.width = `${Math.min(bullTotal / maxScore * 50, 50)}%`;
    document.getElementById("scoreFillBear").style.width = `${Math.min(bearTotal / maxScore * 50, 50)}%`;
    const netScore = ov.net_score || 0;
    const pointerPos = 50 + (netScore / maxScore * 50);
    document.getElementById("scorePointer").style.left = `${Math.min(Math.max(pointerPos, 2), 98)}%`;

    // Warnings
    const warningsEl = document.getElementById("verdictWarnings");
    if (ov.warnings && ov.warnings.length > 0) {
        warningsEl.innerHTML = ov.warnings.map(w => `<div class="warning-item">${w}</div>`).join("");
    } else {
        warningsEl.innerHTML = "";
    }

    // Order Book
    const ob = data.orderbook || {};
    if (ob.imbalance != null) {
        document.getElementById("orderbookSection").style.display = "block";
        const obBadge = document.getElementById("obBadge");
        obBadge.textContent = ob.label || "--";
        obBadge.className = `ind-badge ${getSignalClass(ob.signal)}`;
        document.getElementById("obBidFill").style.width = `${ob.imbalance}%`;
        document.getElementById("obAskFill").style.width = `${100 - ob.imbalance}%`;
        document.getElementById("obBidText").textContent = `Alım: %${ob.imbalance}`;
        document.getElementById("obAskText").textContent = `Satım: %${(100 - ob.imbalance).toFixed(1)}`;
        document.getElementById("obSpread").textContent = `Spread: %${ob.spread_pct || 0}`;
        document.getElementById("obDesc").textContent = ob.desc || "";
    } else {
        document.getElementById("orderbookSection").style.display = "none";
    }

    // Show TF section
    document.getElementById("tfSection").style.display = "block";

    // Set active tab
    document.querySelectorAll(".tf-tab").forEach(t => t.classList.toggle("active", t.dataset.tf === "15m"));

    renderModalTf("15m");

    // Render Trading Scenario
    renderScenario(data.scenario);
}

// =================== AI TRADING SENARYOSU ===================

function renderScenario(scenario) {
    const section = document.getElementById("scenarioSection");
    const container = document.getElementById("scenarioContainer");
    if (!scenario) {
        section.style.display = "none";
        return;
    }
    section.style.display = "block";

    // Badge
    const badge = document.getElementById("scenarioBadge");
    const recMap = {
        "LONG": { text: "LONG ÖNCELİKLİ", cls: "bull" },
        "LONG_CAUTIOUS": { text: "HAFİF LONG", cls: "weak-bull" },
        "SHORT": { text: "SHORT ÖNCELİKLİ", cls: "bear" },
        "SHORT_CAUTIOUS": { text: "HAFİF SHORT", cls: "weak-bear" },
        "WAIT": { text: "BEKLE", cls: "neutral" }
    };
    const rec = recMap[scenario.recommended] || recMap["WAIT"];
    badge.textContent = rec.text;
    badge.className = `scenario-badge ${rec.cls}`;

    // Strategy lines
    const stratEl = document.getElementById("scenarioStrategy");
    stratEl.innerHTML = scenario.strategy.map(line => `<div class="strat-line">${line}</div>`).join("");

    // Wait conditions
    const waitEl = document.getElementById("scenarioWait");
    if (scenario.wait_conditions && scenario.wait_conditions.length > 0) {
        waitEl.innerHTML = `<div class="wait-header"><i class="fas fa-clock"></i> Giriş Onay Koşulları</div>` +
            scenario.wait_conditions.map(line => `<div class="wait-line">${line}</div>`).join("");
        waitEl.style.display = "block";
    } else {
        waitEl.style.display = "none";
    }

    // Key Levels
    const levelsEl = document.getElementById("scenarioLevels");
    if (scenario.key_levels && scenario.key_levels.length > 0) {
        let levelsHtml = `<div class="levels-header"><i class="fas fa-map-marker-alt"></i> Anahtar Seviyeler</div>`;
        levelsHtml += `<div class="levels-grid">`;
        scenario.key_levels.forEach(level => {
            const isAbove = level.price_raw > scenario.current_price;
            const dist = ((level.price_raw - scenario.current_price) / scenario.current_price * 100).toFixed(2);
            const distStr = isAbove ? `+${dist}%` : `${dist}%`;
            const typeClass = level.type === "resistance" ? "resistance" : "support";
            const isCurrent = Math.abs(level.price_raw - scenario.current_price) / scenario.current_price < 0.001;
            levelsHtml += `<div class="level-item ${typeClass} ${isCurrent ? 'current' : ''}">
                <span class="level-name">${level.name}</span>
                <span class="level-price">${level.price}</span>
                <span class="level-dist ${isAbove ? 'above' : 'below'}">${distStr}</span>
            </div>`;
        });
        // Current price marker
        levelsHtml += `<div class="level-item current-price">
            <span class="level-name">📍 Mevcut Fiyat</span>
            <span class="level-price">${scenario.current_price_fmt}</span>
            <span class="level-dist">--</span>
        </div>`;
        levelsHtml += `</div>`;
        levelsEl.innerHTML = levelsHtml;
        levelsEl.style.display = "block";
    } else {
        levelsEl.style.display = "none";
    }

    // Long Scenario
    renderScenarioCard("longBody", scenario.long, "longQuality");

    // Short Scenario
    renderScenarioCard("shortBody", scenario.short, "shortQuality");
}

function renderScenarioCard(bodyId, scenarioData, qualityId) {
    const body = document.getElementById(bodyId);
    const qualityEl = document.getElementById(qualityId);

    if (!scenarioData) {
        body.innerHTML = "<p>Veri yok</p>";
        return;
    }

    const quality = scenarioData.quality || 0;
    qualityEl.textContent = `${Math.min(quality, 100)}%`;

    if (quality >= 60) {
        qualityEl.className = "scenario-quality high";
    } else if (quality >= 30) {
        qualityEl.className = "scenario-quality medium";
    } else {
        qualityEl.className = "scenario-quality low";
    }

    let html = "";
    (scenarioData.sections || []).forEach(section => {
        html += `<div class="scenario-section">`;
        html += `<div class="scenario-section-title">${section.title}</div>`;
        (section.lines || []).forEach(line => {
            html += `<div class="scenario-line">${line}</div>`;
        });
        html += `</div>`;
    });

    body.innerHTML = html;
}

function switchModalTf(tf) {
    currentModalTf = tf;
    document.querySelectorAll(".tf-tab").forEach(t => t.classList.toggle("active", t.dataset.tf === tf));
    renderModalTf(tf);
}

function renderModalTf(tf) {
    if (!coinDetailData || !coinDetailData.timeframes) return;

    const tfData = coinDetailData.timeframes[tf];
    if (!tfData) return;

    // TF Verdict chip with score
    const verdictRow = document.getElementById("tfVerdictRow");
    const chipClass = {
        "STRONG_BULLISH": "bull", "BULLISH": "bull", "BEARISH": "bear", "STRONG_BEARISH": "bear",
        "NEUTRAL": "neutral", "LEANING_BULLISH": "weak-bull", "LEANING_BEARISH": "weak-bear", "UNKNOWN": "neutral"
    }[tfData.verdict] || "neutral";

    const verdictEmoji = {
        "STRONG_BULLISH": "🟢🟢", "BULLISH": "🟢", "BEARISH": "🔴", "STRONG_BEARISH": "🔴🔴",
        "NEUTRAL": "⚪", "LEANING_BULLISH": "🟡", "LEANING_BEARISH": "🟠"
    }[tfData.verdict] || "⚪";

    const netScore = tfData.net_score || 0;
    const confidence = tfData.confidence || 0;

    verdictRow.innerHTML = `
        <span class="tf-verdict-chip ${chipClass}">${verdictEmoji} ${tfData.verdict_label || tfData.verdict}</span>
        <span class="tf-verdict-text">Skor: ${netScore > 0 ? '+' : ''}${netScore} | Güven: ${confidence}/100</span>
        <span class="tf-verdict-text">Boğa: ${tfData.bull_score || 0} | Ayı: ${tfData.bear_score || 0}</span>
    `;

    // Indicator Score Summary (mini bar chart)
    const scores = tfData.indicator_scores || {};
    const scoreNames = { trend: "Trend", adx: "ADX", macd: "MACD", rsi: "RSI", stoch_rsi: "StRSI",
                         volume: "Hacim", obv: "OBV", bollinger: "BB", fvg: "FVG", divergence: "Div" };
    const summaryEl = document.getElementById("indScoreSummary");
    let summaryHtml = '<div class="score-chips">';
    for (const [key, name] of Object.entries(scoreNames)) {
        const s = scores[key];
        if (s) {
            const cls = s.direction === "BULL" ? "score-bull" : "score-bear";
            summaryHtml += `<span class="score-chip ${cls}">${name}: ${s.direction === "BULL" ? "+" : "-"}${s.score}</span>`;
        }
    }
    summaryHtml += '</div>';
    summaryEl.innerHTML = summaryHtml;

    // RSI
    const rsi = tfData.rsi || {};
    const rsiBadge = document.getElementById("rsiBadge");
    rsiBadge.textContent = rsi.label || "--";
    rsiBadge.className = `ind-badge ${getSignalClass(rsi.signal)}`;
    document.getElementById("rsiValue").textContent = rsi.value != null ? rsi.value.toFixed(1) : "--";
    document.getElementById("rsiValue").style.color = getSignalColor(rsi.signal);
    document.getElementById("rsiDesc").textContent = rsi.desc || "";
    const needle = document.getElementById("rsiNeedle");
    if (rsi.value != null) {
        needle.style.left = `${Math.min(Math.max(rsi.value, 0), 100)}%`;
        needle.style.display = "block";
    } else {
        needle.style.display = "none";
    }

    // Stochastic RSI
    const stoch = tfData.stoch_rsi || {};
    const stochBadge = document.getElementById("stochRsiBadge");
    stochBadge.textContent = stoch.label || "--";
    stochBadge.className = `ind-badge ${getSignalClass(stoch.signal)}`;
    document.getElementById("stochK").textContent = stoch.k != null ? stoch.k.toFixed(1) : "--";
    document.getElementById("stochK").style.color = stoch.k > 80 ? "var(--red)" : stoch.k < 20 ? "var(--green)" : "var(--text-primary)";
    document.getElementById("stochD").textContent = stoch.d != null ? stoch.d.toFixed(1) : "--";
    document.getElementById("stochRsiDesc").textContent = stoch.desc || "";
    const stochNeedle = document.getElementById("stochNeedle");
    if (stoch.k != null) {
        stochNeedle.style.left = `${Math.min(Math.max(stoch.k, 0), 100)}%`;
        stochNeedle.style.display = "block";
    } else {
        stochNeedle.style.display = "none";
    }

    // MACD
    const macd = tfData.macd || {};
    const macdBadge = document.getElementById("macdBadge");
    macdBadge.textContent = macd.label || "--";
    macdBadge.className = `ind-badge ${getSignalClass(macd.signal_type)}`;
    document.getElementById("macdLine").textContent = macd.macd != null ? macd.macd.toFixed(6) : "--";
    document.getElementById("macdLine").style.color = macd.macd > 0 ? "var(--green)" : macd.macd < 0 ? "var(--red)" : "var(--text-primary)";
    document.getElementById("macdSignal").textContent = macd.signal != null ? macd.signal.toFixed(6) : "--";
    document.getElementById("macdHist").textContent = macd.histogram != null ? macd.histogram.toFixed(6) : "--";
    document.getElementById("macdHist").style.color = macd.histogram > 0 ? "var(--green)" : macd.histogram < 0 ? "var(--red)" : "var(--text-primary)";
    document.getElementById("macdDesc").textContent = macd.desc || "";

    // Bollinger Bands
    const bb = tfData.bollinger || {};
    const bbBadge = document.getElementById("bbBadge");
    bbBadge.textContent = bb.label || "--";
    bbBadge.className = `ind-badge ${getSignalClass(bb.signal)}`;
    document.getElementById("bbUpper").textContent = bb.upper != null ? formatPrice(bb.upper) : "--";
    document.getElementById("bbMiddle").textContent = bb.middle != null ? formatPrice(bb.middle) : "--";
    document.getElementById("bbLower").textContent = bb.lower != null ? formatPrice(bb.lower) : "--";
    document.getElementById("bbDesc").textContent = bb.desc || "";
    document.getElementById("bbSqueeze").textContent = bb.squeeze_desc || "";
    // BB position meter
    const bbPos = document.getElementById("bbPosition");
    if (bb.pct_b != null) {
        bbPos.style.left = `${Math.min(Math.max(bb.pct_b, 0), 100)}%`;
        bbPos.style.display = "block";
        bbPos.style.backgroundColor = bb.pct_b > 80 ? "var(--red)" : bb.pct_b < 20 ? "var(--green)" : "var(--blue)";
    }

    // ADX
    const adx = tfData.adx || {};
    const adxBadge = document.getElementById("adxBadge");
    adxBadge.textContent = adx.label || "--";
    adxBadge.className = `ind-badge ${getSignalClass(adx.signal)}`;
    document.getElementById("adxValue").textContent = adx.adx != null ? adx.adx.toFixed(1) : "--";
    document.getElementById("adxPlusDi").textContent = adx.plus_di != null ? adx.plus_di.toFixed(1) : "--";
    document.getElementById("adxMinusDi").textContent = adx.minus_di != null ? adx.minus_di.toFixed(1) : "--";
    document.getElementById("adxDesc").textContent = adx.desc || "";
    // ADX strength bar
    const adxFill = document.getElementById("adxFill");
    if (adx.adx != null) {
        adxFill.style.width = `${Math.min(adx.adx, 75) / 75 * 100}%`;
        adxFill.style.backgroundColor = adx.adx >= 25 ? (adx.di_direction === "BULLISH" ? "var(--green)" : "var(--red)") : "var(--text-muted)";
    }

    // ATR
    const atr = tfData.atr || {};
    const atrBadge = document.getElementById("atrBadge");
    atrBadge.textContent = atr.label || "--";
    atrBadge.className = `ind-badge ${atr.signal === "HIGH" ? "bear" : atr.signal === "LOW" ? "weak-bull" : "neutral"}`;
    document.getElementById("atrValue").textContent = atr.atr != null ? formatPrice(atr.atr) : "--";
    document.getElementById("atrPct").textContent = atr.atr_pct != null ? `%${atr.atr_pct}` : "--";
    document.getElementById("atrDesc").textContent = atr.desc || "";

    // Volume
    const vol = tfData.volume || {};
    const volBadge = document.getElementById("volBadge");
    volBadge.textContent = vol.label || "--";
    volBadge.className = `ind-badge ${vol.signal === "HIGH" ? "bull" : vol.signal === "LOW" ? "weak-bear" : "neutral"}`;
    document.getElementById("volRatio").textContent = vol.ratio != null ? `${vol.ratio}x` : "--";
    document.getElementById("volRatio").style.color = vol.ratio >= 1.5 ? "var(--green)" : vol.ratio < 0.8 ? "var(--red)" : "var(--text-primary)";
    document.getElementById("volTrend").textContent = vol.trend || "--";
    document.getElementById("volTrend").style.color = vol.trend === "ARTIYOR" ? "var(--green)" : vol.trend === "AZALIYOR" ? "var(--red)" : "var(--text-primary)";
    document.getElementById("volDesc").textContent = vol.desc || "";
    document.getElementById("volHarmony").textContent = vol.price_vol_harmony || "";

    // OBV
    const obv = tfData.obv || {};
    const obvBadge = document.getElementById("obvBadge");
    obvBadge.textContent = obv.label || "--";
    obvBadge.className = `ind-badge ${getSignalClass(obv.signal)}`;
    document.getElementById("obvDesc").textContent = obv.desc || "";

    // FVG
    const fvg = tfData.fvg || {};
    const fvgBadge = document.getElementById("fvgBadge");
    fvgBadge.textContent = fvg.label || "--";
    fvgBadge.className = `ind-badge ${getSignalClass(fvg.signal)}`;
    document.getElementById("fvgBull").textContent = fvg.unfilled_bullish || 0;
    document.getElementById("fvgBear").textContent = fvg.unfilled_bearish || 0;
    document.getElementById("fvgDesc").textContent = fvg.desc || "";

    // Support/Resistance
    const sr = tfData.support_resistance || {};
    const srBadge = document.getElementById("srBadge");
    srBadge.textContent = sr.label || "--";
    srBadge.className = `ind-badge ${getSignalClass(sr.signal)}`;
    document.getElementById("srResistance").textContent = sr.nearest_resistance != null ? formatPrice(sr.nearest_resistance) : "--";
    document.getElementById("srSupport").textContent = sr.nearest_support != null ? formatPrice(sr.nearest_support) : "--";
    const currentPrice = coinDetailData?.price?.last || 0;
    document.getElementById("srCurrent").textContent = currentPrice ? formatPrice(currentPrice) : "--";
    document.getElementById("srDesc").textContent = sr.desc || "";

    // Divergence
    const div = tfData.divergence || {};
    const divBadge = document.getElementById("divBadge");
    divBadge.textContent = div.label || "--";
    const divClass = div.type === "BULLISH" ? "bull" : div.type === "BEARISH" ? "bear" : "neutral";
    divBadge.className = `ind-badge ${divClass}`;
    document.getElementById("divDesc").textContent = div.desc || "";

    // Trend / EMA
    const trendBadge = document.getElementById("trendBadge");
    trendBadge.textContent = tfData.trend_label || tfData.trend || "--";
    const trendClass = {
        "BULLISH": "bull", "BEARISH": "bear", "WEAKENING_BEAR": "weak-bull",
        "WEAKENING_BULL": "weak-bear", "UNKNOWN": "neutral"
    }[tfData.trend] || "neutral";
    trendBadge.className = `ind-badge ${trendClass}`;

    const ema = tfData.ema || {};
    document.getElementById("ema8Val").textContent = ema.ema8 != null ? formatPrice(ema.ema8) : "--";
    document.getElementById("ema21Val").textContent = ema.ema21 != null ? formatPrice(ema.ema21) : "--";
    document.getElementById("ema50Val").textContent = ema.ema50 != null ? formatPrice(ema.ema50) : "--";
    document.getElementById("ema200Val").textContent = ema.ema200 != null ? formatPrice(ema.ema200) : "--";

    // EMA colors
    if (ema.ema8 != null && ema.ema21 != null) {
        document.getElementById("ema8Val").style.color = ema.ema8 > ema.ema21 ? "var(--green)" : "var(--red)";
        document.getElementById("ema21Val").style.color = ema.ema21 > (ema.ema50 || 0) ? "var(--green)" : "var(--red)";
    }
    if (ema.ema50 != null) {
        document.getElementById("ema50Val").style.color = currentPrice > ema.ema50 ? "var(--green)" : "var(--red)";
    }
    if (ema.ema200 != null) {
        document.getElementById("ema200Val").style.color = currentPrice > ema.ema200 ? "var(--green)" : "var(--red)";
    }

    // EMA Order badge
    const emaOrderBadge = document.getElementById("emaOrderBadge");
    if (ema.order === "BULL") {
        emaOrderBadge.textContent = "EMA Sıralaması: Boğa (8>21>50)";
        emaOrderBadge.className = "ema-order-badge bull";
    } else if (ema.order === "BEAR") {
        emaOrderBadge.textContent = "EMA Sıralaması: Ayı (8<21<50)";
        emaOrderBadge.className = "ema-order-badge bear";
    } else {
        emaOrderBadge.textContent = "EMA Sıralaması: Karışık";
        emaOrderBadge.className = "ema-order-badge neutral";
    }

    document.getElementById("trendDesc").textContent = tfData.trend_desc || "";
}

function getSignalClass(signal) {
    switch (signal) {
        case "BULLISH": return "bull";
        case "BEARISH": return "bear";
        case "WEAKENING_BULL": return "weak-bear";
        case "WEAKENING_BEAR": return "weak-bull";
        default: return "neutral";
    }
}

function getSignalColor(signal) {
    switch (signal) {
        case "BULLISH": return "var(--green)";
        case "BEARISH": return "var(--red)";
        default: return "var(--text-primary)";
    }
}

// =================== AUTO REFRESH ===================

let autoRefreshActive = false;
let autoRefreshInterval = null;
let autoRefreshCountdown = 30;
let autoRefreshCountdownInterval = null;
let lastScenarioRecommended = null;

function toggleAutoRefresh() {
    if (autoRefreshActive) {
        stopAutoRefresh();
    } else {
        startAutoRefresh();
    }
}

function startAutoRefresh() {
    if (!coinDetailData?._symbol) return;
    autoRefreshActive = true;
    autoRefreshCountdown = 30;
    lastScenarioRecommended = coinDetailData?.scenario?.recommended || null;

    const toggle = document.getElementById("autoRefreshToggle");
    toggle.classList.add("active");
    document.getElementById("arSwitchDot").classList.add("on");
    updateCountdownDisplay();

    // Countdown
    autoRefreshCountdownInterval = setInterval(() => {
        autoRefreshCountdown--;
        updateCountdownDisplay();
        if (autoRefreshCountdown <= 0) {
            autoRefreshCountdown = 30;
            refreshCoinDetail();
        }
    }, 1000);
}

function stopAutoRefresh() {
    autoRefreshActive = false;
    if (autoRefreshCountdownInterval) clearInterval(autoRefreshCountdownInterval);
    autoRefreshCountdownInterval = null;

    const toggle = document.getElementById("autoRefreshToggle");
    toggle.classList.remove("active");
    document.getElementById("arSwitchDot").classList.remove("on");
    document.getElementById("arCountdown").textContent = "";
}

function updateCountdownDisplay() {
    document.getElementById("arCountdown").textContent = `${autoRefreshCountdown}s`;
}

async function refreshCoinDetail() {
    if (!coinDetailData?._symbol || !autoRefreshActive) return;

    try {
        const data = await apiFetch(`/api/coin-detail/${coinDetailData._symbol}`);
        if (!data || data.error) return;

        const oldRec = coinDetailData?.scenario?.recommended;
        coinDetailData = data;
        coinDetailData._symbol = coinDetailData._symbol || data.symbol;

        // Update price
        const price = data.price || {};
        const change = price.change24h || 0;
        const changeStr = change >= 0 ? `+${change.toFixed(2)}%` : `${change.toFixed(2)}%`;
        const changeColor = change >= 0 ? "var(--green)" : "var(--red)";
        document.getElementById("modalCoinPrice").innerHTML =
            `${formatPrice(price.last)} <span style="color:${changeColor};font-size:12px;font-weight:600;margin-left:8px">${changeStr}</span>`;

        // Update verdict
        const ov = data.overall || {};
        const verdictType = ov.verdict || "NEUTRAL";
        document.getElementById("modalOverallVerdict").className = `detail-verdict-card verdict-${verdictType}`;
        document.getElementById("verdictLabel").textContent = `🤖 ${ov.label || "Bilinmiyor"}`;
        document.getElementById("verdictDesc").textContent = ov.description || "";

        // Score bar
        const bullTotal = ov.bull_total || 0;
        const bearTotal = ov.bear_total || 0;
        const maxScore = 50;
        document.getElementById("scoreFillBull").style.width = `${Math.min(bullTotal / maxScore * 50, 50)}%`;
        document.getElementById("scoreFillBear").style.width = `${Math.min(bearTotal / maxScore * 50, 50)}%`;
        const netScore = ov.net_score || 0;
        const pointerPos = 50 + (netScore / maxScore * 50);
        document.getElementById("scorePointer").style.left = `${Math.min(Math.max(pointerPos, 2), 98)}%`;

        // Warnings
        if (ov.warnings && ov.warnings.length > 0) {
            document.getElementById("verdictWarnings").innerHTML = ov.warnings.map(w => `<div class="warning-item">${w}</div>`).join("");
        }

        // Scenario
        renderScenario(data.scenario);

        // Re-render current TF
        renderModalTf(currentModalTf);

        // Check scenario change
        const newRec = data.scenario?.recommended;
        if (oldRec && newRec && oldRec !== newRec) {
            const recLabels = { LONG: "LONG", SHORT: "SHORT", LONG_CAUTIOUS: "Hafif LONG", SHORT_CAUTIOUS: "Hafif SHORT", WAIT: "BEKLE" };
            showToast(`⚡ Senaryo değişti: ${recLabels[oldRec] || oldRec} → ${recLabels[newRec] || newRec}`, "info");
        }
    } catch (e) {
        console.error("Auto-refresh error:", e);
    }
}

// =================== BACKTEST ===================

async function loadBacktestCoins() {
    const select = document.getElementById("btSymbol");
    if (select.options.length > 1) return; // Already loaded

    const data = await apiFetch("/api/coins");
    if (!data || !data.coins) return;

    data.coins.forEach(coin => {
        const opt = document.createElement("option");
        opt.value = coin.symbol;
        opt.textContent = coin.symbol.replace("-USDT-SWAP", "");
        select.appendChild(opt);
    });
}

async function runBacktest() {
    const symbol = document.getElementById("btSymbol").value;
    const timeframe = document.getElementById("btTimeframe").value;
    const period = document.getElementById("btPeriod").value;
    const minScore = document.getElementById("btMinScore").value;

    if (!symbol) {
        showToast("Lütfen bir coin seçin", "error");
        return;
    }

    document.getElementById("btLoading").style.display = "flex";
    document.getElementById("btResults").style.display = "none";
    document.getElementById("btRunBtn").disabled = true;

    try {
        const data = await apiFetch(`/api/backtest/${symbol}?tf=${timeframe}&limit=${period}&min_score=${minScore}`);
        document.getElementById("btLoading").style.display = "none";
        document.getElementById("btRunBtn").disabled = false;

        if (!data || data.error) {
            showToast(data?.error || "Backtest başarısız", "error");
            return;
        }

        renderBacktestResults(data);
    } catch (e) {
        document.getElementById("btLoading").style.display = "none";
        document.getElementById("btRunBtn").disabled = false;
        showToast("Backtest hatası: " + e.message, "error");
    }
}

function renderBacktestResults(data) {
    document.getElementById("btResults").style.display = "block";

    // Summary
    const wr = data.win_rate || 0;
    document.getElementById("btWinRate").textContent = `${wr.toFixed(1)}%`;
    document.getElementById("btWinRate").style.color = wr >= 50 ? "var(--green)" : "var(--red)";

    document.getElementById("btTotalTrades").textContent = data.total_trades || 0;

    const pnl = data.total_pnl || 0;
    document.getElementById("btTotalPnl").textContent = `${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}%`;
    document.getElementById("btTotalPnl").style.color = pnl >= 0 ? "var(--green)" : "var(--red)";

    document.getElementById("btAvgRR").textContent = `1:${(data.avg_rr || 0).toFixed(1)}`;

    const best = data.best_trade || 0;
    document.getElementById("btBestTrade").textContent = `+${best.toFixed(2)}%`;
    document.getElementById("btBestTrade").style.color = "var(--green)";

    const worst = data.worst_trade || 0;
    document.getElementById("btWorstTrade").textContent = `${worst.toFixed(2)}%`;
    document.getElementById("btWorstTrade").style.color = "var(--red)";

    // Win/Loss bar
    const wins = data.wins || 0;
    const losses = data.losses || 0;
    const total = wins + losses || 1;
    document.getElementById("btWinFill").style.width = `${(wins / total) * 100}%`;
    document.getElementById("btLossFill").style.width = `${(losses / total) * 100}%`;
    document.getElementById("btWinCount").textContent = `${wins} Kazanç`;
    document.getElementById("btLossCount").textContent = `${losses} Kayıp`;

    // Equity curve bars
    const equityEl = document.getElementById("btEquityBars");
    if (data.equity_curve && data.equity_curve.length > 0) {
        const maxAbs = Math.max(...data.equity_curve.map(e => Math.abs(e)));
        equityEl.innerHTML = data.equity_curve.map((val, i) => {
            const h = Math.max((Math.abs(val) / (maxAbs || 1)) * 100, 3);
            const cls = val >= 0 ? "eq-bar-win" : "eq-bar-loss";
            return `<div class="eq-bar ${cls}" style="height:${h}%" title="İşlem ${i + 1}: ${val >= 0 ? '+' : ''}${val.toFixed(2)}%"></div>`;
        }).join("");
    } else {
        equityEl.innerHTML = "<p style='color:var(--text-muted);text-align:center;padding:20px'>İşlem yok</p>";
    }

    // Trades table
    const tbody = document.getElementById("btTradesBody");
    if (data.trades && data.trades.length > 0) {
        tbody.innerHTML = data.trades.map((t, i) => {
            const dirIcon = t.direction === "LONG" ? "↑" : "↓";
            const dirClass = t.direction === "LONG" ? "bull" : "bear";
            const resultIcon = t.result === "WIN" ? "✅" : "❌";
            const pnlClass = t.pnl >= 0 ? "bull" : "bear";
            return `<tr>
                <td>${i + 1}</td>
                <td class="${dirClass}">${dirIcon} ${t.direction}</td>
                <td>${t.entry_price}</td>
                <td>${t.exit_price}</td>
                <td>${t.sl_price}</td>
                <td>${t.tp_price}</td>
                <td>${resultIcon} ${t.result}</td>
                <td class="${pnlClass}">${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}%</td>
                <td>${t.score}</td>
            </tr>`;
        }).join("");
    } else {
        tbody.innerHTML = "<tr><td colspan='9' style='text-align:center;color:var(--text-muted)'>Bu ayarlarla işlem bulunamadı</td></tr>";
    }

    // Analysis
    const analysisEl = document.getElementById("btAnalysis");
    if (data.analysis && data.analysis.length > 0) {
        analysisEl.innerHTML = data.analysis.map(line => `<div class="bt-analysis-line">${line}</div>`).join("");
    }
}

// =================== FOREX & ALTIN ICT ===================

let forexCurrentTf = "1h";
let forexScanData = null;
let forexDetailData = null;

async function loadForexKillZones() {
    const data = await apiFetch("/api/forex/kill-zones");
    if (!data) return;

    const bar = document.getElementById("fxKillZoneBar");
    bar.classList.toggle("active", data.is_kill_zone);
    document.getElementById("kzLabel").textContent = data.is_kill_zone ? `${data.active_zone} Kill Zone Aktif` : "Kill Zone Dışında";
    document.getElementById("kzDesc").textContent = data.desc;

    const zonesEl = document.getElementById("kzZones");
    zonesEl.innerHTML = data.zones.map(z =>
        `<span class="kz-zone-pill ${z.active ? 'active' : ''}">${z.name.split(' ')[0]} ${z.hours}</span>`
    ).join("");
}

function switchForexTf(tf) {
    forexCurrentTf = tf;
    document.querySelectorAll(".fx-tf-btn").forEach(b => b.classList.toggle("active", b.dataset.fxtf === tf));
    // If we have data, rescan
    if (forexScanData) scanForex();
}

async function scanForex() {
    const grid = document.getElementById("fxSignalsGrid");
    const loading = document.getElementById("fxLoading");
    const emptyState = document.getElementById("fxEmptyState");
    const scanBtn = document.getElementById("fxScanBtn");

    if (emptyState) emptyState.style.display = "none";
    loading.style.display = "flex";
    scanBtn.classList.add("scanning");
    grid.innerHTML = "";
    grid.appendChild(loading);

    // Also refresh kill zones
    loadForexKillZones();

    const data = await apiFetch(`/api/forex/scan?tf=${forexCurrentTf}`);
    loading.style.display = "none";
    scanBtn.classList.remove("scanning");

    if (!data || !data.results || data.results.length === 0) {
        grid.innerHTML = `<div class="fx-empty-state"><i class="fas fa-exclamation-triangle"></i><h3>Veri alınamadı</h3><p>Forex verileri şu an erişilemez durumda. Lütfen tekrar deneyin.</p></div>`;
        return;
    }

    forexScanData = data.results;

    // Sort: strong signals first
    const signalOrder = { "STRONG_LONG": 0, "STRONG_SHORT": 1, "LONG": 2, "SHORT": 3, "WAIT": 4 };
    data.results.sort((a, b) => (signalOrder[a.signal] || 9) - (signalOrder[b.signal] || 9));

    grid.innerHTML = data.results.map(r => renderForexCard(r)).join("");
}

function renderForexCard(r) {
    const totalScore = r.bull_score + r.bear_score || 1;
    const bullPct = (r.bull_score / totalScore * 100).toFixed(0);
    const bearPct = (r.bear_score / totalScore * 100).toFixed(0);

    const tags = [];
    if (r.market_structure.trend !== "NEUTRAL") {
        const trendCls = r.market_structure.trend === "BULLISH" ? "bull" : "bear";
        tags.push(`<span class="fx-card-tag ${trendCls}">${r.market_structure.trend === "BULLISH" ? "↑ Yükseliş" : "↓ Düşüş"}</span>`);
    }
    if (r.market_structure.choch) {
        const chCls = r.market_structure.choch.type.includes("BULL") ? "bull" : "bear";
        tags.push(`<span class="fx-card-tag ${chCls}">CHoCH</span>`);
    }
    if (r.market_structure.bos.length > 0) {
        tags.push(`<span class="fx-card-tag ${r.market_structure.bos[0].type.includes('BULL') ? 'bull' : 'bear'}">BOS</span>`);
    }
    if (r.displacement && r.displacement.length > 0) {
        const lastD = r.displacement[r.displacement.length - 1];
        tags.push(`<span class="fx-card-tag ${lastD.type.includes('BULL') ? 'bull' : 'bear'}">DISP</span>`);
    }
    if (r.fvg && (r.fvg.bull > 0 || r.fvg.bear > 0)) {
        const fvgDir = r.fvg.bull > r.fvg.bear ? "bull" : "bear";
        tags.push(`<span class="fx-card-tag ${fvgDir}">FVG</span>`);
    }
    if (r.premium_discount.zone === "DISCOUNT") {
        tags.push(`<span class="fx-card-tag bull">Discount</span>`);
    } else if (r.premium_discount.zone === "PREMIUM") {
        tags.push(`<span class="fx-card-tag bear">Premium</span>`);
    }
    if (r.ote) {
        tags.push(`<span class="fx-card-tag ${r.ote.direction === 'LONG' ? 'bull' : 'bear'}">OTE</span>`);
    }
    if (r.amd) {
        tags.push(`<span class="fx-card-tag ${r.amd.direction === 'LONG' ? 'bull' : 'bear'}">AMD</span>`);
    }
    if (r.judas) {
        tags.push(`<span class="fx-card-tag ${r.judas.type.includes('BULL') ? 'bull' : 'bear'}">Judas</span>`);
    }
    if (r.smart_money_trap) {
        tags.push(`<span class="fx-card-tag ${r.smart_money_trap.type === 'BEAR_TRAP' ? 'bull' : 'bear'}">SMT</span>`);
    }

    const fmtPrice = (v) => {
        if (v >= 100) return v.toFixed(2);
        if (v >= 1) return v.toFixed(4);
        return v.toFixed(5);
    };

    return `
    <div class="fx-signal-card signal-${r.signal}" onclick="openForexDetail('${r.instrument}')">
        <div class="fx-card-top">
            <div class="fx-card-info">
                <div class="fx-card-icon">${r.icon}</div>
                <div>
                    <div class="fx-card-name">${r.name}</div>
                    <div class="fx-card-desc">${r.desc}</div>
                </div>
            </div>
            <span class="fx-card-signal ${r.signal}">${r.label}</span>
        </div>
        <div class="fx-card-price">${fmtPrice(r.price)}</div>
        <div class="fx-card-details">${tags.join("")}</div>
        <div class="fx-card-score">
            <div class="fx-card-score-bear" style="width:${bearPct}%"></div>
            <div class="fx-card-score-bull" style="width:${bullPct}%"></div>
        </div>
        <div class="fx-card-footer">
            <span>RSI: ${r.indicators.rsi.toFixed(1)}</span>
            <span>Net: ${r.net_score > 0 ? '+' : ''}${r.net_score}</span>
            <span>Conf: ${Math.max(r.confluence_bull||0, r.confluence_bear||0)}</span>
        </div>
    </div>`;
}

async function openForexDetail(instrument) {
    const grid = document.getElementById("fxSignalsGrid");
    const panel = document.getElementById("fxDetailPanel");
    const controls = document.querySelector(".fx-controls");
    const killBar = document.getElementById("fxKillZoneBar");

    grid.style.display = "none";
    controls.style.display = "none";
    killBar.style.display = "none";
    panel.style.display = "block";

    // Fetch fresh data
    const data = await apiFetch(`/api/forex/signal/${instrument}?tf=${forexCurrentTf}`);
    if (!data || data.error) {
        panel.innerHTML = `<p style="color:var(--red)">Veri alınamadı: ${data?.error || "Bilinmeyen hata"}</p>`;
        return;
    }

    forexDetailData = data;

    const fmtPrice = (v) => {
        if (v >= 100) return v.toFixed(2);
        if (v >= 1) return v.toFixed(4);
        return v.toFixed(5);
    };

    // Header
    document.getElementById("fxDetIcon").textContent = data.icon;
    document.getElementById("fxDetName").textContent = data.name;
    document.getElementById("fxDetPrice").textContent = fmtPrice(data.price);

    const sigBadge = document.getElementById("fxDetSignalBadge");
    sigBadge.textContent = data.label;
    sigBadge.className = `fx-detail-signal fx-card-signal ${data.signal}`;

    // Verdict
    document.getElementById("fxDetVerdictLabel").textContent = data.label;
    document.getElementById("fxDetVerdictDesc").textContent = data.description;

    const totalScore = data.bull_score + data.bear_score || 1;
    document.getElementById("fxScoreBear").style.width = `${data.bear_score / totalScore * 100}%`;
    document.getElementById("fxScoreBull").style.width = `${data.bull_score / totalScore * 100}%`;
    document.getElementById("fxScoreBearVal").textContent = data.bear_score;
    document.getElementById("fxScoreBullVal").textContent = data.bull_score;

    // SL / TP
    const slTpSection = document.getElementById("fxSlTpSection");
    const levelsGrid = document.getElementById("fxLevelsGrid");
    if (data.sl_tp) {
        slTpSection.style.display = "block";
        levelsGrid.innerHTML = `
            <div class="fx-level-item">
                <div class="fx-level-label entry">Giriş</div>
                <div class="fx-level-val">${fmtPrice(data.price)}</div>
            </div>
            <div class="fx-level-item">
                <div class="fx-level-label sl">Stop Loss</div>
                <div class="fx-level-val">${fmtPrice(data.sl_tp.sl)}</div>
            </div>
            <div class="fx-level-item">
                <div class="fx-level-label tp">Take Profit 1</div>
                <div class="fx-level-val">${fmtPrice(data.sl_tp.tp1)}</div>
            </div>
            <div class="fx-level-item">
                <div class="fx-level-label tp">Take Profit 2</div>
                <div class="fx-level-val">${fmtPrice(data.sl_tp.tp2)}</div>
            </div>
        `;
    } else {
        slTpSection.style.display = "none";
    }

    // Market Structure
    const msCard = document.getElementById("fxMsCard");
    const ms = data.market_structure;
    const trendIcon = ms.trend === "BULLISH" ? "fa-arrow-trend-up" : ms.trend === "BEARISH" ? "fa-arrow-trend-down" : "fa-arrows-left-right";
    const trendColor = ms.trend === "BULLISH" ? "var(--green)" : ms.trend === "BEARISH" ? "var(--red)" : "var(--yellow)";
    let msHtml = `
        <div class="fx-ms-row">
            <i class="fas ${trendIcon}" style="color:${trendColor}"></i>
            <span class="label">Piyasa Yapısı</span>
            <span class="value" style="color:${trendColor}">${ms.trend === "BULLISH" ? "Yükseliş Trendi (HH+HL)" : ms.trend === "BEARISH" ? "Düşüş Trendi (LH+LL)" : "Nötr / Kararsız"}</span>
        </div>`;

    for (const bos of ms.bos) {
        const bosColor = bos.type.includes("BULL") ? "var(--green)" : "var(--red)";
        msHtml += `<div class="fx-ms-row"><i class="fas fa-bolt" style="color:${bosColor}"></i><span class="label">Break of Structure</span><span class="value" style="color:${bosColor}">${bos.type.replace("_", " ")} @ ${fmtPrice(bos.level)}</span></div>`;
    }
    if (ms.choch) {
        const chochColor = ms.choch.type.includes("BULL") ? "var(--green)" : "var(--red)";
        msHtml += `<div class="fx-ms-row"><i class="fas fa-rotate" style="color:${chochColor}"></i><span class="label">Change of Character</span><span class="value" style="color:${chochColor}">${ms.choch.desc}</span></div>`;
    }

    if (ms.swing_highs && ms.swing_highs.length > 0) {
        const lastSH = ms.swing_highs[ms.swing_highs.length - 1];
        msHtml += `<div class="fx-ms-row"><i class="fas fa-arrow-up" style="color:var(--green)"></i><span class="label">Son Swing High</span><span class="value">${fmtPrice(lastSH.price)}</span></div>`;
    }
    if (ms.swing_lows && ms.swing_lows.length > 0) {
        const lastSL = ms.swing_lows[ms.swing_lows.length - 1];
        msHtml += `<div class="fx-ms-row"><i class="fas fa-arrow-down" style="color:var(--red)"></i><span class="label">Son Swing Low</span><span class="value">${fmtPrice(lastSL.price)}</span></div>`;
    }
    msCard.innerHTML = msHtml;

    // Order Blocks
    const obList = document.getElementById("fxObList");
    if (data.order_blocks && data.order_blocks.length > 0) {
        obList.innerHTML = data.order_blocks.map(ob => `
            <div class="fx-ob-item ${ob.type.includes('BULL') ? 'bull' : 'bear'}">
                <span class="ob-type">${ob.type.includes("BULL") ? "🟢 Bullish" : "🔴 Bearish"}</span>
                <span class="ob-range">${fmtPrice(ob.low)} — ${fmtPrice(ob.high)}</span>
                <span class="ob-str">Güç: ${ob.strength}x</span>
            </div>`).join("");
    } else {
        obList.innerHTML = `<div style="color:var(--text-muted);font-size:12px;padding:8px">Aktif Order Block bulunamadı</div>`;
    }

    // Breaker Blocks
    const breakerSection = document.getElementById("fxBreakerSection");
    const breakerList = document.getElementById("fxBreakerList");
    if (data.breaker_blocks && data.breaker_blocks.length > 0) {
        breakerSection.style.display = "block";
        breakerList.innerHTML = data.breaker_blocks.map(bb => `
            <div class="fx-ob-item ${bb.type.includes('BULL') ? 'bull' : 'bear'}">
                <span class="ob-type">${bb.type.includes("BULL") ? "🟢 Bullish Breaker" : "🔴 Bearish Breaker"}</span>
                <span class="ob-range">${fmtPrice(bb.low)} — ${fmtPrice(bb.high)}</span>
                <span class="ob-str" style="font-size:11px">${bb.desc}</span>
            </div>`).join("");
    } else {
        breakerSection.style.display = "none";
    }

    // FVG & CE
    const fvgCard = document.getElementById("fxFvgCard");
    let fvgHtml = "";
    if (data.fvg) {
        const fvg = data.fvg;
        if (fvg.bull > 0) fvgHtml += `<div class="fx-ms-row"><i class="fas fa-layer-group" style="color:var(--green)"></i><span class="label">Bullish FVG</span><span class="value" style="color:var(--green)">${fvg.bull} adet${fvg.ce_bull > 0 ? ' (' + fvg.ce_bull + ' CE test edildi)' : ''}</span></div>`;
        if (fvg.bear > 0) fvgHtml += `<div class="fx-ms-row"><i class="fas fa-layer-group" style="color:var(--red)"></i><span class="label">Bearish FVG</span><span class="value" style="color:var(--red)">${fvg.bear} adet${fvg.ce_bear > 0 ? ' (' + fvg.ce_bear + ' CE test edildi)' : ''}</span></div>`;
        if (fvg.active && fvg.active.length > 0) {
            for (const f of fvg.active.slice(0, 3)) {
                const fc = f.type.includes("BULL") ? "var(--green)" : "var(--red)";
                fvgHtml += `<div class="fx-ms-row"><i class="fas fa-arrow-right" style="color:${fc}"></i><span class="label">${f.type.replace("_", " ")}</span><span class="value">${fmtPrice(f.bottom)} — ${fmtPrice(f.top)} | CE: ${fmtPrice(f.ce_level)}</span></div>`;
            }
        }
    }
    if (!fvgHtml) fvgHtml = `<div style="color:var(--text-muted);font-size:12px;padding:8px">Aktif FVG bulunamadı</div>`;
    fvgCard.innerHTML = fvgHtml;

    // Displacement & Liquidity
    const dispLiqCard = document.getElementById("fxDispLiqCard");
    let dlHtml = "";
    if (data.displacement && data.displacement.length > 0) {
        for (const d of data.displacement) {
            const dc = d.type.includes("BULL") ? "var(--green)" : "var(--red)";
            dlHtml += `<div class="fx-ms-row"><i class="fas fa-bolt-lightning" style="color:${dc}"></i><span class="label">Displacement</span><span class="value" style="color:${dc}">${d.desc}</span></div>`;
        }
    }
    if (data.liquidity_sweeps && data.liquidity_sweeps.length > 0) {
        for (const s of data.liquidity_sweeps) {
            const sc = s.type === "BUY_SIDE_SWEEP" ? "var(--green)" : "var(--red)";
            dlHtml += `<div class="fx-ms-row"><i class="fas fa-water" style="color:${sc}"></i><span class="label">Likidite Avi</span><span class="value" style="color:${sc}">${s.desc}</span></div>`;
        }
    }
    if (data.inducement && data.inducement.length > 0) {
        for (const ind of data.inducement) {
            const ic = ind.type.includes("BULL") ? "var(--green)" : "var(--red)";
            dlHtml += `<div class="fx-ms-row"><i class="fas fa-magnet" style="color:${ic}"></i><span class="label">Inducement</span><span class="value" style="color:${ic}">${ind.desc}</span></div>`;
        }
    }
    if (data.smart_money_trap) {
        const stc = data.smart_money_trap.type === "BEAR_TRAP" ? "var(--green)" : "var(--red)";
        dlHtml += `<div class="fx-ms-row"><i class="fas fa-skull-crossbones" style="color:${stc}"></i><span class="label">Smart Money Trap</span><span class="value" style="color:${stc}">${data.smart_money_trap.desc}</span></div>`;
    }
    if (!dlHtml) dlHtml = `<div style="color:var(--text-muted);font-size:12px;padding:8px">Displacement/Likidite verisi yok</div>`;
    dispLiqCard.innerHTML = dlHtml;

    // Daily Bias & AMD
    const biasAmdCard = document.getElementById("fxBiasAmdCard");
    let baHtml = "";
    if (data.daily_bias) {
        const biasColor = data.daily_bias.bias === "BULLISH" ? "var(--green)" : data.daily_bias.bias === "BEARISH" ? "var(--red)" : "var(--yellow)";
        baHtml += `<div class="fx-ms-row"><i class="fas fa-compass" style="color:${biasColor}"></i><span class="label">Daily Bias</span><span class="value" style="color:${biasColor}">${data.daily_bias.desc}</span></div>`;
    }
    if (data.amd) {
        const amdColor = data.amd.direction === "LONG" ? "var(--green)" : "var(--red)";
        baHtml += `<div class="fx-ms-row"><i class="fas fa-recycle" style="color:${amdColor}"></i><span class="label">Power of 3 (AMD)</span><span class="value" style="color:${amdColor}">${data.amd.desc}</span></div>`;
    }
    if (data.judas) {
        const jColor = data.judas.type.includes("BULL") ? "var(--green)" : "var(--red)";
        baHtml += `<div class="fx-ms-row"><i class="fas fa-masks-theater" style="color:${jColor}"></i><span class="label">Judas Swing</span><span class="value" style="color:${jColor}">${data.judas.desc}</span></div>`;
    }
    if (data.asian_breakout) {
        const abColor = data.asian_breakout.type === "BULLISH_BREAKOUT" ? "var(--green)" : data.asian_breakout.type === "BEARISH_BREAKOUT" ? "var(--red)" : "var(--yellow)";
        baHtml += `<div class="fx-ms-row"><i class="fas fa-sunrise" style="color:${abColor}"></i><span class="label">Asian Range</span><span class="value" style="color:${abColor}">${data.asian_breakout.desc}</span></div>`;
    }
    if (data.kill_zones && data.kill_zones.is_kill_zone) {
        baHtml += `<div class="fx-ms-row"><i class="fas fa-clock" style="color:var(--yellow)"></i><span class="label">Kill Zone</span><span class="value" style="color:var(--yellow)">${data.kill_zones.desc}</span></div>`;
    }
    if (data.silver_bullet && data.silver_bullet.is_active) {
        baHtml += `<div class="fx-ms-row"><i class="fas fa-bullseye" style="color:var(--yellow)"></i><span class="label">Silver Bullet</span><span class="value" style="color:var(--yellow)">${data.silver_bullet.desc}</span></div>`;
    }
    if (!baHtml) baHtml = `<div style="color:var(--text-muted);font-size:12px;padding:8px">Aktif bias/pattern verisi yok</div>`;
    biasAmdCard.innerHTML = baHtml;

    // Premium/Discount
    const pdCard = document.getElementById("fxPdCard");
    const pd = data.premium_discount;
    const pdPct = pd.zone === "PREMIUM" ? 50 + pd.zone_pct / 2 : 50 - pd.zone_pct / 2;
    const pdColor = pd.zone === "DISCOUNT" ? "var(--green)" : "var(--red)";
    let pdHtml = `
        <div class="fx-pd-meter"><div class="fx-pd-pointer" style="left:${pdPct}%"></div></div>
        <div class="fx-pd-labels"><span style="color:var(--green)">Discount</span><span>Equilibrium: ${fmtPrice(pd.equilibrium)}</span><span style="color:var(--red)">Premium</span></div>
        <div class="fx-ms-row"><i class="fas fa-percentage" style="color:${pdColor}"></i><span class="label">Bölge</span><span class="value" style="color:${pdColor}">${pd.zone} (%${pd.zone_pct})</span></div>
        <div class="fx-ms-row"><i class="fas fa-arrows-up-down"></i><span class="label">Range</span><span class="value">${fmtPrice(pd.range_low)} — ${fmtPrice(pd.range_high)}</span></div>`;

    if (data.ote) {
        const oteColor = data.ote.direction === "LONG" ? "var(--green)" : "var(--red)";
        pdHtml += `<div class="fx-ms-row"><i class="fas fa-crosshairs" style="color:${oteColor}"></i><span class="label">OTE Bölgesi</span><span class="value" style="color:${oteColor}">${data.ote.desc}</span></div>`;
    }
    pdCard.innerHTML = pdHtml;

    // Reasons
    const reasonsEl = document.getElementById("fxReasons");
    let reasonsHtml = "";
    for (const r of data.reasons_bull) {
        reasonsHtml += `<div class="fx-reason-item bull">${r}</div>`;
    }
    for (const r of data.reasons_bear) {
        reasonsHtml += `<div class="fx-reason-item bear">${r}</div>`;
    }
    if (!reasonsHtml) {
        reasonsHtml = `<div style="color:var(--text-muted);font-size:12px;padding:8px">Net bir sinyal gerekçesi yok</div>`;
    }
    reasonsEl.innerHTML = reasonsHtml;

    // Indicators
    const indEl = document.getElementById("fxIndicators");
    const ind = data.indicators;
    indEl.innerHTML = `
        <div class="fx-ind-item"><div class="fx-ind-label">RSI (14)</div><div class="fx-ind-val" style="color:${ind.rsi > 70 ? 'var(--red)' : ind.rsi < 30 ? 'var(--green)' : 'var(--text-primary)'}">${ind.rsi.toFixed(1)}</div></div>
        <div class="fx-ind-item"><div class="fx-ind-label">EMA 20</div><div class="fx-ind-val">${fmtPrice(ind.ema20)}</div></div>
        <div class="fx-ind-item"><div class="fx-ind-label">EMA 50</div><div class="fx-ind-val">${fmtPrice(ind.ema50)}</div></div>
        ${ind.ema200 ? `<div class="fx-ind-item"><div class="fx-ind-label">EMA 200</div><div class="fx-ind-val">${fmtPrice(ind.ema200)}</div></div>` : ''}
        <div class="fx-ind-item"><div class="fx-ind-label">ATR</div><div class="fx-ind-val">${fmtPrice(ind.atr)}</div></div>
        <div class="fx-ind-item"><div class="fx-ind-label">ATR %</div><div class="fx-ind-val">${ind.atr_pct}%</div></div>
    `;
}

function closeFxDetail() {
    document.getElementById("fxDetailPanel").style.display = "none";
    document.getElementById("fxSignalsGrid").style.display = "grid";
    document.querySelector(".fx-controls").style.display = "flex";
    document.getElementById("fxKillZoneBar").style.display = "flex";
}
