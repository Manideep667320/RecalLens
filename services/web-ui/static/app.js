/**
 * RecalLens — Context-Driven Personal Document Intelligence
 * Client-Side Application Engine (Double-Bezel UI, Real-Time SSE, Grounded Proof)
 */

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const clearSearchBtn = document.getElementById('clearSearchBtn');
const contentDiv = document.getElementById('content');
const queryExpansionCheckbox = document.getElementById('queryExpansion');
const rerankingCheckbox = document.getElementById('reranking');
const advancedToggle = document.getElementById('advancedToggle');
const advancedOptions = document.getElementById('advancedOptions');
const toggleIcon = document.getElementById('toggleIcon');
const recallensLink = document.getElementById('recallensLink') || document.getElementById('paperlessLink');
const paperlessLink = recallensLink;

// Sliders & Values
const recencyWeight = document.getElementById('recencyWeight');
const recencyWeightValue = document.getElementById('recencyWeightValue');
const recencyDecay = document.getElementById('recencyDecay');
const recencyDecayValue = document.getElementById('recencyDecayValue');
const hybridAlpha = document.getElementById('hybridAlpha');
const hybridAlphaValue = document.getElementById('hybridAlphaValue');
const rerankInfluence = document.getElementById('rerankInfluence');
const rerankInfluenceValue = document.getElementById('rerankInfluenceValue');
const retrievalTopK = document.getElementById('retrievalTopK');
const retrievalTopKValue = document.getElementById('retrievalTopKValue');
const topK = document.getElementById('topK');
const topKValue = document.getElementById('topKValue');

// Filter Elements
const yearFrom = document.getElementById('yearFrom');
const yearTo = document.getElementById('yearTo');
const tagsToggle = document.getElementById('tagsToggle');
const correspondentsToggle = document.getElementById('correspondentsToggle');
const tagsPanel = document.getElementById('tagsPanel');
const correspondentsPanel = document.getElementById('correspondentsPanel');
const clearFiltersBtn = document.getElementById('clearFilters');
const tagsList = document.getElementById('tagsList');
const correspondentsList = document.getElementById('correspondentsList');
const tagSearch = document.getElementById('tagSearch');
const correspondentSearch = document.getElementById('correspondentSearch');
const logicButtons = document.querySelectorAll('.logic-btn');

// State
let isSearching = false;
let recallensBaseUrl = 'http://localhost:8000';
let paperlessBaseUrl = recallensBaseUrl;
let allTags = [];
let allCorrespondents = [];
let selectedTags = new Set();
let selectedCorrespondent = null;
let tagLogic = 'any';
let currentQuery = '';
let currentQueryContext = null;
let progressiveResults = [];
let currentSearchResults = [];

// 1. Initial Configuration & Setup
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            const config = await response.json();
            recallensBaseUrl = config.recallens_url || config.paperless_url || recallensBaseUrl;
            paperlessBaseUrl = recallensBaseUrl;
            if (recallensLink) {
                recallensLink.href = recallensBaseUrl;
            }
        }
    } catch (error) {
        console.warn('Could not load backend config, using defaults:', error);
    }
}
loadConfig();

// Populate Year Selector Dropdowns
function populateYearDropdowns() {
    const currentYear = new Date().getFullYear();
    for (let year = currentYear + 1; year >= currentYear - 15; year--) {
        const optFrom = document.createElement('option');
        optFrom.value = year;
        optFrom.textContent = year;
        yearFrom.appendChild(optFrom);

        const optTo = document.createElement('option');
        optTo.value = year;
        optTo.textContent = year;
        yearTo.appendChild(optTo);
    }
}
populateYearDropdowns();

// 2. Preset Query Chips (Student & Everyday Records)
document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const queryText = chip.getAttribute('data-query');
        if (queryText) {
            searchInput.value = queryText;
            updateClearButtonVisibility();
            performSearch();
        }
    });
});

// Clear Search Input Button
searchInput.addEventListener('input', updateClearButtonVisibility);
function updateClearButtonVisibility() {
    if (searchInput.value.trim().length > 0) {
        clearSearchBtn.style.display = 'block';
    } else {
        clearSearchBtn.style.display = 'none';
    }
}
clearSearchBtn.addEventListener('click', () => {
    searchInput.value = '';
    updateClearButtonVisibility();
    searchInput.focus();
});

// Global Keyboard Shortcuts (/ to focus search, Esc to clear)
document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
    } else if (e.key === 'Escape' && document.activeElement === searchInput) {
        searchInput.value = '';
        updateClearButtonVisibility();
        searchInput.blur();
    }
});

// 3. Filter Drawer Toggles & Logic
tagsToggle.addEventListener('click', () => {
    tagsPanel.classList.toggle('collapsed');
    tagsToggle.classList.toggle('active');
    correspondentsPanel.classList.add('collapsed');
    correspondentsToggle.classList.remove('active');
});

correspondentsToggle.addEventListener('click', () => {
    correspondentsPanel.classList.toggle('collapsed');
    correspondentsToggle.classList.toggle('active');
    tagsPanel.classList.add('collapsed');
    tagsToggle.classList.remove('active');
});

logicButtons.forEach(button => {
    button.addEventListener('click', () => {
        tagLogic = button.getAttribute('data-logic') || 'any';
        logicButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    });
});

clearFiltersBtn.addEventListener('click', () => {
    yearFrom.value = '';
    yearTo.value = '';
    selectedTags.clear();
    selectedCorrespondent = null;
    renderTags(allTags);
    renderCorrespondents(allCorrespondents);
});

// Load Tags from API
async function loadTags() {
    try {
        const response = await fetch('/api/recallens/tags');
        if (response.ok) {
            const data = await response.json();
            allTags = data.tags || [];
            renderTags(allTags);
        }
    } catch (error) {
        tagsList.innerHTML = '<div class="loading-state">Could not load tags</div>';
    }
}

function renderTags(tags) {
    tagsList.innerHTML = '';
    if (!tags || tags.length === 0) {
        tagsList.innerHTML = '<div class="loading-state">No tags available</div>';
        return;
    }

    tags.forEach(tag => {
        const isSelected = selectedTags.has(tag.name);
        const chip = document.createElement('div');
        chip.className = `filter-chip-item ${isSelected ? 'selected' : ''}`;
        chip.innerHTML = `
            <span>${escapeHtml(tag.name)}</span>
            <span class="doc-count">${tag.document_count || 0}</span>
        `;
        chip.addEventListener('click', () => {
            if (selectedTags.has(tag.name)) {
                selectedTags.delete(tag.name);
                chip.classList.remove('selected');
            } else {
                selectedTags.add(tag.name);
                chip.classList.add('selected');
            }
        });
        tagsList.appendChild(chip);
    });
}

tagSearch.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = allTags.filter(t => t.name.toLowerCase().includes(term));
    renderTags(filtered);
});

// Load Correspondents
async function loadCorrespondents() {
    try {
        const response = await fetch('/api/recallens/correspondents');
        if (response.ok) {
            const data = await response.json();
            allCorrespondents = data.correspondents || [];
            renderCorrespondents(allCorrespondents);
        }
    } catch (error) {
        correspondentsList.innerHTML = '<div class="loading-state">Could not load correspondents</div>';
    }
}

function renderCorrespondents(corrs) {
    correspondentsList.innerHTML = '';
    if (!corrs || corrs.length === 0) {
        correspondentsList.innerHTML = '<div class="loading-state">No correspondents available</div>';
        return;
    }

    corrs.forEach(corr => {
        const isSelected = selectedCorrespondent === corr.name;
        const chip = document.createElement('div');
        chip.className = `filter-chip-item ${isSelected ? 'selected' : ''}`;
        chip.innerHTML = `
            <span>${escapeHtml(corr.name)}</span>
            <span class="doc-count">${corr.document_count || 0}</span>
        `;
        chip.addEventListener('click', () => {
            if (selectedCorrespondent === corr.name) {
                selectedCorrespondent = null;
                chip.classList.remove('selected');
            } else {
                selectedCorrespondent = corr.name;
                renderCorrespondents(allCorrespondents);
            }
        });
        correspondentsList.appendChild(chip);
    });
}

correspondentSearch.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = allCorrespondents.filter(c => c.name.toLowerCase().includes(term));
    renderCorrespondents(filtered);
});

loadTags();
loadCorrespondents();

// Advanced Accordion Toggle
advancedToggle.addEventListener('click', () => {
    advancedOptions.classList.toggle('show');
    toggleIcon.textContent = advancedOptions.classList.contains('show') ? '▼' : '▶';
});

// Slider updates
recencyWeight.addEventListener('input', (e) => recencyWeightValue.textContent = e.target.value);
recencyDecay.addEventListener('input', (e) => recencyDecayValue.textContent = `${e.target.value}d`);
hybridAlpha.addEventListener('input', (e) => hybridAlphaValue.textContent = e.target.value);
rerankInfluence.addEventListener('input', (e) => rerankInfluenceValue.textContent = e.target.value);
retrievalTopK.addEventListener('input', (e) => retrievalTopKValue.textContent = e.target.value);
topK.addEventListener('input', (e) => topKValue.textContent = e.target.value);

// Search Triggers
searchButton.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// 4. Search Execution Engine
function buildFilters() {
    const filters = {};
    if (yearFrom.value || yearTo.value) {
        filters.created_date = {};
        if (yearFrom.value) filters.created_date.gte = `${yearFrom.value}-01-01`;
        if (yearTo.value) filters.created_date.lte = `${yearTo.value}-12-31`;
    }
    if (selectedTags.size > 0) {
        filters.tags = Array.from(selectedTags);
        filters.tag_logic = tagLogic;
    }
    if (selectedCorrespondent !== null) {
        filters.correspondent_name = selectedCorrespondent;
    }
    return Object.keys(filters).length > 0 ? filters : null;
}

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query || isSearching) return;

    isSearching = true;
    currentQuery = query;
    currentQueryContext = null;
    progressiveResults = [];
    currentSearchResults = [];

    searchButton.disabled = true;
    searchButton.querySelector('.btn-text').textContent = 'Searching...';

    // Show modern 3-stage progress deck
    contentDiv.innerHTML = `
        <div class="progress-container">
            <div class="progress-stage" id="progress-query">
                <div class="progress-stage-header">
                    <span class="progress-stage-name">
                        <span class="progress-icon">✨</span>
                        Understanding Context & Expanding Query
                    </span>
                    <span class="progress-stage-status">Analyzing entities & dates...</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill indeterminate"></div>
                </div>
            </div>
            <div class="progress-stage" id="progress-retrieval">
                <div class="progress-stage-header">
                    <span class="progress-stage-name">
                        <span class="progress-icon">⚡</span>
                        Dual-Stream Retrieval (Qdrant Vectors + Structured Database)
                    </span>
                    <span class="progress-stage-status">Waiting...</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="progress-stage" id="progress-reranking">
                <div class="progress-stage-header">
                    <span class="progress-stage-name">
                        <span class="progress-icon">🎯</span>
                        Multi-Signal Reranking (S = w_s·S_s + w_e·S_e + w_v·S_v + w_t·S_t + w_r·S_r)
                    </span>
                    <span class="progress-stage-status">Waiting...</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>
    `;

    try {
        const filters = buildFilters();
        const searchPayload = {
            query: query,
            enable_query_expansion: queryExpansionCheckbox.checked,
            enable_reranking: rerankingCheckbox.checked,
            top_k: parseInt(topK.value),
            retrieval_top_k: parseInt(retrievalTopK.value),
            hybrid_alpha: parseFloat(hybridAlpha.value),
            rerank_influence: parseFloat(rerankInfluence.value),
            recency_weight: parseFloat(recencyWeight.value),
            recency_decay_days: parseInt(recencyDecay.value)
        };

        if (filters) {
            searchPayload.filters = filters;
        }

        const response = await fetch('/api/search/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchPayload)
        });

        if (!response.ok) {
            throw new Error(`Search request failed (HTTP ${response.status})`);
        }

        // Process SSE Stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    handleProgressUpdate(data);
                }
            }
        }

    } catch (error) {
        contentDiv.innerHTML = `
            <div class="no-results-card">
                <div class="no-results-emoji">⚠️</div>
                <h3>Search Failed</h3>
                <p>${escapeHtml(error.message)}</p>
            </div>
        `;
    } finally {
        isSearching = false;
        searchButton.disabled = false;
        searchButton.querySelector('.btn-text').textContent = 'Find Document';
    }
}

// 5. SSE Streaming Handler
function handleProgressUpdate(data) {
    if (data.type === 'progress') {
        requestAnimationFrame(() => {
            updateProgressStage(data.stage, data.message, data.current, data.total, data.done);
        });
    } else if (data.type === 'initial_results') {
        currentQuery = data.data.query;
        currentQueryContext = data.data.query_context || null;
        progressiveResults = data.data.results || [];
        currentSearchResults = [...progressiveResults];
        renderResults(data.data.query, data.data.expanded_query, data.data.retrieval_time_ms);
    } else if (data.type === 'result_update') {
        updateResult(data);
    } else if (data.type === 'highlights') {
        const highlightsMap = data.highlights || {};
        for (const [docId, highlights] of Object.entries(highlightsMap)) {
            const item = progressiveResults.find(r => (r.recallens_id || r.paperless_id) == docId);
            if (item) {
                item.highlights = highlights;
            }
        }
        renderResults(currentQuery, null, null);
    } else if (data.type === 'done') {
        const progressEl = document.querySelector('.progress-container');
        if (progressEl) {
            progressEl.style.display = 'none';
        }
    } else if (data.type === 'error') {
        contentDiv.innerHTML = `
            <div class="no-results-card">
                <div class="no-results-emoji">⚠️</div>
                <h3>Error During Search</h3>
                <p>${escapeHtml(data.message)}</p>
            </div>
        `;
    }
}

function updateResult(data) {
    const updateId = data.recallens_id || data.paperless_id;
    const result = progressiveResults.find(r => (r.recallens_id || r.paperless_id) === updateId);
    if (!result) return;

    result.rerank_score = data.rerank_score;
    result.final_score = data.final_score;
    result.recency_boost = data.recency_boost;
    result.context_score = data.context_score;
    result.matched_entities = data.matched_entities || [];
    result.matched_event = data.matched_event;
    result.matched_relationship = data.matched_relationship;
    result.evidence = data.evidence;

    const currentResult = currentSearchResults.find(r => (r.recallens_id || r.paperless_id) === updateId);
    if (currentResult) {
        Object.assign(currentResult, result);
    }

    renderResults(currentQuery, null, null);
}

function updateProgressStage(stage, message, current, total, done) {
    const stageMap = {
        'query_expansion': 'progress-query',
        'retrieval': 'progress-retrieval',
        'reranking': 'progress-reranking',
        'finalize': 'progress-reranking'
    };

    const stageId = stageMap[stage];
    if (!stageId) return;

    const stageEl = document.getElementById(stageId);
    if (!stageEl) return;

    const statusEl = stageEl.querySelector('.progress-stage-status');
    const fillEl = stageEl.querySelector('.progress-bar-fill');
    const iconEl = stageEl.querySelector('.progress-icon');

    if (done) {
        statusEl.textContent = 'Completed';
        fillEl.style.width = '100%';
        fillEl.classList.remove('indeterminate');
        if (iconEl) iconEl.textContent = '✅';
    } else if (current !== undefined && total) {
        const percent = Math.min(100, Math.round((current / total) * 100));
        statusEl.textContent = message || `${current} / ${total}`;
        fillEl.style.width = `${percent}%`;
        fillEl.classList.remove('indeterminate');
    } else {
        statusEl.textContent = message || 'Processing...';
        fillEl.classList.add('indeterminate');
    }
}

// 6. Modern Result Cards Rendering
function renderResults(query, expandedQuery, retrievalTimeMs) {
    // Sort by final_score (calibrated multi-signal score)
    progressiveResults.sort((a, b) => (b.final_score || b.score) - (a.final_score || a.score));

    const currentTopK = parseInt(topK.value) || 10;
    const displayResults = progressiveResults.slice(0, currentTopK);

    let html = '';

    // Render Context Understanding Banner (The "Brain" of RecalLens)
    if (currentQueryContext) {
        html += renderContextBanner(currentQueryContext);
    }

    // Results Header Bar
    if (progressiveResults.length > 0) {
        const timeText = retrievalTimeMs ? `in ${(retrievalTimeMs / 1000).toFixed(2)}s` : '';
        html += `
            <div class="results-header-bar">
                <div class="results-count-text">
                    Showing <span class="results-count-highlight">${displayResults.length}</span> of ${progressiveResults.length} relevant documents ${timeText}
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="no-results-card">
                <div class="no-results-emoji">🔍</div>
                <h3>No documents matched your query</h3>
                <p>Try searching with broader terms, or adjusting your entity or year filters above.</p>
            </div>
        `;
        replaceResultsHTML(html);
        return;
    }

    // Results Stack
    html += '<div class="results-stack">';

    displayResults.forEach((result, index) => {
        const activeDocId = result.recallens_id || result.paperless_id;
        const documentUrl = `${recallensBaseUrl}/documents/${activeDocId}`;
        const downloadUrl = `${recallensBaseUrl}/api/documents/${activeDocId}/download/`;
        const finalScore = result.final_score !== undefined && result.final_score !== null ? result.final_score : result.score;
        const matchPercent = Math.min(100, Math.max(10, Math.round(finalScore * 100)));

        // Determine gauge tier
        let gaugeClass = 'gauge-high';
        if (matchPercent < 70) gaugeClass = 'gauge-low';
        else if (matchPercent < 85) gaugeClass = 'gauge-mid';

        // Document Category Pill
        const docCategory = inferCategory(result);

        // Grounded Evidence Text
        const evidenceText = result.evidence || buildClientEvidence(result);

        html += `
            <article class="result-card-outer" data-recallens-id="${activeDocId}" data-paperless-id="${activeDocId}">
                <div class="result-card-inner">
                    <!-- Top Row: Category, Page & Context Gauge -->
                    <div class="card-top-row">
                        <div class="card-badge-cluster">
                            <span class="doc-type-badge ${docCategory.cssClass}">
                                <span>${docCategory.icon}</span>
                                <span>${escapeHtml(docCategory.label)}</span>
                            </span>
                            <span class="page-indicator-badge">
                                📄 Page ${result.page_number || 1}
                            </span>
                        </div>

                        <div class="context-match-gauge ${gaugeClass}" title="Multi-signal context score based on entity, event, date proximity & semantic match">
                            <span class="gauge-bar-track">
                                <span class="gauge-bar-fill" style="width: ${matchPercent}%"></span>
                            </span>
                            <span>${matchPercent}% Context Match</span>
                        </div>
                    </div>

                    <!-- Document Title -->
                    <a href="${paperlessUrl}" target="_blank" class="card-doc-title">
                        <span class="title-rank-num">#${index + 1}</span>
                        <span>${escapeHtml(result.title)}</span>
                        <span class="title-external-icon">↗</span>
                    </a>

                    <!-- Metadata Row -->
                    <div class="card-meta-row">
                        ${result.correspondent_name ? `
                            <span class="meta-chip-item">
                                <span>👤</span>
                                <strong>${escapeHtml(result.correspondent_name)}</strong>
                            </span>
                        ` : ''}
                        ${result.created_date ? `
                            <span class="meta-chip-item">
                                <span>📅</span>
                                <span>${formatDate(result.created_date)}</span>
                            </span>
                        ` : ''}
                        ${result.recency_boost && result.recency_boost > 0 ? `
                            <span class="meta-chip-item" style="color: var(--emerald-text);">
                                <span>⚡ Recency +${(result.recency_boost * 100).toFixed(0)}%</span>
                            </span>
                        ` : ''}
                    </div>

                    <!-- Grounded Evidence Box (Why this matched) -->
                    ${evidenceText ? `
                        <div class="grounded-evidence-box">
                            <div class="evidence-sparkle-icon">✨</div>
                            <div class="evidence-content">
                                <div class="evidence-label">Grounded Context Proof</div>
                                <div class="evidence-text">${escapeHtml(evidenceText)}</div>
                            </div>
                        </div>
                    ` : ''}

                    <!-- Highlights & Content Excerpts -->
                    ${result.highlights && result.highlights.length > 0 ? `
                        <div class="card-highlights">
                            ${result.highlights.map(h => `
                                <div class="highlight-pill-snippet">...${formatHighlightText(h)}...</div>
                            `).join('')}
                        </div>
                    ` : (result.content ? `
                        <div class="card-highlights">
                            <div class="highlight-pill-snippet">${escapeHtml(result.content.slice(0, 240))}...</div>
                        </div>
                    ` : '')}

                    <!-- Document Tags -->
                    ${result.tags && result.tags.length > 0 ? `
                        <div class="card-tags-row">
                            ${result.tags.map(t => `<span class="card-tag-pill">#${escapeHtml(t)}</span>`).join('')}
                        </div>
                    ` : ''}

                    <!-- Actions Bar -->
                    <div class="card-actions-bar">
                        <a href="${documentUrl}" target="_blank" class="btn-action-primary">
                            <span>📄 View in RecalLens</span>
                            <span>↗</span>
                        </a>
                        <a href="${downloadUrl}" target="_blank" class="btn-action-secondary">
                            <span>⬇️ Download PDF</span>
                        </a>
                        <button class="btn-action-secondary" onclick="toggleExplanation(${activeDocId}, event)">
                            <span>💡 Explain Match</span>
                        </button>
                        <button class="btn-action-secondary" onclick="copySnippet(${activeDocId}, this)">
                            <span>📋 Copy Text</span>
                        </button>
                        <button class="btn-action-secondary" onclick="toggleBreakdown(${activeDocId})">
                            <span>⚙️ Score Details</span>
                        </button>
                    </div>

                    <!-- Interactive AI Explanation Drawer -->
                    <div class="drawer-explanation" id="explanation-${activeDocId}">
                        <div class="drawer-header-row">AI Relevance Justification</div>
                        <div class="explanation-body-text" id="explanation-text-${activeDocId}">
                            Loading intelligent rationale...
                        </div>
                    </div>

                    <!-- Technical Score Breakdown Drawer -->
                    <div class="drawer-breakdown" id="breakdown-${activeDocId}">
                        <div class="drawer-header-row">Multi-Signal Formula Breakdown</div>
                        <div class="breakdown-grid">
                            <div class="breakdown-stat-card">
                                <div class="breakdown-stat-name">Final Score</div>
                                <div class="breakdown-stat-val" style="color: var(--brand-primary);">${finalScore.toFixed(4)}</div>
                            </div>
                            <div class="breakdown-stat-card">
                                <div class="breakdown-stat-name">Initial Vector</div>
                                <div class="breakdown-stat-val">${result.score.toFixed(4)}</div>
                            </div>
                            <div class="breakdown-stat-card">
                                <div class="breakdown-stat-name">Rerank Score</div>
                                <div class="breakdown-stat-val">${result.rerank_score ? (result.rerank_score).toFixed(2) + '/10' : 'N/A'}</div>
                            </div>
                            <div class="breakdown-stat-card">
                                <div class="breakdown-stat-name">Context Score</div>
                                <div class="breakdown-stat-val">${result.context_score ? result.context_score.toFixed(4) : finalScore.toFixed(4)}</div>
                            </div>
                            <div class="breakdown-stat-card">
                                <div class="breakdown-stat-name">Chunk Index</div>
                                <div class="breakdown-stat-val">${result.chunk_index || 0}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </article>
        `;
    });

    html += '</div>';
    replaceResultsHTML(html);
}

// Render Context Understanding Banner
function renderContextBanner(ctx) {
    const slots = [];
    if (ctx.item || ctx.entity || ctx.person) {
        const entVal = ctx.item || ctx.entity || ctx.person;
        slots.push(`<div class="context-slot-pill slot-entity"><span class="slot-key">Entity</span> <strong>${escapeHtml(entVal)}</strong></div>`);
    }
    if (ctx.event) {
        slots.push(`<div class="context-slot-pill slot-event"><span class="slot-key">Event</span> <strong>${escapeHtml(ctx.event)}</strong></div>`);
    }
    if (ctx.resolved_date_start || ctx.target_date || ctx.time_reference) {
        let dateText = ctx.time_reference || '';
        if (ctx.resolved_date_start && ctx.resolved_date_end) {
            dateText = `${ctx.resolved_date_start} → ${ctx.resolved_date_end}`;
        } else if (ctx.target_date) {
            dateText = ctx.target_date;
        }
        slots.push(`<div class="context-slot-pill slot-temporal"><span class="slot-key">Time</span> <strong>${escapeHtml(dateText)}</strong></div>`);
    }
    if (ctx.relationship) {
        slots.push(`<div class="context-slot-pill slot-relation"><span class="slot-key">Relationship</span> <strong>${escapeHtml(ctx.relationship)}</strong></div>`);
    }

    if (slots.length === 0) return '';

    return `
        <div class="context-understanding-banner">
            <div class="context-banner-header">
                <div class="context-banner-title">
                    <span>🧠 Context Reasoning</span>
                </div>
                <div style="font-size: 0.75rem; color: var(--text-muted); font-weight: 600;">
                    Extracted Semantic Signals
                </div>
            </div>
            <div class="context-slots-row">
                ${slots.join('')}
            </div>
        </div>
    `;
}

// Category Inferencer
function inferCategory(result) {
    const title = (result.title || '').toLowerCase();
    const docType = (result.document_type || '').toLowerCase();
    const tags = (result.tags || []).map(t => t.toLowerCase());

    if (title.includes('receipt') || docType.includes('receipt') || title.includes('purchase') || tags.includes('receipt')) {
        return { label: 'Receipt / Purchase', icon: '🧾', cssClass: 'type-receipt' };
    }
    if (title.includes('warranty') || docType.includes('warranty') || title.includes('service') || tags.includes('warranty')) {
        return { label: 'Warranty & Service', icon: '🛡️', cssClass: 'type-warranty' };
    }
    if (title.includes('lease') || title.includes('contract') || title.includes('agreement') || tags.includes('contract')) {
        return { label: 'Contract / Lease', icon: '📄', cssClass: 'type-contract' };
    }
    if (title.includes('prescription') || title.includes('medical') || title.includes('doctor') || tags.includes('medical')) {
        return { label: 'Medical Record', icon: '🏥', cssClass: 'type-medical' };
    }
    if (title.includes('tuition') || title.includes('grade') || title.includes('transcript') || tags.includes('education')) {
        return { label: 'Academic Record', icon: '🎓', cssClass: 'type-receipt' };
    }
    if (title.includes('flight') || title.includes('ticket') || title.includes('travel') || tags.includes('travel')) {
        return { label: 'Travel Booking', icon: '✈️', cssClass: 'type-service' };
    }
    return { label: result.document_type || 'Document', icon: '📋', cssClass: 'type-default' };
}

function buildClientEvidence(result) {
    const parts = [];
    if (result.matched_entities && result.matched_entities.length > 0) {
        parts.push(`Entity: ${result.matched_entities.join(', ')}`);
    }
    if (result.matched_event) {
        parts.push(`Event: ${result.matched_event}`);
    }
    if (result.created_date) {
        parts.push(`Date: ${formatDate(result.created_date)}`);
    }
    if (result.matched_relationship) {
        parts.push(`Relation: ${result.matched_relationship}`);
    }
    return parts.join(' | ');
}

function replaceResultsHTML(html) {
    const progressContainer = contentDiv.querySelector('.progress-container');
    if (progressContainer) {
        let sibling = progressContainer.nextElementSibling;
        while (sibling) {
            const next = sibling.nextElementSibling;
            sibling.remove();
            sibling = next;
        }
        progressContainer.insertAdjacentHTML('afterend', html);
    } else {
        contentDiv.innerHTML = html;
    }
}

// 7. Interactive Drawers & Actions
window.toggleExplanation = async function(docId, event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }
    const drawer = document.getElementById(`explanation-${docId}`);
    const textEl = document.getElementById(`explanation-text-${docId}`);
    if (!drawer || !textEl) return;

    if (drawer.classList.contains('show')) {
        drawer.classList.remove('show');
        return;
    }

    drawer.classList.add('show');

    const result = currentSearchResults.find(r => (r.recallens_id || r.paperless_id) == docId);
    if (!result) return;

    textEl.textContent = 'Generating grounded AI explanation...';

    try {
        const response = await fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: currentQuery,
                title: result.title,
                content: result.content,
                summary: result.summary,
                tags: result.tags || [],
                correspondent_name: result.correspondent_name,
                matched_entities: result.matched_entities || [],
                matched_event: result.matched_event || null,
                matched_relationship: result.matched_relationship || null
            })
        });

        if (response.ok) {
            const data = await response.json();
            textEl.innerHTML = escapeHtml(data.explanation).replace(/\n/g, '<br>');
        } else {
            textEl.textContent = 'Could not generate explanation at this time.';
        }
    } catch (err) {
        textEl.textContent = `Explanation unavailable: ${err.message}`;
    }
};

window.toggleBreakdown = function(docId) {
    const drawer = document.getElementById(`breakdown-${docId}`);
    if (drawer) {
        drawer.classList.toggle('show');
    }
};

window.copySnippet = function(docId, button) {
    const result = currentSearchResults.find(r => (r.recallens_id || r.paperless_id) == docId);
    if (!result) return;

    const textToCopy = (result.highlights && result.highlights.length > 0)
        ? result.highlights.join('\n')
        : (result.content || result.title);

    navigator.clipboard.writeText(textToCopy).then(() => {
        const originalText = button.innerHTML;
        button.innerHTML = '<span>✅ Copied!</span>';
        setTimeout(() => {
            button.innerHTML = originalText;
        }, 1500);
    }).catch(err => {
        console.error('Failed to copy excerpt:', err);
    });
};

// 8. Helper Utilities
function formatDate(dateString) {
    if (!dateString) return '';
    try {
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return String(dateString).slice(0, 10);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    } catch (e) {
        return String(dateString).slice(0, 10);
    }
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function formatHighlightText(text) {
    const escaped = escapeHtml(text);
    return escaped.replace(/==([^=]+)==/g, '<mark class="kw-mark">$1</mark>');
}
