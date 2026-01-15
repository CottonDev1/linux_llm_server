/**
 * LLM Settings Client-Side JavaScript
 * Handles all interactions with the LLM configuration API
 *
 * Features:
 * - Load and display available LLM models
 * - Configure generation parameters (temperature, top_p, etc.)
 * - Test generation with current settings
 * - Save/load configuration from server
 * - Real-time validation and feedback
 */

// State management
let isDirty = false;
let currentConfig = null;
let availableModels = [];

/**
 * Initialize the page on load
 */
document.addEventListener('DOMContentLoaded', async () => {
    console.log('LLM Settings page initializing...');

    // Initialize sidebar if available
    if (typeof initSidebar === 'function') {
        await initSidebar();
    }

    // Load configuration and models
    await Promise.all([
        loadConfiguration(),
        loadModels(),
        refreshRunningModels()
    ]);

    // Initialize all slider values
    initializeSliderDisplays();
});

/**
 * Load current LLM configuration from server
 */
async function loadConfiguration() {
    try {
        const response = await fetch('/api/llm/config');
        const data = await response.json();

        if (data.success) {
            currentConfig = data.config;
            applyConfigToForm(currentConfig);
            showToast('Configuration loaded', 'success');
        } else {
            showToast('Failed to load configuration: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error loading configuration:', error);
        showToast('Error loading configuration: ' + error.message, 'error');
    }
}

/**
 * Apply configuration values to form elements
 */
function applyConfigToForm(config) {
    // Model selection
    if (config.model) {
        setSelectValue('defaultModel', config.model);
    }
    if (config.sqlModel) {
        setSelectValue('sqlModel', config.sqlModel);
    }

    // Sampling parameters
    setSliderValue('temperature', config.temperature);
    setSliderValue('top_p', config.top_p);
    setInputValue('top_k', config.top_k);
    setSliderValue('min_p', config.min_p);

    // Mirostat
    setSelectValue('mirostat', config.mirostat);
    setSliderValue('mirostat_eta', config.mirostat_eta);
    setSliderValue('mirostat_tau', config.mirostat_tau);

    // Repetition control
    setSliderValue('repeat_penalty', config.repeat_penalty);
    setInputValue('repeat_last_n', config.repeat_last_n);
    setSliderValue('frequency_penalty', config.frequency_penalty);
    setSliderValue('presence_penalty', config.presence_penalty);

    // Context settings
    setSelectValue('num_ctx', config.num_ctx);
    setInputValue('num_predict', config.num_predict);

    // Performance settings
    setInputValue('num_gpu', config.num_gpu);
    setInputValue('num_thread', config.num_thread);
    setInputValue('num_batch', config.num_batch);
    setCheckboxValue('low_vram', config.low_vram);

    // Advanced settings
    setInputValue('seed', config.seed);
    setInputValue('keep_alive', config.keep_alive);
    setInputValue('stop', Array.isArray(config.stop) ? config.stop.join(', ') : config.stop || '');
    setInputValue('system_prompt', config.system_prompt || '');

    // Reset dirty state
    isDirty = false;
    updateUnsavedIndicator();
}

/**
 * Load available models from LLM server
 */
async function loadModels() {
    try {
        const response = await fetch('/api/llm/models');
        const data = await response.json();

        if (data.success) {
            availableModels = data.models;
            renderModelList(availableModels);
            populateModelSelects(availableModels);
        } else {
            showToast('Failed to load models: ' + data.error, 'error');
            renderModelList([]);
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showToast('Error loading models. Is the LLM server running?', 'error');
        renderModelList([]);
    }
}

/**
 * Refresh running models status
 */
async function refreshRunningModels() {
    const container = document.getElementById('runningModels');
    if (!container) return;

    try {
        const response = await fetch('/api/llm/running');
        const data = await response.json();

        if (data.success && data.models.length > 0) {
            container.innerHTML = data.models.map(model => `
                <div class="running-model">
                    <span class="running-indicator"></span>
                    <span class="model-name">${model.name}</span>
                    <span style="color: var(--text-muted); font-size: 12px;">
                        ${model.sizeVramFormatted} VRAM
                    </span>
                    <button class="btn btn-sm btn-secondary" onclick="unloadModel('${model.name}')" title="Unload from memory">
                        Unload
                    </button>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<span style="color: var(--text-muted);">No models currently loaded in memory</span>';
        }
    } catch (error) {
        console.error('Error fetching running models:', error);
        container.innerHTML = '<span style="color: var(--text-muted);">Could not fetch running models</span>';
    }
}

/**
 * Render the model list
 */
function renderModelList(models) {
    const container = document.getElementById('modelList');
    if (!container) return;

    if (models.length === 0) {
        container.innerHTML = '<div style="padding: 24px; text-align: center; color: var(--text-muted);">No models found. Pull a model to get started.</div>';
        return;
    }

    container.innerHTML = models.map(model => `
        <div class="model-item ${model.isDefault ? 'selected' : ''}" onclick="selectModel('${model.name}')">
            <div class="model-info">
                <div class="model-name">${model.name}</div>
                <div class="model-meta">
                    <span>${model.sizeFormatted}</span>
                    ${model.quantization ? `<span>${model.quantization}</span>` : ''}
                    <span>${formatDate(model.modified)}</span>
                </div>
            </div>
            <div class="model-actions">
                <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); editModel('${model.name}')" title="Edit parameters">
                    Edit
                </button>
                <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); loadModel('${model.name}')" title="Load into memory">
                    Load
                </button>
                <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); viewModelDetails('${model.name}')" title="View details">
                    Info
                </button>
                <button class="btn btn-sm" style="background: #dc2626; color: white;" onclick="event.stopPropagation(); deleteModel('${model.name}')" title="Delete model">
                    Delete
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * Populate model selection dropdowns
 */
function populateModelSelects(models) {
    const defaultSelect = document.getElementById('defaultModel');
    const sqlSelect = document.getElementById('sqlModel');

    if (!defaultSelect || !sqlSelect) return;

    const options = models.map(m => `<option value="${m.name}">${m.name} (${m.sizeFormatted})</option>`).join('');

    defaultSelect.innerHTML = '<option value="">Select a model...</option>' + options;
    sqlSelect.innerHTML = '<option value="">Select a model...</option>' + options;

    // Restore current config values
    if (currentConfig) {
        if (currentConfig.model) {
            setSelectValue('defaultModel', currentConfig.model);
        }
        if (currentConfig.sqlModel) {
            setSelectValue('sqlModel', currentConfig.sqlModel);
        }
    }
}

/**
 * Select a model from the list
 */
function selectModel(modelName) {
    // Update the default model select
    setSelectValue('defaultModel', modelName);

    // Update visual selection
    document.querySelectorAll('.model-item').forEach(item => {
        item.classList.remove('selected');
        if (item.querySelector('.model-name').textContent === modelName) {
            item.classList.add('selected');
        }
    });

    markDirty();
}

/**
 * Load a model into memory
 */
async function loadModel(modelName) {
    showToast(`Loading ${modelName} into memory...`, 'info');

    try {
        const response = await fetch('/api/llm/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Model ${modelName} loaded successfully`, 'success');
            await refreshRunningModels();
        } else {
            showToast(`Failed to load model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error loading model: ${error.message}`, 'error');
    }
}

/**
 * Unload a model from memory
 */
async function unloadModel(modelName) {
    try {
        const response = await fetch('/api/llm/unload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Model ${modelName} unloaded`, 'success');
            await refreshRunningModels();
        } else {
            showToast(`Failed to unload model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error unloading model: ${error.message}`, 'error');
    }
}

/**
 * View model details
 */
async function viewModelDetails(modelName) {
    try {
        const response = await fetch(`/api/llm/model/${encodeURIComponent(modelName)}`);
        const data = await response.json();

        if (data.success) {
            const model = data.model;
            alert(`Model: ${model.name}\n\nParameters:\n${model.parameters || 'Default'}\n\nTemplate:\n${model.template || 'Default'}`);
        } else {
            showToast(`Failed to get model info: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error getting model info: ${error.message}`, 'error');
    }
}

/**
 * Delete a model
 */
async function deleteModel(modelName) {
    if (!confirm(`Are you sure you want to delete ${modelName}? This cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`/api/llm/model/${encodeURIComponent(modelName)}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Model ${modelName} deleted`, 'success');
            await loadModels();
        } else {
            showToast(`Failed to delete model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error deleting model: ${error.message}`, 'error');
    }
}

/**
 * Pull a new model
 */
async function pullModel() {
    const input = document.getElementById('pullModelName');
    const modelName = input.value.trim();

    if (!modelName) {
        showToast('Please enter a model name', 'error');
        return;
    }

    showToast(`Pulling model ${modelName}... This may take a while.`, 'info');

    try {
        const response = await fetch('/api/llm/pull', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelName })
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Model ${modelName} pulled successfully!`, 'success');
            input.value = '';
            await loadModels();
        } else {
            showToast(`Failed to pull model: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error pulling model: ${error.message}`, 'error');
    }
}

/**
 * Save all settings to server
 */
async function saveAllSettings() {
    const config = getFormValues();

    try {
        const response = await fetch('/api/llm/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            currentConfig = data.config;
            isDirty = false;
            updateUnsavedIndicator();
            showToast('Settings saved successfully!', 'success');

            // Also save to localStorage for client-side persistence
            localStorage.setItem('llmConfig', JSON.stringify(config));
        } else {
            showToast(`Failed to save: ${data.error}`, 'error');
            if (data.validationErrors) {
                data.validationErrors.forEach(err => showToast(err, 'error'));
            }
        }
    } catch (error) {
        showToast(`Error saving settings: ${error.message}`, 'error');
    }
}

/**
 * Reset to default settings
 */
async function resetToDefaults() {
    if (!confirm('Reset all settings to defaults? This will discard your current configuration.')) {
        return;
    }

    try {
        const response = await fetch('/api/llm/config/reset', {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            currentConfig = data.config;
            applyConfigToForm(currentConfig);
            showToast('Settings reset to defaults', 'success');
        } else {
            showToast(`Failed to reset: ${data.error}`, 'error');
        }
    } catch (error) {
        showToast(`Error resetting settings: ${error.message}`, 'error');
    }
}

/**
 * Run test generation
 */
async function runTest() {
    const prompt = document.getElementById('testPrompt').value.trim();
    const button = document.getElementById('testButton');
    const resultsContainer = document.getElementById('testResults');
    const responseText = document.getElementById('testResponseText');
    const statsContainer = document.getElementById('testStats');

    if (!prompt) {
        showToast('Please enter a test prompt', 'error');
        return;
    }

    // Show loading state
    button.disabled = true;
    button.textContent = 'Generating...';
    resultsContainer.style.display = 'block';
    responseText.textContent = 'Generating response...';
    statsContainer.innerHTML = '';

    try {
        // Gather current form options for the test
        const options = {
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('top_p').value),
            top_k: parseInt(document.getElementById('top_k').value),
            repeat_penalty: parseFloat(document.getElementById('repeat_penalty').value),
            num_ctx: parseInt(document.getElementById('num_ctx').value),
            num_predict: parseInt(document.getElementById('num_predict').value),
            seed: parseInt(document.getElementById('seed').value),
            mirostat: parseInt(document.getElementById('mirostat').value),
            mirostat_eta: parseFloat(document.getElementById('mirostat_eta').value),
            mirostat_tau: parseFloat(document.getElementById('mirostat_tau').value)
        };

        const response = await fetch('/api/llm/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model: document.getElementById('defaultModel').value,
                options
            })
        });

        const data = await response.json();

        if (data.success) {
            responseText.textContent = data.response;

            // Display stats
            statsContainer.innerHTML = `
                <div class="test-stat">
                    <span class="test-stat-value">${data.stats.durationFormatted}</span>
                    <span class="test-stat-label">Duration</span>
                </div>
                <div class="test-stat">
                    <span class="test-stat-value">${data.stats.evalCount || '-'}</span>
                    <span class="test-stat-label">Tokens</span>
                </div>
                <div class="test-stat">
                    <span class="test-stat-value">${data.stats.tokensPerSecond || '-'}</span>
                    <span class="test-stat-label">Tokens/sec</span>
                </div>
                <div class="test-stat">
                    <span class="test-stat-value">${data.stats.promptEvalCount || '-'}</span>
                    <span class="test-stat-label">Prompt Tokens</span>
                </div>
            `;
        } else {
            responseText.textContent = `Error: ${data.error}`;
        }
    } catch (error) {
        responseText.textContent = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
        button.textContent = 'Run Test';
    }
}

/**
 * Toggle collapsible section
 */
function toggleSection(sectionId) {
    const content = document.getElementById(`${sectionId}Content`);
    const icon = document.getElementById(`${sectionId}Icon`);

    if (content && icon) {
        content.classList.toggle('collapsed');
        icon.classList.toggle('collapsed');
    }
}

/**
 * Update slider value display and mark dirty
 */
function updateSliderValue(sliderId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(`${sliderId}Value`);

    if (slider && display) {
        let value = parseFloat(slider.value);
        // Format based on step
        if (slider.step && parseFloat(slider.step) < 1) {
            const decimals = slider.step.split('.')[1]?.length || 1;
            value = value.toFixed(decimals);
        }
        display.textContent = value;
    }

    markDirty();
}

/**
 * Initialize all slider displays
 */
function initializeSliderDisplays() {
    const sliders = ['temperature', 'top_p', 'min_p', 'mirostat_eta', 'mirostat_tau',
                     'repeat_penalty', 'frequency_penalty', 'presence_penalty'];

    sliders.forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            updateSliderValue(id);
        }
    });
}

/**
 * Mark form as having unsaved changes
 */
function markDirty() {
    isDirty = true;
    updateUnsavedIndicator();
}

/**
 * Update the unsaved changes indicator
 */
function updateUnsavedIndicator() {
    const indicator = document.getElementById('unsavedIndicator');
    if (indicator) {
        indicator.classList.toggle('visible', isDirty);
    }
}

/**
 * Get all form values as config object
 */
function getFormValues() {
    return {
        model: document.getElementById('defaultModel')?.value || '',
        sqlModel: document.getElementById('sqlModel')?.value || '',
        temperature: parseFloat(document.getElementById('temperature')?.value || 0.7),
        top_p: parseFloat(document.getElementById('top_p')?.value || 0.9),
        top_k: parseInt(document.getElementById('top_k')?.value || 40),
        min_p: parseFloat(document.getElementById('min_p')?.value || 0),
        mirostat: parseInt(document.getElementById('mirostat')?.value || 0),
        mirostat_eta: parseFloat(document.getElementById('mirostat_eta')?.value || 0.1),
        mirostat_tau: parseFloat(document.getElementById('mirostat_tau')?.value || 5.0),
        repeat_penalty: parseFloat(document.getElementById('repeat_penalty')?.value || 1.1),
        repeat_last_n: parseInt(document.getElementById('repeat_last_n')?.value || 64),
        frequency_penalty: parseFloat(document.getElementById('frequency_penalty')?.value || 0),
        presence_penalty: parseFloat(document.getElementById('presence_penalty')?.value || 0),
        num_ctx: parseInt(document.getElementById('num_ctx')?.value || 4096),
        num_predict: parseInt(document.getElementById('num_predict')?.value || -1),
        num_gpu: parseInt(document.getElementById('num_gpu')?.value || -1),
        num_thread: parseInt(document.getElementById('num_thread')?.value || 0),
        num_batch: parseInt(document.getElementById('num_batch')?.value || 512),
        low_vram: document.getElementById('low_vram')?.checked || false,
        seed: parseInt(document.getElementById('seed')?.value || -1),
        keep_alive: document.getElementById('keep_alive')?.value || '5m',
        stop: document.getElementById('stop')?.value || '',
        system_prompt: document.getElementById('system_prompt')?.value || ''
    };
}

// Helper functions
function setSliderValue(id, value) {
    const slider = document.getElementById(id);
    const display = document.getElementById(`${id}Value`);
    if (slider) {
        slider.value = value;
        if (display) {
            let displayValue = parseFloat(value);
            if (slider.step && parseFloat(slider.step) < 1) {
                const decimals = slider.step.split('.')[1]?.length || 1;
                displayValue = displayValue.toFixed(decimals);
            }
            display.textContent = displayValue;
        }
    }
}

function setInputValue(id, value) {
    const input = document.getElementById(id);
    if (input) {
        input.value = value;
    }
}

function setSelectValue(id, value) {
    const select = document.getElementById(id);
    if (select) {
        select.value = value;
    }
}

function setCheckboxValue(id, checked) {
    const checkbox = document.getElementById(id);
    if (checkbox) {
        checkbox.checked = checked;
    }
}

function formatDate(dateString) {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleDateString();
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Remove after 5 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Logout function
 */
function logout() {
    if (typeof AuthClient !== 'undefined') {
        const auth = new AuthClient();
        auth.logout();
    } else {
        window.location.href = '/';
    }
}

// Warn about unsaved changes when leaving
window.addEventListener('beforeunload', (e) => {
    if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// ============================================================================
// MODEL EDITOR MODAL FUNCTIONALITY
// ============================================================================

let currentEditingModel = null;
let modelEditorConfig = {};

/**
 * Open the model editor modal for a specific model
 */
async function editModel(modelName) {
    currentEditingModel = modelName;

    // Show the modal
    const modal = document.getElementById('modelEditModal');
    const modalTitle = document.getElementById('modalModelName');
    const editorContent = document.getElementById('modelEditorContent');

    modalTitle.textContent = modelName;
    modal.classList.add('active');

    // Show loading state
    editorContent.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-muted);">Loading model configuration...</div>';

    try {
        // Get current global config as defaults
        const response = await fetch('/api/llm/config');
        const data = await response.json();

        if (data.success) {
            modelEditorConfig = { ...data.config };
            renderModelEditor();
        } else {
            throw new Error(data.error);
        }
    } catch (error) {
        editorContent.innerHTML = `<div style="text-align: center; padding: 40px; color: #ef4444;">Error loading configuration: ${error.message}</div>`;
    }
}

/**
 * Render the model editor form
 */
function renderModelEditor() {
    const editorContent = document.getElementById('modelEditorContent');

    editorContent.innerHTML = `
        <!-- Sampling Parameters -->
        <div class="param-section">
            <div class="param-section-title">Sampling Parameters</div>

            <div class="param-row">
                <label class="param-label">Temperature</label>
                <span class="param-description">Controls randomness. Lower = more focused and deterministic, Higher = more creative and varied. Range: 0.0 - 2.0</span>
                <div class="param-slider-group">
                    <input type="range" id="edit_temperature" class="slider" min="0" max="2" step="0.1" value="${modelEditorConfig.temperature || 0.7}" oninput="updateEditSlider('temperature')">
                    <span id="edit_temperature_value" class="slider-value">${(modelEditorConfig.temperature || 0.7).toFixed(1)}</span>
                </div>
            </div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">Top P (Nucleus Sampling)</label>
                    <span class="param-description">Cumulative probability threshold. Range: 0.0 - 1.0</span>
                    <div class="param-slider-group">
                        <input type="range" id="edit_top_p" class="slider" min="0" max="1" step="0.05" value="${modelEditorConfig.top_p || 0.9}" oninput="updateEditSlider('top_p')">
                        <span id="edit_top_p_value" class="slider-value">${(modelEditorConfig.top_p || 0.9).toFixed(2)}</span>
                    </div>
                </div>

                <div class="param-row">
                    <label class="param-label">Top K</label>
                    <span class="param-description">Limits vocabulary to top K tokens. Range: 1 - 100</span>
                    <input type="number" id="edit_top_k" class="param-input" min="1" max="100" value="${modelEditorConfig.top_k || 40}">
                </div>
            </div>

            <div class="param-row">
                <label class="param-label">Min P</label>
                <span class="param-description">Minimum probability threshold for token consideration. Range: 0.0 - 0.5</span>
                <div class="param-slider-group">
                    <input type="range" id="edit_min_p" class="slider" min="0" max="0.5" step="0.01" value="${modelEditorConfig.min_p || 0}" oninput="updateEditSlider('min_p')">
                    <span id="edit_min_p_value" class="slider-value">${(modelEditorConfig.min_p || 0).toFixed(2)}</span>
                </div>
            </div>
        </div>

        <!-- Context Settings -->
        <div class="param-section">
            <div class="param-section-title">Context & Generation Settings</div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">Context Window Size</label>
                    <span class="param-description">Maximum context tokens. Larger = more memory usage.</span>
                    <select id="edit_num_ctx" class="param-input">
                        <option value="2048" ${modelEditorConfig.num_ctx === 2048 ? 'selected' : ''}>2,048 (Small)</option>
                        <option value="4096" ${modelEditorConfig.num_ctx === 4096 ? 'selected' : ''}>4,096 (Default)</option>
                        <option value="8192" ${modelEditorConfig.num_ctx === 8192 ? 'selected' : ''}>8,192 (Large)</option>
                        <option value="16384" ${modelEditorConfig.num_ctx === 16384 ? 'selected' : ''}>16,384 (Very Large)</option>
                        <option value="32768" ${modelEditorConfig.num_ctx === 32768 ? 'selected' : ''}>32,768 (Huge)</option>
                    </select>
                </div>

                <div class="param-row">
                    <label class="param-label">Max Tokens to Generate</label>
                    <span class="param-description">-1 = unlimited, -2 = fill context window</span>
                    <input type="number" id="edit_num_predict" class="param-input" min="-2" max="131072" value="${modelEditorConfig.num_predict || -1}">
                </div>
            </div>
        </div>

        <!-- Repetition Control -->
        <div class="param-section">
            <div class="param-section-title">Repetition Control</div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">Repeat Penalty</label>
                    <span class="param-description">Penalty for repeating tokens. 1.0 = no penalty. Range: 1.0 - 2.0</span>
                    <div class="param-slider-group">
                        <input type="range" id="edit_repeat_penalty" class="slider" min="1" max="2" step="0.05" value="${modelEditorConfig.repeat_penalty || 1.1}" oninput="updateEditSlider('repeat_penalty')">
                        <span id="edit_repeat_penalty_value" class="slider-value">${(modelEditorConfig.repeat_penalty || 1.1).toFixed(2)}</span>
                    </div>
                </div>

                <div class="param-row">
                    <label class="param-label">Repeat Last N</label>
                    <span class="param-description">Tokens to consider for repetition penalty. Range: 0 - 2048</span>
                    <input type="number" id="edit_repeat_last_n" class="param-input" min="0" max="2048" value="${modelEditorConfig.repeat_last_n || 64}">
                </div>
            </div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">Frequency Penalty</label>
                    <span class="param-description">Penalize frequently used tokens. Range: 0.0 - 2.0</span>
                    <div class="param-slider-group">
                        <input type="range" id="edit_frequency_penalty" class="slider" min="0" max="2" step="0.1" value="${modelEditorConfig.frequency_penalty || 0}" oninput="updateEditSlider('frequency_penalty')">
                        <span id="edit_frequency_penalty_value" class="slider-value">${(modelEditorConfig.frequency_penalty || 0).toFixed(1)}</span>
                    </div>
                </div>

                <div class="param-row">
                    <label class="param-label">Presence Penalty</label>
                    <span class="param-description">Encourage discussing new topics. Range: 0.0 - 2.0</span>
                    <div class="param-slider-group">
                        <input type="range" id="edit_presence_penalty" class="slider" min="0" max="2" step="0.1" value="${modelEditorConfig.presence_penalty || 0}" oninput="updateEditSlider('presence_penalty')">
                        <span id="edit_presence_penalty_value" class="slider-value">${(modelEditorConfig.presence_penalty || 0).toFixed(1)}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Mirostat -->
        <div class="param-section">
            <div class="param-section-title">Mirostat (Adaptive Sampling)</div>

            <div class="param-row">
                <label class="param-label">Mirostat Mode</label>
                <span class="param-description">Adaptive sampling algorithm. 0 = Disabled, 1 = Mirostat, 2 = Mirostat 2.0</span>
                <select id="edit_mirostat" class="param-input">
                    <option value="0" ${modelEditorConfig.mirostat === 0 ? 'selected' : ''}>Disabled</option>
                    <option value="1" ${modelEditorConfig.mirostat === 1 ? 'selected' : ''}>Mirostat 1</option>
                    <option value="2" ${modelEditorConfig.mirostat === 2 ? 'selected' : ''}>Mirostat 2.0</option>
                </select>
            </div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">Mirostat Eta (Learning Rate)</label>
                    <span class="param-description">How quickly the algorithm adapts. Range: 0.0 - 1.0</span>
                    <div class="param-slider-group">
                        <input type="range" id="edit_mirostat_eta" class="slider" min="0" max="1" step="0.01" value="${modelEditorConfig.mirostat_eta || 0.1}" oninput="updateEditSlider('mirostat_eta')">
                        <span id="edit_mirostat_eta_value" class="slider-value">${(modelEditorConfig.mirostat_eta || 0.1).toFixed(2)}</span>
                    </div>
                </div>

                <div class="param-row">
                    <label class="param-label">Mirostat Tau (Target Entropy)</label>
                    <span class="param-description">Lower = more focused, Higher = more varied. Range: 0.0 - 10.0</span>
                    <div class="param-slider-group">
                        <input type="range" id="edit_mirostat_tau" class="slider" min="0" max="10" step="0.5" value="${modelEditorConfig.mirostat_tau || 5}" oninput="updateEditSlider('mirostat_tau')">
                        <span id="edit_mirostat_tau_value" class="slider-value">${(modelEditorConfig.mirostat_tau || 5).toFixed(1)}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Performance Settings -->
        <div class="param-section">
            <div class="param-section-title">Performance Settings</div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">GPU Layers</label>
                    <span class="param-description">-1 = auto, 0 = CPU only, or specific layer count</span>
                    <input type="number" id="edit_num_gpu" class="param-input" min="-1" max="999" value="${modelEditorConfig.num_gpu || -1}">
                </div>

                <div class="param-row">
                    <label class="param-label">CPU Threads</label>
                    <span class="param-description">0 = auto-detect optimal thread count</span>
                    <input type="number" id="edit_num_thread" class="param-input" min="0" max="256" value="${modelEditorConfig.num_thread || 0}">
                </div>
            </div>

            <div class="param-grid">
                <div class="param-row">
                    <label class="param-label">Batch Size</label>
                    <span class="param-description">Tokens processed in parallel. Range: 1 - 2048</span>
                    <input type="number" id="edit_num_batch" class="param-input" min="1" max="2048" value="${modelEditorConfig.num_batch || 512}">
                </div>

                <div class="param-row">
                    <label class="param-label">Keep Alive</label>
                    <span class="param-description">How long to keep model loaded (e.g., "5m", "1h", "0")</span>
                    <input type="text" id="edit_keep_alive" class="param-input" value="${modelEditorConfig.keep_alive || '5m'}">
                </div>
            </div>
        </div>

        <!-- Advanced Settings -->
        <div class="param-section">
            <div class="param-section-title">Advanced Settings</div>

            <div class="param-row">
                <label class="param-label">Random Seed</label>
                <span class="param-description">-1 = random each time. Set specific value for reproducibility.</span>
                <input type="number" id="edit_seed" class="param-input" min="-1" value="${modelEditorConfig.seed || -1}">
            </div>

            <div class="param-row">
                <label class="param-label">Stop Sequences</label>
                <span class="param-description">Comma-separated. Generation stops when these tokens are produced.</span>
                <input type="text" id="edit_stop" class="param-input" placeholder="e.g., </code>, [END]" value="${Array.isArray(modelEditorConfig.stop) ? modelEditorConfig.stop.join(', ') : modelEditorConfig.stop || ''}">
            </div>

            <div class="param-row">
                <label class="param-label">Custom System Prompt</label>
                <span class="param-description">Override the default system prompt for this model. Leave empty to use defaults.</span>
                <textarea id="edit_system_prompt" class="param-input param-textarea" placeholder="Enter a custom system prompt...">${modelEditorConfig.system_prompt || ''}</textarea>
            </div>
        </div>

        <!-- Test Model -->
        <div class="test-model-section">
            <h4>Test Model with Current Settings</h4>
            <textarea id="editTestPrompt" class="param-input param-textarea" placeholder="Enter a test prompt..." style="margin-bottom: 12px;">Hello! Please respond with a brief greeting.</textarea>
            <button class="btn btn-primary" onclick="testModelInEditor()" id="editTestBtn" style="width: 100%;">Test Model</button>
            <div id="editTestResults" style="display: none; margin-top: 16px;">
                <div style="background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px; padding: 16px;">
                    <pre id="editTestResponse" style="white-space: pre-wrap; word-wrap: break-word; font-family: 'Courier New', monospace; font-size: 13px; color: var(--text-secondary); margin: 0;"></pre>
                </div>
                <div id="editTestStats" style="display: flex; gap: 16px; margin-top: 12px; font-size: 12px; color: var(--text-muted);"></div>
            </div>
        </div>
    `;
}

/**
 * Update slider value display in editor
 */
function updateEditSlider(name) {
    const slider = document.getElementById(`edit_${name}`);
    const display = document.getElementById(`edit_${name}_value`);

    if (slider && display) {
        let value = parseFloat(slider.value);
        if (slider.step && parseFloat(slider.step) < 1) {
            const decimals = slider.step.split('.')[1]?.length || 1;
            value = value.toFixed(decimals);
        }
        display.textContent = value;
    }
}

/**
 * Close the model editor modal
 */
function closeModelEditor() {
    const modal = document.getElementById('modelEditModal');
    modal.classList.remove('active');
    currentEditingModel = null;
    modelEditorConfig = {};
}

/**
 * Save model configuration
 */
async function saveModelConfig() {
    if (!currentEditingModel) return;

    const saveBtn = document.getElementById('saveModelBtn');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';

    try {
        // Gather values from the editor form
        const config = {
            model: currentEditingModel,
            temperature: parseFloat(document.getElementById('edit_temperature')?.value || 0.7),
            top_p: parseFloat(document.getElementById('edit_top_p')?.value || 0.9),
            top_k: parseInt(document.getElementById('edit_top_k')?.value || 40),
            min_p: parseFloat(document.getElementById('edit_min_p')?.value || 0),
            num_ctx: parseInt(document.getElementById('edit_num_ctx')?.value || 4096),
            num_predict: parseInt(document.getElementById('edit_num_predict')?.value || -1),
            repeat_penalty: parseFloat(document.getElementById('edit_repeat_penalty')?.value || 1.1),
            repeat_last_n: parseInt(document.getElementById('edit_repeat_last_n')?.value || 64),
            frequency_penalty: parseFloat(document.getElementById('edit_frequency_penalty')?.value || 0),
            presence_penalty: parseFloat(document.getElementById('edit_presence_penalty')?.value || 0),
            mirostat: parseInt(document.getElementById('edit_mirostat')?.value || 0),
            mirostat_eta: parseFloat(document.getElementById('edit_mirostat_eta')?.value || 0.1),
            mirostat_tau: parseFloat(document.getElementById('edit_mirostat_tau')?.value || 5),
            num_gpu: parseInt(document.getElementById('edit_num_gpu')?.value || -1),
            num_thread: parseInt(document.getElementById('edit_num_thread')?.value || 0),
            num_batch: parseInt(document.getElementById('edit_num_batch')?.value || 512),
            keep_alive: document.getElementById('edit_keep_alive')?.value || '5m',
            seed: parseInt(document.getElementById('edit_seed')?.value || -1),
            stop: document.getElementById('edit_stop')?.value || '',
            system_prompt: document.getElementById('edit_system_prompt')?.value || ''
        };

        // Save to server
        const response = await fetch('/api/llm/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (data.success) {
            showToast(`Configuration saved for ${currentEditingModel}`, 'success');
            closeModelEditor();

            // Reload the main configuration
            await loadConfiguration();
        } else {
            showToast(`Failed to save: ${data.error}`, 'error');
            if (data.validationErrors) {
                data.validationErrors.forEach(err => showToast(err, 'error'));
            }
        }
    } catch (error) {
        showToast(`Error saving configuration: ${error.message}`, 'error');
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save Configuration';
    }
}

/**
 * Test model with current editor settings
 */
async function testModelInEditor() {
    const prompt = document.getElementById('editTestPrompt')?.value.trim();
    const testBtn = document.getElementById('editTestBtn');
    const resultsContainer = document.getElementById('editTestResults');
    const responseText = document.getElementById('editTestResponse');
    const statsContainer = document.getElementById('editTestStats');

    if (!prompt) {
        showToast('Please enter a test prompt', 'error');
        return;
    }

    testBtn.disabled = true;
    testBtn.textContent = 'Testing...';
    resultsContainer.style.display = 'block';
    responseText.textContent = 'Generating response...';
    statsContainer.innerHTML = '';

    try {
        // Gather current editor settings
        const options = {
            temperature: parseFloat(document.getElementById('edit_temperature')?.value || 0.7),
            top_p: parseFloat(document.getElementById('edit_top_p')?.value || 0.9),
            top_k: parseInt(document.getElementById('edit_top_k')?.value || 40),
            repeat_penalty: parseFloat(document.getElementById('edit_repeat_penalty')?.value || 1.1),
            num_ctx: parseInt(document.getElementById('edit_num_ctx')?.value || 4096),
            num_predict: parseInt(document.getElementById('edit_num_predict')?.value || -1),
            seed: parseInt(document.getElementById('edit_seed')?.value || -1),
            mirostat: parseInt(document.getElementById('edit_mirostat')?.value || 0),
            mirostat_eta: parseFloat(document.getElementById('edit_mirostat_eta')?.value || 0.1),
            mirostat_tau: parseFloat(document.getElementById('edit_mirostat_tau')?.value || 5)
        };

        const response = await fetch('/api/llm/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                model: currentEditingModel,
                options
            })
        });

        const data = await response.json();

        if (data.success) {
            responseText.textContent = data.response;

            statsContainer.innerHTML = `
                <span>Duration: <strong>${data.stats.durationFormatted}</strong></span>
                <span>Tokens: <strong>${data.stats.evalCount || '-'}</strong></span>
                <span>Speed: <strong>${data.stats.tokensPerSecond || '-'} tok/s</strong></span>
            `;
        } else {
            responseText.textContent = `Error: ${data.error}`;
        }
    } catch (error) {
        responseText.textContent = `Error: ${error.message}`;
    } finally {
        testBtn.disabled = false;
        testBtn.textContent = 'Test Model';
    }
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
    const modal = document.getElementById('modelEditModal');
    if (e.target === modal) {
        closeModelEditor();
    }
});
