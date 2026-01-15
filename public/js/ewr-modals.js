/**
 * EWR Modal System - Reusable Modal Components
 *
 * Usage:
 *   // Display Modal (read-only info)
 *   EwrModal.display({
 *       title: 'SQL Query',
 *       sections: [
 *           { label: 'Question', content: 'How many users?' },
 *           { label: 'Generated SQL', content: 'SELECT COUNT(*) FROM Users', isCode: true }
 *       ],
 *       buttons: [
 *           { text: 'Copy', class: 'primary', onClick: () => copyToClipboard() },
 *           { text: 'Close', class: 'secondary', close: true }
 *       ]
 *   });
 *
 *   // Input Modal (single input/textarea)
 *   EwrModal.input({
 *       title: 'Provide Feedback',
 *       label: 'What went wrong?',
 *       placeholder: 'Describe the issue...',
 *       type: 'textarea',
 *       onSubmit: (value) => submitFeedback(value)
 *   });
 *
 *   // Form Modal (multiple inputs)
 *   EwrModal.form({
 *       title: 'Add Rule',
 *       size: 'large',
 *       fields: [
 *           { name: 'name', label: 'Rule Name', type: 'text', required: true },
 *           { name: 'sql', label: 'SQL Template', type: 'textarea', fullWidth: true }
 *       ],
 *       onSubmit: (data) => saveRule(data)
 *   });
 *
 *   // Close any open modal
 *   EwrModal.close();
 */

const EwrModal = (function() {
    // Private: Current modal element
    let currentModal = null;
    let modalContainer = null;

    // Initialize container
    function ensureContainer() {
        if (!modalContainer) {
            modalContainer = document.createElement('div');
            modalContainer.id = 'ewr-modal-container';
            document.body.appendChild(modalContainer);
        }
        return modalContainer;
    }

    // Create base modal structure
    function createBaseModal(options = {}) {
        const size = options.size || 'medium';

        const overlay = document.createElement('div');
        overlay.className = 'ewr-modal-overlay';
        overlay.onclick = (e) => {
            if (e.target === overlay && options.closeOnOverlay !== false) {
                close();
            }
        };

        const modal = document.createElement('div');
        modal.className = `ewr-modal ${size}`;
        modal.onclick = (e) => e.stopPropagation();

        // Header
        const header = document.createElement('div');
        header.className = 'ewr-modal-header';
        header.innerHTML = `
            <h3 class="ewr-modal-title">${options.title || 'Modal'}</h3>
            <button class="ewr-modal-close" onclick="EwrModal.close()">&times;</button>
        `;

        // Body
        const body = document.createElement('div');
        body.className = 'ewr-modal-body';

        // Footer
        const footer = document.createElement('div');
        footer.className = 'ewr-modal-footer';

        modal.appendChild(header);
        modal.appendChild(body);
        modal.appendChild(footer);
        overlay.appendChild(modal);

        return { overlay, modal, header, body, footer };
    }

    // Create buttons
    function createButtons(footer, buttons, context = {}) {
        footer.innerHTML = '';
        buttons.forEach(btn => {
            const button = document.createElement('button');
            button.className = `ewr-modal-btn ewr-modal-btn-${btn.class || 'secondary'}`;
            button.innerHTML = btn.text;
            button.onclick = () => {
                if (btn.onClick) {
                    btn.onClick(context);
                }
                if (btn.close) {
                    close();
                }
            };
            footer.appendChild(button);
        });
    }

    // Public: Display Modal
    function display(options) {
        close(); // Close any existing modal

        const { overlay, body, footer } = createBaseModal(options);

        // Add sections
        options.sections?.forEach(section => {
            const sectionEl = document.createElement('div');
            sectionEl.className = 'ewr-modal-section';

            if (section.label) {
                const label = document.createElement('label');
                label.className = 'ewr-modal-label';
                label.textContent = section.label;
                sectionEl.appendChild(label);
            }

            const contentBox = document.createElement('div');
            contentBox.className = section.isCode ? 'ewr-modal-code-box' : 'ewr-modal-display-box';
            contentBox.textContent = section.content || '';
            contentBox.id = section.id || '';
            sectionEl.appendChild(contentBox);

            body.appendChild(sectionEl);
        });

        // Add buttons
        const buttons = options.buttons || [
            { text: 'Close', class: 'secondary', close: true }
        ];
        createButtons(footer, buttons, { getContent: () => options.sections });

        // Show modal
        ensureContainer().appendChild(overlay);
        currentModal = overlay;

        // Trigger animation
        requestAnimationFrame(() => {
            overlay.classList.add('active');
        });

        // Handle escape key
        document.addEventListener('keydown', handleEscape);

        return { overlay, close };
    }

    // Public: Input Modal
    function input(options) {
        close();

        const { overlay, body, footer } = createBaseModal(options);

        // Create input section
        const section = document.createElement('div');
        section.className = 'ewr-modal-section';

        if (options.label) {
            const label = document.createElement('label');
            label.className = 'ewr-modal-label';
            label.textContent = options.label;
            section.appendChild(label);
        }

        let inputEl;
        if (options.type === 'textarea') {
            inputEl = document.createElement('textarea');
            inputEl.className = 'ewr-modal-textarea';
            inputEl.rows = options.rows || 4;
        } else if (options.type === 'select') {
            inputEl = document.createElement('select');
            inputEl.className = 'ewr-modal-select';
            options.options?.forEach(opt => {
                const optEl = document.createElement('option');
                optEl.value = opt.value;
                optEl.textContent = opt.label;
                inputEl.appendChild(optEl);
            });
        } else {
            inputEl = document.createElement('input');
            inputEl.className = 'ewr-modal-input';
            inputEl.type = options.type || 'text';
        }

        inputEl.placeholder = options.placeholder || '';
        inputEl.value = options.value || '';
        inputEl.id = 'ewr-modal-input-value';
        section.appendChild(inputEl);

        if (options.helper) {
            const helper = document.createElement('div');
            helper.className = 'ewr-modal-helper';
            helper.textContent = options.helper;
            section.appendChild(helper);
        }

        body.appendChild(section);

        // Buttons
        const buttons = options.buttons || [
            {
                text: options.submitText || 'Submit',
                class: 'primary',
                onClick: () => {
                    if (options.onSubmit) {
                        options.onSubmit(inputEl.value);
                    }
                    close();
                }
            },
            { text: 'Cancel', class: 'secondary', close: true }
        ];
        createButtons(footer, buttons);

        ensureContainer().appendChild(overlay);
        currentModal = overlay;

        requestAnimationFrame(() => {
            overlay.classList.add('active');
            inputEl.focus();
        });

        document.addEventListener('keydown', handleEscape);

        return { overlay, close, getInput: () => inputEl.value };
    }

    // Public: Form Modal
    function form(options) {
        close();

        const { overlay, body, footer } = createBaseModal(options);

        // Create form
        const formEl = document.createElement('form');
        formEl.className = options.twoColumn ? 'ewr-modal-form-grid two-column' : 'ewr-modal-form-grid';
        formEl.onsubmit = (e) => e.preventDefault();

        options.fields?.forEach(field => {
            const section = document.createElement('div');
            section.className = `ewr-modal-section${field.fullWidth ? ' full-width' : ''}`;

            if (field.label) {
                const label = document.createElement('label');
                label.className = 'ewr-modal-label';
                label.textContent = field.label + (field.required ? ' *' : '');
                section.appendChild(label);
            }

            let inputEl;
            if (field.type === 'textarea') {
                inputEl = document.createElement('textarea');
                inputEl.className = 'ewr-modal-textarea';
                inputEl.rows = field.rows || 4;
            } else if (field.type === 'select') {
                inputEl = document.createElement('select');
                inputEl.className = 'ewr-modal-select';
                field.options?.forEach(opt => {
                    const optEl = document.createElement('option');
                    optEl.value = opt.value;
                    optEl.textContent = opt.label;
                    inputEl.appendChild(optEl);
                });
            } else if (field.type === 'radio') {
                inputEl = document.createElement('div');
                inputEl.className = 'ewr-modal-options';
                field.options?.forEach(opt => {
                    const optDiv = document.createElement('label');
                    optDiv.className = 'ewr-modal-option';
                    optDiv.innerHTML = `
                        <input type="radio" name="${field.name}" value="${opt.value}" ${opt.checked ? 'checked' : ''}>
                        <span class="ewr-modal-option-label">${opt.label}</span>
                    `;
                    inputEl.appendChild(optDiv);
                });
            } else {
                inputEl = document.createElement('input');
                inputEl.className = 'ewr-modal-input';
                inputEl.type = field.type || 'text';
            }

            if (inputEl.tagName !== 'DIV') {
                inputEl.name = field.name;
                inputEl.placeholder = field.placeholder || '';
                inputEl.value = field.value || '';
                if (field.required) inputEl.required = true;
            }
            section.appendChild(inputEl);

            if (field.helper) {
                const helper = document.createElement('div');
                helper.className = 'ewr-modal-helper';
                helper.textContent = field.helper;
                section.appendChild(helper);
            }

            formEl.appendChild(section);
        });

        body.appendChild(formEl);

        // Collect form data
        function getFormData() {
            const data = {};
            options.fields?.forEach(field => {
                if (field.type === 'radio') {
                    const checked = formEl.querySelector(`input[name="${field.name}"]:checked`);
                    data[field.name] = checked ? checked.value : null;
                } else {
                    const input = formEl.querySelector(`[name="${field.name}"]`);
                    data[field.name] = input ? input.value : '';
                }
            });
            return data;
        }

        // Buttons
        const buttons = options.buttons || [
            {
                text: options.submitText || 'Save',
                class: 'primary',
                onClick: () => {
                    if (options.onSubmit) {
                        options.onSubmit(getFormData());
                    }
                    close();
                }
            },
            { text: 'Cancel', class: 'secondary', close: true }
        ];
        createButtons(footer, buttons);

        ensureContainer().appendChild(overlay);
        currentModal = overlay;

        requestAnimationFrame(() => {
            overlay.classList.add('active');
            const firstInput = formEl.querySelector('input, textarea, select');
            if (firstInput) firstInput.focus();
        });

        document.addEventListener('keydown', handleEscape);

        return { overlay, close, getFormData };
    }

    // Public: Confirm Modal
    function confirm(options) {
        close();

        const { overlay, body, footer } = createBaseModal({
            ...options,
            size: options.size || 'small'
        });

        // Message
        const messageEl = document.createElement('div');
        messageEl.className = 'ewr-modal-display-box';
        messageEl.style.textAlign = 'center';
        messageEl.textContent = options.message || 'Are you sure?';
        body.appendChild(messageEl);

        // Buttons
        const buttons = [
            {
                text: options.confirmText || 'Confirm',
                class: options.danger ? 'danger' : 'primary',
                onClick: () => {
                    if (options.onConfirm) options.onConfirm();
                    close();
                }
            },
            {
                text: options.cancelText || 'Cancel',
                class: 'secondary',
                onClick: () => {
                    if (options.onCancel) options.onCancel();
                    close();
                }
            }
        ];
        createButtons(footer, buttons);

        ensureContainer().appendChild(overlay);
        currentModal = overlay;

        requestAnimationFrame(() => {
            overlay.classList.add('active');
        });

        document.addEventListener('keydown', handleEscape);

        return { overlay, close };
    }

    // Close modal
    function close() {
        if (currentModal) {
            currentModal.classList.remove('active');
            setTimeout(() => {
                if (currentModal && currentModal.parentNode) {
                    currentModal.parentNode.removeChild(currentModal);
                }
                currentModal = null;
            }, 200);
        }
        document.removeEventListener('keydown', handleEscape);
    }

    // Handle escape key
    function handleEscape(e) {
        if (e.key === 'Escape') {
            close();
        }
    }

    // Public API
    return {
        display,
        input,
        form,
        confirm,
        close
    };
})();

// Make globally available
if (typeof window !== 'undefined') {
    window.EwrModal = EwrModal;
}
