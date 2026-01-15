/**
 * Markdown Viewer Component
 * Reusable tree view and document viewer for markdown files
 */

class MarkdownViewer {
    constructor(options = {}) {
        // Configuration
        this.basePath = options.basePath || '/docs';
        this.manifestPath = options.manifestPath || `${this.basePath}/docs-manifest.json`;
        this.rootFolder = options.rootFolder || null; // If set, restricts to this folder
        this.canNavigateUp = options.canNavigateUp !== false; // Allow navigating to parent folders
        this.enableDragDrop = options.enableDragDrop !== false;
        this.enableDelete = options.enableDelete !== false;
        this.enableCreateFolder = options.enableCreateFolder !== false;
        this.title = options.title || 'Documents';

        // Element IDs (configurable)
        this.treeContainerId = options.treeContainerId || 'treeContainer';
        this.contentBodyId = options.contentBodyId || 'contentBody';
        this.breadcrumbId = options.breadcrumbId || 'breadcrumb';
        this.tocListId = options.tocListId || 'tocList';
        this.tocPanelId = options.tocPanelId || 'tocPanel';
        this.searchInputId = options.searchInputId || 'searchInput';
        this.deleteBtnId = options.deleteBtnId || 'deleteBtn';
        this.deleteModalId = options.deleteModalId || 'deleteModal';
        this.createFolderBtnId = options.createFolderBtnId || 'createFolderBtn';
        this.createFolderModalId = options.createFolderModalId || 'createFolderModal';

        // Callbacks
        this.onDocumentLoad = options.onDocumentLoad || null;
        this.onError = options.onError || null;
        this.onFolderCreate = options.onFolderCreate || null;

        // State
        this.state = {
            manifest: null,
            currentPath: null,
            searchTerm: '',
            expandedFolders: new Set(),
            selectedFolderPath: null  // Track selected folder for new folder creation
        };

        // Drag and drop state
        this.draggedItem = null;
        this.draggedPath = null;
    }

    /**
     * Initialize the markdown viewer
     */
    async init() {
        try {
            await this.loadManifest();
            this.renderTree();

            // Check URL for initial path
            const urlParams = new URLSearchParams(window.location.search);
            const initialPath = urlParams.get('path');

            if (initialPath) {
                this.loadDocument(initialPath);
            } else {
                const readme = this.findDocument('README.md') || this.findFirstDocument();
                if (readme) {
                    this.loadDocument(readme.path);
                }
            }

            this.setupEventListeners();

        } catch (error) {
            console.error('Markdown viewer initialization error:', error);
            this.showError('Failed to initialize documentation viewer');
            if (this.onError) this.onError(error);
        }
    }

    /**
     * Load the manifest file
     */
    async loadManifest() {
        try {
            const response = await fetch(this.manifestPath);
            if (!response.ok) {
                throw new Error('Failed to load manifest');
            }
            this.state.manifest = await response.json();

            // If rootFolder is set, filter structure to only show that folder
            if (this.rootFolder && this.state.manifest.structure) {
                const rootItem = this.state.manifest.structure.find(
                    item => item.type === 'folder' && item.name === this.rootFolder
                );
                if (rootItem && rootItem.children) {
                    this.state.manifest.structure = rootItem.children;
                }
            }
        } catch (error) {
            console.error('Manifest loading error:', error);
            throw error;
        }
    }

    /**
     * Render the file tree
     */
    renderTree() {
        const container = document.getElementById(this.treeContainerId);
        if (!container) return;

        container.innerHTML = '';

        if (!this.state.manifest || !this.state.manifest.structure) {
            container.innerHTML = '<div class="docs-empty-container"><p class="docs-empty-text">No documentation found</p></div>';
            return;
        }

        const tree = this.buildTree(this.state.manifest.structure);
        container.appendChild(tree);
    }

    /**
     * Build tree DOM recursively
     */
    buildTree(items, parentPath = '') {
        const ul = document.createElement('div');
        ul.className = 'tree-children expanded';

        const filteredItems = this.state.searchTerm
            ? items.filter(item => this.matchesSearch(item))
            : items;

        filteredItems.forEach(item => {
            const li = document.createElement('div');
            li.className = 'tree-item';

            const content = document.createElement('div');
            content.className = 'tree-item-content';

            // Build path (include rootFolder if set)
            let currentPath;
            if (this.rootFolder) {
                currentPath = parentPath
                    ? `${this.rootFolder}/${parentPath}/${item.name}`
                    : `${this.rootFolder}/${item.name}`;
            } else {
                currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
            }

            // Internal path for tree display (without rootFolder prefix)
            const displayPath = parentPath ? `${parentPath}/${item.name}` : item.name;

            if (item.type === 'folder') {
                content.classList.add('folder');

                const toggle = document.createElement('span');
                toggle.className = 'tree-item-toggle';
                toggle.innerHTML = '<i class="fas fa-caret-right"></i>';

                if (this.state.expandedFolders.has(displayPath) || this.state.searchTerm) {
                    toggle.classList.add('expanded');
                }

                content.appendChild(toggle);

                const icon = document.createElement('span');
                icon.className = 'tree-item-icon';
                icon.innerHTML = '<i class="fas fa-folder"></i>';
                content.appendChild(icon);

                const label = document.createElement('span');
                label.className = 'tree-item-label';
                label.textContent = item.name;
                content.appendChild(label);

                content.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.toggleFolder(displayPath, li);
                });

                // Set up drag and drop for folder
                if (this.enableDragDrop) {
                    this.setupDragAndDrop(content, currentPath, true);
                }

                li.appendChild(content);

                if (item.children && item.children.length > 0) {
                    const children = this.buildTree(item.children, displayPath);

                    if (!this.state.expandedFolders.has(displayPath) && !this.state.searchTerm) {
                        children.classList.remove('expanded');
                    }

                    li.appendChild(children);
                }

            } else if (item.type === 'file') {
                const icon = document.createElement('span');
                icon.className = 'tree-item-icon';
                icon.innerHTML = '<i class="fas fa-file-alt"></i>';
                content.appendChild(icon);

                const label = document.createElement('span');
                label.className = 'tree-item-label';
                label.textContent = item.name.replace('.md', '');
                content.appendChild(label);

                content.dataset.path = currentPath;

                if (this.state.currentPath === currentPath) {
                    content.classList.add('active');
                }

                content.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.loadDocument(currentPath);
                });

                // Set up drag and drop for file
                if (this.enableDragDrop) {
                    this.setupDragAndDrop(content, currentPath, false);
                }

                li.appendChild(content);
            }

            ul.appendChild(li);
        });

        return ul;
    }

    /**
     * Toggle folder expanded/collapsed state
     */
    toggleFolder(path, element) {
        if (this.state.expandedFolders.has(path)) {
            this.state.expandedFolders.delete(path);
        } else {
            this.state.expandedFolders.add(path);
        }

        const toggle = element.querySelector('.tree-item-toggle');
        const children = element.querySelector('.tree-children');

        if (toggle) toggle.classList.toggle('expanded');
        if (children) children.classList.toggle('expanded');
    }

    /**
     * Check if item matches search term
     */
    matchesSearch(item) {
        if (!this.state.searchTerm) return true;

        const term = this.state.searchTerm.toLowerCase();

        if (item.name.toLowerCase().includes(term)) return true;

        if (item.type === 'folder' && item.children) {
            return item.children.some(child => this.matchesSearch(child));
        }

        return false;
    }

    /**
     * Load and display a document
     */
    async loadDocument(path) {
        const contentBody = document.getElementById(this.contentBodyId);
        if (!contentBody) return;

        contentBody.innerHTML = `
            <div class="docs-loading-container">
                <div class="docs-loading-spinner"></div>
                <p class="docs-loading-text">Loading document...</p>
            </div>
        `;

        try {
            const response = await fetch(`${this.basePath}/${path}`);
            if (!response.ok) {
                throw new Error('Document not found');
            }

            const markdown = await response.text();
            const html = marked.parse(markdown);

            contentBody.innerHTML = `<div class="markdown-content">${html}</div>`;

            // Highlight code blocks
            contentBody.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
                this.wrapCodeBlock(block);
            });

            this.state.currentPath = path;
            this.updateURL(path);
            this.updateBreadcrumb(path);
            this.updateTreeActiveState(path);
            this.generateTOC();
            contentBody.scrollTop = 0;

            // Enable delete button
            const deleteBtn = document.getElementById(this.deleteBtnId);
            if (deleteBtn && this.enableDelete) {
                deleteBtn.disabled = false;
            }

            if (this.onDocumentLoad) {
                this.onDocumentLoad(path);
            }

        } catch (error) {
            console.error('Document loading error:', error);
            this.showError('Failed to load document');
            const deleteBtn = document.getElementById(this.deleteBtnId);
            if (deleteBtn) deleteBtn.disabled = true;
        }
    }

    /**
     * Wrap code block with header and copy button
     */
    wrapCodeBlock(codeElement) {
        const pre = codeElement.parentElement;
        const language = codeElement.className.match(/language-(\w+)/)?.[1] || 'text';

        const header = document.createElement('div');
        header.className = 'code-header';

        const languageLabel = document.createElement('span');
        languageLabel.className = 'code-language';
        languageLabel.textContent = language;

        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-button';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
        copyBtn.addEventListener('click', () => this.copyCode(codeElement.textContent, copyBtn));

        header.appendChild(languageLabel);
        header.appendChild(copyBtn);

        pre.insertBefore(header, codeElement);
    }

    /**
     * Copy code to clipboard
     */
    async copyCode(code, button) {
        try {
            await navigator.clipboard.writeText(code);

            const originalHTML = button.innerHTML;
            button.innerHTML = '<i class="fas fa-check"></i> Copied!';
            button.classList.add('copied');

            setTimeout(() => {
                button.innerHTML = originalHTML;
                button.classList.remove('copied');
            }, 2000);

        } catch (error) {
            console.error('Copy failed:', error);
        }
    }

    /**
     * Update breadcrumb navigation
     */
    updateBreadcrumb(path) {
        const breadcrumb = document.getElementById(this.breadcrumbId);
        if (!breadcrumb) return;

        const parts = path.split('/');

        // If rootFolder, don't show it in breadcrumb
        const displayParts = this.rootFolder
            ? parts.filter(p => p !== this.rootFolder)
            : parts;

        let html = `
            <span class="docs-breadcrumb-item">
                <i class="fas fa-home"></i>
                <span>${this.title}</span>
            </span>
        `;

        displayParts.forEach((part, index) => {
            html += '<span class="docs-breadcrumb-separator"><i class="fas fa-chevron-right"></i></span>';
            html += `<span class="docs-breadcrumb-item"><span>${part.replace('.md', '')}</span></span>`;
        });

        breadcrumb.innerHTML = html;
    }

    /**
     * Generate table of contents
     */
    generateTOC() {
        const content = document.querySelector('.markdown-content');
        const tocList = document.getElementById(this.tocListId);
        const tocPanel = document.getElementById(this.tocPanelId);

        if (!content || !tocList || !tocPanel) return;

        const headings = content.querySelectorAll('h1, h2, h3');

        if (headings.length <= 1) {
            tocPanel.classList.remove('visible');
            return;
        }

        let html = '';

        headings.forEach((heading, index) => {
            const level = parseInt(heading.tagName.substring(1));
            const text = heading.textContent;
            const id = `heading-${index}`;

            heading.id = id;

            html += `
                <li class="docs-toc-item">
                    <a href="#${id}" class="docs-toc-link level-${level}">${text}</a>
                </li>
            `;
        });

        tocList.innerHTML = html;
        tocPanel.classList.add('visible');

        // Setup click handlers
        tocList.querySelectorAll('.docs-toc-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.getElementById(link.getAttribute('href').substring(1));
                if (target) {
                    const contentBody = document.getElementById(this.contentBodyId);
                    contentBody.scrollTo({
                        top: target.offsetTop - 80,
                        behavior: 'smooth'
                    });
                }
            });
        });

        this.setupScrollSpy();
    }

    /**
     * Setup scroll spy for TOC
     */
    setupScrollSpy() {
        const contentBody = document.getElementById(this.contentBodyId);
        const tocLinks = document.querySelectorAll('.docs-toc-link');

        if (!contentBody || tocLinks.length === 0) return;

        contentBody.addEventListener('scroll', () => {
            const headings = document.querySelectorAll('.markdown-content h1, .markdown-content h2, .markdown-content h3');

            let currentHeading = null;

            headings.forEach(heading => {
                const rect = heading.getBoundingClientRect();
                if (rect.top <= 150) {
                    currentHeading = heading;
                }
            });

            tocLinks.forEach(link => {
                link.classList.remove('active');
                if (currentHeading && link.getAttribute('href') === `#${currentHeading.id}`) {
                    link.classList.add('active');
                }
            });
        });
    }

    /**
     * Update URL with current path
     */
    updateURL(path) {
        const url = new URL(window.location);
        url.searchParams.set('path', path);
        window.history.pushState({}, '', url);
    }

    /**
     * Update tree active state
     */
    updateTreeActiveState(path) {
        document.querySelectorAll('.tree-item-content').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.path === path) {
                item.classList.add('active');

                let parent = item.parentElement;
                while (parent) {
                    if (parent.classList.contains('tree-children')) {
                        parent.classList.add('expanded');
                        const toggle = parent.previousElementSibling?.querySelector('.tree-item-toggle');
                        if (toggle) toggle.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        });
    }

    /**
     * Find a document by name
     */
    findDocument(name) {
        const search = (items, parentPath = '') => {
            for (const item of items) {
                let currentPath;
                if (this.rootFolder) {
                    currentPath = parentPath
                        ? `${this.rootFolder}/${parentPath}/${item.name}`
                        : `${this.rootFolder}/${item.name}`;
                } else {
                    currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
                }

                if (item.type === 'file' && item.name === name) {
                    return { ...item, path: currentPath };
                }

                if (item.type === 'folder' && item.children) {
                    const displayPath = parentPath ? `${parentPath}/${item.name}` : item.name;
                    const found = search(item.children, displayPath);
                    if (found) return found;
                }
            }
            return null;
        };

        return this.state.manifest?.structure ? search(this.state.manifest.structure) : null;
    }

    /**
     * Find the first document in the tree
     */
    findFirstDocument() {
        const search = (items, parentPath = '') => {
            for (const item of items) {
                let currentPath;
                if (this.rootFolder) {
                    currentPath = parentPath
                        ? `${this.rootFolder}/${parentPath}/${item.name}`
                        : `${this.rootFolder}/${item.name}`;
                } else {
                    currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
                }

                if (item.type === 'file') {
                    return { ...item, path: currentPath };
                }

                if (item.type === 'folder' && item.children) {
                    const displayPath = parentPath ? `${parentPath}/${item.name}` : item.name;
                    const found = search(item.children, displayPath);
                    if (found) return found;
                }
            }
            return null;
        };

        return this.state.manifest?.structure ? search(this.state.manifest.structure) : null;
    }

    /**
     * Show error message
     */
    showError(message) {
        const contentBody = document.getElementById(this.contentBodyId);
        if (!contentBody) return;

        contentBody.innerHTML = `
            <div class="docs-error-container">
                <i class="fas fa-exclamation-triangle"></i>
                <p class="docs-error-text">${message}</p>
            </div>
        `;
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'success') {
        const existingToast = document.querySelector('.docs-toast');
        if (existingToast) {
            existingToast.remove();
        }

        const toast = document.createElement('div');
        toast.className = `docs-toast ${type}`;
        toast.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    /**
     * Setup drag and drop
     */
    setupDragAndDrop(element, path, isFolder) {
        if (!isFolder) {
            element.setAttribute('draggable', 'true');
            element.classList.add('draggable');

            element.addEventListener('dragstart', (e) => {
                this.draggedItem = element;
                this.draggedPath = path;
                element.classList.add('dragging');
                e.dataTransfer.setData('text/plain', path);
                e.dataTransfer.effectAllowed = 'move';
            });

            element.addEventListener('dragend', () => {
                element.classList.remove('dragging');
                this.draggedItem = null;
                this.draggedPath = null;

                document.querySelectorAll('.drag-over').forEach(el => {
                    el.classList.remove('drag-over');
                });
            });
        }

        if (isFolder) {
            element.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'move';
                element.classList.add('drag-over');
            });

            element.addEventListener('dragleave', (e) => {
                if (!element.contains(e.relatedTarget)) {
                    element.classList.remove('drag-over');
                }
            });

            element.addEventListener('drop', async (e) => {
                e.preventDefault();
                element.classList.remove('drag-over');

                if (!this.draggedPath) return;

                const sourcePath = this.draggedPath;
                const targetFolder = path;

                const sourceFolder = sourcePath.substring(0, sourcePath.lastIndexOf('/')) || '';
                if (sourceFolder === targetFolder) {
                    this.showToast('File is already in this folder', 'error');
                    return;
                }

                await this.moveDocument(sourcePath, targetFolder);
            });
        }
    }

    /**
     * Move document to a different folder
     */
    async moveDocument(sourcePath, targetFolder) {
        try {
            const response = await fetch('/api/docs/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ sourcePath, targetFolder })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.showToast(`Moved to ${targetFolder || 'root'}`, 'success');

                await this.loadManifest();
                this.renderTree();

                if (this.state.currentPath === sourcePath) {
                    const fileName = sourcePath.split('/').pop();
                    this.state.currentPath = targetFolder ? `${targetFolder}/${fileName}` : fileName;
                    this.updateURL(this.state.currentPath);
                    this.updateTreeActiveState(this.state.currentPath);
                }

            } else {
                throw new Error(result.error || 'Failed to move document');
            }

        } catch (error) {
            console.error('Move error:', error);
            this.showToast(error.message || 'Failed to move document', 'error');
        }
    }

    /**
     * Delete current document
     */
    async deleteDocument() {
        if (!this.state.currentPath) return;

        try {
            const response = await fetch('/api/docs/delete', {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: this.state.currentPath })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.showToast('Document deleted successfully', 'success');

                await this.loadManifest();
                this.renderTree();

                this.state.currentPath = null;
                const deleteBtn = document.getElementById(this.deleteBtnId);
                if (deleteBtn) deleteBtn.disabled = true;

                const contentBody = document.getElementById(this.contentBodyId);
                if (contentBody) {
                    contentBody.innerHTML = `
                        <div class="docs-empty-container">
                            <i class="fas fa-file-alt"></i>
                            <p class="docs-empty-text">Select a document from the sidebar to get started</p>
                        </div>
                    `;
                }

                this.updateURL('');

            } else {
                throw new Error(result.error || 'Failed to delete document');
            }

        } catch (error) {
            console.error('Delete error:', error);
            this.showToast(error.message || 'Failed to delete document', 'error');
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Search
        const searchInput = document.getElementById(this.searchInputId);
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.state.searchTerm = e.target.value;
                this.renderTree();
            });
        }

        // Browser history
        window.addEventListener('popstate', () => {
            const urlParams = new URLSearchParams(window.location.search);
            const path = urlParams.get('path');
            if (path) {
                this.loadDocument(path);
            }
        });
    }

    /**
     * Get current document path
     */
    getCurrentPath() {
        return this.state.currentPath;
    }

    /**
     * Get current document filename
     */
    getCurrentFileName() {
        if (!this.state.currentPath) return null;
        return this.state.currentPath.split('/').pop();
    }

    /**
     * Get selected folder path for new folder creation
     */
    getSelectedFolderPath() {
        return this.state.selectedFolderPath;
    }

    /**
     * Set selected folder path
     */
    setSelectedFolderPath(path) {
        this.state.selectedFolderPath = path;
    }

    /**
     * Create a new folder
     */
    async createFolder(folderName, parentPath = null) {
        if (!this.enableCreateFolder) {
            this.showToast('Folder creation is disabled', 'error');
            return false;
        }

        if (!folderName || !folderName.trim()) {
            this.showToast('Folder name is required', 'error');
            return false;
        }

        try {
            const response = await fetch('/api/docs/folder', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    folderName: folderName.trim(),
                    parentPath: parentPath || this.state.selectedFolderPath || null
                })
            });

            const result = await response.json();

            if (response.ok && result.success) {
                this.showToast(`Folder "${folderName}" created successfully`, 'success');

                // Reload manifest and tree
                await this.loadManifest();
                this.renderTree();

                // Expand the parent folder to show the new folder
                if (parentPath || this.state.selectedFolderPath) {
                    this.state.expandedFolders.add(parentPath || this.state.selectedFolderPath);
                }

                // Also expand the new folder's parent
                if (result.path) {
                    const newFolderParent = result.path.substring(0, result.path.lastIndexOf('/'));
                    if (newFolderParent) {
                        this.state.expandedFolders.add(newFolderParent);
                    }
                }

                this.renderTree();

                if (this.onFolderCreate) {
                    this.onFolderCreate(result.path);
                }

                return true;

            } else {
                throw new Error(result.error || 'Failed to create folder');
            }

        } catch (error) {
            console.error('Create folder error:', error);
            this.showToast(error.message || 'Failed to create folder', 'error');
            return false;
        }
    }

    /**
     * Get all folder paths for selection dropdown
     */
    getAllFolderPaths() {
        const folders = [];

        const collectFolders = (items, parentPath = '') => {
            for (const item of items) {
                if (item.type === 'folder') {
                    const currentPath = parentPath ? `${parentPath}/${item.name}` : item.name;
                    folders.push({
                        path: currentPath,
                        name: item.name,
                        depth: parentPath.split('/').filter(p => p).length
                    });

                    if (item.children) {
                        collectFolders(item.children, currentPath);
                    }
                }
            }
        };

        if (this.state.manifest?.structure) {
            collectFolders(this.state.manifest.structure);
        }

        return folders;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MarkdownViewer;
}

// Make available globally
if (typeof window !== 'undefined') {
    window.MarkdownViewer = MarkdownViewer;
}
