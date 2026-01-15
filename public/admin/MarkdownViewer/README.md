# Markdown Documentation Viewer

A modern, feature-rich documentation viewer with dark theme and advanced navigation capabilities.

## Features

### Core Functionality
- **Tree Navigation**: Expandable/collapsible folder structure in left sidebar
- **Markdown Rendering**: GitHub-flavored markdown with syntax highlighting
- **Table of Contents**: Auto-generated from headings with scroll-spy
- **Deep Linking**: URL parameters for direct document access
- **Search**: Filter documents by name
- **Responsive Design**: Mobile-friendly with hamburger menu

### Visual Features
- **Dark Theme**: Gradient background (#1a1a2e to #16213e) with teal accents (#4ecdc4)
- **Code Blocks**: Syntax highlighting with copy buttons
- **Tables**: Styled GitHub-flavored markdown tables
- **Breadcrumbs**: Current document path navigation
- **Loading States**: Smooth transitions and loading indicators

### Developer Features
- **Print Friendly**: Optimized print styles
- **Share Links**: Copy current document URL
- **Error Handling**: Graceful fallbacks for missing content
- **Performance**: Efficient rendering and navigation

## Usage

### Accessing the Viewer

Navigate to:
```
http://localhost:3000/admin/MarkdownViewer/
```

### Deep Linking

Link directly to a specific document:
```
http://localhost:3000/admin/MarkdownViewer/?path=Getting%20Started/README.md
```

### Adding Documentation

1. **Add Markdown Files**: Place `.md` files in `/public/docs/`

```bash
/public/docs/
├── Getting Started/
│   ├── README.md
│   └── Installation.md
├── Architecture/
│   ├── Overview.md
│   └── Pipelines/
│       └── SQL_QUERY_PIPELINE.md
└── API/
    └── REST Endpoints.md
```

2. **Generate Manifest**: Run the manifest generator

```bash
node scripts/generate-docs-manifest.js
```

This creates `/public/docs/docs-manifest.json` with the directory structure.

3. **Refresh**: Reload the viewer to see new documents

### Manifest Structure

The `docs-manifest.json` file defines the navigation tree:

```json
{
  "version": "1.0.0",
  "generated": "2025-12-30T00:00:00Z",
  "structure": [
    {
      "name": "Getting Started",
      "type": "folder",
      "children": [
        {
          "name": "README.md",
          "type": "file"
        }
      ]
    }
  ]
}
```

## Markdown Features

### Supported Elements

- **Headings** (h1-h6) - Auto-indexed for TOC
- **Paragraphs** - Standard text blocks
- **Lists** - Ordered and unordered
- **Code Blocks** - With syntax highlighting and copy button
- **Inline Code** - Styled with teal accent
- **Tables** - GitHub-flavored markdown tables
- **Blockquotes** - Styled with left border
- **Links** - Internal and external
- **Images** - Max-width contained
- **Horizontal Rules** - Section dividers

### Code Blocks

Code blocks support syntax highlighting for:
- JavaScript/TypeScript
- Python
- SQL
- JSON
- Bash/Shell
- HTML/CSS
- And many more via highlight.js

Example:
````markdown
```javascript
function hello() {
  console.log('Hello, World!');
}
```
````

### Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
```

## Customization

### Theme Colors

Edit the CSS variables in `index.html`:

```css
:root {
    --bg-primary: #1a1a2e;
    --bg-secondary: #16213e;
    --accent: #4ecdc4;
    --text-primary: #e4e4e7;
    /* ... */
}
```

### Sidebar Width

```css
:root {
    --sidebar-width: 280px;
    --toc-width: 250px;
}
```

### Search Behavior

Modify the `matchesSearch()` function to customize search logic.

## File Structure

```
/public/admin/MarkdownViewer/
├── index.html          # Main viewer (complete, standalone)
└── README.md           # This file

/public/docs/
├── docs-manifest.json  # Auto-generated navigation manifest
├── Getting Started/
│   └── README.md
└── [other docs]/

/scripts/
└── generate-docs-manifest.js  # Manifest generator utility
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (responsive design)

## Dependencies

All dependencies are loaded via CDN:

- **marked.js** (v11.1.1) - Markdown parsing
- **highlight.js** (v11.9.0) - Syntax highlighting
- **Font Awesome** (v6.5.1) - Icons

No build process required - runs standalone in the browser.

## Keyboard Shortcuts

- **Arrow Keys**: Navigate tree when focused
- **Enter**: Open selected document
- **Escape**: Close mobile menu
- **Ctrl/Cmd + P**: Print current document

## Mobile Experience

On mobile devices (≤768px):

- Sidebar becomes full-screen overlay
- Hamburger menu button appears (top-left)
- TOC sidebar hidden
- Touch-optimized controls

## Performance

### Optimizations
- Lazy rendering of tree items
- Debounced search input
- Efficient DOM updates
- Scroll-based TOC activation

### Caching
The viewer fetches documents on demand. Consider adding service worker caching for offline support.

## Troubleshooting

### Documents Not Loading

1. **Check manifest exists**: `/public/docs/docs-manifest.json`
2. **Verify file paths**: Paths are relative to `/public/docs/`
3. **Check console**: Look for network errors
4. **Regenerate manifest**: Run `generate-docs-manifest.js`

### Search Not Working

1. **Case sensitive**: Search is case-insensitive by default
2. **File extensions**: `.md` extension not required in search
3. **Clear input**: Click X or clear manually

### Styling Issues

1. **Clear cache**: Hard refresh (Ctrl+Shift+R)
2. **Check CSS**: Custom properties properly set
3. **Browser compatibility**: Use modern browser

## Future Enhancements

Potential improvements:

- [ ] Full-text search (not just filenames)
- [ ] Version selector for documentation versions
- [ ] Dark/light theme toggle
- [ ] Export to PDF
- [ ] Collaborative annotations
- [ ] Search result highlighting
- [ ] Bookmark favorite documents
- [ ] Recently viewed history

## License

Part of the LLM Website project.

## Support

For issues or questions, see the main project documentation.
