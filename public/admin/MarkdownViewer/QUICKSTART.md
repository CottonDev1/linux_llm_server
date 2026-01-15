# Markdown Viewer - Quick Start Guide

Get started with the documentation viewer in 2 minutes.

## Step 1: Access the Viewer

Start your server and navigate to:

```
http://localhost:3000/admin/MarkdownViewer/
```

You should see:
- Left sidebar with documentation tree
- Main content area
- Right sidebar with table of contents (when applicable)

## Step 2: Navigate Documents

**Click folders** to expand/collapse:
- "Getting Started" folder contains README.md
- "Architecture" folder contains Overview.md

**Click files** to view:
- Click "README.md" to see the main documentation
- Click "Overview.md" to see the architecture guide

**Use breadcrumbs** at the top to track your location.

## Step 3: Try Features

### Search
Type in the search box (left sidebar) to filter documents:
- Try searching for "SQL"
- Try searching for "Architecture"

### Code Blocks
Documents with code blocks have a "Copy" button:
1. Find a code block
2. Click the "Copy" button
3. Paste anywhere to use the code

### Share Links
Click "Share Link" button to copy the current document URL:
1. Navigate to any document
2. Click "Share Link"
3. Paste the URL to share

### Print
Click "Print" button to print the current document:
1. Navigation is automatically hidden
2. Code wraps for readability
3. Optimized for paper

## Step 4: Add Your Own Documentation

### Quick Method

1. **Create a markdown file:**
   ```bash
   echo "# My First Doc" > /mnt/c/Projects/llm_website/public/docs/MyDoc.md
   ```

2. **Regenerate manifest:**
   ```bash
   node /mnt/c/Projects/llm_website/scripts/generate-docs-manifest.js
   ```

3. **Refresh the viewer** - Your document appears in the tree!

### Organized Method

1. **Create a folder structure:**
   ```bash
   mkdir -p /mnt/c/Projects/llm_website/public/docs/Guides
   ```

2. **Add markdown files:**
   ```bash
   cat > /mnt/c/Projects/llm_website/public/docs/Guides/MyGuide.md << 'EOF'
   # My Guide

   This is my documentation guide.

   ## Section 1

   Content here...

   ## Section 2

   More content...
   EOF
   ```

3. **Generate manifest:**
   ```bash
   node /mnt/c/Projects/llm_website/scripts/generate-docs-manifest.js
   ```

4. **Refresh** - "Guides" folder appears with "MyGuide.md"!

## Markdown Tips

### Headings
```markdown
# H1 - Main Title
## H2 - Section
### H3 - Subsection
```

### Code Blocks
````markdown
```javascript
function hello() {
  console.log('Hello!');
}
```
````

### Tables
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

### Blockquotes
```markdown
> This is a blockquote
> With multiple lines
```

### Links
```markdown
[Link Text](URL)
[Internal Link](../Other/Document.md)
```

## File Organization

**Best practices:**

```
/public/docs/
├── Getting Started/
│   ├── README.md           # Start here
│   ├── Installation.md
│   └── Quick Start.md
├── Architecture/
│   ├── Overview.md
│   └── Components/
│       ├── Backend.md
│       └── Frontend.md
├── API/
│   ├── REST.md
│   └── WebSocket.md
├── Guides/
│   ├── How To X.md
│   └── How To Y.md
└── FAQ.md
```

## Troubleshooting

### Document not showing up?
1. Check filename ends with `.md`
2. Run manifest generator
3. Refresh browser

### Styling looks wrong?
1. Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
2. Clear browser cache
3. Check browser console for errors

### Search not working?
1. Make sure you're typing in the sidebar search box
2. Search is case-insensitive
3. Clear search to see all documents

## Next Steps

- **[Full Documentation](README.md)**: Complete feature guide
- **[Example Docs](../../docs/Getting%20Started/README.md)**: See markdown examples
- **[FAQ](../../docs/FAQ.md)**: Common questions

## Support

For issues or questions:
1. Check the [README](README.md)
2. Review example documents
3. Check browser console for errors

---

That's it! You're ready to use the Markdown Viewer.
