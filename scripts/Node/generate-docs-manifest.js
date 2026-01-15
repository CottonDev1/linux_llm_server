#!/usr/bin/env node

/**
 * Generate docs-manifest.json from /public/docs directory structure
 *
 * This script scans the /public/docs directory and creates a JSON manifest
 * that the Markdown Viewer uses for navigation.
 *
 * Usage:
 *   node scripts/generate-docs-manifest.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const DOCS_DIR = path.join(__dirname, '../public/docs');
const OUTPUT_FILE = path.join(DOCS_DIR, 'docs-manifest.json');
const IGNORE_FILES = ['.DS_Store', 'Thumbs.db', 'docs-manifest.json'];
const IGNORE_DIRS = ['.git', 'node_modules'];

/**
 * Recursively build directory structure
 */
function buildStructure(dirPath, relativePath = '') {
    const items = [];

    try {
        const entries = fs.readdirSync(dirPath, { withFileTypes: true });

        // Sort: directories first, then files, alphabetically
        entries.sort((a, b) => {
            if (a.isDirectory() && !b.isDirectory()) return -1;
            if (!a.isDirectory() && b.isDirectory()) return 1;
            return a.name.localeCompare(b.name);
        });

        for (const entry of entries) {
            const fullPath = path.join(dirPath, entry.name);
            const relPath = relativePath ? `${relativePath}/${entry.name}` : entry.name;

            if (entry.isDirectory()) {
                // Skip ignored directories
                if (IGNORE_DIRS.includes(entry.name)) {
                    continue;
                }

                // Recursively process directory
                const children = buildStructure(fullPath, relPath);

                // Only include non-empty directories
                if (children.length > 0) {
                    items.push({
                        name: entry.name,
                        type: 'folder',
                        children: children
                    });
                }

            } else if (entry.isFile()) {
                // Skip ignored files and non-markdown files
                if (IGNORE_FILES.includes(entry.name) || !entry.name.endsWith('.md')) {
                    continue;
                }

                items.push({
                    name: entry.name,
                    type: 'file'
                });
            }
        }

    } catch (error) {
        console.error(`Error reading directory ${dirPath}:`, error.message);
    }

    return items;
}

/**
 * Count total files and folders
 */
function countItems(structure) {
    let files = 0;
    let folders = 0;

    function count(items) {
        for (const item of items) {
            if (item.type === 'folder') {
                folders++;
                if (item.children) {
                    count(item.children);
                }
            } else {
                files++;
            }
        }
    }

    count(structure);
    return { files, folders };
}

/**
 * Main execution
 */
function main() {
    console.log('Generating documentation manifest...\n');

    // Check if docs directory exists
    if (!fs.existsSync(DOCS_DIR)) {
        console.error(`Error: Documentation directory not found: ${DOCS_DIR}`);
        console.error('Please create the directory and add markdown files.');
        process.exit(1);
    }

    // Build structure
    console.log(`Scanning: ${DOCS_DIR}`);
    const structure = buildStructure(DOCS_DIR);

    // Count items
    const { files, folders } = countItems(structure);

    // Create manifest
    const manifest = {
        version: '1.0.0',
        generated: new Date().toISOString(),
        stats: {
            totalFiles: files,
            totalFolders: folders
        },
        structure: structure
    };

    // Write to file
    try {
        fs.writeFileSync(
            OUTPUT_FILE,
            JSON.stringify(manifest, null, 2),
            'utf8'
        );

        console.log(`\n✓ Manifest generated successfully!`);
        console.log(`  Location: ${OUTPUT_FILE}`);
        console.log(`  Files: ${files}`);
        console.log(`  Folders: ${folders}`);
        console.log(`  Generated: ${manifest.generated}`);

    } catch (error) {
        console.error(`Error writing manifest file:`, error.message);
        process.exit(1);
    }
}

// Run the script
main();

export { buildStructure, countItems };
