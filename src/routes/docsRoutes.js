/**
 * Documentation Routes Module
 *
 * API endpoints for managing documentation files:
 * - DELETE /api/docs/delete - Delete a document
 * - POST /api/docs/move - Move a document to a different folder
 */

import express from 'express';
import path from 'path';
import fs from 'fs/promises';
import { execSync } from 'child_process';

const router = express.Router();

// Base path for documentation files
const DOCS_BASE_PATH = path.join(process.cwd(), 'public', 'docs');

/**
 * Validate that a path is within the docs directory (prevent path traversal)
 */
function isValidDocsPath(filePath) {
    const normalizedPath = path.normalize(filePath);
    const resolvedPath = path.resolve(DOCS_BASE_PATH, normalizedPath);
    return resolvedPath.startsWith(DOCS_BASE_PATH);
}

/**
 * Regenerate the docs manifest after file operations
 */
async function regenerateManifest() {
    try {
        const scriptPath = path.join(process.cwd(), 'scripts', 'generate-docs-manifest.js');
        execSync(`node "${scriptPath}"`, {
            cwd: process.cwd(),
            stdio: 'pipe'
        });
        console.log('📄 Docs manifest regenerated');
        return true;
    } catch (error) {
        console.error('Failed to regenerate manifest:', error.message);
        return false;
    }
}

/**
 * Initialize docs routes with required dependencies
 */
export default function createDocsRoutes(dependencies = {}) {
    const { requireAuthOrAdmin } = dependencies;

    /**
     * DELETE /api/docs/delete
     * Delete a documentation file
     */
    router.delete('/delete', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
        try {
            const { path: docPath } = req.body;

            if (!docPath) {
                return res.status(400).json({
                    success: false,
                    error: 'Document path is required'
                });
            }

            // Validate path
            if (!isValidDocsPath(docPath)) {
                return res.status(400).json({
                    success: false,
                    error: 'Invalid document path'
                });
            }

            const fullPath = path.join(DOCS_BASE_PATH, docPath);

            // Check if file exists
            try {
                const stats = await fs.stat(fullPath);
                if (!stats.isFile()) {
                    return res.status(400).json({
                        success: false,
                        error: 'Path is not a file'
                    });
                }
            } catch (err) {
                return res.status(404).json({
                    success: false,
                    error: 'Document not found'
                });
            }

            // Delete the file
            await fs.unlink(fullPath);
            console.log(`🗑️ Deleted document: ${docPath}`);

            // Check if parent directory is empty and clean up
            const parentDir = path.dirname(fullPath);
            if (parentDir !== DOCS_BASE_PATH) {
                try {
                    const files = await fs.readdir(parentDir);
                    if (files.length === 0) {
                        await fs.rmdir(parentDir);
                        console.log(`📁 Removed empty directory: ${path.dirname(docPath)}`);
                    }
                } catch (err) {
                    // Directory cleanup is optional, don't fail on this
                }
            }

            // Regenerate manifest
            await regenerateManifest();

            res.json({
                success: true,
                message: `Document "${docPath}" deleted successfully`
            });

        } catch (error) {
            console.error('Delete error:', error);
            res.status(500).json({
                success: false,
                error: error.message || 'Failed to delete document'
            });
        }
    });

    /**
     * POST /api/docs/folder
     * Create a new folder
     */
    router.post('/folder', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
        try {
            const { parentPath, folderName } = req.body;

            if (!folderName) {
                return res.status(400).json({
                    success: false,
                    error: 'Folder name is required'
                });
            }

            // Validate folder name (no special characters except spaces, hyphens, underscores)
            const validFolderName = /^[a-zA-Z0-9\s\-_]+$/.test(folderName);
            if (!validFolderName) {
                return res.status(400).json({
                    success: false,
                    error: 'Folder name can only contain letters, numbers, spaces, hyphens, and underscores'
                });
            }

            // Validate parent path if provided
            if (parentPath && !isValidDocsPath(parentPath)) {
                return res.status(400).json({
                    success: false,
                    error: 'Invalid parent path'
                });
            }

            const newFolderPath = parentPath
                ? path.join(DOCS_BASE_PATH, parentPath, folderName)
                : path.join(DOCS_BASE_PATH, folderName);

            // Validate the new folder path
            if (!isValidDocsPath(parentPath ? `${parentPath}/${folderName}` : folderName)) {
                return res.status(400).json({
                    success: false,
                    error: 'Invalid folder path'
                });
            }

            // Check if folder already exists
            try {
                await fs.stat(newFolderPath);
                return res.status(400).json({
                    success: false,
                    error: 'A folder with this name already exists'
                });
            } catch (err) {
                // Folder doesn't exist, which is what we want
            }

            // Create the folder
            await fs.mkdir(newFolderPath, { recursive: true });
            console.log(`📁 Created folder: ${parentPath ? `${parentPath}/${folderName}` : folderName}`);

            // Regenerate manifest
            await regenerateManifest();

            res.json({
                success: true,
                message: `Folder "${folderName}" created successfully`,
                path: parentPath ? `${parentPath}/${folderName}` : folderName
            });

        } catch (error) {
            console.error('Create folder error:', error);
            res.status(500).json({
                success: false,
                error: error.message || 'Failed to create folder'
            });
        }
    });

    /**
     * POST /api/docs/move
     * Move a document to a different folder
     */
    router.post('/move', requireAuthOrAdmin || ((req, res, next) => next()), async (req, res) => {
        try {
            const { sourcePath, targetFolder } = req.body;

            if (!sourcePath) {
                return res.status(400).json({
                    success: false,
                    error: 'Source path is required'
                });
            }

            // Validate paths
            if (!isValidDocsPath(sourcePath)) {
                return res.status(400).json({
                    success: false,
                    error: 'Invalid source path'
                });
            }

            if (targetFolder && !isValidDocsPath(targetFolder)) {
                return res.status(400).json({
                    success: false,
                    error: 'Invalid target folder'
                });
            }

            const sourceFullPath = path.join(DOCS_BASE_PATH, sourcePath);
            const fileName = path.basename(sourcePath);
            const targetDir = targetFolder
                ? path.join(DOCS_BASE_PATH, targetFolder)
                : DOCS_BASE_PATH;
            const targetFullPath = path.join(targetDir, fileName);

            // Check if source exists
            try {
                const stats = await fs.stat(sourceFullPath);
                if (!stats.isFile()) {
                    return res.status(400).json({
                        success: false,
                        error: 'Source is not a file'
                    });
                }
            } catch (err) {
                return res.status(404).json({
                    success: false,
                    error: 'Source document not found'
                });
            }

            // Check if target already exists
            try {
                await fs.stat(targetFullPath);
                return res.status(400).json({
                    success: false,
                    error: 'A file with this name already exists in the target folder'
                });
            } catch (err) {
                // File doesn't exist, which is what we want
            }

            // Ensure target directory exists
            await fs.mkdir(targetDir, { recursive: true });

            // Move the file
            await fs.rename(sourceFullPath, targetFullPath);
            console.log(`📦 Moved document: ${sourcePath} → ${targetFolder || 'root'}/${fileName}`);

            // Clean up empty source directory
            const sourceDir = path.dirname(sourceFullPath);
            if (sourceDir !== DOCS_BASE_PATH) {
                try {
                    const files = await fs.readdir(sourceDir);
                    if (files.length === 0) {
                        await fs.rmdir(sourceDir);
                        console.log(`📁 Removed empty directory: ${path.dirname(sourcePath)}`);
                    }
                } catch (err) {
                    // Directory cleanup is optional
                }
            }

            // Regenerate manifest
            await regenerateManifest();

            const newPath = targetFolder ? `${targetFolder}/${fileName}` : fileName;

            res.json({
                success: true,
                message: `Document moved successfully`,
                newPath: newPath
            });

        } catch (error) {
            console.error('Move error:', error);
            res.status(500).json({
                success: false,
                error: error.message || 'Failed to move document'
            });
        }
    });

    return router;
}
