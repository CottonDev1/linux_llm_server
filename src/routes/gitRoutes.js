/**
 * Git Routes Module
 *
 * Handles all Git-related operations:
 * - Pull updates from repositories
 * - Update context from Git repositories
 * - List and manage Git repositories
 * - Sync individual repositories
 */

import express from 'express';
import { exec, execSync } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const router = express.Router();
const execAsync = promisify(exec);

/**
 * Initialize Git routes with required dependencies
 *
 * @param {Object} dependencies - Service and middleware dependencies
 * @param {Function} dependencies.requireAuth - Authentication middleware
 * @param {Object} dependencies.ewraiDatabase - EWRAIDatabase instance
 * @param {Array} dependencies.gitRepositories - Git repositories configuration
 * @param {Function} dependencies.scanForGitRepositories - Function to scan for git repos
 * @param {Function} dependencies.saveGitRepoConfig - Function to save git repo config
 * @param {Function} dependencies.getRecentCommits - Function to get recent commits
 * @param {Function} dependencies.getLastPullTime - Function to get last pull time
 * @param {boolean} dependencies.userDBInitialized - Flag if user DB is initialized
 */
export default function initGitRoutes(dependencies) {
  const {
    requireAuth,
    ewraiDatabase,
    gitRepositories,
    scanForGitRepositories,
    saveGitRepoConfig,
    getRecentCommits,
    getLastPullTime,
    userDBInitialized
  } = dependencies;

  // ============================================================================
  // Git Operations
  // ============================================================================

  /**
   * POST /api/admin/git/pull
   * Execute Git operations (pull updates from repository)
   */
  router.post('/pull', requireAuth, async (req, res) => {
    try {
      console.log('🔄 Executing git pull for all repositories...');

      // Pull all 4 repositories
      const GIT_ROOT = process.env.GIT_ROOT;
      const repos = [
        { name: 'Gin', path: `${GIT_ROOT}\\Gin`, dbName: 'gin' },
        { name: 'Marketing', path: `${GIT_ROOT}\\Marketing`, dbName: 'marketing' },
        { name: 'Warehouse', path: `${GIT_ROOT}\\Warehouse`, dbName: 'warehouse' },
        { name: 'EWR Library', path: `${GIT_ROOT}\\EWR Library`, dbName: 'ewrlibrary' }
      ];

      const results = [];

      for (const repo of repos) {
        try {
          console.log(`\n📦 Processing ${repo.name}...`);

          // Git pull
          console.log(`  🔄 Git pull...`);
          const { stdout: pullOutput } = await execAsync(`git pull`, {
            cwd: repo.path,
            timeout: 60000
          });

          const hasChanges = !pullOutput.includes('Already up to date');

          if (hasChanges) {
            console.log(`  📝 Changes detected, running Roslyn analysis...`);

            // Run Roslyn analysis
            const { tmpdir } = await import('os');
            const { join } = await import('path');
            const { promises: fs } = await import('fs');

            const outputFile = join(tmpdir(), `roslyn-analysis-${repo.dbName}-${Date.now()}.json`);
            const analyzerDll = 'roslyn-analyzer/RoslynCodeAnalyzer/bin/Debug/net8.0/RoslynCodeAnalyzer.dll';

            const analyzeCmd = `dotnet ${analyzerDll} "${repo.path}" "${outputFile}"`;
            await execAsync(analyzeCmd, {
              maxBuffer: 50 * 1024 * 1024,
              cwd: process.cwd(),
              timeout: 300000
            });

            // Import to vector database
            console.log(`  💾 Importing to vector database...`);
            const importCmd = `node code-import-pipeline.js import "${outputFile}" "${repo.dbName}"`;
            await execAsync(importCmd, {
              maxBuffer: 50 * 1024 * 1024,
              cwd: process.cwd(),
              timeout: 300000
            });

            // Clean up
            await fs.unlink(outputFile).catch(() => {});

            results.push(`✅ ${repo.name}: Pulled and analyzed`);
          } else {
            results.push(`✅ ${repo.name}: Already up to date`);
          }

        } catch (error) {
          console.error(`  ❌ Error processing ${repo.name}:`, error.message);
          results.push(`❌ ${repo.name}: ${error.message}`);
        }
      }

      res.json({
        success: true,
        message: 'Git pull and analysis completed',
        results: results
      });

    } catch (error) {
      console.error('❌ Git pull process failed:', error.message);
      res.status(500).json({
        success: false,
        error: 'Git pull process failed',
        details: error.message
      });
    }
  });

  /**
   * POST /api/admin/git/update-context
   * Update Git context from repositories
   */
  router.post('/update-context', requireAuth, async (req, res) => {
    try {
      const { repo, changedOnly = true } = req.body;

      console.log('🔄 Updating context from Git repositories...');

      let command = 'node update-context-from-git.js';
      if (repo) command += ` --repo ${repo}`;
      if (changedOnly) command += ' --changed-only';

      const { stdout, stderr } = await execAsync(command, {
        cwd: process.cwd(),
        maxBuffer: 10 * 1024 * 1024
      });

      res.json({
        success: true,
        message: 'Context update completed',
        output: stdout,
        errors: stderr || null
      });

    } catch (error) {
      console.error('❌ Context update failed:', error.message);
      res.status(500).json({
        success: false,
        error: 'Context update failed',
        details: error.message
      });
    }
  });

  /**
   * GET /api/admin/git/repositories
   * List all repositories from filesystem scan with enriched status
   * Primary source: Filesystem scan (standard repos in GIT_ROOT)
   * Secondary source: Database (admin-added non-standard repos only)
   */
  router.get('/repositories', requireAuth, async (req, res) => {
    try {
      // Primary: Scan filesystem for repos - this is the source of truth
      const scannedRepos = scanForGitRepositories ? scanForGitRepositories() : gitRepositories;
      const repoMap = new Map();

      // Add scanned repos first (these are the standard ones from filesystem)
      // Use lowercase name as key to avoid case-sensitive duplicates
      for (const repo of scannedRepos) {
        const key = repo.name.toLowerCase();
        repoMap.set(key, {
          id: repo.name,
          name: repo.name,
          path: repo.path,
          branch: 'master',
          displayName: repo.displayName || repo.name,
          projectName: repo.projectName || null,
          source: 'filesystem'
        });
      }

      // Secondary: Add admin-added repos from database (only non-standard locations)
      if (userDBInitialized && ewraiDatabase) {
        const dbRepos = ewraiDatabase.getAllGitRepositories();
        for (const repo of dbRepos) {
          // Use lowercase name as key to avoid case-sensitive duplicates
          const key = repo.name.toLowerCase();
          // Only add if not already found in filesystem scan
          if (!repoMap.has(key)) {
            repoMap.set(key, {
              id: repo.repo_id || repo.name,
              name: repo.name,
              path: repo.path,
              branch: repo.branch || 'master',
              displayName: repo.display_name || repo.name,
              projectName: repo.project_name || null,
              accessToken: repo.access_token ? '***MASKED***' : null,
              lastSync: repo.last_pull_time || null,
              lastCommitHash: repo.last_commit_hash || null,
              source: 'database'
            });
          }
        }
      }

      // Enrich repository data with current status
      const enrichedRepos = [];
      for (const repo of repoMap.values()) {
        try {
          const commits = await getRecentCommits(repo.path, 5);
          const lastPull = await getLastPullTime(repo.path);

          enrichedRepos.push({
            id: repo.id,
            name: repo.name,
            path: repo.path,
            branch: repo.branch,
            displayName: repo.displayName || repo.name,
            projectName: repo.projectName,
            accessToken: repo.accessToken,
            lastSync: lastPull || repo.lastSync,
            lastCommitHash: repo.lastCommitHash,
            recentCommits: commits,
            status: commits.length > 0 ? 'active' : 'error',
            source: repo.source
          });
        } catch (repoError) {
          console.error(`Error enriching ${repo.name}:`, repoError.message);
          enrichedRepos.push({
            ...repo,
            recentCommits: [],
            status: 'error'
          });
        }
      }

      res.json({
        success: true,
        repositories: enrichedRepos
      });
    } catch (error) {
      console.error('❌ Error listing repositories:', error);
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * POST /api/admin/git/repositories
   * Add new repository
   */
  router.post('/repositories', requireAuth, async (req, res) => {
    try {
      const { name, path, branch = 'main', syncInterval = 60 } = req.body;

      if (!name || !path) {
        return res.status(400).json({ error: 'Name and path are required' });
      }

      // Check if repository already exists
      if (gitRepositories.find(r => r.path === path)) {
        return res.status(400).json({ error: 'Repository already exists' });
      }

      // Verify repository exists
      try {
        execSync(`cd "${path}" && git status`, { encoding: 'utf8' });
      } catch (error) {
        return res.status(400).json({ error: 'Invalid git repository path' });
      }

      const newRepo = {
        id: Date.now().toString(),
        name,
        path,
        branch,
        syncInterval,
        addedAt: new Date().toISOString(),
        status: 'active'
      };

      gitRepositories.push(newRepo);
      await saveGitRepoConfig();

      console.log(`✅ Added repository: ${name}`);

      res.json({
        success: true,
        repository: newRepo
      });
    } catch (error) {
      console.error('❌ Error adding repository:', error);
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * POST /api/admin/git/repositories/:id/sync
   * Sync a repository (pull latest changes)
   */
  router.post('/repositories/:id/sync', requireAuth, async (req, res) => {
    try {
      const { id } = req.params;
      const repo = gitRepositories.find(r => r.id === id);

      if (!repo) {
        return res.status(404).json({ error: 'Repository not found' });
      }

      console.log(`🔄 Syncing repository: ${repo.name}`);

      // Pull latest changes
      try {
        const output = execSync(
          `cd "${repo.path}" && git pull origin ${repo.branch}`,
          { encoding: 'utf8' }
        );
        console.log(`   Git pull output: ${output}`);
      } catch (error) {
        console.error(`   Git pull failed: ${error.message}`);
        return res.status(500).json({ error: 'Git pull failed: ' + error.message });
      }

      // TODO: Trigger re-indexing of the repository
      // This would call your existing git analysis code

      console.log(`✅ Synced repository: ${repo.name}`);

      res.json({
        success: true,
        message: `Repository ${repo.name} synced successfully`
      });
    } catch (error) {
      console.error('❌ Error syncing repository:', error);
      res.status(500).json({ error: error.message });
    }
  });

  /**
   * DELETE /api/admin/git/repositories/:id
   * Remove a repository from monitoring
   */
  router.delete('/repositories/:id', requireAuth, async (req, res) => {
    try {
      const { id } = req.params;
      const index = gitRepositories.findIndex(r => r.id === id);

      if (index === -1) {
        return res.status(404).json({ error: 'Repository not found' });
      }

      const removedRepo = gitRepositories[index];
      gitRepositories.splice(index, 1);
      await saveGitRepoConfig();

      console.log(`🗑️ Removed repository: ${removedRepo.name}`);

      res.json({
        success: true,
        message: `Repository ${removedRepo.name} removed`
      });
    } catch (error) {
      console.error('❌ Error removing repository:', error);
      res.status(500).json({ error: error.message });
    }
  });

  return router;
}
