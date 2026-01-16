import express from 'express';

/**
 * Navigation Routes
 *
 * Provides dynamic navigation structure by scanning the public directory.
 * Directories become dropdown categories, HTML files become page links.
 *
 * @param {Object} dependencies - Optional dependencies (none required currently)
 * @returns {express.Router} Configured Express router
 */
export default function initNavigationRoutes(dependencies = {}) {
  const router = express.Router();

  // ============================================================================
  // Navigation Structure - Dynamic Sidebar Population
  // ============================================================================

  /**
   * GET /api/admin/navigation
   * Scans the public directory structure and returns navigation categories/pages
   * Directories become dropdown categories, HTML files become page links
   */
  router.get('/navigation', async (req, res) => {
    try {
      const fs = await import('fs');
      const path = await import('path');

      const publicDir = path.default.join(process.cwd(), 'public');
      const configPath = path.default.join(process.cwd(), 'config', 'nav_config.json');

      // Load navigation config (page metadata, icons, roles, etc.)
      let navConfig = {};
      try {
        const configContent = fs.default.readFileSync(configPath, 'utf8');
        navConfig = JSON.parse(configContent);
      } catch (e) {
        console.warn('No nav_config.json found, using defaults');
      }

      const categories = [];
      const excludeDirs = navConfig.excludeDirs || ['css', 'js', 'data', 'images', 'fonts'];
      const excludeFiles = navConfig.excludeFiles || ['login.html', 'index.html'];

      // Helper functions to reduce filter duplication
      // Filters directories: excludes those in excludeDirs list and those starting with '_'
      const filterDirs = (entries) => entries
        .filter(d => d.isDirectory() && !excludeDirs.includes(d.name) && !d.name.startsWith('_'));

      // Filters HTML files: excludes those in excludeFiles list and those starting with '_'
      const filterHtmlFiles = (files) => files
        .filter(f => f.endsWith('.html') && !excludeFiles.includes(f) && !f.startsWith('_'));

      // Helper function to get page info from an HTML file
      const getPageInfo = (filePath, pageId, pageConfig, dirConfig, urlPath) => {
        let displayName = pageConfig.name || pageId.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        try {
          const htmlContent = fs.default.readFileSync(filePath, 'utf8');
          const titleMatch = htmlContent.match(/<title[^>]*>([^<]+)</i);
          if (titleMatch && !pageConfig.name) {
            displayName = titleMatch[1].replace(' - EWR Admin', '').replace(' - EWR', '').trim();
          }
        } catch (e) { /* ignore */ }

        return {
          id: pageId,
          name: displayName,
          url: urlPath,
          icon: pageConfig.icon || navConfig.defaultIcon || '<svg viewBox="0 0 24 24"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm-1 7V3.5L18.5 9H13z"/></svg>',
          requiredRole: pageConfig.requiredRole || dirConfig.defaultRole || 'user',
          description: pageConfig.description || '',
          disabled: pageConfig.disabled || false
        };
      };

      // Scan admin directory
      const adminDir = path.default.join(publicDir, 'admin');
      const adminConfig = navConfig.categories?.['admin'] || {};

      // Scan admin subdirectories as separate categories
      const adminSubdirs = filterDirs(fs.default.readdirSync(adminDir, { withFileTypes: true }));

      for (const subdir of adminSubdirs) {
        const subdirPath = path.default.join(adminDir, subdir.name);
        const subdirConfig = navConfig.categories?.[subdir.name] || {};

        const subFiles = filterHtmlFiles(fs.default.readdirSync(subdirPath));

        if (subFiles.length === 0) continue;

        const subPages = subFiles.map(file => {
          const pageId = file.replace('.html', '');
          const pageConfig = subdirConfig.pages?.[pageId] || {};
          const filePath = path.default.join(subdirPath, file);
          const urlPath = `${subdir.name}/${file}`;

          return getPageInfo(filePath, pageId, pageConfig, subdirConfig, urlPath);
        });

        // Sort pages by order
        subPages.sort((a, b) => {
          const orderA = subdirConfig.pages?.[a.id]?.order ?? 999;
          const orderB = subdirConfig.pages?.[b.id]?.order ?? 999;
          if (orderA !== orderB) return orderA - orderB;
          return a.name.localeCompare(b.name);
        });

        categories.push({
          id: subdir.name,
          name: subdirConfig.name || subdir.name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
          icon: subdirConfig.icon || navConfig.defaultCategoryIcon || '<svg viewBox="0 0 24 24"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>',
          expanded: subdirConfig.expanded !== false,
          order: subdirConfig.order ?? 999,
          pages: subPages,
          isAdmin: true  // Mark as admin category
        });
      }

      // Scan non-admin (public root) directories for user-accessible pages
      const publicSubdirs = filterDirs(fs.default.readdirSync(publicDir, { withFileTypes: true }))
        .filter(d => d.name !== 'admin'); // Exclude admin directory

      for (const subdir of publicSubdirs) {
        const subdirPath = path.default.join(publicDir, subdir.name);
        const subdirConfig = navConfig.categories?.[subdir.name] || {};

        const subFiles = filterHtmlFiles(fs.default.readdirSync(subdirPath));

        if (subFiles.length === 0) continue;

        const subPages = subFiles.map(file => {
          const pageId = file.replace('.html', '');
          const pageConfig = subdirConfig.pages?.[pageId] || {};
          const filePath = path.default.join(subdirPath, file);
          const urlPath = `/${subdir.name}/${file}`;

          return getPageInfo(filePath, pageId, pageConfig, subdirConfig, urlPath);
        });

        // Sort pages by order
        subPages.sort((a, b) => {
          const orderA = subdirConfig.pages?.[a.id]?.order ?? 999;
          const orderB = subdirConfig.pages?.[b.id]?.order ?? 999;
          if (orderA !== orderB) return orderA - orderB;
          return a.name.localeCompare(b.name);
        });

        categories.push({
          id: `public-${subdir.name}`,
          name: subdirConfig.name || subdir.name.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
          icon: subdirConfig.icon || navConfig.defaultCategoryIcon || '<svg viewBox="0 0 24 24"><path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>',
          expanded: subdirConfig.expanded !== false,
          order: (subdirConfig.order ?? 999) + 100, // Place after admin categories
          pages: subPages,
          isAdmin: false  // Mark as non-admin category
        });
      }

      // Also scan root-level public HTML files (non-admin)
      const publicRootFiles = filterHtmlFiles(fs.default.readdirSync(publicDir));
      if (publicRootFiles.length > 0) {
        const publicRootPages = publicRootFiles.map(file => {
          const pageId = file.replace('.html', '');
          const pageConfig = navConfig.categories?.['public-root']?.pages?.[pageId] || {};
          const filePath = path.default.join(publicDir, file);

          return getPageInfo(filePath, pageId, pageConfig, {}, `/${file}`);
        });

        // Sort pages
        publicRootPages.sort((a, b) => a.name.localeCompare(b.name));

        if (publicRootPages.length > 0) {
          categories.push({
            id: 'public-root',
            name: 'Main Site',
            icon: navConfig.defaultCategoryIcon,
            expanded: false,
            order: 200,
            pages: publicRootPages,
            isAdmin: false
          });
        }
      }

      // Also scan root-level admin HTML files (like roslyn.html, project.html)
      const rootFiles = filterHtmlFiles(fs.default.readdirSync(adminDir));

      if (rootFiles.length > 0) {
        const rootPages = rootFiles.map(file => {
          const pageId = file.replace('.html', '');
          const pageConfig = adminConfig.pages?.[pageId] || {};
          const filePath = path.default.join(adminDir, file);

          return getPageInfo(filePath, pageId, pageConfig, adminConfig, file);
        });

        // Sort root pages
        rootPages.sort((a, b) => {
          const orderA = adminConfig.pages?.[a.id]?.order ?? 999;
          const orderB = adminConfig.pages?.[b.id]?.order ?? 999;
          if (orderA !== orderB) return orderA - orderB;
          return a.name.localeCompare(b.name);
        });

        // Add admin root pages category
        if (rootPages.length > 0) {
          categories.push({
            id: 'admin',
            name: adminConfig.name || 'Administration',
            icon: adminConfig.icon || navConfig.defaultCategoryIcon,
            expanded: adminConfig.expanded !== false,
            order: adminConfig.order ?? 1,
            pages: rootPages
          });
        }
      }

      // Add external link categories from config (like Prefect)
      if (navConfig.categories) {
        for (const [catId, catConfig] of Object.entries(navConfig.categories)) {
          // Only add if it has an external link and isn't already in categories
          if (catConfig.externalLink && !categories.find(c => c.id === catId)) {
            categories.push({
              id: catId,
              name: catConfig.name || catId,
              icon: catConfig.icon || navConfig.defaultCategoryIcon,
              expanded: catConfig.expanded !== false,
              order: catConfig.order ?? 999,
              externalLink: catConfig.externalLink,
              openInNewTab: catConfig.openInNewTab !== false,
              pages: []
            });
          }
        }
      }

      // Sort categories by order
      categories.sort((a, b) => a.order - b.order);

      res.json({
        success: true,
        categories,
        config: {
          excludeDirs,
          excludeFiles,
          lastScanned: new Date().toISOString()
        }
      });

    } catch (error) {
      console.error('Failed to scan navigation structure:', error.message);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  });

  return router;
}
