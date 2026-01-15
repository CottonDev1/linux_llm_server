/**
 * Analyze all 4 C# projects using Roslyn analyzer
 * and import results into vector database
 */

import { analyzeCode } from './roslyn-wrapper.js'; // Same directory
import { exec } from 'child_process';
import { promisify } from 'util';
import { promises as fs } from 'fs';
import path from 'path';
import os from 'os';

const execAsync = promisify(exec);

// Validate required environment variables
const GIT_ROOT = process.env.GIT_ROOT;
const TEMP_DIR = process.env.TEMP_DIR;
const IMPORT_SCRIPT = process.env.IMPORT_SCRIPT;

const missingVars = [];
if (!GIT_ROOT) missingVars.push('GIT_ROOT');
if (!TEMP_DIR) missingVars.push('TEMP_DIR');
if (!IMPORT_SCRIPT) missingVars.push('IMPORT_SCRIPT');

if (missingVars.length > 0) {
  console.error('ERROR: Required environment variables are missing:');
  missingVars.forEach(varName => console.error(`  - ${varName}`));
  console.error('\nPlease set these environment variables before running this script.');
  process.exit(1);
}

const projects = [
  { name: 'gin', path: path.join(GIT_ROOT, 'Gin', 'WindowsUI') },
  { name: 'warehouse', path: path.join(GIT_ROOT, 'Warehouse', 'WindowsUI') },
  { name: 'marketing', path: path.join(GIT_ROOT, 'Marketing', 'WindowsUI') },
  { name: 'ewrlibrary', path: path.join(GIT_ROOT, 'EWR Library') }  // FIXED: Now analyzes ALL sub-projects (1,068 files instead of 723)
];

async function analyzeAll() {
  console.log('='.repeat(80));
  console.log('🔍 Starting Roslyn C# Analysis for all 4 projects');
  console.log('='.repeat(80));
  console.log();

  const results = [];

  for (const project of projects) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`📦 Analyzing: ${project.name}`);
    console.log('='.repeat(80));

    // Use appropriate temp directory based on platform
    const outputPath = path.join(TEMP_DIR, `roslyn-analysis-${project.name}.json`);

    try {
      console.log(`   Source: ${project.path}`);
      console.log(`   Output: ${outputPath}`);
      console.log();

      // Run Roslyn analysis - returns data in memory
      const result = await analyzeCode(project.path);

      if (!result.success) {
        throw new Error(result.error || 'Analysis failed');
      }

      // Write the result data to the output file
      await fs.writeFile(outputPath, JSON.stringify(result.data, null, 2), 'utf-8');
      console.log(`\n✅ ${project.name} analysis complete`);

      // Import to vector database
      console.log(`\n🔄 Importing ${project.name} to vector database...`);
      const importCmd = `node "${IMPORT_SCRIPT}" import "${outputPath}" "${project.name}"`;
      const { stdout, stderr } = await execAsync(importCmd);

      if (stdout) console.log(stdout);
      if (stderr) console.error('Import warnings:', stderr);

      console.log(`✅ ${project.name} imported to vector database`);

      results.push({ project: project.name, status: 'success' });
    } catch (error) {
      console.error(`\n❌ ${project.name} failed: ${error.message}`);
      results.push({ project: project.name, status: 'failed', error: error.message });
    }
  }

  // Summary
  console.log('\n' + '='.repeat(80));
  console.log('📊 Analysis Summary');
  console.log('='.repeat(80));

  results.forEach(r => {
    const status = r.status === 'success' ? '✅' : '❌';
    console.log(`   ${status} ${r.project}: ${r.status}`);
    if (r.error) {
      console.log(`      Error: ${r.error}`);
    }
  });

  const successCount = results.filter(r => r.status === 'success').length;
  console.log(`\n   Total: ${results.length} projects, ${successCount} successful`);
  console.log('='.repeat(80));
}

analyzeAll().catch(err => {
  console.error('\n❌ Fatal error:', err);
  process.exit(1);
});
