/**
 * C# Code Analysis Script with LLM
 *
 * Analyzes all C# files in the repositories and generates:
 * - File purpose and functionality
 * - Class/interface descriptions
 * - Method summaries
 * - Dependencies and relationships
 * - Business logic and patterns
 *
 * Stores analysis in vector database code context
 */
import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { promisify } from 'util';
import fetch from 'node-fetch';

const execAsync = promisify(exec);

// Validate required environment variables
const GIT_ROOT = process.env.GIT_ROOT;

if (!GIT_ROOT) {
  console.error('ERROR: GIT_ROOT environment variable is required');
  console.error('Please set this environment variable before running this script.');
  process.exit(1);
}

// LLM configuration with reasonable defaults
const LLM_URL = process.env.LLM_URL || 'http://localhost:11434';
const LLM_MODEL = process.env.LLM_MODEL || 'qwen2.5-coder:7b'; // Best for code analysis

// Repository configurations
const REPOSITORIES = [
  {
    name: 'Warehouse',
    path: path.join(GIT_ROOT, 'Warehouse'),
    context: 'warehouse',
    description: 'Warehouse management system'
  },
  {
    name: 'Marketing',
    path: path.join(GIT_ROOT, 'Marketing'),
    context: 'marketing',
    description: 'Marketing and sales management system'
  },
  {
    name: 'Gin',
    path: path.join(GIT_ROOT, 'Gin'),
    context: 'gin',
    description: 'Cotton gin management system'
  },
  {
    name: 'EWRLibrary',
    path: path.join(GIT_ROOT, 'EWR Library'),
    context: 'ewrlibrary',
    description: 'Shared library and common utilities'
  }
];

// File patterns to exclude
const EXCLUDE_PATTERNS = [
  'node_modules',
  'bin',
  'obj',
  '.git',
  'packages',
  'TestResults',
  '.vs',
  'Debug',
  'Release'
];

/**
 * Find all C# files in a repository
 */
async function findCSharpFiles(repoPath) {
  const excludeArgs = EXCLUDE_PATTERNS.map(p => `-not -path "*/${p}/*"`).join(' ');

  try {
    const { stdout } = await execAsync(
      `find "${repoPath}" -type f -name "*.cs" ${excludeArgs} 2>/dev/null`
    );

    return stdout.trim().split('\n').filter(line => line.length > 0);
  } catch (error) {
    console.error(`   ✗ Error finding C# files: ${error.message}`);
    return [];
  }
}

/**
 * Read file content
 */
async function readFileContent(filePath) {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    return content;
  } catch (error) {
    console.error(`   ✗ Error reading ${filePath}: ${error.message}`);
    return null;
  }
}

/**
 * Extract namespace and class names from C# code
 */
function extractCodeStructure(code) {
  const namespaceMatch = code.match(/namespace\s+([\w.]+)/);
  const classMatches = [...code.matchAll(/(?:public|private|internal|protected)?\s*(?:static|abstract|sealed)?\s*(?:partial)?\s*(?:class|interface|struct|enum)\s+([\w<>]+)/g)];

  return {
    namespace: namespaceMatch ? namespaceMatch[1] : 'Unknown',
    classes: classMatches.map(m => m[1])
  };
}

/**
 * Analyze C# file with LLM (llama.cpp)
 */
async function analyzeWithLLM(filePath, code, repoName) {
  const structure = extractCodeStructure(code);

  // Truncate very large files for analysis
  const codeSnippet = code.length > 4000 ? code.substring(0, 4000) + '\n... (truncated)' : code;

  const prompt = `Analyze this C# file from the ${repoName} repository and provide a structured analysis.

File: ${filePath}
Namespace: ${structure.namespace}
Classes/Interfaces: ${structure.classes.join(', ') || 'None detected'}

Code:
${codeSnippet}

Provide a concise analysis with:
1. PURPOSE: What is this file's primary purpose? (1-2 sentences)
2. KEY COMPONENTS: What are the main classes/interfaces and what do they do?
3. DEPENDENCIES: What external types or namespaces does it depend on?
4. BUSINESS LOGIC: What business rules or domain logic does it implement?
5. PATTERNS: What design patterns or architectural patterns are used?

Keep the response concise and focused on searchable information.`;

  try {
    const response = await fetch(`${LLM_URL}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: LLM_MODEL,
        prompt: prompt,
        stream: false,
        options: {
          temperature: 0.3,
          num_predict: 400
        }
      })
    });

    if (!response.ok) {
      throw new Error(`LLM request failed: ${response.statusText}`);
    }

    const result = await response.json();
    return result.response;
  } catch (error) {
    console.error(`   ✗ LLM analysis failed: ${error.message}`);
    return null;
  }
}

/**
 * Format analysis for storage
 */
function formatAnalysis(filePath, repoName, repoContext, structure, analysis, code) {
  // Create relative path by removing the GIT_ROOT prefix
  const gitRootRegex = new RegExp(`^${GIT_ROOT.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/[^/]+/`);
  const relativePath = filePath.replace(gitRootRegex, '');

  let content = `Repository: ${repoName}\n`;
  content += `File: ${relativePath}\n`;
  content += `Namespace: ${structure.namespace}\n`;

  if (structure.classes.length > 0) {
    content += `Components: ${structure.classes.join(', ')}\n`;
  }

  content += `\n`;

  if (analysis) {
    content += `${analysis}\n`;
  } else {
    // Fallback analysis
    content += `PURPOSE: C# file containing ${structure.classes.join(', ') || 'code'}\n`;
    content += `NAMESPACE: ${structure.namespace}\n`;
  }

  return content;
}

/**
 * Process all C# files in a repository
 */
async function processRepository(repoConfig, codeDB) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`📦 Processing Repository: ${repoConfig.name}`);
  console.log(`   Path: ${repoConfig.path}`);
  console.log(`   Context: ${repoConfig.context}`);
  console.log('='.repeat(80));

  // Find all C# files
  console.log(`\n🔍 Finding C# files...`);
  const files = await findCSharpFiles(repoConfig.path);

  if (files.length === 0) {
    console.log(`   ⚠️  No C# files found`);
    return { processed: 0, failed: 0 };
  }

  console.log(`   Found ${files.length} C# files`);
  console.log(`   Using LLM model: ${LLM_MODEL}\n`);

  let processed = 0;
  let failed = 0;
  let skipped = 0;

  for (const filePath of files) {
    try {
      // Read file content
      const code = await readFileContent(filePath);

      if (!code) {
        skipped++;
        continue;
      }

      // Skip very small files (likely auto-generated or empty)
      if (code.length < 100) {
        skipped++;
        continue;
      }

      // Extract structure
      const structure = extractCodeStructure(code);

      // Create relative path for display
      const relativePath = filePath.replace(repoConfig.path + '/', '');

      console.log(`   🔍 Analyzing ${relativePath}...`);

      // Analyze with LLM
      const analysis = await analyzeWithLLM(filePath, code, repoConfig.name);

      // Format analysis
      const content = formatAnalysis(
        filePath,
        repoConfig.name,
        repoConfig.context,
        structure,
        analysis,
        code
      );

      // Create document ID
      const documentId = `code_${repoConfig.context}_${relativePath.replace(/[/\\]/g, '_')}`;

      // Store in vector database
      await codeDB.store(documentId, content, {
        repository: repoConfig.name,
        project: repoConfig.context,
        filePath: filePath,
        relativePath: relativePath,
        namespace: structure.namespace,
        classes: structure.classes,
        category: 'code',
        type: 'csharp_file',
        codeLength: code.length,
        updated: new Date().toISOString()
      });

      processed++;

      if (processed % 10 === 0) {
        console.log(`   ✅ Progress: ${processed}/${files.length}`);
      }

    } catch (error) {
      console.error(`   ✗ Failed to process ${filePath}: ${error.message}`);
      failed++;
    }
  }

  console.log(`\n   ✅ Completed ${repoConfig.name}:`);
  console.log(`      Processed: ${processed}`);
  console.log(`      Failed: ${failed}`);
  console.log(`      Skipped: ${skipped}`);

  return { processed, failed, skipped };
}

/**
 * Main execution
 */
async function main() {
  console.log('================================================================================');
  console.log('🤖 C# Code Analysis with LLM');
  console.log('================================================================================');
  console.log(`Model: ${LLM_MODEL}`);
  console.log(`LLM URL: ${LLM_URL}`);
  console.log(`Repositories: ${REPOSITORIES.length}\n`);

  // Test LLM connection
  console.log('🔍 Testing LLM connection...');
  try {
    const testResponse = await fetch(`${LLM_URL}/api/tags`);
    if (!testResponse.ok) {
      throw new Error('LLM server not responding');
    }
    console.log('✅ LLM server is running\n');
  } catch (error) {
    console.error('❌ Cannot connect to LLM server. Please start llama.cpp server first.');
    console.error('   Check that the server is running on port 11434');
    process.exit(1);
  }

  // Initialize Code Context Database
  console.log('🔧 Initializing Code Context Database...');
  // Note: Database initialization removed - update to use appropriate vector DB
  // const codeDB = new VectorDB({
  //   dbPath: './data',
  //   tableName: 'code_context'
  // });
  // await codeDB.initialize();

  const startTime = Date.now();
  const stats = {
    totalProcessed: 0,
    totalFailed: 0,
    totalSkipped: 0
  };

  // Process each repository
  for (const repoConfig of REPOSITORIES) {
    try {
      const result = await processRepository(repoConfig, codeDB);
      stats.totalProcessed += result.processed;
      stats.totalFailed += result.failed;
      stats.totalSkipped += result.skipped;
    } catch (error) {
      console.error(`\n❌ Failed to process ${repoConfig.name}:`, error.message);
    }
  }

  const duration = ((Date.now() - startTime) / 1000 / 60).toFixed(2);

  console.log('\n' + '='.repeat(80));
  console.log('✅ C# Code Analysis Complete!');
  console.log(`   Total time: ${duration} minutes`);
  console.log('='.repeat(80));

  console.log('\n📊 Summary:');
  console.log(`   Files Processed: ${stats.totalProcessed}`);
  console.log(`   Files Failed: ${stats.totalFailed}`);
  console.log(`   Files Skipped: ${stats.totalSkipped}`);
  console.log(`   Total Files: ${stats.totalProcessed + stats.totalFailed + stats.totalSkipped}`);
}

main().catch(error => {
  console.error('❌ Fatal error:', error);
  process.exit(1);
});
