/**
 * Roslyn C# Code Analyzer Wrapper
 *
 * Provides a Node.js interface to execute the Roslyn C# analyzer
 * and retrieve structured JSON output.
 */

import { spawn } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Validate required environment variable
const ANALYZER_ROOT = process.env.ROSLYN_ANALYZER_ROOT;

if (!ANALYZER_ROOT) {
  console.error('ERROR: ROSLYN_ANALYZER_ROOT environment variable is required');
  console.error('Please set this environment variable before running this script.');
  process.exit(1);
}

// Try .NET 9.0 first, fallback to 8.0
// Use relative paths from ANALYZER_ROOT or absolute paths if provided
const ANALYZER_DLL_NET9 = process.env.ROSLYN_ANALYZER_DLL || path.join(ANALYZER_ROOT, 'bin', 'Debug', 'net9.0', 'RoslynCodeAnalyzer.dll');
const ANALYZER_DLL_NET8 = path.join(ANALYZER_ROOT, 'bin', 'Debug', 'net8.0', 'RoslynCodeAnalyzer.dll');

/**
 * Get the analyzer DLL path, checking which .NET version is available
 */
async function getAnalyzerDll() {
  try {
    await fs.access(ANALYZER_DLL_NET9);
    return ANALYZER_DLL_NET9;
  } catch {
    return ANALYZER_DLL_NET8;
  }
}

export class RoslynAnalyzer {
  constructor(options = {}) {
    this.options = {
      timeout: options.timeout || 300000, // 5 minutes default
      verbose: options.verbose || false,
      includePrivate: options.includePrivate || false,
      ...options
    };
    this.analyzerDll = null;
  }

  /**
   * Analyze C# code at the specified path
   * @param {string} inputPath - Path to C# file or directory
   * @param {Object} options - Additional options
   * @returns {Promise<Object>} - Parsed JSON analysis results
   */
  async analyzeCode(inputPath, options = {}) {
    const startTime = Date.now();
    const opts = { ...this.options, ...options };

    // Get the analyzer DLL path (net9.0 or net8.0)
    if (!this.analyzerDll) {
      this.analyzerDll = await getAnalyzerDll();
    }

    // Create temporary output file
    const tempDir = os.tmpdir();
    const tempFile = path.join(tempDir, `roslyn-output-${Date.now()}.json`);

    try {
      if (opts.verbose) {
        console.log(`🔍 Analyzing: ${inputPath}`);
        console.log(`📄 Output will be written to: ${tempFile}`);
      }

      // Verify input path exists (convert WSL path to Windows if needed)
      const normalizedPath = await this._normalizePath(inputPath);

      if (opts.verbose) {
        console.log(`📂 Normalized path: ${normalizedPath}`);
        console.log(`🔧 Using analyzer: ${this.analyzerDll}`);
      }

      // Build command arguments
      const args = [
        this.analyzerDll,
        normalizedPath,
        tempFile
      ];

      if (opts.verbose) {
        args.push('--verbose');
      }

      if (opts.includePrivate) {
        args.push('--include-private');
      }

      // Execute analyzer
      const output = await this._executeDotnet(args, opts);

      // Read and parse JSON output
      const jsonContent = await fs.readFile(tempFile, 'utf-8');
      const result = JSON.parse(jsonContent);

      // Clean up temp file
      await fs.unlink(tempFile).catch(() => {});

      const duration = Date.now() - startTime;

      if (opts.verbose) {
        console.log(`✅ Analysis completed in ${duration}ms`);
        console.log(`📊 Found ${result.namespaces?.length || 0} namespaces`);
        console.log(`📊 Found ${result.classes?.length || 0} classes`);
        console.log(`📊 Found ${result.methods?.length || 0} methods`);
      }

      return {
        success: true,
        data: result,
        duration,
        inputPath: normalizedPath,
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      // Clean up temp file on error
      await fs.unlink(tempFile).catch(() => {});

      return {
        success: false,
        error: error.message,
        duration: Date.now() - startTime,
        inputPath
      };
    }
  }

  /**
   * Analyze multiple paths in parallel
   * @param {string[]} paths - Array of paths to analyze
   * @param {Object} options - Options
   * @returns {Promise<Object[]>} - Array of analysis results
   */
  async analyzeMultiple(paths, options = {}) {
    const opts = { ...this.options, ...options };

    if (opts.verbose) {
      console.log(`🔍 Analyzing ${paths.length} paths...`);
    }

    const results = await Promise.allSettled(
      paths.map(p => this.analyzeCode(p, { ...opts, verbose: false }))
    );

    return results.map((result, index) => {
      if (result.status === 'fulfilled') {
        return result.value;
      } else {
        return {
          success: false,
          error: result.reason.message,
          inputPath: paths[index]
        };
      }
    });
  }

  /**
   * Execute dotnet command
   * @private
   */
  async _executeDotnet(args, options) {
    return new Promise((resolve, reject) => {
      // Debug: Log the exact command being executed
      if (options.verbose) {
        console.log('Executing:', 'dotnet', args.join(' '));
      }
      const process = spawn('dotnet', args);

      let stdout = '';
      let stderr = '';
      let killed = false;

      // Set timeout
      const timeoutId = setTimeout(() => {
        killed = true;
        process.kill('SIGTERM');
        reject(new Error(`Analysis timeout after ${options.timeout}ms`));
      }, options.timeout);

      process.stdout.on('data', (data) => {
        const text = data.toString();
        stdout += text;
        if (options.verbose) {
          console.log(text.trim());
        }
      });

      process.stderr.on('data', (data) => {
        const text = data.toString();
        stderr += text;
        if (options.verbose) {
          console.error(text.trim());
        }
      });

      process.on('close', (code) => {
        clearTimeout(timeoutId);

        if (killed) {
          return; // Already rejected
        }

        if (code !== 0) {
          reject(new Error(`Analyzer exited with code ${code}\n${stderr}`));
        } else {
          resolve({ stdout, stderr });
        }
      });

      process.on('error', (error) => {
        clearTimeout(timeoutId);
        reject(new Error(`Failed to execute dotnet: ${error.message}`));
      });
    });
  }

  /**
   * Normalize path for cross-platform compatibility
   * @private
   */
  async _normalizePath(inputPath) {
    // Convert WSL paths to Windows paths since we're running on Windows
    if (inputPath.startsWith('/mnt/')) {
      // Convert /mnt/c/... to C:\...
      const windowsPath = inputPath
        .replace(/^\/mnt\/([a-z])\//, '$1:\\')
        .replace(/\//g, '\\');
      return windowsPath;
    }
    return inputPath;
  }

  /**
   * Get analyzer version and status
   */
  async getStatus() {
    try {
      // Get the analyzer DLL path (net9.0 or net8.0)
      if (!this.analyzerDll) {
        this.analyzerDll = await getAnalyzerDll();
      }

      const args = [this.analyzerDll, '--help'];
      const { stdout, stderr } = await this._executeDotnet(args, {
        timeout: 5000,
        verbose: false
      });

      return {
        available: true,
        dll: this.analyzerDll,
        output: stdout || stderr
      };
    } catch (error) {
      return {
        available: false,
        dll: this.analyzerDll || 'not found',
        error: error.message
      };
    }
  }
}

// Export convenience function for direct usage
export async function analyzeCode(inputPath, options = {}) {
  const analyzer = new RoslynAnalyzer(options);
  return analyzer.analyzeCode(inputPath, options);
}

// CLI support
if (import.meta.url === `file://${process.argv[1]}`) {
  const inputPath = process.argv[2];
  const verbose = process.argv.includes('--verbose') || process.argv.includes('-v');

  if (!inputPath) {
    console.error('Usage: node roslyn-wrapper.js <path> [--verbose]');
    process.exit(1);
  }

  const analyzer = new RoslynAnalyzer({ verbose });

  (async () => {
    const result = await analyzer.analyzeCode(inputPath);

    if (result.success) {
      console.log(JSON.stringify(result.data, null, 2));
      process.exit(0);
    } else {
      console.error('Analysis failed:', result.error);
      process.exit(1);
    }
  })();
}
