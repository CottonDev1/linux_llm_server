#!/usr/bin/env node

/**
 * Analyze and Import CLI Tool
 *
 * End-to-end tool to analyze C# code using Roslyn and import results into vector database.
 * Combines roslyn-wrapper and code-import-pipeline into a single workflow.
 */

import { Command } from 'commander';
import { promises as fs } from 'fs';
import path from 'path';
import chalk from 'chalk';

const program = new Command();

program
  .name('analyze-and-import')
  .description('Analyze C# code with Roslyn and import into vector database')
  .version('1.0.0')
  .argument('<path>', 'Path to C# file or directory to analyze')
  .option('-p, --project <name>', 'Project name for categorization', 'unknown')
  .option('-o, --output <file>', 'Save analysis JSON to file (optional)')
  .option('-v, --verbose', 'Enable verbose output', false)
  .option('--include-private', 'Include private members in analysis', false)
  .option('--analyze-only', 'Only analyze, do not import', false)
  .option('--import-only <file>', 'Import existing JSON file without analyzing')
  .option('--timeout <ms>', 'Analysis timeout in milliseconds', '300000')
  .action(async (inputPath, options) => {
    try {
      console.error(chalk.red('\n❌ Error: Import functionality has been removed'));
      console.error(chalk.yellow('This tool now only supports analysis. Use --analyze-only flag.'));
      process.exit(1);
    } catch (error) {
      console.error(chalk.red(`\n❌ Error: ${error.message}`));
      process.exit(1);
    }
  });

async function runAnalysisAndImport(inputPath, options) {
  const startTime = Date.now();

  console.log(chalk.bold.blue('\n🚀 C# Code Analysis and Import Pipeline\n'));
  console.log(chalk.gray('─'.repeat(60)));

  let analysisResult;

  // Step 1: Analysis (or load from file)
  if (options.importOnly) {
    console.log(chalk.cyan('\n📂 Loading analysis from file...'));
    console.log(`   File: ${options.importOnly}`);

    const content = await fs.readFile(options.importOnly, 'utf-8');
    const data = JSON.parse(content);

    analysisResult = {
      success: true,
      data,
      inputPath: options.importOnly,
      timestamp: new Date().toISOString()
    };

    console.log(chalk.green('✅ Loaded successfully'));
  } else {
    console.log(chalk.cyan('\n🔍 Step 1: Analyzing C# code with Roslyn...'));
    console.log(`   Path: ${inputPath}`);
    console.log(`   Project: ${options.project}`);

    const analyzer = new RoslynAnalyzer({
      verbose: options.verbose,
      includePrivate: options.includePrivate,
      timeout: parseInt(options.timeout)
    });

    analysisResult = await analyzer.analyzeCode(inputPath);

    if (!analysisResult.success) {
      throw new Error(`Analysis failed: ${analysisResult.error}`);
    }

    console.log(chalk.green(`✅ Analysis completed in ${analysisResult.duration}ms`));
    console.log(chalk.gray(`   Namespaces: ${analysisResult.data.namespaces?.length || 0}`));
    console.log(chalk.gray(`   Classes: ${analysisResult.data.classes?.length || 0}`));
    console.log(chalk.gray(`   Methods: ${analysisResult.data.methods?.length || 0}`));
    console.log(chalk.gray(`   Properties: ${analysisResult.data.properties?.length || 0}`));

    // Save to file if requested
    if (options.output) {
      await fs.writeFile(options.output, JSON.stringify(analysisResult.data, null, 2));
      console.log(chalk.gray(`   Saved to: ${options.output}`));
    }
  }

  // Step 2: Import removed - vector database import functionality deprecated
  if (!options.analyzeOnly) {
    console.log(chalk.yellow('\n⚠️  Import functionality has been removed'));
    console.log(chalk.gray('   Analysis results saved to file only'));
  }

  const totalDuration = Date.now() - startTime;

  console.log(chalk.gray('\n' + '─'.repeat(60)));
  console.log(chalk.bold.green(`\n✨ Complete! Total time: ${totalDuration}ms\n`));
}

// Batch analysis command
program
  .command('batch')
  .description('Analyze multiple directories in batch')
  .argument('<paths...>', 'Paths to analyze')
  .option('-p, --project <name>', 'Project name for categorization', 'unknown')
  .option('-v, --verbose', 'Enable verbose output', false)
  .option('--include-private', 'Include private members in analysis', false)
  .action(async (paths, options) => {
    console.log(chalk.bold.blue('\n🚀 Batch Analysis Mode\n'));
    console.log(chalk.gray(`Analyzing ${paths.length} paths...\n`));

    const results = [];

    for (let i = 0; i < paths.length; i++) {
      const inputPath = paths[i];

      console.log(chalk.cyan(`\n[${i + 1}/${paths.length}] ${inputPath}`));
      console.log(chalk.gray('─'.repeat(60)));

      try {
        await runAnalysisAndImport(inputPath, options);
        results.push({ path: inputPath, success: true });
      } catch (error) {
        console.error(chalk.red(`❌ Failed: ${error.message}`));
        results.push({ path: inputPath, success: false, error: error.message });
      }
    }

    // Print batch summary
    console.log(chalk.bold.blue('\n📊 Batch Summary\n'));
    console.log(chalk.gray('─'.repeat(60)));

    const successful = results.filter(r => r.success).length;
    const failed = results.filter(r => !r.success).length;

    console.log(chalk.green(`✅ Successful: ${successful}`));
    console.log(chalk.red(`❌ Failed: ${failed}`));

    if (failed > 0) {
      console.log(chalk.yellow('\nFailed paths:'));
      results.filter(r => !r.success).forEach(r => {
        console.log(chalk.gray(`  • ${r.path}: ${r.error}`));
      });
    }

    console.log();
  });

// Status check command
program
  .command('status')
  .description('Check Roslyn analyzer status')
  .action(async () => {
    console.log(chalk.bold.blue('\n🔍 System Status\n'));
    console.log(chalk.gray('─'.repeat(60)));

    // Check Roslyn
    console.log(chalk.cyan('\n📋 Roslyn Analyzer:'));
    console.log(chalk.yellow('  ⚠️  Roslyn analyzer status check removed'));
    console.log(chalk.gray('  Vector database import functionality has been deprecated'))

    // Check LLM (llama.cpp)
    console.log(chalk.cyan('\n🤖 LLM (llama.cpp):'));
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      if (response.ok) {
        console.log(chalk.green('  ✅ Connected'));
        const data = await response.json();
        if (data.models) {
          console.log(chalk.gray(`  Models: ${data.models.length}`));
          const qwen = data.models.find(m => m.name.includes('qwen2.5-coder'));
          if (qwen) {
            console.log(chalk.green(`  ✅ qwen2.5-coder:1.5b available`));
          } else {
            console.log(chalk.yellow(`  ⚠️  qwen2.5-coder:1.5b not found`));
          }
        }
      } else {
        console.log(chalk.red('  ❌ Not responding'));
      }
    } catch (error) {
      console.log(chalk.red(`  ❌ Error: ${error.message}`));
    }

    console.log(chalk.gray('\n' + '─'.repeat(60) + '\n'));
  });

// Project stats command
program
  .command('stats')
  .description('Show statistics for a specific project')
  .argument('<project>', 'Project name')
  .action(async (projectName) => {
    console.log(chalk.bold.blue(`\n📊 Statistics for project: ${projectName}\n`));
    console.log(chalk.gray('─'.repeat(60)));
    console.log(chalk.yellow('\n⚠️  Statistics functionality has been removed'));
    console.log(chalk.gray('Vector database import functionality has been deprecated\n'));
  });

// Parse arguments
program.parse();
