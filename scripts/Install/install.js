#!/usr/bin/env node
/**
 * Cross-Platform Installation Script for RAG Server
 *
 * This script installs all dependencies for the RAG server project,
 * working on both Windows and Linux.
 *
 * Usage:
 *   node scripts/Install/install.js [options]
 *
 * Options:
 *   --skip-node      Skip Node.js dependency installation
 *   --skip-python    Skip Python dependency installation
 *   --skip-audio     Skip audio processing dependencies (torch, funasr)
 *   --skip-prefect   Skip Prefect workflow dependencies
 *   --dev            Install development dependencies
 *   --help           Show this help message
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Colors for console output
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    cyan: '\x1b[36m'
};

const log = {
    info: (msg) => console.log(`${colors.blue}[INFO]${colors.reset} ${msg}`),
    success: (msg) => console.log(`${colors.green}[OK]${colors.reset} ${msg}`),
    warn: (msg) => console.log(`${colors.yellow}[WARN]${colors.reset} ${msg}`),
    error: (msg) => console.log(`${colors.red}[ERROR]${colors.reset} ${msg}`),
    header: (msg) => console.log(`\n${colors.cyan}${colors.bright}=== ${msg} ===${colors.reset}\n`)
};

// Platform detection
const isWindows = os.platform() === 'win32';
const projectRoot = path.resolve(__dirname, '..', '..');
const pythonServicesDir = path.join(projectRoot, 'python_services');

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
    skipNode: args.includes('--skip-node'),
    skipPython: args.includes('--skip-python'),
    skipAudio: args.includes('--skip-audio'),
    skipPrefect: args.includes('--skip-prefect'),
    dev: args.includes('--dev'),
    help: args.includes('--help') || args.includes('-h')
};

function showHelp() {
    console.log(`
${colors.cyan}RAG Server Installation Script${colors.reset}

Usage: node scripts/Install/install.js [options]

Options:
  --skip-node      Skip Node.js dependency installation
  --skip-python    Skip Python dependency installation
  --skip-audio     Skip audio processing dependencies (torch, funasr)
  --skip-prefect   Skip Prefect workflow dependencies
  --dev            Install development dependencies
  --help, -h       Show this help message

Examples:
  node scripts/Install/install.js              # Full installation
  node scripts/Install/install.js --skip-audio # Skip large audio deps
  node scripts/Install/install.js --dev        # Include dev dependencies
`);
}

function run(command, options = {}) {
    const { cwd = projectRoot, silent = false, ignoreError = false } = options;

    if (!silent) {
        log.info(`Running: ${command}`);
    }

    try {
        const result = execSync(command, {
            cwd,
            encoding: 'utf8',
            stdio: silent ? 'pipe' : 'inherit',
            shell: true
        });
        return { success: true, output: result };
    } catch (error) {
        if (ignoreError) {
            return { success: false, error: error.message };
        }
        throw error;
    }
}

function commandExists(command) {
    try {
        const checkCmd = isWindows ? `where ${command}` : `which ${command}`;
        execSync(checkCmd, { stdio: 'pipe' });
        return true;
    } catch {
        return false;
    }
}

function getPythonCommand() {
    // Check for python3 first (Linux), then python (Windows)
    if (commandExists('python3')) return 'python3';
    if (commandExists('python')) return 'python';
    return null;
}

function getPoetryCommand() {
    if (commandExists('poetry')) return 'poetry';
    // Check if poetry is installed via pip
    const python = getPythonCommand();
    if (python) {
        try {
            execSync(`${python} -m poetry --version`, { stdio: 'pipe' });
            return `${python} -m poetry`;
        } catch {
            // Poetry not installed via pip
        }
    }
    return null;
}

function checkPrerequisites() {
    log.header('Checking Prerequisites');

    const issues = [];

    // Check Node.js
    if (commandExists('node')) {
        const nodeVersion = execSync('node --version', { encoding: 'utf8' }).trim();
        log.success(`Node.js ${nodeVersion}`);
    } else {
        issues.push('Node.js is not installed');
    }

    // Check npm
    if (commandExists('npm')) {
        const npmVersion = execSync('npm --version', { encoding: 'utf8' }).trim();
        log.success(`npm ${npmVersion}`);
    } else {
        issues.push('npm is not installed');
    }

    // Check Python
    const python = getPythonCommand();
    if (python) {
        const pythonVersion = execSync(`${python} --version`, { encoding: 'utf8' }).trim();
        log.success(pythonVersion);
    } else {
        issues.push('Python 3 is not installed');
    }

    // Check Poetry
    const poetry = getPoetryCommand();
    if (poetry) {
        try {
            const poetryVersion = execSync(`${poetry} --version`, { encoding: 'utf8' }).trim();
            log.success(`Poetry: ${poetryVersion}`);
        } catch {
            log.warn('Poetry found but version check failed');
        }
    } else {
        log.warn('Poetry not found - will install via pip');
    }

    // Check Git
    if (commandExists('git')) {
        const gitVersion = execSync('git --version', { encoding: 'utf8' }).trim();
        log.success(gitVersion);
    } else {
        issues.push('Git is not installed');
    }

    if (issues.length > 0) {
        log.error('Missing prerequisites:');
        issues.forEach(issue => console.log(`  - ${issue}`));
        process.exit(1);
    }

    log.success('All prerequisites met');
}

function installNodeDependencies() {
    log.header('Installing Node.js Dependencies');

    // Check if package.json exists
    const packageJson = path.join(projectRoot, 'package.json');
    if (!fs.existsSync(packageJson)) {
        log.warn('No package.json found in project root');
        return;
    }

    // Install dependencies
    run('npm install', { cwd: projectRoot });

    // Rebuild native modules (better-sqlite3, sharp, etc.)
    log.info('Rebuilding native modules...');
    run('npm rebuild', { cwd: projectRoot, ignoreError: true });

    log.success('Node.js dependencies installed');
}

function installPoetry() {
    log.info('Installing Poetry...');
    const python = getPythonCommand();

    try {
        run(`${python} -m pip install poetry`, { silent: true });
        log.success('Poetry installed');
        return `${python} -m poetry`;
    } catch (error) {
        log.error('Failed to install Poetry');
        throw error;
    }
}

function installPythonDependencies() {
    log.header('Installing Python Dependencies');

    // Check if pyproject.toml exists
    const pyprojectToml = path.join(pythonServicesDir, 'pyproject.toml');
    if (!fs.existsSync(pyprojectToml)) {
        log.warn('No pyproject.toml found in python_services');
        return;
    }

    // Get or install Poetry
    let poetry = getPoetryCommand();
    if (!poetry) {
        poetry = installPoetry();
    }

    // Configure Poetry to create virtualenv in project directory
    log.info('Configuring Poetry...');
    run(`${poetry} config virtualenvs.in-project true`, {
        cwd: pythonServicesDir,
        ignoreError: true,
        silent: true
    });

    // Build the install command
    let installCmd = `${poetry} install`;

    // Handle optional dependencies
    if (options.skipAudio || options.skipPrefect) {
        // If skipping optional deps, we need to modify pyproject.toml or use pip fallback
        log.info('Using pip for selective installation (skipping optional deps)...');
        installWithPip();
        return;
    }

    if (!options.dev) {
        installCmd += ' --without dev';
    }

    // Install dependencies
    log.info('Installing Python dependencies with Poetry...');
    try {
        run(installCmd, { cwd: pythonServicesDir });
        log.success('Python dependencies installed');

        // Install Prefect separately (Poetry resolver has issues with Prefect's self-dependency)
        if (!options.skipPrefect) {
            installPrefect(poetry);
        }
    } catch (error) {
        log.warn('Poetry install failed, falling back to pip...');
        installWithPip();
    }
}

function installPrefect(poetry) {
    log.info('Installing Prefect (separate due to Poetry resolver issues)...');
    const python = getPythonCommand();

    try {
        // Use pip within the Poetry environment to install Prefect
        if (poetry) {
            run(`${poetry} run pip install "prefect>=3.0.0"`, {
                cwd: pythonServicesDir,
                ignoreError: true
            });
        } else {
            run(`${python} -m pip install "prefect>=3.0.0"`, {
                cwd: pythonServicesDir,
                ignoreError: true
            });
        }
        log.success('Prefect installed');
    } catch (error) {
        log.warn('Failed to install Prefect - workflow orchestration will not be available');
    }
}

function installWithPip() {
    log.info('Installing Python dependencies with pip...');

    const python = getPythonCommand();
    const requirementsFile = path.join(pythonServicesDir, 'requirements.txt');

    if (!fs.existsSync(requirementsFile)) {
        log.error('No requirements.txt found');
        return;
    }

    // Read requirements and filter based on options
    let requirements = fs.readFileSync(requirementsFile, 'utf8')
        .split('\n')
        .filter(line => line.trim() && !line.startsWith('#'));

    if (options.skipAudio) {
        const audioPackages = ['funasr', 'torch', 'torchaudio', 'mutagen', 'pydub', 'soundfile'];
        requirements = requirements.filter(req =>
            !audioPackages.some(pkg => req.toLowerCase().startsWith(pkg))
        );
        log.info('Skipping audio processing packages');
    }

    if (options.skipPrefect) {
        requirements = requirements.filter(req => !req.toLowerCase().startsWith('prefect'));
        log.info('Skipping Prefect packages');
    }

    // Create temp requirements file
    const tempReq = path.join(pythonServicesDir, 'requirements.temp.txt');
    fs.writeFileSync(tempReq, requirements.join('\n'));

    try {
        run(`${python} -m pip install -r requirements.temp.txt`, { cwd: pythonServicesDir });
        log.success('Python dependencies installed');
    } finally {
        // Clean up temp file
        if (fs.existsSync(tempReq)) {
            fs.unlinkSync(tempReq);
        }
    }
}

function generatePoetryLock() {
    log.header('Generating Poetry Lock File');

    const poetry = getPoetryCommand();
    if (!poetry) {
        log.warn('Poetry not available, skipping lock file generation');
        return;
    }

    const pyprojectToml = path.join(pythonServicesDir, 'pyproject.toml');
    if (!fs.existsSync(pyprojectToml)) {
        log.warn('No pyproject.toml found');
        return;
    }

    try {
        log.info('Running poetry lock (this may take a while)...');
        run(`${poetry} lock`, { cwd: pythonServicesDir });
        log.success('Poetry lock file generated');
    } catch (error) {
        log.warn('Failed to generate poetry.lock - dependencies can still be installed');
    }
}

function verifyInstallation() {
    log.header('Verifying Installation');

    let allGood = true;

    // Check Node modules
    const nodeModules = path.join(projectRoot, 'node_modules');
    if (fs.existsSync(nodeModules)) {
        log.success('Node modules installed');
    } else {
        log.warn('Node modules not found');
        allGood = false;
    }

    // Check Python virtual environment
    const venvDir = isWindows
        ? path.join(pythonServicesDir, '.venv', 'Scripts')
        : path.join(pythonServicesDir, '.venv', 'bin');

    if (fs.existsSync(venvDir)) {
        log.success('Python virtual environment created');
    } else {
        log.warn('Python virtual environment not found');
        allGood = false;
    }

    // Test critical imports
    const python = getPythonCommand();
    if (python) {
        const testImports = ['fastapi', 'uvicorn', 'pymongo', 'pydantic', 'prefect'];
        for (const pkg of testImports) {
            const result = run(`${python} -c "import ${pkg}"`, {
                cwd: pythonServicesDir,
                silent: true,
                ignoreError: true
            });
            if (result.success) {
                log.success(`Python package: ${pkg}`);
            } else {
                log.warn(`Python package missing: ${pkg}`);
                allGood = false;
            }
        }
    }

    if (allGood) {
        log.success('All verifications passed');
    } else {
        log.warn('Some verifications failed - check warnings above');
    }
}

async function main() {
    console.log(`
${colors.cyan}${colors.bright}
 ____      _    ____   ____
|  _ \\    / \\  / ___| / ___|  ___ _ ____   _____ _ __
| |_) |  / _ \\| |  _  \\___ \\ / _ \\ '__\\ \\ / / _ \\ '__|
|  _ <  / ___ \\ |_| |  ___) |  __/ |   \\ V /  __/ |
|_| \\_\\/_/   \\_\\____| |____/ \\___|_|    \\_/ \\___|_|

${colors.reset}${colors.cyan}Installation Script - Cross Platform${colors.reset}
`);

    if (options.help) {
        showHelp();
        process.exit(0);
    }

    console.log(`Platform: ${os.platform()} (${os.arch()})`);
    console.log(`Project Root: ${projectRoot}`);
    console.log('');

    try {
        checkPrerequisites();

        if (!options.skipNode) {
            installNodeDependencies();
        } else {
            log.info('Skipping Node.js dependencies (--skip-node)');
        }

        if (!options.skipPython) {
            installPythonDependencies();
            generatePoetryLock();
        } else {
            log.info('Skipping Python dependencies (--skip-python)');
        }

        verifyInstallation();

        log.header('Installation Complete');
        console.log(`
${colors.green}Installation completed successfully!${colors.reset}

Next steps:
  1. Start the Node.js server:    ${colors.cyan}node rag-server.js${colors.reset}
  2. Start Python services:       ${colors.cyan}cd python_services && poetry run python main.py${colors.reset}

For development:
  - Run tests:                    ${colors.cyan}npm test${colors.reset}
  - Start with hot reload:        ${colors.cyan}npm run dev${colors.reset}
`);

    } catch (error) {
        log.error(`Installation failed: ${error.message}`);
        process.exit(1);
    }
}

main();
