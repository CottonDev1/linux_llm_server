#!/usr/bin/env python3
"""
Roslyn Analysis Runner

This script runs the Roslyn C# analyzer on all configured C# projects and
updates the MongoDB code_methods collection with properly structured calls_to arrays.

Usage:
    python rerun_roslyn_analysis.py [--project PROJECT_NAME] [--dry-run] [--verbose]

Options:
    --project PROJECT_NAME    Only analyze specific project (gin, warehouse, marketing, ewr-library)
    --dry-run                 Show what would be done without making changes
    --verbose                 Enable verbose output
    --clear                   Clear existing code_methods before importing

Example:
    python rerun_roslyn_analysis.py
    python rerun_roslyn_analysis.py --project gin
    python rerun_roslyn_analysis.py --dry-run --verbose
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python_services"))

try:
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import BulkWriteError
except ImportError:
    print("ERROR: pymongo is required. Install with: pip install pymongo")
    sys.exit(1)

# Configuration
ROSLYN_ANALYZER_DLL = Path(__file__).parent.parent / "roslyn-analyzer" / "RoslynCodeAnalyzer" / "bin" / "Debug" / "net8.0" / "RoslynCodeAnalyzer.dll"
GIT_ROOT = Path(r"C:\Projects\Git")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "rag_server")

# Project configurations
PROJECTS = {
    "gin": {
        "name": "Gin",
        "path": GIT_ROOT / "Gin",
        "enabled": True
    },
    "warehouse": {
        "name": "Warehouse",
        "path": GIT_ROOT / "Warehouse",
        "enabled": True
    },
    "marketing": {
        "name": "Marketing",
        "path": GIT_ROOT / "Marketing",
        "enabled": True
    },
    "ewr-library": {
        "name": "EWR Library",
        "path": GIT_ROOT / "EWR Library",
        "enabled": True
    },
    "provider": {
        "name": "Provider",
        "path": GIT_ROOT / "Provider",
        "enabled": True
    }
}

# MongoDB Collections
COLLECTION_CODE_METHODS = "code_methods"
COLLECTION_CODE_CLASSES = "code_classes"
COLLECTION_CODE_CALLGRAPH = "code_callgraph"


def generate_method_id(project: str, full_name: str) -> str:
    """Generate a unique ID for a method based on project and full name."""
    key = f"{project}:{full_name}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def generate_class_id(project: str, full_name: str) -> str:
    """Generate a unique ID for a class based on project and full name."""
    key = f"class:{project}:{full_name}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def generate_callgraph_id(caller_project: str, caller_class: str, caller_method: str,
                          callee_class: str, callee_method: str, call_site_line: int) -> str:
    """Generate a unique ID for a call graph entry."""
    key = f"call:{caller_project}:{caller_class}:{caller_method}:{callee_class}:{callee_method}:{call_site_line}"
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class RoslynAnalysisRunner:
    """Runs Roslyn analysis and updates MongoDB with results."""

    def __init__(self, mongodb_uri: str, database: str, verbose: bool = False, dry_run: bool = False):
        self.mongodb_uri = mongodb_uri
        self.database_name = database
        self.verbose = verbose
        self.dry_run = dry_run
        self.client = None
        self.db = None
        self.stats = {
            "projects_analyzed": 0,
            "methods_imported": 0,
            "classes_imported": 0,
            "callgraph_entries": 0,
            "errors": []
        }

    def connect(self):
        """Connect to MongoDB."""
        if self.dry_run:
            self.log("DRY RUN: Would connect to MongoDB")
            return True

        try:
            self.client = MongoClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            # Test connection
            self.client.admin.command('ping')
            self.log(f"Connected to MongoDB: {self.database_name}")
            return True
        except Exception as e:
            self.log(f"ERROR: Failed to connect to MongoDB: {e}")
            return False

    def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            self.log("Disconnected from MongoDB")

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{timestamp}] [{level}]"

        if level == "DEBUG" and not self.verbose:
            return

        print(f"{prefix} {message}")

    def run_roslyn_analyzer(self, project_path: Path, output_file: Path) -> bool:
        """Run the Roslyn analyzer on a project directory."""
        if not ROSLYN_ANALYZER_DLL.exists():
            self.log(f"ERROR: Roslyn analyzer not found at {ROSLYN_ANALYZER_DLL}")
            self.log("Please build the analyzer first: cd roslyn-analyzer/RoslynCodeAnalyzer && dotnet build")
            return False

        # Convert paths for Windows
        project_path_str = str(project_path).replace("/mnt/c", "C:")
        output_file_str = str(output_file).replace("/mnt/c", "C:")
        dll_path_str = str(ROSLYN_ANALYZER_DLL).replace("/mnt/c", "C:")

        cmd = f'dotnet "{dll_path_str}" "{project_path_str}" "{output_file_str}"'

        self.log(f"Running Roslyn analyzer on {project_path.name}...", "INFO")
        self.log(f"Command: {cmd}", "DEBUG")

        if self.dry_run:
            self.log("DRY RUN: Would execute Roslyn analyzer")
            return False

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=str(Path(__file__).parent.parent)
            )

            if result.returncode != 0:
                self.log(f"Roslyn analyzer error: {result.stderr}", "ERROR")
                return False

            if self.verbose and result.stdout:
                self.log(f"Roslyn output: {result.stdout}", "DEBUG")

            return True

        except subprocess.TimeoutExpired:
            self.log(f"ERROR: Roslyn analyzer timed out for {project_path.name}")
            return False
        except Exception as e:
            self.log(f"ERROR: Failed to run Roslyn analyzer: {e}")
            return False

    def parse_roslyn_output(self, output_file: Path) -> Optional[Dict]:
        """Parse the Roslyn analyzer JSON output."""
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"ERROR: Failed to parse Roslyn output: {e}")
            return None

    def transform_calls_to(self, calls_to: List[Dict]) -> List[Dict]:
        """
        Transform calls_to array to ensure proper structure:
        - File: full file path
        - Class: class name only (not prefixed to method)
        - Method: method name only (not ClassName.Method)
        """
        transformed = []

        for call in calls_to:
            method_name = call.get("Method", "")
            class_name = call.get("Class", "")

            # Fix: If Method contains "ClassName.Method", extract just the method name
            if "." in method_name and class_name in ["Unresolved", "Unknown", ""]:
                parts = method_name.rsplit(".", 1)
                if len(parts) == 2:
                    # The class is the first part, method is the second
                    class_name = parts[0]
                    method_name = parts[1]

            transformed.append({
                "Project": call.get("Project", "Unknown"),
                "File": call.get("File", "Unknown"),
                "Class": class_name,
                "Method": method_name,
                "Line": call.get("Line", 0),
                "CallType": call.get("CallType", "Unknown")
            })

        return transformed

    def transform_called_by(self, called_by: List[Dict]) -> List[Dict]:
        """Transform called_by array to ensure proper structure."""
        transformed = []

        for caller in called_by:
            transformed.append({
                "Project": caller.get("Project", "Unknown"),
                "File": caller.get("File", "Unknown"),
                "Class": caller.get("Class", "Unknown"),
                "Method": caller.get("Method", "Unknown"),
                "Line": caller.get("Line", 0),
                "CallType": caller.get("CallType", "Unknown")
            })

        return transformed

    def import_methods(self, analysis_data: Dict, project_name: str) -> int:
        """Import methods from analysis data into MongoDB."""
        methods = analysis_data.get("Methods", [])

        if not methods:
            self.log(f"No methods found in analysis for {project_name}", "DEBUG")
            return 0

        if self.dry_run:
            self.log(f"DRY RUN: Would import {len(methods)} methods for {project_name}")
            return len(methods)

        operations = []

        for method in methods:
            # Transform the calls_to and called_by arrays
            calls_to = self.transform_calls_to(method.get("CallsTo", []))
            called_by = self.transform_called_by(method.get("CalledBy", []))

            full_name = method.get("FullName", "")

            # Generate unique ID for this method
            method_id = generate_method_id(project_name, full_name)

            doc = {
                "id": method_id,  # Required by existing unique index
                "project": project_name,
                "method_name": method.get("MethodName", ""),
                "class_name": method.get("ClassName", ""),
                "namespace": method.get("Namespace", ""),
                "full_name": full_name,
                "file_path": method.get("FilePath", ""),
                "line_number": method.get("LineNumber", 0),
                "line_count": method.get("LineCount", 0),
                "accessibility": method.get("Accessibility", ""),
                "is_static": method.get("IsStatic", False),
                "is_async": method.get("IsAsync", False),
                "is_virtual": method.get("IsVirtual", False),
                "is_override": method.get("IsOverride", False),
                "return_type": method.get("ReturnType", ""),
                "parameters": method.get("Parameters", []),
                "cyclomatic_complexity": method.get("CyclomaticComplexity", 0),
                "local_variables": method.get("LocalVariables", []),
                "calls_to": calls_to,
                "called_by": called_by,
                "sql_calls": method.get("SqlCalls", []),
                "updated_at": datetime.utcnow()
            }

            # Use upsert based on id (matches unique index)
            filter_doc = {
                "id": method_id
            }

            operations.append(UpdateOne(filter_doc, {"$set": doc}, upsert=True))

        if operations:
            try:
                result = self.db[COLLECTION_CODE_METHODS].bulk_write(operations)
                imported = result.upserted_count + result.modified_count
                self.log(f"Imported {imported} methods for {project_name} (upserted: {result.upserted_count}, modified: {result.modified_count})")
                return imported
            except BulkWriteError as e:
                self.log(f"ERROR: Bulk write error for methods: {e.details}", "ERROR")
                return 0

        return 0

    def import_classes(self, analysis_data: Dict, project_name: str) -> int:
        """Import classes from analysis data into MongoDB."""
        classes = analysis_data.get("Classes", [])

        if not classes:
            self.log(f"No classes found in analysis for {project_name}", "DEBUG")
            return 0

        if self.dry_run:
            self.log(f"DRY RUN: Would import {len(classes)} classes for {project_name}")
            return len(classes)

        operations = []

        for cls in classes:
            full_name = cls.get("FullName", "")

            # Generate unique ID for this class
            class_id = generate_class_id(project_name, full_name)

            doc = {
                "id": class_id,  # Required by existing unique index
                "project": project_name,
                "class_name": cls.get("ClassName", ""),
                "namespace": cls.get("Namespace", ""),
                "full_name": full_name,
                "file_path": cls.get("FilePath", ""),
                "line_number": cls.get("LineNumber", 0),
                "accessibility": cls.get("Accessibility", ""),
                "is_static": cls.get("IsStatic", False),
                "is_abstract": cls.get("IsAbstract", False),
                "is_sealed": cls.get("IsSealed", False),
                "base_class": cls.get("BaseClass", ""),
                "interfaces": cls.get("Interfaces", []),
                "fields": cls.get("Fields", []),
                "properties": cls.get("Properties", []),
                "methods": cls.get("Methods", []),
                "updated_at": datetime.utcnow()
            }

            # Use upsert based on id (matches unique index)
            filter_doc = {
                "id": class_id
            }

            operations.append(UpdateOne(filter_doc, {"$set": doc}, upsert=True))

        if operations:
            try:
                result = self.db[COLLECTION_CODE_CLASSES].bulk_write(operations)
                imported = result.upserted_count + result.modified_count
                self.log(f"Imported {imported} classes for {project_name}")
                return imported
            except BulkWriteError as e:
                self.log(f"ERROR: Bulk write error for classes: {e.details}", "ERROR")
                return 0

        return 0

    def import_callgraph(self, analysis_data: Dict, project_name: str) -> int:
        """Import call graph relationships into MongoDB."""
        methods = analysis_data.get("Methods", [])

        if not methods:
            return 0

        if self.dry_run:
            total_calls = sum(len(m.get("CallsTo", [])) for m in methods)
            self.log(f"DRY RUN: Would import {total_calls} call graph entries for {project_name}")
            return total_calls

        operations = []

        for method in methods:
            caller_info = {
                "project": project_name,
                "class": method.get("ClassName", ""),
                "method": method.get("MethodName", ""),
                "namespace": method.get("Namespace", ""),
                "file": method.get("FilePath", ""),
                "line": method.get("LineNumber", 0)
            }

            for call in method.get("CallsTo", []):
                # Transform the method name if needed
                method_name = call.get("Method", "")
                class_name = call.get("Class", "")

                if "." in method_name and class_name in ["Unresolved", "Unknown", ""]:
                    parts = method_name.rsplit(".", 1)
                    if len(parts) == 2:
                        class_name = parts[0]
                        method_name = parts[1]

                call_site_line = call.get("Line", 0)

                # Generate unique ID for this call graph entry
                callgraph_id = generate_callgraph_id(
                    project_name,
                    caller_info["class"],
                    caller_info["method"],
                    class_name,
                    method_name,
                    call_site_line
                )

                doc = {
                    "id": callgraph_id,  # Required by existing unique index
                    "caller_project": project_name,
                    "caller_class": caller_info["class"],
                    "caller_method": caller_info["method"],
                    "caller_namespace": caller_info["namespace"],
                    "caller_file": caller_info["file"],
                    "caller_line": caller_info["line"],
                    "callee_project": call.get("Project", "Unknown"),
                    "callee_class": class_name,
                    "callee_method": method_name,
                    "callee_file": call.get("File", ""),
                    "call_site_line": call_site_line,
                    "call_type": call.get("CallType", "Unknown"),
                    "updated_at": datetime.utcnow()
                }

                # Use upsert based on id (matches unique index)
                filter_doc = {
                    "id": callgraph_id
                }

                operations.append(UpdateOne(filter_doc, {"$set": doc}, upsert=True))

        if operations:
            try:
                # Process in batches to avoid memory issues
                batch_size = 1000
                total_imported = 0

                for i in range(0, len(operations), batch_size):
                    batch = operations[i:i + batch_size]
                    result = self.db[COLLECTION_CODE_CALLGRAPH].bulk_write(batch)
                    total_imported += result.upserted_count + result.modified_count

                self.log(f"Imported {total_imported} call graph entries for {project_name}")
                return total_imported
            except BulkWriteError as e:
                self.log(f"ERROR: Bulk write error for call graph: {e.details}", "ERROR")
                return 0

        return 0

    def clear_project_data(self, project_name: str):
        """Clear existing data for a project before re-importing."""
        if self.dry_run:
            self.log(f"DRY RUN: Would clear data for {project_name}")
            return

        self.log(f"Clearing existing data for {project_name}...")

        self.db[COLLECTION_CODE_METHODS].delete_many({"project": project_name})
        self.db[COLLECTION_CODE_CLASSES].delete_many({"project": project_name})
        self.db[COLLECTION_CODE_CALLGRAPH].delete_many({"caller_project": project_name})

        self.log(f"Cleared existing data for {project_name}")

    def analyze_project(self, project_key: str, project_config: Dict, clear_existing: bool = False) -> bool:
        """Analyze a single project."""
        project_name = project_config["name"]
        project_path = project_config["path"]

        if not project_path.exists():
            self.log(f"WARNING: Project path does not exist: {project_path}")
            self.stats["errors"].append(f"{project_name}: Path not found")
            return False

        self.log(f"\n{'='*60}")
        self.log(f"Analyzing project: {project_name}")
        self.log(f"Path: {project_path}")
        self.log(f"{'='*60}")

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = Path(f.name)

        try:
            # Run Roslyn analyzer
            if not self.run_roslyn_analyzer(project_path, output_file):
                if not self.dry_run:
                    self.stats["errors"].append(f"{project_name}: Roslyn analysis failed")
                    return False

            if self.dry_run:
                self.stats["projects_analyzed"] += 1
                return True

            # Parse output
            analysis_data = self.parse_roslyn_output(output_file)
            if not analysis_data:
                self.stats["errors"].append(f"{project_name}: Failed to parse output")
                return False

            # Clear existing data if requested
            if clear_existing:
                self.clear_project_data(project_name)

            # Import data
            methods_count = self.import_methods(analysis_data, project_name)
            classes_count = self.import_classes(analysis_data, project_name)
            callgraph_count = self.import_callgraph(analysis_data, project_name)

            self.stats["methods_imported"] += methods_count
            self.stats["classes_imported"] += classes_count
            self.stats["callgraph_entries"] += callgraph_count
            self.stats["projects_analyzed"] += 1

            self.log(f"\nProject {project_name} complete:")
            self.log(f"  - Methods: {methods_count}")
            self.log(f"  - Classes: {classes_count}")
            self.log(f"  - Call graph entries: {callgraph_count}")

            return True

        finally:
            # Cleanup temp file
            try:
                output_file.unlink()
            except:
                pass

    def run(self, project_filter: Optional[str] = None, clear_existing: bool = False):
        """Run analysis on all configured projects."""
        self.log("="*60)
        self.log("Roslyn Analysis Runner")
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.dry_run:
            self.log("MODE: DRY RUN (no changes will be made)")
        self.log("="*60)

        # Connect to MongoDB
        if not self.connect():
            return False

        try:
            # Filter projects if specified
            projects_to_analyze = {}

            if project_filter:
                if project_filter.lower() in PROJECTS:
                    projects_to_analyze[project_filter.lower()] = PROJECTS[project_filter.lower()]
                else:
                    self.log(f"ERROR: Unknown project '{project_filter}'")
                    self.log(f"Available projects: {', '.join(PROJECTS.keys())}")
                    return False
            else:
                projects_to_analyze = {k: v for k, v in PROJECTS.items() if v.get("enabled", True)}

            self.log(f"\nProjects to analyze: {', '.join(projects_to_analyze.keys())}")

            # Analyze each project
            for project_key, project_config in projects_to_analyze.items():
                self.analyze_project(project_key, project_config, clear_existing)

            # Print summary
            self.log("\n" + "="*60)
            self.log("ANALYSIS COMPLETE")
            self.log("="*60)
            self.log(f"Projects analyzed: {self.stats['projects_analyzed']}")
            self.log(f"Methods imported: {self.stats['methods_imported']}")
            self.log(f"Classes imported: {self.stats['classes_imported']}")
            self.log(f"Call graph entries: {self.stats['callgraph_entries']}")

            if self.stats["errors"]:
                self.log(f"\nErrors ({len(self.stats['errors'])}):")
                for error in self.stats["errors"]:
                    self.log(f"  - {error}")

            return len(self.stats["errors"]) == 0

        finally:
            self.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="Run Roslyn analysis on C# projects and update MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python rerun_roslyn_analysis.py                    # Analyze all projects
    python rerun_roslyn_analysis.py --project gin     # Analyze only Gin project
    python rerun_roslyn_analysis.py --dry-run         # Show what would be done
    python rerun_roslyn_analysis.py --clear           # Clear existing data first
    python rerun_roslyn_analysis.py --verbose         # Enable verbose output
        """
    )

    parser.add_argument(
        "--project", "-p",
        type=str,
        help=f"Only analyze specific project ({', '.join(PROJECTS.keys())})"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="Clear existing data for projects before importing"
    )

    parser.add_argument(
        "--mongodb-uri",
        type=str,
        default=MONGODB_URI,
        help=f"MongoDB connection URI (default: {MONGODB_URI})"
    )

    parser.add_argument(
        "--database",
        type=str,
        default=MONGODB_DATABASE,
        help=f"MongoDB database name (default: {MONGODB_DATABASE})"
    )

    args = parser.parse_args()

    runner = RoslynAnalysisRunner(
        mongodb_uri=args.mongodb_uri,
        database=args.database,
        verbose=args.verbose,
        dry_run=args.dry_run
    )

    success = runner.run(
        project_filter=args.project,
        clear_existing=args.clear
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
