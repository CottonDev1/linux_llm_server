"""
Roslyn MongoDB Service - Store C# code analysis with vector embeddings

This service stores Roslyn analyzer output in MongoDB with proper vector embeddings
for semantic search. It follows best practices for vector storage:

1. Separate collections for different entity types (classes, methods, call graph)
2. Embed semantic descriptions, not raw code
3. Store full code/details in metadata
4. Create proper indexes for efficient retrieval
"""

import asyncio
import json
import subprocess
import os
import re
import hashlib
import tempfile
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    COLLECTION_CODE_METHODS,
    COLLECTION_CODE_CLASSES,
    COLLECTION_CODE_CALLGRAPH,
    DEFAULT_SEARCH_LIMIT,
    SIMILARITY_THRESHOLD
)
from embedding_service import get_embedding_service
from mongodb import get_mongodb_service
import numpy as np


# Additional collections for Roslyn-specific data
COLLECTION_CODE_EVENTHANDLERS = "code_eventhandlers"
COLLECTION_CODE_DBOPERATIONS = "code_dboperations"


class RoslynMongoDBService:
    """
    Service for storing and querying Roslyn C# code analysis in MongoDB.

    Stores:
    - Classes with inheritance and member information
    - Methods with signatures, complexity, and SQL operations
    - Call graph relationships (who calls what)
    - Event handlers (UI to code mapping)
    - Database operations (SQL commands and stored procedures)

    All entities are stored with vector embeddings for semantic search.
    """

    _instance: Optional['RoslynMongoDBService'] = None

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.embedding_service = None
        self.is_initialized = False

        # Path to Roslyn analyzer
        self.analyzer_paths = [
            r"C:\Projects\Python_NodeServer\roslyn-analyzer\RoslynCodeAnalyzer\bin\Debug\net9.0\RoslynCodeAnalyzer.dll",
            r"C:\Projects\Python_NodeServer\roslyn-analyzer\RoslynCodeAnalyzer\bin\Debug\net8.0\RoslynCodeAnalyzer.dll",
        ]

    @classmethod
    def get_instance(cls) -> 'RoslynMongoDBService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize MongoDB connection and create indexes"""
        if self.is_initialized:
            return

        print(f"Connecting to MongoDB: {MONGODB_URI}")
        self.client = AsyncIOMotorClient(MONGODB_URI)
        self.db = self.client[MONGODB_DATABASE]

        # Test connection
        await self.client.admin.command('ping')
        print("Connected to MongoDB successfully")

        # Initialize embedding service
        self.embedding_service = get_embedding_service()
        await self.embedding_service.initialize()
        print("Embedding service ready")

        # Create indexes
        await self._create_indexes()

        self.is_initialized = True
        print("Roslyn MongoDB service initialized")

    async def _create_indexes(self):
        """Create indexes for efficient queries"""

        # Code Methods collection
        methods_collection = self.db[COLLECTION_CODE_METHODS]
        await methods_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("project", ASCENDING)]),
            IndexModel([("namespace", ASCENDING)]),
            IndexModel([("class_name", ASCENDING)]),
            IndexModel([("method_name", ASCENDING)]),
            IndexModel([("file_path", ASCENDING)]),
            IndexModel([("has_sql_operations", ASCENDING)]),
            IndexModel([("cyclomatic_complexity", DESCENDING)]),
        ])

        # Code Classes collection
        classes_collection = self.db[COLLECTION_CODE_CLASSES]
        await classes_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("project", ASCENDING)]),
            IndexModel([("namespace", ASCENDING)]),
            IndexModel([("class_name", ASCENDING)]),
            IndexModel([("base_class", ASCENDING)]),
            IndexModel([("file_path", ASCENDING)]),
        ])

        # Call Graph collection
        callgraph_collection = self.db[COLLECTION_CODE_CALLGRAPH]
        await callgraph_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("caller_class", ASCENDING)]),
            IndexModel([("caller_method", ASCENDING)]),
            IndexModel([("callee_class", ASCENDING)]),
            IndexModel([("callee_method", ASCENDING)]),
            IndexModel([("project", ASCENDING)]),
            IndexModel([("is_sql_operation", ASCENDING)]),
        ])

        # Event Handlers collection
        events_collection = self.db[COLLECTION_CODE_EVENTHANDLERS]
        await events_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("project", ASCENDING)]),
            IndexModel([("event_name", ASCENDING)]),
            IndexModel([("handler_method", ASCENDING)]),
            IndexModel([("ui_element_type", ASCENDING)]),
        ])

        # Database Operations collection
        dbops_collection = self.db[COLLECTION_CODE_DBOPERATIONS]
        await dbops_collection.create_indexes([
            IndexModel([("id", ASCENDING)], unique=True),
            IndexModel([("project", ASCENDING)]),
            IndexModel([("operation_type", ASCENDING)]),
            IndexModel([("table_name", ASCENDING)]),
            IndexModel([("stored_procedure", ASCENDING)]),
        ])

        print("Roslyn collection indexes created")

    def _get_analyzer_path(self) -> Optional[str]:
        """Get the path to the Roslyn analyzer DLL"""
        for path in self.analyzer_paths:
            if os.path.exists(path):
                return path
        return None


    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content for change detection"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    async def analyze_code(self, input_path: str, project_name: str = "Unknown") -> Dict:
        """
        Analyze C# code and return structured results.

        First tries the Roslyn DLL if available, then falls back to Python regex parsing.

        Args:
            input_path: Path to C# file or directory
            project_name: Project name for tagging

        Returns:
            Dict with analysis results
        """
        analyzer_dll = self._get_analyzer_path()

        # Try Roslyn DLL first
        if analyzer_dll:
            result = await self._analyze_with_roslyn(input_path, project_name, analyzer_dll)
            if result.get('success'):
                return result
            print(f"Roslyn DLL failed, falling back to Python parser: {result.get('error')}")

        # Fall back to Python-based analysis
        return await self._analyze_with_python(input_path, project_name)

    async def _analyze_with_roslyn(self, input_path: str, project_name: str, analyzer_dll: str) -> Dict:
        """Run Roslyn DLL analyzer."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            cmd = [
                "dotnet",
                analyzer_dll,
                input_path,
                output_path,
                "--project", project_name
            ]

            print(f"Running Roslyn analyzer: dotnet {analyzer_dll} {input_path}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Analyzer failed: {result.stderr}",
                    "stdout": result.stdout
                }

            with open(output_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            return {
                "success": True,
                "data": analysis_data,
                "input_path": input_path,
                "project": project_name,
                "analyzer": "roslyn"
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Analyzer timed out after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            try:
                os.unlink(output_path)
            except:
                pass

    async def analyze_code_with_filelist(self, filelist_path: str, project_name: str = "Unknown") -> Dict:
        """
        Analyze C# code using a file list (--filelist mode).

        This allows targeted analysis of specific files rather than scanning
        an entire directory. Used for incremental updates when only certain
        files need to be re-analyzed.

        Args:
            filelist_path: Path to a text file containing one .cs file path per line
            project_name: Project name for tagging

        Returns:
            Dict with analysis results
        """
        analyzer_dll = self._get_analyzer_path()

        if analyzer_dll:
            result = await self._analyze_with_roslyn_filelist(filelist_path, project_name, analyzer_dll)
            if result.get('success'):
                return result
            print(f"Roslyn DLL filelist mode failed, falling back to Python parser: {result.get('error')}")

        # Fall back to Python-based analysis with file list
        return await self._analyze_filelist_with_python(filelist_path, project_name)

    async def _analyze_with_roslyn_filelist(self, filelist_path: str, project_name: str, analyzer_dll: str) -> Dict:
        """Run Roslyn DLL analyzer with --filelist mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            cmd = [
                "dotnet",
                analyzer_dll,
                filelist_path,
                output_path,
                "--filelist",
                "--project", project_name
            ]

            print(f"Running Roslyn analyzer (filelist mode): dotnet {analyzer_dll} {filelist_path}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for larger file lists
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Analyzer failed: {result.stderr}",
                    "stdout": result.stdout
                }

            with open(output_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            return {
                "success": True,
                "data": analysis_data,
                "filelist_path": filelist_path,
                "project": project_name,
                "analyzer": "roslyn-filelist"
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Analyzer timed out after 10 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            try:
                os.unlink(output_path)
            except:
                pass

    async def _analyze_filelist_with_python(self, filelist_path: str, project_name: str) -> Dict:
        """
        Python-based C# analysis using regex patterns for files in a file list.
        Fallback when Roslyn DLL is not available.
        """
        from git_service import CodeAnalyzer

        print(f"Using Python regex parser for file list: {filelist_path}")

        # Read the file list
        with open(filelist_path, 'r') as f:
            cs_files = [Path(line.strip()) for line in f if line.strip() and line.strip().endswith('.cs')]

        if not cs_files:
            return {
                "success": False,
                "error": f"No C# files found in file list {filelist_path}"
            }

        print(f"Processing {len(cs_files)} C# files from file list")

        analyzer = CodeAnalyzer()
        classes = []
        methods = []
        call_graph = []
        event_handlers = []
        db_operations = []

        for cs_file in cs_files:
            try:
                if not cs_file.exists():
                    print(f"  Skipping missing file: {cs_file}")
                    continue

                content = cs_file.read_text(encoding='utf-8', errors='replace')
                relative_path = str(cs_file.name)

                # Extract namespace
                namespace_match = re.search(r'namespace\s+([\w.]+)', content)
                namespace = namespace_match.group(1) if namespace_match else ""

                # Extract classes using regex
                class_matches = re.finditer(
                    r'(?:public|private|protected|internal)?\s*'
                    r'(?:static|abstract|sealed|partial)?\s*'
                    r'(?:class|interface)\s+(\w+)(?:\s*:\s*([^\{]+))?',
                    content
                )

                for match in class_matches:
                    class_name = match.group(1)
                    base_class = match.group(2).split(',')[0].strip() if match.group(2) else None

                    classes.append({
                        "ClassName": class_name,
                        "Namespace": namespace,
                        "FilePath": str(cs_file),
                        "BaseClass": base_class,
                        "ProjectName": project_name
                    })

                # Extract methods using regex
                method_matches = re.finditer(
                    r'(?:public|private|protected|internal)\s+'
                    r'(?:static\s+)?(?:async\s+)?(?:virtual\s+)?(?:override\s+)?'
                    r'(\w+(?:<[\w,\s]+>)?)\s+(\w+)\s*\(([^)]*)\)',
                    content
                )

                current_class = None
                for class_match in re.finditer(r'class\s+(\w+)', content):
                    current_class = class_match.group(1)

                for match in method_matches:
                    return_type = match.group(1)
                    method_name = match.group(2)
                    parameters = match.group(3)

                    # Skip common keywords
                    if method_name in ['if', 'for', 'while', 'switch', 'using', 'return', 'new', 'class']:
                        continue

                    methods.append({
                        "MethodName": method_name,
                        "ClassName": current_class or "Unknown",
                        "Namespace": namespace,
                        "FilePath": str(cs_file),
                        "ReturnType": return_type,
                        "Parameters": parameters,
                        "ProjectName": project_name
                    })

            except Exception as e:
                print(f"  Error processing {cs_file}: {e}")

        return {
            "success": True,
            "data": {
                "Classes": classes,
                "Methods": methods,
                "CallGraph": call_graph,
                "EventHandlers": event_handlers,
                "DatabaseOperations": db_operations
            },
            "filelist_path": filelist_path,
            "project": project_name,
            "analyzer": "python-filelist"
        }

    async def _analyze_with_python(self, input_path: str, project_name: str) -> Dict:
        """
        Python-based C# analysis using regex patterns.
        Fallback when Roslyn DLL is not available.
        """
        from git_service import CodeAnalyzer

        print(f"Using Python regex parser for: {input_path}")

        input_path_obj = Path(input_path)
        analyzer = CodeAnalyzer()

        # Collect all C# files
        cs_files = []
        if input_path_obj.is_file() and input_path_obj.suffix.lower() == '.cs':
            cs_files = [input_path_obj]
        elif input_path_obj.is_dir():
            cs_files = list(input_path_obj.rglob('*.cs'))

        if not cs_files:
            return {
                "success": False,
                "error": f"No C# files found in {input_path}"
            }

        print(f"Found {len(cs_files)} C# files to analyze")

        classes = []
        methods = []
        call_graph = []
        event_handlers = []
        db_operations = []

        for cs_file in cs_files:
            try:
                content = cs_file.read_text(encoding='utf-8', errors='replace')
                relative_path = str(cs_file.relative_to(input_path_obj) if input_path_obj.is_dir() else cs_file.name)

                # Extract namespace
                namespace_match = re.search(r'namespace\s+([\w.]+)', content)
                namespace = namespace_match.group(1) if namespace_match else ""

                # Extract classes
                class_matches = re.finditer(
                    r'(?:public|private|protected|internal)?\s*'
                    r'(?:static|abstract|sealed|partial)?\s*'
                    r'(?:class|interface)\s+(\w+)(?:\s*:\s*([^\{]+))?',
                    content
                )

                for match in class_matches:
                    class_name = match.group(1)
                    inheritance = match.group(2) if match.group(2) else ""

                    # Parse inheritance
                    base_class = ""
                    interfaces = []
                    if inheritance:
                        parts = [p.strip() for p in inheritance.split(',')]
                        for i, part in enumerate(parts):
                            if i == 0 and not part.startswith('I'):
                                base_class = part
                            else:
                                interfaces.append(part)

                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1

                    # Extract methods for this class
                    class_methods = analyzer.extract_csharp_methods(content)

                    classes.append({
                        "ClassName": class_name,
                        "Namespace": namespace,
                        "BaseClass": base_class,
                        "Interfaces": interfaces,
                        "Methods": class_methods,
                        "Properties": [],
                        "FilePath": relative_path,
                        "LineNumber": line_num,
                        "Summary": ""
                    })

                # Extract methods with more detail
                method_pattern = re.compile(
                    r'(?:public|private|protected|internal)\s+'
                    r'(?:static\s+)?'
                    r'(?:async\s+)?'
                    r'(?:virtual\s+|override\s+|abstract\s+)?'
                    r'(\w+(?:<[^>]+>)?)\s+'  # Return type
                    r'(\w+)\s*'              # Method name
                    r'\(([^)]*)\)'           # Parameters
                    r'[^{;]*[{;]',           # Body start or semicolon
                    re.MULTILINE
                )

                for match in method_pattern.finditer(content):
                    return_type = match.group(1)
                    method_name = match.group(2)
                    params_str = match.group(3)

                    # Skip constructors and property accessors
                    if method_name in analyzer.CSHARP_KEYWORDS:
                        continue

                    line_num = content[:match.start()].count('\n') + 1

                    # Parse parameters
                    params = []
                    if params_str.strip():
                        for param in params_str.split(','):
                            parts = param.strip().split()
                            if len(parts) >= 2:
                                params.append({
                                    "Type": ' '.join(parts[:-1]),
                                    "Name": parts[-1]
                                })

                    # Check for SQL calls
                    sql_calls = []
                    # Look for SqlCommand, ExecuteReader, stored procedure calls
                    if 'SqlCommand' in content or 'ExecuteReader' in content or 'ExecuteNonQuery' in content:
                        sql_pattern = re.compile(
                            r'(?:new\s+SqlCommand\s*\(\s*"([^"]+)"|'
                            r'CommandText\s*=\s*"([^"]+)"|'
                            r'\.CommandText\s*=\s*"([^"]+)")',
                            re.IGNORECASE
                        )
                        for sql_match in sql_pattern.finditer(content):
                            cmd_text = sql_match.group(1) or sql_match.group(2) or sql_match.group(3)
                            if cmd_text:
                                sql_calls.append({
                                    "CommandText": cmd_text[:200],
                                    "StoredProcedure": cmd_text if not cmd_text.upper().startswith('SELECT') else ""
                                })

                    methods.append({
                        "MethodName": method_name,
                        "ClassName": classes[-1]["ClassName"] if classes else "",
                        "Namespace": namespace,
                        "ReturnType": return_type,
                        "Parameters": params,
                        "IsStatic": 'static' in match.group(0),
                        "IsAsync": 'async' in match.group(0),
                        "FilePath": relative_path,
                        "LineNumber": line_num,
                        "SqlCalls": sql_calls,
                        "Summary": ""
                    })

                # Extract event handlers
                event_pattern = re.compile(
                    r'(\w+)\s*\.\s*(\w+)\s*\+=\s*(?:new\s+\w+\s*\(\s*)?(\w+)',
                    re.MULTILINE
                )
                for match in event_pattern.finditer(content):
                    element_name = match.group(1)
                    event_name = match.group(2)
                    handler_method = match.group(3)

                    event_handlers.append({
                        "ElementName": element_name,
                        "EventName": event_name,
                        "HandlerMethod": handler_method,
                        "HandlerClass": classes[-1]["ClassName"] if classes else "",
                        "Namespace": namespace,
                        "FilePath": relative_path,
                        "LineNumber": content[:match.start()].count('\n') + 1
                    })

            except Exception as e:
                print(f"Error analyzing {cs_file}: {e}")

        return {
            "success": True,
            "data": {
                "Classes": classes,
                "Methods": methods,
                "CallGraph": call_graph,
                "EventHandlers": event_handlers,
                "DatabaseOperations": db_operations
            },
            "input_path": input_path,
            "project": project_name,
            "analyzer": "python_regex",
            "files_analyzed": len(cs_files)
        }

    # =========================================================================
    # Embedding Text Formatters
    # Key insight: Embed semantic descriptions, not raw code
    # =========================================================================

    def _format_method_embedding_text(self, method: Dict) -> str:
        """
        Format method information for embedding.
        Creates semantic description optimized for search.
        """
        parts = []

        # Method identification
        class_name = method.get('ClassName', method.get('class_name', 'Unknown'))
        method_name = method.get('MethodName', method.get('method_name', 'Unknown'))
        parts.append(f"Method: {class_name}.{method_name}")

        # Return type and parameters
        return_type = method.get('ReturnType', method.get('return_type', 'void'))
        params = method.get('Parameters', method.get('parameters', []))
        if params:
            param_str = ', '.join([f"{p.get('Type', p.get('type', ''))} {p.get('Name', p.get('name', ''))}" for p in params])
            parts.append(f"Signature: {return_type} {method_name}({param_str})")
        else:
            parts.append(f"Returns: {return_type}")

        # Summary/documentation
        summary = method.get('Summary', method.get('summary', ''))
        if summary:
            parts.append(f"Description: {summary}")

        # SQL operations (important for finding database-related code)
        sql_calls = method.get('SqlCalls', method.get('sql_calls', []))
        if sql_calls:
            sql_info = []
            for sql in sql_calls:
                if sql.get('StoredProcedure') or sql.get('stored_procedure'):
                    sql_info.append(f"calls stored procedure {sql.get('StoredProcedure') or sql.get('stored_procedure')}")
                elif sql.get('CommandText') or sql.get('command_text'):
                    cmd = sql.get('CommandText') or sql.get('command_text', '')
                    if 'SELECT' in cmd.upper():
                        sql_info.append("reads from database")
                    elif 'INSERT' in cmd.upper():
                        sql_info.append("inserts into database")
                    elif 'UPDATE' in cmd.upper():
                        sql_info.append("updates database")
                    elif 'DELETE' in cmd.upper():
                        sql_info.append("deletes from database")
            if sql_info:
                parts.append(f"Database: {', '.join(sql_info)}")

        # Complexity indicator
        complexity = method.get('CyclomaticComplexity', method.get('cyclomatic_complexity', 0))
        if complexity > 10:
            parts.append(f"Complex logic (complexity: {complexity})")

        return "\n".join(parts)

    def _format_class_embedding_text(self, cls: Dict) -> str:
        """
        Format class information for embedding.
        """
        parts = []

        # Class identification
        class_name = cls.get('ClassName', cls.get('class_name', 'Unknown'))
        namespace = cls.get('Namespace', cls.get('namespace', ''))
        parts.append(f"Class: {namespace}.{class_name}" if namespace else f"Class: {class_name}")

        # Inheritance
        base_class = cls.get('BaseClass', cls.get('base_class', ''))
        if base_class:
            parts.append(f"Inherits from: {base_class}")

        # Interfaces
        interfaces = cls.get('Interfaces', cls.get('interfaces', []))
        if interfaces:
            parts.append(f"Implements: {', '.join(interfaces)}")

        # Summary
        summary = cls.get('Summary', cls.get('summary', ''))
        if summary:
            parts.append(f"Description: {summary}")

        # Methods overview
        methods = cls.get('Methods', cls.get('methods', []))
        if methods:
            parts.append(f"Methods: {', '.join(methods[:10])}")  # Limit to first 10

        # Properties overview
        properties = cls.get('Properties', cls.get('properties', []))
        if properties:
            parts.append(f"Properties: {', '.join(properties[:10])}")

        return "\n".join(parts)

    def _format_callgraph_embedding_text(self, call: Dict) -> str:
        """
        Format call relationship for embedding.
        """
        caller = f"{call.get('CallerClass', call.get('caller_class', ''))}.{call.get('CallerMethod', call.get('caller_method', ''))}"
        callee = f"{call.get('CalleeClass', call.get('callee_class', ''))}.{call.get('CalleeMethod', call.get('callee_method', ''))}"

        parts = [f"Call relationship: {caller} calls {callee}"]

        call_type = call.get('CallType', call.get('call_type', ''))
        if call_type:
            parts.append(f"Type: {call_type}")

        if call.get('IsSqlOperation', call.get('is_sql_operation')):
            parts.append("This is a database operation call")
            sp = call.get('StoredProcedureName', call.get('stored_procedure_name', ''))
            if sp:
                parts.append(f"Stored procedure: {sp}")

        return "\n".join(parts)

    def _format_eventhandler_embedding_text(self, handler: Dict) -> str:
        """
        Format event handler for embedding.
        """
        parts = []

        event_name = handler.get('EventName', handler.get('event_name', 'Unknown'))
        handler_method = handler.get('HandlerMethod', handler.get('handler_method', 'Unknown'))

        parts.append(f"Event handler: {event_name} -> {handler_method}")

        ui_element = handler.get('UIElementType', handler.get('ui_element_type', ''))
        if ui_element:
            parts.append(f"UI Element: {ui_element}")

        element_name = handler.get('ElementName', handler.get('element_name', ''))
        if element_name:
            parts.append(f"Element: {element_name}")

        # Make it searchable by common terms
        parts.append(f"When user clicks {element_name or 'button'}, {handler_method} is executed")

        return "\n".join(parts)

    # =========================================================================
    # Storage Methods
    # =========================================================================

    async def store_analysis(self, analysis_result: Dict) -> Dict:
        """
        Store complete Roslyn analysis in MongoDB.

        Args:
            analysis_result: Output from analyze_code()

        Returns:
            Dict with storage statistics
        """
        if not self.is_initialized:
            await self.initialize()

        if not analysis_result.get('success'):
            return {"success": False, "error": analysis_result.get('error')}

        data = analysis_result['data']
        project = analysis_result.get('project', 'Unknown')

        stats = {
            "classes_stored": 0,
            "methods_stored": 0,
            "callgraph_stored": 0,
            "eventhandlers_stored": 0,
            "dboperations_stored": 0
        }

        # Store classes
        classes = data.get('Classes', [])
        for cls in classes:
            await self._store_class(cls, project)
            stats["classes_stored"] += 1

        # Store methods
        methods = data.get('Methods', [])
        for method in methods:
            await self._store_method(method, project)
            stats["methods_stored"] += 1

        # Store call graph
        callgraph = data.get('CallGraph', [])
        for call in callgraph:
            await self._store_callgraph_edge(call, project)
            stats["callgraph_stored"] += 1

        # Store event handlers
        event_handlers = data.get('EventHandlers', [])
        for handler in event_handlers:
            await self._store_eventhandler(handler, project)
            stats["eventhandlers_stored"] += 1

        # Store database operations
        db_operations = data.get('DatabaseOperations', [])
        for op in db_operations:
            await self._store_dboperation(op, project)
            stats["dboperations_stored"] += 1

        stats["success"] = True
        stats["project"] = project

        print(f"Stored analysis for {project}:")
        print(f"  Classes: {stats['classes_stored']}")
        print(f"  Methods: {stats['methods_stored']}")
        print(f"  Call Graph Edges: {stats['callgraph_stored']}")
        print(f"  Event Handlers: {stats['eventhandlers_stored']}")
        print(f"  DB Operations: {stats['dboperations_stored']}")

        return stats

    async def _store_class(self, cls: Dict, project: str):
        """Store a class with embedding"""
        collection = self.db[COLLECTION_CODE_CLASSES]

        class_name = cls.get('ClassName', '')
        namespace = cls.get('Namespace', '')
        file_path = cls.get('FilePath', '')

        # Generate ID
        class_id = f"{project}:{namespace}.{class_name}"

        # Generate embedding
        embedding_text = self._format_class_embedding_text(cls)
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        # Compute content hash for change detection
        hash_content = f"{class_name}:{namespace}:{cls.get('BaseClass', '')}:{str(cls.get('Interfaces', []))}:{str(cls.get('Methods', []))}"
        content_hash = self._compute_content_hash(hash_content)


        document = {
            "id": class_id,
            "project": project,
            "namespace": namespace,
            "class_name": class_name,
            "base_class": cls.get('BaseClass', ''),
            "interfaces": cls.get('Interfaces', []),
            "methods": cls.get('Methods', []),
            "properties": cls.get('Properties', []),
            "fields": cls.get('Fields', []),
            "is_static": cls.get('IsStatic', False),
            "is_abstract": cls.get('IsAbstract', False),
            "is_sealed": cls.get('IsSealed', False),
            "accessibility": cls.get('Accessibility', ''),
            "summary": cls.get('Summary', ''),
            "file_path": file_path,
            "line_number": cls.get('LineNumber', 0),
            "embedding_text": embedding_text,
            "content_hash": content_hash,
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        # Use update with $set and $setOnInsert for created_at
        await collection.update_one(
            {"id": class_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

    async def _store_method(self, method: Dict, project: str):
        """Store a method with embedding"""
        collection = self.db[COLLECTION_CODE_METHODS]

        class_name = method.get('ClassName', '')
        method_name = method.get('MethodName', '')
        namespace = method.get('Namespace', '')

        # Generate ID
        method_id = f"{project}:{namespace}.{class_name}.{method_name}"

        # Generate embedding
        embedding_text = self._format_method_embedding_text(method)
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        # Check for SQL operations
        sql_calls = method.get('SqlCalls', [])
        has_sql = len(sql_calls) > 0

        # Compute content hash for change detection
        hash_content = f"{method_name}:{method.get('ReturnType', 'void')}:{str(method.get('Parameters', []))}:{method.get('Body', '')}"
        content_hash = self._compute_content_hash(hash_content)


        document = {
            "id": method_id,
            "project": project,
            "namespace": namespace,
            "class_name": class_name,
            "method_name": method_name,
            "return_type": method.get('ReturnType', 'void'),
            "parameters": method.get('Parameters', []),
            "is_static": method.get('IsStatic', False),
            "is_async": method.get('IsAsync', False),
            "is_virtual": method.get('IsVirtual', False),
            "is_override": method.get('IsOverride', False),
            "accessibility": method.get('Accessibility', ''),
            "cyclomatic_complexity": method.get('CyclomaticComplexity', 1),
            "line_count": method.get('LineCount', 0),
            "line_number": method.get('LineNumber', 0),
            "summary": method.get('Summary', ''),
            "body": method.get('Body', ''),  # Store full code for reference
            "sql_calls": sql_calls,
            "has_sql_operations": has_sql,
            "calls_to": method.get('CallsTo', []),
            "called_by": method.get('CalledBy', []),
            "file_path": method.get('FilePath', ''),
            "embedding_text": embedding_text,
            "content_hash": content_hash,
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        # Use update with $set and $setOnInsert for created_at
        await collection.update_one(
            {"id": method_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

    async def _store_callgraph_edge(self, call: Dict, project: str):
        """Store a call graph edge with embedding"""
        collection = self.db[COLLECTION_CODE_CALLGRAPH]

        caller_class = call.get('CallerClass', '')
        caller_method = call.get('CallerMethod', '')
        callee_class = call.get('CalleeClass', '')
        callee_method = call.get('CalleeMethod', '')

        # Generate ID
        call_id = f"{project}:{caller_class}.{caller_method}->{callee_class}.{callee_method}"

        # Generate embedding
        embedding_text = self._format_callgraph_embedding_text(call)
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        # Compute content hash for change detection
        call_type = call.get('CallType', 'Direct')
        hash_content = f"{caller_class}.{caller_method}->{callee_class}.{callee_method}:{call_type}:{call.get('SqlCommandText', '')}"
        content_hash = self._compute_content_hash(hash_content)


        document = {
            "id": call_id,
            "project": project,
            "caller_namespace": call.get('CallerNamespace', ''),
            "caller_class": caller_class,
            "caller_method": caller_method,
            "caller_file": call.get('CallerFilePath', ''),
            "caller_line": call.get('CallerLineNumber', 0),
            "callee_namespace": call.get('CalleeNamespace', ''),
            "callee_class": callee_class,
            "callee_method": callee_method,
            "callee_file": call.get('CalleeFilePath', ''),
            "call_type": call.get('CallType', 'Direct'),
            "is_sql_operation": call.get('IsSqlOperation', False),
            "stored_procedure_name": call.get('StoredProcedureName', ''),
            "sql_command_text": call.get('SqlCommandText', ''),
            "call_site_line": call.get('CallSiteLineNumber', 0),
            "embedding_text": embedding_text,
            "content_hash": content_hash,
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        # Use update with $set and $setOnInsert for created_at
        await collection.update_one(
            {"id": call_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

    async def _store_eventhandler(self, handler: Dict, project: str):
        """Store an event handler with embedding"""
        collection = self.db[COLLECTION_CODE_EVENTHANDLERS]

        event_name = handler.get('EventName', '')
        handler_method = handler.get('HandlerMethod', '')
        handler_class = handler.get('HandlerClass', '')

        # Generate ID
        handler_id = f"{project}:{handler_class}.{handler_method}:{event_name}"

        # Generate embedding
        embedding_text = self._format_eventhandler_embedding_text(handler)
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        # Compute content hash for change detection
        element_name = handler.get('ElementName', '')
        hash_content = f"{event_name}:{handler_method}:{handler_class}:{element_name}"
        content_hash = self._compute_content_hash(hash_content)


        document = {
            "id": handler_id,
            "project": project,
            "event_name": event_name,
            "handler_method": handler_method,
            "handler_class": handler_class,
            "namespace": handler.get('Namespace', ''),
            "ui_element_type": handler.get('UIElementType', ''),
            "element_name": handler.get('ElementName', ''),
            "subscription_type": handler.get('SubscriptionType', ''),
            "file_path": handler.get('FilePath', ''),
            "line_number": handler.get('LineNumber', 0),
            "embedding_text": embedding_text,
            "content_hash": content_hash,
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        # Use update with $set and $setOnInsert for created_at
        await collection.update_one(
            {"id": handler_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

    async def _store_dboperation(self, op: Dict, project: str):
        """Store a database operation with embedding"""
        collection = self.db[COLLECTION_CODE_DBOPERATIONS]

        method_name = op.get('MethodName', '')
        class_name = op.get('ClassName', '')
        line = op.get('LineNumber', 0)

        # Generate ID
        op_id = f"{project}:{class_name}.{method_name}:{line}"

        # Generate embedding text
        parts = [f"Database operation in {class_name}.{method_name}"]

        op_type = op.get('OperationType', '')
        if op_type:
            parts.append(f"Type: {op_type}")

        table_name = op.get('TableName', '')
        if table_name:
            parts.append(f"Table: {table_name}")

        sp = op.get('StoredProcedure', '')
        if sp:
            parts.append(f"Stored Procedure: {sp}")

        cmd = op.get('CommandText', '')
        if cmd:
            parts.append(f"SQL: {cmd[:200]}")  # Limit length

        embedding_text = "\n".join(parts)
        embedding = await self.embedding_service.generate_embedding(embedding_text)

        # Compute content hash for change detection
        hash_content = f"{class_name}.{method_name}:{op_type}:{table_name}:{sp}:{cmd}"
        content_hash = self._compute_content_hash(hash_content)


        document = {
            "id": op_id,
            "project": project,
            "class_name": class_name,
            "method_name": method_name,
            "operation_type": op_type,
            "table_name": table_name,
            "stored_procedure": sp,
            "command_text": cmd,
            "command_type": op.get('CommandType', ''),
            "parameters": op.get('Parameters', []),
            "file_path": op.get('FilePath', ''),
            "line_number": line,
            "embedding_text": embedding_text,
            "content_hash": content_hash,
            "updated_at": datetime.utcnow().isoformat(),
            "vector": embedding
        }

        # Use update with $set and $setOnInsert for created_at
        await collection.update_one(
            {"id": op_id},
            {
                "$set": document,
                "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}
            },
            upsert=True
        )

    # =========================================================================
    # Search Methods
    # =========================================================================

    async def _vector_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
        filter_query: Optional[Dict] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """
        Perform vector similarity search.

        Delegates to main MongoDB service for native $vectorSearch support,
        falls back to in-memory calculation if not available.
        """
        # Try to use native vector search from main mongodb_service
        try:
            main_service = get_mongodb_service()
            if main_service.is_initialized and main_service._vector_search_available:
                # Check if this collection has a vector index
                index_name = main_service.VECTOR_INDEX_NAMES.get(collection_name)
                if index_name:
                    return await main_service._vector_search_native(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        index_name=index_name,
                        limit=limit,
                        filter_query=filter_query,
                        threshold=threshold
                    )
        except Exception as e:
            print(f"Native vector search not available, using in-memory: {e}")

        # Fall back to in-memory search
        return await self._vector_search_inmemory(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter_query=filter_query,
            threshold=threshold
        )

    async def _vector_search_inmemory(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
        filter_query: Optional[Dict] = None,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Dict]:
        """In-memory vector similarity search using cosine similarity."""
        collection = self.db[collection_name]

        query = filter_query or {}
        cursor = collection.find(query)
        documents = await cursor.to_list(length=10000)

        if not documents:
            return []

        results = []
        query_vec = np.array(query_vector)
        query_norm = np.linalg.norm(query_vec)

        for doc in documents:
            if 'vector' not in doc:
                continue

            doc_vec = np.array(doc['vector'])
            doc_norm = np.linalg.norm(doc_vec)

            if query_norm == 0 or doc_norm == 0:
                continue

            similarity = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))

            if similarity >= threshold:
                doc['_similarity'] = similarity
                # Remove vector from results to reduce payload
                doc.pop('vector', None)
                results.append(doc)

        results.sort(key=lambda x: x['_similarity'], reverse=True)
        return results[:limit]

    async def search_methods(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 10,
        include_sql_only: bool = False
    ) -> List[Dict]:
        """
        Search for methods by semantic similarity.

        Args:
            query: Natural language query
            project: Optional project filter
            limit: Max results
            include_sql_only: Only return methods with SQL operations
        """
        if not self.is_initialized:
            await self.initialize()

        query_vector = await self.embedding_service.generate_embedding(query)

        filter_query = {}
        if project:
            filter_query["project"] = project
        if include_sql_only:
            filter_query["has_sql_operations"] = True

        results = await self._vector_search(
            COLLECTION_CODE_METHODS,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None
        )

        return [{
            "id": doc.get("id"),
            "method_name": doc.get("method_name"),
            "class_name": doc.get("class_name"),
            "namespace": doc.get("namespace"),
            "signature": f"{doc.get('return_type', 'void')} {doc.get('method_name')}({', '.join([p.get('name', '') for p in doc.get('parameters', [])])})",
            "summary": doc.get("summary"),
            "sql_calls": doc.get("sql_calls", []),
            "cyclomatic_complexity": doc.get("cyclomatic_complexity"),
            "file_path": doc.get("file_path"),
            "line_number": doc.get("line_number"),
            "similarity": doc.get("_similarity"),
            "project": doc.get("project")
        } for doc in results]

    async def search_classes(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search for classes by semantic similarity"""
        if not self.is_initialized:
            await self.initialize()

        query_vector = await self.embedding_service.generate_embedding(query)

        filter_query = {}
        if project:
            filter_query["project"] = project

        results = await self._vector_search(
            COLLECTION_CODE_CLASSES,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None
        )

        return [{
            "id": doc.get("id"),
            "class_name": doc.get("class_name"),
            "namespace": doc.get("namespace"),
            "base_class": doc.get("base_class"),
            "interfaces": doc.get("interfaces", []),
            "methods": doc.get("methods", []),
            "properties": doc.get("properties", []),
            "summary": doc.get("summary"),
            "file_path": doc.get("file_path"),
            "similarity": doc.get("_similarity"),
            "project": doc.get("project")
        } for doc in results]

    async def search_event_handlers(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search for event handlers (UI to code mapping)"""
        if not self.is_initialized:
            await self.initialize()

        query_vector = await self.embedding_service.generate_embedding(query)

        filter_query = {}
        if project:
            filter_query["project"] = project

        results = await self._vector_search(
            COLLECTION_CODE_EVENTHANDLERS,
            query_vector,
            limit=limit,
            filter_query=filter_query if filter_query else None
        )

        return [{
            "id": doc.get("id"),
            "event_name": doc.get("event_name"),
            "handler_method": doc.get("handler_method"),
            "handler_class": doc.get("handler_class"),
            "ui_element_type": doc.get("ui_element_type"),
            "element_name": doc.get("element_name"),
            "file_path": doc.get("file_path"),
            "similarity": doc.get("_similarity"),
            "project": doc.get("project")
        } for doc in results]

    async def get_call_chain(
        self,
        method_name: str,
        class_name: str,
        project: Optional[str] = None,
        direction: str = "both",  # "callers", "callees", or "both"
        max_depth: int = 3
    ) -> Dict:
        """
        Get the call chain for a method.

        Args:
            method_name: Method to trace
            class_name: Containing class
            project: Project filter
            direction: "callers" (who calls this), "callees" (what it calls), or "both"
            max_depth: Max levels to traverse

        Returns:
            Dict with callers and/or callees
        """
        if not self.is_initialized:
            await self.initialize()

        collection = self.db[COLLECTION_CODE_CALLGRAPH]
        result = {"method": f"{class_name}.{method_name}", "project": project}

        if direction in ("callers", "both"):
            # Find who calls this method
            query = {
                "callee_class": class_name,
                "callee_method": method_name
            }
            if project:
                query["project"] = project

            cursor = collection.find(query)
            callers = await cursor.to_list(length=100)
            result["callers"] = [{
                "class": c.get("caller_class"),
                "method": c.get("caller_method"),
                "file": c.get("caller_file"),
                "line": c.get("call_site_line")
            } for c in callers]

        if direction in ("callees", "both"):
            # Find what this method calls
            query = {
                "caller_class": class_name,
                "caller_method": method_name
            }
            if project:
                query["project"] = project

            cursor = collection.find(query)
            callees = await cursor.to_list(length=100)
            result["callees"] = [{
                "class": c.get("callee_class"),
                "method": c.get("callee_method"),
                "file": c.get("callee_file"),
                "is_sql": c.get("is_sql_operation", False),
                "stored_procedure": c.get("stored_procedure_name")
            } for c in callees]

        return result

    async def get_comprehensive_code_context(
        self,
        query: str,
        project: Optional[str] = None,
        method_limit: int = 10,
        class_limit: int = 5,
        callgraph_limit: int = 10
    ) -> Dict:
        """
        Get comprehensive code context for answering developer questions.
        Combines methods, classes, and call relationships.
        """
        if not self.is_initialized:
            await self.initialize()

        import asyncio

        # Run searches in parallel
        methods_task = self.search_methods(query, project=project, limit=method_limit)
        classes_task = self.search_classes(query, project=project, limit=class_limit)

        methods, classes = await asyncio.gather(methods_task, classes_task)

        return {
            "methods": methods,
            "classes": classes,
            "query": query,
            "project": project
        }

    async def get_stats(self) -> Dict:
        """Get statistics for all Roslyn collections"""
        if not self.is_initialized:
            await self.initialize()

        collections = {
            "classes": COLLECTION_CODE_CLASSES,
            "methods": COLLECTION_CODE_METHODS,
            "callgraph": COLLECTION_CODE_CALLGRAPH,
            "eventhandlers": COLLECTION_CODE_EVENTHANDLERS,
            "dboperations": COLLECTION_CODE_DBOPERATIONS
        }

        stats = {}
        for name, coll_name in collections.items():
            try:
                count = await self.db[coll_name].count_documents({})
                stats[name] = count
            except Exception:
                stats[name] = 0

        stats["total"] = sum(stats.values())

        return stats

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.is_initialized = False


# Singleton getter
def get_roslyn_mongodb_service() -> RoslynMongoDBService:
    """Get the singleton Roslyn MongoDB service instance"""
    return RoslynMongoDBService.get_instance()


# CLI for testing
async def main():
    """Test the Roslyn MongoDB service"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python roslyn_mongodb_service.py <path-to-cs-file-or-directory> [project-name]")
        print("\nExample:")
        print("  python roslyn_mongodb_service.py <path-to-project>\\WindowsUI\\ViewModels <ProjectName>")
        sys.exit(1)

    input_path = sys.argv[1]
    project_name = sys.argv[2] if len(sys.argv) > 2 else "Unknown"

    service = get_roslyn_mongodb_service()

    try:
        await service.initialize()

        print(f"\n{'='*60}")
        print(f"Analyzing: {input_path}")
        print(f"Project: {project_name}")
        print(f"{'='*60}\n")

        # Run analysis
        analysis_result = await service.analyze_code(input_path, project_name)

        if not analysis_result['success']:
            print(f"Analysis failed: {analysis_result.get('error')}")
            sys.exit(1)

        # Store results
        stats = await service.store_analysis(analysis_result)

        print(f"\n{'='*60}")
        print("Storage complete!")
        print(f"{'='*60}")

        # Show current stats
        total_stats = await service.get_stats()
        print("\nTotal documents in MongoDB:")
        for key, value in total_stats.items():
            print(f"  {key}: {value}")

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
