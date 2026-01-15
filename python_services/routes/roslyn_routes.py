"""
Roslyn routes for C# code analysis using Roslyn.

Provides endpoints for:
- C# code analysis with Roslyn
- Semantic search for methods, classes, and event handlers
- Call graph traversal and impact analysis
- Repository statistics
"""
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException, Body

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLLECTION_CODE_CALLGRAPH, COLLECTION_CODE_METHODS
from roslyn_mongodb_service import get_roslyn_mongodb_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/roslyn", tags=["Roslyn"])


# ============================================================================
# Roslyn Analysis Routes
# ============================================================================

@router.post("/analyze")
async def analyze_csharp_code(
    input_path: str = Body(..., description="Path to C# file or directory"),
    project: str = Body("Unknown", description="Project name for tagging"),
    store: bool = Body(True, description="Store results in MongoDB")
):
    """
    Analyze C# code using Roslyn and optionally store results in MongoDB.

    Extracts:
    - Classes with inheritance and members
    - Methods with signatures, complexity, and SQL operations
    - Call graph relationships
    - Event handlers (UI to code mapping)
    - Database operations
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    # Run Roslyn analyzer
    result = await roslyn.analyze_code(input_path, project)

    if not result['success']:
        raise HTTPException(status_code=500, detail=result.get('error', 'Analysis failed'))

    if store:
        # Store in MongoDB with embeddings
        stats = await roslyn.store_analysis(result)
        return {
            "success": True,
            "project": project,
            "input_path": input_path,
            "storage_stats": stats,
            "data_summary": {
                "classes": len(result['data'].get('Classes', [])),
                "methods": len(result['data'].get('Methods', [])),
                "call_graph": len(result['data'].get('CallGraph', [])),
                "event_handlers": len(result['data'].get('EventHandlers', [])),
                "db_operations": len(result['data'].get('DatabaseOperations', []))
            }
        }
    else:
        return {
            "success": True,
            "project": project,
            "input_path": input_path,
            "data": result['data']
        }


@router.get("/search/methods")
async def search_roslyn_methods(
    query: str = Query(..., description="Natural language query"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(10, ge=1, le=100),
    sql_only: bool = Query(False, description="Only return methods with SQL operations")
):
    """
    Search for C# methods using semantic similarity.

    Examples:
    - "methods that save bale data"
    - "button click handlers"
    - "methods that call stored procedures"
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    results = await roslyn.search_methods(query, project, limit, sql_only)
    return {
        "success": True,
        "query": query,
        "count": len(results),
        "results": results
    }


@router.get("/search/classes")
async def search_roslyn_classes(
    query: str = Query(..., description="Natural language query"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Search for C# classes using semantic similarity.

    Examples:
    - "view model for bale entry"
    - "classes that handle shipping orders"
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    results = await roslyn.search_classes(query, project, limit)
    return {
        "success": True,
        "query": query,
        "count": len(results),
        "results": results
    }


@router.get("/search/event-handlers")
async def search_roslyn_event_handlers(
    query: str = Query(..., description="Natural language query"),
    project: Optional[str] = Query(None, description="Filter by project"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Search for event handlers (UI to code mapping).

    Examples:
    - "what happens when Save button is clicked"
    - "button click handlers for shipping"
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    results = await roslyn.search_event_handlers(query, project, limit)
    return {
        "success": True,
        "query": query,
        "count": len(results),
        "results": results
    }


@router.get("/call-chain")
async def get_roslyn_call_chain(
    method_name: str = Query(..., description="Method name"),
    class_name: str = Query(..., description="Class name"),
    project: Optional[str] = Query(None, description="Filter by project"),
    direction: str = Query("both", description="callers, callees, or both"),
    max_depth: int = Query(3, ge=1, le=10, description="Max traversal depth")
):
    """
    Get the call chain for a method.

    Returns who calls this method and/or what it calls.
    Useful for understanding code flow and impact analysis.
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    result = await roslyn.get_call_chain(method_name, class_name, project, direction, max_depth)
    return {
        "success": True,
        **result
    }


@router.get("/stats")
async def get_roslyn_stats():
    """Get statistics for all Roslyn collections in MongoDB."""
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    stats = await roslyn.get_stats()
    return {
        "success": True,
        **stats
    }


@router.post("/analyze-targeted")
async def analyze_targeted_files(
    methods: List[str] = Body(..., description="List of method names to find references for"),
    repo_path: str = Body(..., description="Path to repository root"),
    project: str = Body(..., description="Project name")
):
    """
    Targeted analysis: Find files containing method references and analyze only those.

    This is much faster than full repository analysis when you only need to update
    call graph data for specific methods.

    Steps:
    1. Search repository for files containing method name references
    2. Create a file list of matching C# files
    3. Run Roslyn analyzer on just those files using --filelist mode
    4. Store updated call graph data
    """
    import os

    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists():
        raise HTTPException(status_code=400, detail=f"Repository path does not exist: {repo_path}")

    # Find files containing references to the methods
    files_to_analyze = set()

    for method in methods:
        # Extract just the method name if it's in ClassName.MethodName format
        method_name = method.split('.')[-1] if '.' in method else method

        try:
            # Use grep to find files containing the method name
            result = subprocess.run(
                ['grep', '-rl', '--include=*.cs', method_name, str(repo_path_obj)],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                for file_path in result.stdout.strip().split('\n'):
                    if file_path and file_path.endswith('.cs'):
                        # Exclude obj and bin directories
                        if '\\obj\\' not in file_path and '/obj/' not in file_path:
                            if '\\bin\\' not in file_path and '/bin/' not in file_path:
                                files_to_analyze.add(file_path)
        except subprocess.TimeoutExpired:
            print(f"Grep timeout for method: {method_name}")
        except Exception as e:
            print(f"Error searching for {method_name}: {e}")

    if not files_to_analyze:
        return {
            "success": True,
            "message": "No files found containing method references",
            "methods_searched": methods,
            "files_analyzed": 0
        }

    # Create a temporary file list for Roslyn analyzer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for file_path in files_to_analyze:
            f.write(file_path + '\n')
        file_list_path = f.name

    try:
        # Run Roslyn analyzer with --filelist mode
        analysis_result = await roslyn.analyze_code_with_filelist(file_list_path, project)

        if analysis_result.get('success'):
            # Store results in MongoDB
            stats = await roslyn.store_analysis(analysis_result)

            return {
                "success": True,
                "project": project,
                "methods_searched": methods,
                "files_found": len(files_to_analyze),
                "files_analyzed": list(files_to_analyze)[:20],  # Limit for response size
                "files_truncated": len(files_to_analyze) > 20,
                "storage_stats": stats,
                "data_summary": {
                    "classes": len(analysis_result['data'].get('Classes', [])),
                    "methods": len(analysis_result['data'].get('Methods', [])),
                    "call_graph": len(analysis_result['data'].get('CallGraph', []))
                }
            }
        else:
            return {
                "success": False,
                "error": analysis_result.get('error', 'Analysis failed'),
                "files_attempted": len(files_to_analyze)
            }
    finally:
        try:
            import os
            os.unlink(file_list_path)
        except:
            pass


@router.post("/analyze-full")
async def analyze_full_repository(
    repo_path: str = Body(..., description="Path to repository root"),
    project: str = Body(..., description="Project name"),
    generate_embeddings: bool = Body(True, description="Generate vector embeddings")
):
    """
    Full repository analysis: Analyze all C# files and build complete call graph.

    This is a comprehensive analysis that:
    1. Scans all .cs files in the repository
    2. Extracts classes, methods, and call relationships
    3. Generates embeddings for semantic search
    4. Stores everything in MongoDB

    Use this for initial setup or periodic full refresh.
    For incremental updates, use /roslyn/analyze-targeted instead.
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    repo_path_obj = Path(repo_path)
    if not repo_path_obj.exists():
        raise HTTPException(status_code=400, detail=f"Repository path does not exist: {repo_path}")

    # Count files first
    cs_files = list(repo_path_obj.rglob('*.cs'))
    cs_files = [f for f in cs_files if '\\obj\\' not in str(f) and '/obj/' not in str(f)]
    cs_files = [f for f in cs_files if '\\bin\\' not in str(f) and '/bin/' not in str(f)]

    if not cs_files:
        raise HTTPException(status_code=400, detail="No C# files found in repository")

    print(f"Starting full analysis of {project}: {len(cs_files)} files")

    # Run full analysis
    analysis_result = await roslyn.analyze_code(str(repo_path_obj), project)

    if not analysis_result.get('success'):
        raise HTTPException(status_code=500, detail=analysis_result.get('error', 'Analysis failed'))

    # Store in MongoDB
    stats = await roslyn.store_analysis(analysis_result, generate_embeddings)

    return {
        "success": True,
        "project": project,
        "repo_path": repo_path,
        "total_files": len(cs_files),
        "storage_stats": stats,
        "data_summary": {
            "classes": len(analysis_result['data'].get('Classes', [])),
            "methods": len(analysis_result['data'].get('Methods', [])),
            "call_graph": len(analysis_result['data'].get('CallGraph', [])),
            "event_handlers": len(analysis_result['data'].get('EventHandlers', [])),
            "db_operations": len(analysis_result['data'].get('DatabaseOperations', []))
        }
    }


@router.get("/callgraph-status/{project}")
async def get_callgraph_status(project: str):
    """
    Check if call graph data exists for a project.

    Returns counts of call graph entries and when they were last updated.
    Useful for determining if Roslyn analysis needs to be run.
    """
    roslyn = get_roslyn_mongodb_service()
    await roslyn.initialize()

    # Get call graph count for this project
    callgraph_collection = roslyn.db[COLLECTION_CODE_CALLGRAPH]
    methods_collection = roslyn.db[COLLECTION_CODE_METHODS]

    callgraph_count = await callgraph_collection.count_documents({"project": project})
    methods_count = await methods_collection.count_documents({"project": project})

    # Get most recent entry timestamp
    latest_entry = await callgraph_collection.find_one(
        {"project": project},
        sort=[("timestamp", -1)]
    )

    last_updated = latest_entry.get("timestamp") if latest_entry else None

    return {
        "success": True,
        "project": project,
        "has_callgraph": callgraph_count > 0,
        "callgraph_entries": callgraph_count,
        "methods_indexed": methods_count,
        "last_updated": last_updated,
        "needs_analysis": callgraph_count == 0
    }
