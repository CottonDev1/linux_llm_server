"""
EWR Code Intelligence Agent
===========================

Deep code analysis agent with self-validating knowledge.

This is the central intelligence agent that:
1. Performs deep static analysis of codebases
2. Builds call graphs and data flow traces
3. Self-validates its embeddings by querying Code Agent
4. Refines knowledge until answers are accurate
5. Answers complex developer questions
"""

import asyncio
import re
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

try:
    import networkx as nx
except ImportError:
    nx = None

from ewr_agent_core import (
    BaseAgent,
    AgentType,
    AgentCapability,
    TaskResult,
    TaskStatus,
    AgentConfig,
    MessagePriority,
)

from .models import (
    AnalysisStatus,
    EntryPointType,
    CallNode,
    CallEdge,
    CallGraph,
    DataFlowPoint,
    DataFlowTrace,
    EntryPoint,
    DependencyInfo,
    AnalysisResult,
    WorkflowStep,
    WorkflowExplanation,
    KnowledgeGap,
    ValidationResult,
    DeveloperAnswer,
    CodeChunk,
    EmbeddingResult,
    RefinementResult,
)


class CodeIntelAgent(BaseAgent):
    """
    Code Intelligence Agent - Deep code analysis with self-validation.

    This agent provides advanced code understanding by:
    - Building comprehensive call graphs
    - Tracing data flow through applications
    - Identifying entry points and key components
    - Self-validating knowledge accuracy
    - Answering complex developer questions

    It can delegate tasks to the Code Agent for basic file operations
    and uses its own deep analysis capabilities for advanced understanding.
    """

    def __init__(
        self,
        config: AgentConfig = None,
        storage_backend: str = "memory",  # memory, mongodb
        mongodb_uri: str = None,
        embedding_model: str = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.storage_backend = storage_backend
        self.mongodb_uri = mongodb_uri
        self.embedding_model = embedding_model

        # In-memory storage
        self._analysis_cache: Dict[str, AnalysisResult] = {}
        self._chunks: Dict[str, CodeChunk] = {}
        self._embeddings: Dict[str, EmbeddingResult] = {}
        self._call_graphs: Dict[str, CallGraph] = {}

    @property
    def agent_type(self) -> AgentType:
        return AgentType.CODE

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.CODE_ANALYZE,
            AgentCapability.CODE_EXPLAIN,
            AgentCapability.PROJECT_SCAN,
        ]

    async def _initialize(self) -> None:
        """Initialize the code intel agent."""
        self.logger.info("Code Intelligence Agent initialized")
        if nx is None:
            self.logger.warning("networkx not installed - some graph features disabled")

    async def handle_task(self, task: Dict[str, Any]) -> TaskResult:
        """Handle incoming tasks."""
        task_id = task.get("task_id", "unknown")
        task_type = task.get("task_type", "")
        params = task.get("params", {})

        try:
            if task_type == "analyze_repository":
                result = await self.analyze_repository(params.get("repo_path", "."))
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"analysis": result.model_dump()}
                )

            elif task_type == "build_call_graph":
                graph = await self.build_call_graph(
                    entry_point=params.get("entry_point", ""),
                    repo_path=params.get("repo_path", ".")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"call_graph": graph.model_dump() if graph else None}
                )

            elif task_type == "answer_question":
                answer = await self.answer_question(
                    question=params.get("question", ""),
                    repo_path=params.get("repo_path", ".")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"answer": answer.model_dump()}
                )

            elif task_type == "validate_knowledge":
                result = await self.validate_knowledge(
                    test_questions=params.get("questions", []),
                    repo_path=params.get("repo_path", ".")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"validation": result.model_dump()}
                )

            elif task_type == "explain_workflow":
                explanation = await self.explain_workflow(
                    question=params.get("question", ""),
                    repo_path=params.get("repo_path", ".")
                )
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result={"explanation": explanation.model_dump()}
                )

            else:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task_type}"
                )

        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e)
            )

    # =========================================================================
    # Repository Analysis
    # =========================================================================

    async def analyze_repository(
        self,
        repo_path: str,
        max_files: int = 5000,
        analyze_depth: str = "standard"  # quick, standard, deep
    ) -> AnalysisResult:
        """
        Perform deep analysis of a repository.

        Args:
            repo_path: Path to the repository
            max_files: Maximum files to process
            analyze_depth: Analysis depth level

        Returns:
            AnalysisResult with complete analysis
        """
        repo_path = str(Path(repo_path).resolve())
        repo_name = Path(repo_path).name

        result = AnalysisResult(
            repo_path=repo_path,
            repo_name=repo_name,
            status=AnalysisStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        try:
            # Scan project structure first
            structure = await self._scan_project_structure(repo_path, max_files)
            result.total_files = structure.get("total_files", 0)
            result.total_lines = structure.get("total_lines", 0)
            result.languages = structure.get("languages", {})

            # Find entry points
            result.entry_points = await self._find_entry_points(repo_path, structure)

            # Analyze dependencies
            result.dependencies = await self._analyze_dependencies(repo_path)

            # Build call graphs for key entry points
            if analyze_depth in ("standard", "deep"):
                for ep in result.entry_points[:10]:  # Limit to top 10
                    try:
                        graph = await self.build_call_graph(ep.name, repo_path)
                        if graph:
                            result.call_graphs[ep.id] = graph
                    except Exception as e:
                        self.logger.warning(f"Failed to build call graph for {ep.name}: {e}")

            # Identify key components
            result.key_classes = structure.get("key_classes", [])
            result.key_functions = structure.get("key_functions", [])

            # Generate architecture notes using LLM
            result.architecture_notes = await self._generate_architecture_notes(result)

            result.status = AnalysisStatus.COMPLETED
            result.completed_at = datetime.utcnow()

            # Cache the result
            self._analysis_cache[repo_path] = result

        except Exception as e:
            result.status = AnalysisStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.utcnow()

        return result

    async def _scan_project_structure(
        self,
        repo_path: str,
        max_files: int
    ) -> Dict[str, Any]:
        """Scan project and gather structure information."""
        import os

        structure = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "key_classes": [],
            "key_functions": [],
            "files": []
        }

        ignore_dirs = {
            "node_modules", "__pycache__", ".git", ".svn", "venv",
            ".venv", "dist", "build", "bin", "obj"
        }
        ignore_extensions = {".exe", ".dll", ".so", ".pyc", ".pyo", ".class"}

        extension_to_lang = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".cs": "C#", ".java": "Java", ".go": "Go", ".rs": "Rust",
            ".cpp": "C++", ".c": "C", ".rb": "Ruby", ".php": "PHP"
        }

        class_patterns = {
            "Python": re.compile(r"^\s*class\s+(\w+)"),
            "JavaScript": re.compile(r"^\s*class\s+(\w+)"),
            "TypeScript": re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"),
            "C#": re.compile(r"^\s*(?:public|internal|private)?\s*class\s+(\w+)"),
        }

        function_patterns = {
            "Python": re.compile(r"^\s*(?:async\s+)?def\s+(\w+)"),
            "JavaScript": re.compile(r"^\s*(?:async\s+)?function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?\("),
            "C#": re.compile(r"^\s*(?:public|private|protected)\s+(?:static\s+)?(?:async\s+)?\w+\s+(\w+)\s*\("),
        }

        for root, dirs, files in os.walk(repo_path):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                if structure["total_files"] >= max_files:
                    break

                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext in ignore_extensions:
                    continue

                structure["total_files"] += 1
                lang = extension_to_lang.get(ext, "Other")
                structure["languages"][lang] = structure["languages"].get(lang, 0) + 1

                # Analyze code files
                if lang in class_patterns:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            lines = content.split("\n")
                            structure["total_lines"] += len(lines)

                            # Find classes
                            for match in class_patterns[lang].finditer(content):
                                class_name = match.group(1)
                                if class_name and not class_name.startswith("_"):
                                    structure["key_classes"].append(f"{file}:{class_name}")

                            # Find functions
                            if lang in function_patterns:
                                for match in function_patterns[lang].finditer(content):
                                    func_name = match.group(1) or (match.group(2) if match.lastindex >= 2 else None)
                                    if func_name and not func_name.startswith("_"):
                                        structure["key_functions"].append(f"{file}:{func_name}")

                    except Exception:
                        pass

        # Limit to most important items
        structure["key_classes"] = structure["key_classes"][:100]
        structure["key_functions"] = structure["key_functions"][:200]

        return structure

    async def _find_entry_points(
        self,
        repo_path: str,
        structure: Dict[str, Any]
    ) -> List[EntryPoint]:
        """Identify entry points in the codebase."""
        entry_points = []

        # Patterns for different entry point types
        patterns = {
            EntryPointType.API_ENDPOINT: [
                # Flask/FastAPI
                (r'@app\.(get|post|put|delete|patch)\s*\(["\']([^"\']+)', "Python"),
                (r'@router\.(get|post|put|delete|patch)\s*\(["\']([^"\']+)', "Python"),
                # Express.js
                (r'app\.(get|post|put|delete|patch)\s*\(["\']([^"\']+)', "JavaScript"),
                (r'router\.(get|post|put|delete|patch)\s*\(["\']([^"\']+)', "JavaScript"),
                # ASP.NET
                (r'\[Http(Get|Post|Put|Delete)\s*\(["\']?([^"\')\]]*)', "C#"),
                (r'\[Route\(["\']([^"\']+)', "C#"),
            ],
            EntryPointType.MAIN_FUNCTION: [
                (r'if\s+__name__\s*==\s*["\']__main__["\']', "Python"),
                (r'static\s+void\s+Main\s*\(', "C#"),
                (r'func\s+main\s*\(\s*\)', "Go"),
            ],
            EntryPointType.EVENT_HANDLER: [
                (r'addEventListener\s*\(["\'](\w+)["\']', "JavaScript"),
                (r'on(\w+)\s*=\s*function', "JavaScript"),
            ],
        }

        import os
        for root, _, files in os.walk(repo_path):
            if any(ig in root for ig in ["node_modules", "__pycache__", ".git"]):
                continue

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                lang = {".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
                       ".cs": "C#", ".go": "Go"}.get(ext)

                if not lang:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        lines = content.split("\n")

                    for ep_type, type_patterns in patterns.items():
                        for pattern, pattern_lang in type_patterns:
                            if pattern_lang != lang:
                                continue

                            for i, line in enumerate(lines):
                                match = re.search(pattern, line)
                                if match:
                                    route = match.group(2) if match.lastindex >= 2 else match.group(1)
                                    method = match.group(1) if ep_type == EntryPointType.API_ENDPOINT else None

                                    entry_points.append(EntryPoint(
                                        id=str(uuid.uuid4()),
                                        name=f"{file}:{i+1}",
                                        entry_type=ep_type,
                                        file_path=str(file_path),
                                        line_number=i + 1,
                                        route=route if ep_type == EntryPointType.API_ENDPOINT else None,
                                        http_method=method.upper() if method else None,
                                    ))

                except Exception:
                    pass

        return entry_points[:50]  # Limit results

    async def _analyze_dependencies(self, repo_path: str) -> List[DependencyInfo]:
        """Analyze project dependencies."""
        dependencies = []
        repo_path = Path(repo_path)

        # Python requirements
        req_files = list(repo_path.glob("requirements*.txt"))
        for req_file in req_files:
            try:
                with open(req_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Parse package==version or package>=version
                            match = re.match(r"([a-zA-Z0-9_-]+)", line)
                            if match:
                                dependencies.append(DependencyInfo(
                                    name=match.group(1),
                                    source="external"
                                ))
            except Exception:
                pass

        # package.json
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                import json
                with open(package_json) as f:
                    data = json.load(f)
                    for name, version in data.get("dependencies", {}).items():
                        dependencies.append(DependencyInfo(
                            name=name,
                            version=version,
                            source="external"
                        ))
            except Exception:
                pass

        return dependencies

    async def _generate_architecture_notes(self, result: AnalysisResult) -> List[str]:
        """Generate architecture notes using LLM."""
        context = f"""Repository: {result.repo_name}
Languages: {', '.join(f'{k}: {v} files' for k, v in result.languages.items())}
Entry Points: {len(result.entry_points)}
Key Classes: {len(result.key_classes)}
Dependencies: {len(result.dependencies)}

Entry Point Types:
{chr(10).join(f'- {ep.entry_type.value}: {ep.name}' for ep in result.entry_points[:10])}

Key Classes:
{chr(10).join(f'- {c}' for c in result.key_classes[:20])}"""

        prompt = f"""Based on this code analysis, provide 3-5 brief architecture notes:

{context}

Focus on:
1. Overall architecture pattern (MVC, microservices, monolith, etc.)
2. Key components and their roles
3. Notable patterns or frameworks used
4. Potential areas of interest for developers"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system="You are a software architecture expert. Provide concise, actionable insights.",
                max_tokens=500
            )
            notes = [line.strip() for line in response.content.split("\n")
                    if line.strip() and not line.strip().startswith("#")]
            return notes[:5]
        except Exception as e:
            self.logger.warning(f"Failed to generate architecture notes: {e}")
            return []

    # =========================================================================
    # Call Graph Building
    # =========================================================================

    async def build_call_graph(
        self,
        entry_point: str,
        repo_path: str = ".",
        max_depth: int = 10
    ) -> CallGraph:
        """
        Build a call graph starting from an entry point.

        Args:
            entry_point: Function/method name to start from
            repo_path: Repository path
            max_depth: Maximum call depth to trace

        Returns:
            CallGraph with nodes and edges
        """
        graph = CallGraph(depth=max_depth)

        # For now, use a simplified approach with regex
        # In production, this would use AST parsing or Roslyn for C#

        visited: Set[str] = set()
        to_visit = [(entry_point, 0)]

        while to_visit:
            current, depth = to_visit.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            # Find the function definition
            definition = await self._find_function_definition(current, repo_path)

            if definition:
                node_id = str(uuid.uuid4())
                node = CallNode(
                    id=node_id,
                    name=current,
                    full_name=definition.get("full_name", current),
                    file_path=definition.get("file_path", ""),
                    line_number=definition.get("line_number", 0),
                    node_type=definition.get("type", "function"),
                    class_name=definition.get("class_name"),
                    is_async=definition.get("is_async", False),
                )
                graph.nodes[node_id] = node

                if graph.root_id is None:
                    graph.root_id = node_id

                # Find calls from this function
                calls = await self._find_function_calls(definition, repo_path)

                for callee in calls:
                    callee_id = str(uuid.uuid4())
                    graph.edges.append(CallEdge(
                        caller_id=node_id,
                        callee_id=callee_id,
                    ))
                    graph.total_calls += 1

                    if callee not in visited:
                        to_visit.append((callee, depth + 1))

        return graph

    async def _find_function_definition(
        self,
        func_name: str,
        repo_path: str
    ) -> Optional[Dict[str, Any]]:
        """Find a function definition in the codebase."""
        import os

        patterns = {
            ".py": re.compile(rf"^\s*(async\s+)?def\s+{re.escape(func_name)}\s*\("),
            ".js": re.compile(rf"(?:function\s+{re.escape(func_name)}|{re.escape(func_name)}\s*[=:]\s*(?:async\s+)?(?:function|\())"),
            ".cs": re.compile(rf"(?:public|private|protected)\s+(?:static\s+)?(?:async\s+)?\w+\s+{re.escape(func_name)}\s*\("),
        }

        for root, _, files in os.walk(repo_path):
            if any(ig in root for ig in ["node_modules", "__pycache__", ".git"]):
                continue

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext not in patterns:
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines):
                        if patterns[ext].search(line):
                            return {
                                "name": func_name,
                                "full_name": f"{file}:{func_name}",
                                "file_path": str(file_path),
                                "line_number": i + 1,
                                "type": "function",
                                "is_async": "async" in line,
                            }
                except Exception:
                    pass

        return None

    async def _find_function_calls(
        self,
        definition: Dict[str, Any],
        repo_path: str
    ) -> List[str]:
        """Find functions called from a function definition."""
        calls = []

        file_path = definition.get("file_path")
        if not file_path:
            return calls

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Simple call detection (would use AST in production)
            call_pattern = re.compile(r"(\w+)\s*\(")
            matches = call_pattern.findall(content)

            # Filter out common non-function calls
            ignore = {"if", "while", "for", "return", "print", "len", "str", "int", "list", "dict"}
            calls = [m for m in matches if m not in ignore and not m.startswith("_")]

        except Exception:
            pass

        return list(set(calls))[:20]  # Limit

    # =========================================================================
    # Data Flow Tracing
    # =========================================================================

    async def trace_data_flow(
        self,
        variable: str,
        file_path: str,
        start_line: int = 1
    ) -> DataFlowTrace:
        """
        Trace how a variable flows through code.

        Args:
            variable: Variable name to trace
            file_path: File to analyze
            start_line: Starting line number

        Returns:
            DataFlowTrace showing data movement
        """
        trace = DataFlowTrace(
            variable_name=variable,
            start_point=DataFlowPoint(
                variable_name=variable,
                file_path=file_path,
                line_number=start_line,
                operation="start"
            )
        )

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            # Track variable through code
            for i, line in enumerate(lines[start_line-1:], start=start_line):
                if variable in line:
                    # Determine operation type
                    if f"{variable} =" in line or f"{variable}=" in line:
                        op = "write"
                    elif f"return {variable}" in line or f"return({variable}" in line:
                        op = "return"
                    elif f"({variable}" in line or f", {variable}" in line:
                        op = "parameter"
                    else:
                        op = "read"

                    trace.flow_points.append(DataFlowPoint(
                        variable_name=variable,
                        file_path=file_path,
                        line_number=i,
                        operation=op
                    ))

        except Exception as e:
            self.logger.warning(f"Data flow trace failed: {e}")

        return trace

    # =========================================================================
    # Knowledge Validation
    # =========================================================================

    async def validate_knowledge(
        self,
        test_questions: List[str],
        repo_path: str = "."
    ) -> ValidationResult:
        """
        Self-validate knowledge by testing questions.

        This is the key self-improvement capability:
        1. Takes test questions about the codebase
        2. Attempts to answer each question
        3. Evaluates answer quality
        4. Identifies gaps in knowledge

        Args:
            test_questions: Questions to test
            repo_path: Repository to validate against

        Returns:
            ValidationResult with accuracy and gaps
        """
        result = ValidationResult(
            total_questions=len(test_questions),
            correct_answers=0
        )

        # Ensure we have analysis
        if repo_path not in self._analysis_cache:
            await self.analyze_repository(repo_path)

        for question in test_questions:
            # Try to answer the question
            answer = await self.answer_question(question, repo_path)

            # Evaluate the answer quality
            evaluation = await self._evaluate_answer(question, answer, repo_path)

            if evaluation["is_correct"]:
                result.correct_answers += 1
            else:
                result.gaps.append(KnowledgeGap(
                    question=question,
                    expected_topic=evaluation.get("expected_topic", "unknown"),
                    actual_answer=answer.answer[:500],
                    missing_context=evaluation.get("missing_context", []),
                    suggested_files=evaluation.get("suggested_files", []),
                    severity=evaluation.get("severity", "medium")
                ))

        # Calculate accuracy
        result.accuracy_score = result.correct_answers / len(test_questions) if test_questions else 0.0
        result.needs_refinement = result.accuracy_score < 0.7

        if result.needs_refinement:
            result.suggestions = [
                f"Re-analyze files: {', '.join(set(f for g in result.gaps for f in g.suggested_files))}",
                "Consider deeper analysis of identified gap areas",
                "Add more context about key components"
            ]

        return result

    async def _evaluate_answer(
        self,
        question: str,
        answer: DeveloperAnswer,
        repo_path: str
    ) -> Dict[str, Any]:
        """Evaluate if an answer is correct/complete."""
        # Use LLM to evaluate
        prompt = f"""Evaluate this answer about a codebase:

Question: {question}

Answer: {answer.answer}

Relevant files referenced: {', '.join(answer.relevant_files[:5])}
Confidence: {answer.confidence}

Evaluate:
1. Is the answer correct and complete? (yes/no)
2. What is the expected topic?
3. What context might be missing?
4. What severity if incomplete? (low/medium/high)

Format:
CORRECT: yes/no
TOPIC: <topic>
MISSING: <missing context>
SEVERITY: low/medium/high"""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system="You are evaluating code documentation quality. Be objective.",
                max_tokens=300
            )

            content = response.content

            return {
                "is_correct": "CORRECT: yes" in content.lower(),
                "expected_topic": self._extract_field(content, "TOPIC"),
                "missing_context": [self._extract_field(content, "MISSING")] if "MISSING:" in content else [],
                "severity": self._extract_field(content, "SEVERITY") or "medium",
                "suggested_files": answer.relevant_files[:3]
            }

        except Exception:
            return {
                "is_correct": answer.confidence > 0.5,
                "expected_topic": "unknown",
                "missing_context": [],
                "severity": "medium",
                "suggested_files": []
            }

    def _extract_field(self, content: str, field: str) -> str:
        """Extract a field from evaluation response."""
        match = re.search(rf"{field}:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    # =========================================================================
    # Developer Question Answering
    # =========================================================================

    async def answer_question(
        self,
        question: str,
        repo_path: str = "."
    ) -> DeveloperAnswer:
        """
        Answer a developer question about the codebase.

        Args:
            question: Natural language question
            repo_path: Repository to answer about

        Returns:
            DeveloperAnswer with response and context
        """
        # Get or build analysis
        analysis = self._analysis_cache.get(repo_path)
        if not analysis:
            analysis = await self.analyze_repository(repo_path)

        # Build context for LLM
        context = self._build_answer_context(analysis, question)

        prompt = f"""Answer this developer question about the codebase:

Question: {question}

Codebase Context:
{context}

Provide:
1. A clear answer to the question
2. Relevant code files/functions to look at
3. Any code examples if helpful
4. Suggestions for follow-up learning"""

        response = await self.llm.generate(
            prompt=prompt,
            system="You are a senior developer helping someone understand a codebase. Be specific and actionable.",
            max_tokens=1000
        )

        # Parse response
        answer = DeveloperAnswer(
            question=question,
            answer=response.content,
            confidence=0.7,  # Base confidence
            relevant_files=self._extract_file_references(response.content, analysis)
        )

        # Increase confidence if we found specific files
        if answer.relevant_files:
            answer.confidence = min(0.9, answer.confidence + 0.1 * len(answer.relevant_files))

        return answer

    def _build_answer_context(self, analysis: AnalysisResult, question: str) -> str:
        """Build context for answering a question."""
        lines = [
            f"Repository: {analysis.repo_name}",
            f"Languages: {', '.join(analysis.languages.keys())}",
            "",
            "Entry Points:",
        ]

        for ep in analysis.entry_points[:10]:
            lines.append(f"  - {ep.entry_type.value}: {ep.name} ({ep.route or 'N/A'})")

        lines.extend(["", "Key Classes:"])
        for cls in analysis.key_classes[:15]:
            lines.append(f"  - {cls}")

        lines.extend(["", "Architecture Notes:"])
        for note in analysis.architecture_notes:
            lines.append(f"  - {note}")

        return "\n".join(lines)

    def _extract_file_references(
        self,
        content: str,
        analysis: AnalysisResult
    ) -> List[str]:
        """Extract file references from answer."""
        files = []

        # Look for file patterns in the response
        file_pattern = re.compile(r"[\w/\\.-]+\.(py|js|ts|cs|java|go)")
        matches = file_pattern.findall(content)

        # Also match against known files
        for ep in analysis.entry_points:
            if ep.file_path and any(part in content.lower() for part in Path(ep.file_path).name.lower().split(".")):
                files.append(ep.file_path)

        return list(set(files))[:10]

    async def explain_workflow(
        self,
        question: str,
        repo_path: str = "."
    ) -> WorkflowExplanation:
        """
        Explain a workflow with full code path.

        Args:
            question: Question like "How does user login work?"
            repo_path: Repository path

        Returns:
            WorkflowExplanation with step-by-step breakdown
        """
        # Get analysis
        analysis = self._analysis_cache.get(repo_path)
        if not analysis:
            analysis = await self.analyze_repository(repo_path)

        # Build prompt
        context = self._build_answer_context(analysis, question)

        prompt = f"""Explain this workflow step by step:

Question: {question}

Codebase Context:
{context}

For each step provide:
1. Step number
2. What happens
3. Which file/function
4. Relevant code location

Format as:
STEP 1: <description>
FILE: <file path>
FUNCTION: <function name>
---
STEP 2: ...

SUMMARY: <overall workflow summary>"""

        response = await self.llm.generate(
            prompt=prompt,
            system="You are explaining code workflows. Be specific about file locations and function names.",
            max_tokens=1500
        )

        # Parse response
        explanation = WorkflowExplanation(
            question=question,
            summary="",
            steps=[]
        )

        content = response.content
        step_blocks = content.split("---")

        for i, block in enumerate(step_blocks):
            if "STEP" in block:
                step = WorkflowStep(
                    step_number=i + 1,
                    description=self._extract_field(block, "STEP \\d+") or block[:200],
                    file_path=self._extract_field(block, "FILE") or "",
                    function_name=self._extract_field(block, "FUNCTION") or "",
                    line_number=0
                )
                explanation.steps.append(step)

        # Extract summary
        summary_match = re.search(r"SUMMARY:\s*(.+)", content, re.DOTALL)
        if summary_match:
            explanation.summary = summary_match.group(1).strip()
        else:
            explanation.summary = f"Workflow explaining: {question}"

        explanation.confidence = 0.7

        return explanation

    # =========================================================================
    # Embedding Refinement
    # =========================================================================

    async def refine_embeddings(
        self,
        gaps: List[KnowledgeGap],
        repo_path: str = "."
    ) -> RefinementResult:
        """
        Refine embeddings based on identified knowledge gaps.

        Args:
            gaps: Knowledge gaps to address
            repo_path: Repository path

        Returns:
            RefinementResult with refinement outcome
        """
        result = RefinementResult(
            chunks_processed=0,
            chunks_added=0,
            chunks_updated=0
        )

        # Identify files to reprocess
        files_to_process = set()
        for gap in gaps:
            files_to_process.update(gap.suggested_files)

        # Process each file
        for file_path in files_to_process:
            try:
                chunks = await self._chunk_file(file_path)
                result.chunks_processed += len(chunks)

                for chunk in chunks:
                    # Generate embedding
                    embedding = await self._generate_embedding(chunk)

                    if chunk.id in self._embeddings:
                        result.chunks_updated += 1
                    else:
                        result.chunks_added += 1

                    self._chunks[chunk.id] = chunk
                    self._embeddings[chunk.id] = embedding

            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")

        # Re-validate
        test_questions = [g.question for g in gaps]
        if test_questions:
            validation = await self.validate_knowledge(test_questions, repo_path)
            result.new_accuracy_score = validation.accuracy_score
            result.remaining_gaps = validation.gaps

        return result

    async def _chunk_file(self, file_path: str) -> List[CodeChunk]:
        """Split a file into chunks for embedding."""
        chunks = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Simple chunking by functions/classes
            # In production, use AST parsing

            lines = content.split("\n")
            chunk_lines = []
            chunk_start = 1

            for i, line in enumerate(lines, 1):
                chunk_lines.append(line)

                # End chunk on function/class boundaries
                if len(chunk_lines) >= 50 or (
                    line.strip() and (
                        line.strip().startswith("def ") or
                        line.strip().startswith("class ") or
                        line.strip().startswith("function ") or
                        "public " in line and "(" in line
                    )
                ):
                    if chunk_lines:
                        chunks.append(CodeChunk(
                            id=str(uuid.uuid4()),
                            content="\n".join(chunk_lines),
                            file_path=file_path,
                            start_line=chunk_start,
                            end_line=i,
                            chunk_type="block",
                            language=Path(file_path).suffix[1:],
                        ))
                        chunk_lines = []
                        chunk_start = i + 1

            # Add remaining
            if chunk_lines:
                chunks.append(CodeChunk(
                    id=str(uuid.uuid4()),
                    content="\n".join(chunk_lines),
                    file_path=file_path,
                    start_line=chunk_start,
                    end_line=len(lines),
                    chunk_type="block",
                    language=Path(file_path).suffix[1:],
                ))

        except Exception as e:
            self.logger.warning(f"Chunking failed for {file_path}: {e}")

        return chunks

    async def _generate_embedding(self, chunk: CodeChunk) -> EmbeddingResult:
        """Generate embedding for a code chunk."""
        try:
            embedding = await self.llm.embed(chunk.content[:4000])
            return EmbeddingResult(
                chunk_id=chunk.id,
                embedding=embedding,
                model=self.embedding_model or self.config.llm_model,
                dimensions=len(embedding)
            )
        except Exception as e:
            self.logger.warning(f"Embedding generation failed: {e}")
            return EmbeddingResult(
                chunk_id=chunk.id,
                embedding=[],
                model=self.embedding_model or "unknown",
                dimensions=0
            )
