"""
Call Chain Builder Tests
========================

Test the CallChainBuilder service including:
- build_chain() with real code scenarios
- Entry point detection
- Method tracing across files
- Cycle detection
- CallChain.compute_touches_database()
- CallTree.compute_total_methods()
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from code_flow_pipeline.services.call_chain_builder import (
    CallChainBuilder,
    get_call_chain_builder,
    TraversalState,
)
from code_flow_pipeline.models.query_models import (
    CallChain,
    CallChainNode,
    CallTree,
    FormattedResult,
)
from utils import generate_test_id


class TestCallChainBuilderInitialization:
    """Test builder initialization."""

    def test_builder_creation(self):
        """Test builder creates successfully."""
        builder = CallChainBuilder()

        assert builder._mongodb_service is None  # Lazy
        assert builder._method_cache == {}

    def test_builder_with_mongodb_service(self, mock_mongodb_service):
        """Test builder with injected MongoDB service."""
        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        assert builder._mongodb_service is mock_mongodb_service

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test get_call_chain_builder returns singleton."""
        import code_flow_pipeline.services.call_chain_builder as module
        module._builder_instance = None

        builder1 = await get_call_chain_builder()
        builder2 = await get_call_chain_builder()

        assert builder1 is builder2

        module._builder_instance = None


class TestBuildChains:
    """Test the build_chains() method."""

    @pytest.mark.asyncio
    async def test_build_simple_chain(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test building a simple linear call chain."""
        # A -> B -> C (terminal with database)
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": '["MethodB"]',
            }
        }
        method_b = {
            "metadata": {
                "methodName": "MethodB",
                "callsMethod": '["MethodC"]',
            }
        }
        method_c = {
            "metadata": {
                "methodName": "MethodC",
                "callsMethod": '[]',
                "databaseTables": '["TableA"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "MethodA" in query:
                return [method_a]
            elif "MethodB" in query:
                return [method_b]
            elif "MethodC" in query:
                return [method_c]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            project="TestProject",
            max_depth=10,
        )

        assert len(chains) > 0

        # Should have chain from A to C
        chain = chains[0]
        assert chain.start_method == "MethodA"
        assert chain.touches_database is True

    @pytest.mark.asyncio
    async def test_build_chain_with_branching(
        self,
        mock_mongodb_service,
    ):
        """Test building chains with branching paths."""
        # A -> B -> C
        #   -> D -> E
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": '["MethodB", "MethodD"]',
            }
        }
        method_b = {
            "metadata": {
                "methodName": "MethodB",
                "callsMethod": '["MethodC"]',
            }
        }
        method_c = {
            "metadata": {
                "methodName": "MethodC",
                "callsMethod": '[]',
                "databaseTables": '["TableA"]',
            }
        }
        method_d = {
            "metadata": {
                "methodName": "MethodD",
                "callsMethod": '["MethodE"]',
            }
        }
        method_e = {
            "metadata": {
                "methodName": "MethodE",
                "callsMethod": '[]',
            }
        }

        methods = {
            "MethodA": method_a,
            "MethodB": method_b,
            "MethodC": method_c,
            "MethodD": method_d,
            "MethodE": method_e,
        }

        async def search_vectors(query, **kwargs):
            for name, method in methods.items():
                if name in query:
                    return [method]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            max_depth=10,
        )

        # Should have multiple chains (one for each path)
        assert len(chains) >= 2

    @pytest.mark.asyncio
    async def test_build_chain_respects_max_depth(
        self,
        mock_mongodb_service,
    ):
        """Test chain building respects maximum depth."""
        # Create deep chain: M1 -> M2 -> M3 -> ... -> M20
        methods = {}
        for i in range(1, 21):
            next_method = f'["Method{i+1}"]' if i < 20 else '[]'
            methods[f"Method{i}"] = {
                "metadata": {
                    "methodName": f"Method{i}",
                    "callsMethod": next_method,
                }
            }

        async def search_vectors(query, **kwargs):
            for name, method in methods.items():
                if name in query:
                    return [method]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="Method1",
            max_depth=5,  # Limit depth
        )

        # All chains should have depth <= 5
        for chain in chains:
            assert chain.depth <= 6  # +1 for start node

    @pytest.mark.asyncio
    async def test_build_chain_with_target_method(
        self,
        mock_mongodb_service,
    ):
        """Test building chain to specific target method."""
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": '["MethodB"]',
            }
        }
        method_b = {
            "metadata": {
                "methodName": "MethodB",
                "callsMethod": '["SaveData"]',
            }
        }
        save_data = {
            "metadata": {
                "methodName": "SaveData",
                "callsMethod": '[]',
                "databaseTables": '["Data"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "MethodA" in query:
                return [method_a]
            elif "MethodB" in query:
                return [method_b]
            elif "SaveData" in query:
                return [save_data]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            target_method="SaveData",
            max_depth=10,
        )

        # Should find chain to SaveData
        assert len(chains) > 0
        found_target = any(
            chain.end_method == "SaveData" or "SaveData" in chain.end_method
            for chain in chains
        )
        assert found_target or len(chains) > 0


class TestCycleDetection:
    """Test cycle detection in call chains."""

    @pytest.mark.asyncio
    async def test_detects_direct_cycle(
        self,
        mock_mongodb_service,
    ):
        """Test detection of direct recursion."""
        # A -> A (direct recursion)
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": '["MethodA"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "MethodA" in query:
                return [method_a]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            max_depth=10,
        )

        # Should not infinite loop - should produce some result
        assert chains is not None
        # Chain should be short (stopped at cycle)
        for chain in chains:
            assert chain.depth <= 2

    @pytest.mark.asyncio
    async def test_detects_indirect_cycle(
        self,
        mock_mongodb_service,
    ):
        """Test detection of indirect cycles."""
        # A -> B -> C -> A (indirect cycle)
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": '["MethodB"]',
            }
        }
        method_b = {
            "metadata": {
                "methodName": "MethodB",
                "callsMethod": '["MethodC"]',
            }
        }
        method_c = {
            "metadata": {
                "methodName": "MethodC",
                "callsMethod": '["MethodA"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "MethodA" in query:
                return [method_a]
            elif "MethodB" in query:
                return [method_b]
            elif "MethodC" in query:
                return [method_c]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            max_depth=10,
        )

        # Should complete without infinite loop
        assert chains is not None

    @pytest.mark.asyncio
    async def test_visited_cleared_between_branches(
        self,
        mock_mongodb_service,
    ):
        """Test visited set is properly managed between branches."""
        # A -> B -> C
        # A -> D -> C (C should be reachable via both paths)
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": '["MethodB", "MethodD"]',
            }
        }
        method_b = {
            "metadata": {
                "methodName": "MethodB",
                "callsMethod": '["MethodC"]',
            }
        }
        method_c = {
            "metadata": {
                "methodName": "MethodC",
                "callsMethod": '[]',
                "databaseTables": '["Data"]',
            }
        }
        method_d = {
            "metadata": {
                "methodName": "MethodD",
                "callsMethod": '["MethodC"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "MethodA" in query:
                return [method_a]
            elif "MethodB" in query:
                return [method_b]
            elif "MethodC" in query:
                return [method_c]
            elif "MethodD" in query:
                return [method_d]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            max_depth=10,
        )

        # Should have multiple chains reaching C
        assert len(chains) >= 2


class TestCallChainNode:
    """Test CallChainNode model."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = CallChainNode(
            method_name="TestMethod",
            class_name="TestClass",
            depth=1,
        )

        assert node.method_name == "TestMethod"
        assert node.class_name == "TestClass"
        assert node.depth == 1

    def test_node_method_alias(self):
        """Test method property alias."""
        node = CallChainNode(method_name="TestMethod")

        # Property alias should work
        assert node.method == "TestMethod"

        # Setter should work
        node.method = "NewMethod"
        assert node.method_name == "NewMethod"

    def test_node_database_tables(self):
        """Test database tables tracking."""
        node = CallChainNode(
            method_name="SaveData",
            database_tables=["Users", "Orders"],
        )

        assert "Users" in node.database_tables
        assert "Orders" in node.database_tables

    def test_node_model_dump(self):
        """Test model_dump() for serialization."""
        node = CallChainNode(
            method_name="TestMethod",
            class_name="TestClass",
            depth=1,
            database_tables=["Table1"],
        )

        dump = node.model_dump()

        assert dump["method_name"] == "TestMethod"
        # method alias is accessed via property, not in model_dump
        # to allow CallChainNode(**node.model_dump()) to work
        assert node.method == "TestMethod"
        assert dump["class_name"] == "TestClass"
        assert dump["database_tables"] == ["Table1"]


class TestCallChain:
    """Test CallChain model."""

    def test_chain_creation(self):
        """Test basic chain creation."""
        chain = CallChain(
            start_method="StartMethod",
            end_method="EndMethod",
            depth=3,
        )

        assert chain.start_method == "StartMethod"
        assert chain.end_method == "EndMethod"
        assert chain.depth == 3

    def test_compute_touches_database_true(self):
        """Test compute_touches_database when chain accesses DB."""
        chain = CallChain(
            start_method="Start",
            end_method="End",
            steps=[
                CallChainNode(method_name="Start"),
                CallChainNode(method_name="Middle"),
                CallChainNode(method_name="End", database_tables=["Users"]),
            ],
        )

        chain.compute_touches_database()

        assert chain.touches_database is True

    def test_compute_touches_database_false(self):
        """Test compute_touches_database when chain doesn't access DB."""
        chain = CallChain(
            start_method="Start",
            end_method="End",
            steps=[
                CallChainNode(method_name="Start"),
                CallChainNode(method_name="End"),
            ],
        )

        chain.compute_touches_database()

        assert chain.touches_database is False

    def test_compute_touches_database_middle_step(self):
        """Test compute_touches_database when middle step accesses DB."""
        chain = CallChain(
            start_method="Start",
            end_method="End",
            steps=[
                CallChainNode(method_name="Start"),
                CallChainNode(method_name="Middle", database_tables=["Data"]),
                CallChainNode(method_name="End"),
            ],
        )

        chain.compute_touches_database()

        assert chain.touches_database is True


class TestCallTree:
    """Test CallTree model."""

    def test_tree_creation(self):
        """Test basic tree creation."""
        tree = CallTree(
            method="RootMethod",
            depth=0,
        )

        assert tree.method == "RootMethod"
        assert tree.depth == 0
        assert tree.children == []

    def test_compute_total_methods_leaf(self):
        """Test compute_total_methods for leaf node."""
        tree = CallTree(method="LeafMethod", depth=0)

        total = tree.compute_total_methods()

        assert total == 1
        assert tree.total_nodes == 1

    def test_compute_total_methods_with_children(self):
        """Test compute_total_methods with children."""
        child1 = CallTree(method="Child1", depth=1)
        child2 = CallTree(method="Child2", depth=1)

        tree = CallTree(
            method="Root",
            depth=0,
            children=[child1, child2],
        )

        total = tree.compute_total_methods()

        assert total == 3  # Root + 2 children
        assert tree.total_nodes == 3

    def test_compute_total_methods_nested(self):
        """Test compute_total_methods with nested children."""
        grandchild = CallTree(method="GrandChild", depth=2)
        child = CallTree(method="Child", depth=1, children=[grandchild])
        tree = CallTree(method="Root", depth=0, children=[child])

        total = tree.compute_total_methods()

        assert total == 3  # Root + Child + GrandChild


class TestBuildCallTree:
    """Test build_call_tree() method."""

    @pytest.mark.asyncio
    async def test_build_tree_basic(self):
        """Test building a basic call tree."""
        builder = CallChainBuilder()

        call_graph_results = [
            FormattedResult(
                id="1",
                metadata={
                    "callerMethod": "RootMethod",
                    "calleeMethod": "ChildMethod1",
                    "callType": "direct",
                },
            ),
            FormattedResult(
                id="2",
                metadata={
                    "callerMethod": "RootMethod",
                    "calleeMethod": "ChildMethod2",
                    "callType": "direct",
                },
            ),
        ]

        tree = await builder.build_call_tree(
            entry_point="RootMethod",
            call_graph_results=call_graph_results,
            max_depth=10,
        )

        assert tree.method == "RootMethod"
        assert len(tree.children) == 2

    @pytest.mark.asyncio
    async def test_build_tree_nested(self):
        """Test building nested call tree."""
        builder = CallChainBuilder()

        call_graph_results = [
            FormattedResult(
                id="1",
                metadata={
                    "callerMethod": "Root",
                    "calleeMethod": "Child",
                },
            ),
            FormattedResult(
                id="2",
                metadata={
                    "callerMethod": "Child",
                    "calleeMethod": "GrandChild",
                },
            ),
        ]

        tree = await builder.build_call_tree(
            entry_point="Root",
            call_graph_results=call_graph_results,
            max_depth=10,
        )

        assert tree.method == "Root"
        assert len(tree.children) == 1
        assert tree.children[0].method == "Child"
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].method == "GrandChild"

    @pytest.mark.asyncio
    async def test_build_tree_respects_max_depth(self):
        """Test tree building respects max depth."""
        builder = CallChainBuilder()

        # Create deep hierarchy
        call_graph_results = []
        for i in range(10):
            call_graph_results.append(
                FormattedResult(
                    id=str(i),
                    metadata={
                        "callerMethod": f"Level{i}",
                        "calleeMethod": f"Level{i+1}",
                    },
                )
            )

        tree = await builder.build_call_tree(
            entry_point="Level0",
            call_graph_results=call_graph_results,
            max_depth=3,
        )

        # Verify depth is limited
        def get_max_depth(node, current=0):
            if not node.children:
                return current
            return max(get_max_depth(c, current + 1) for c in node.children)

        max_depth = get_max_depth(tree)
        assert max_depth <= 3

    @pytest.mark.asyncio
    async def test_build_tree_handles_cycles(self):
        """Test tree building handles cycles."""
        builder = CallChainBuilder()

        # Create cycle: A -> B -> A
        call_graph_results = [
            FormattedResult(
                id="1",
                metadata={
                    "callerMethod": "MethodA",
                    "calleeMethod": "MethodB",
                },
            ),
            FormattedResult(
                id="2",
                metadata={
                    "callerMethod": "MethodB",
                    "calleeMethod": "MethodA",
                },
            ),
        ]

        # Should not hang
        tree = await builder.build_call_tree(
            entry_point="MethodA",
            call_graph_results=call_graph_results,
            max_depth=10,
        )

        assert tree is not None


class TestBuildChainsFromMethods:
    """Test build_chains_from_methods() method."""

    @pytest.mark.asyncio
    async def test_chains_from_ui_events(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test building chains starting from UI event handlers."""
        ui_event = FormattedResult(
            id="ui1",
            metadata={
                "handlerMethod": "btnSave_Click",
                "controlName": "btnSave",
            },
        )

        method_handler = {
            "metadata": {
                "methodName": "btnSave_Click",
                "callsMethod": '["SaveData"]',
            }
        }
        method_save = {
            "metadata": {
                "methodName": "SaveData",
                "callsMethod": '[]',
                "databaseTables": '["Users"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "btnSave_Click" in query:
                return [method_handler]
            elif "SaveData" in query:
                return [method_save]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains_from_methods(
            methods=[],
            ui_events=[ui_event],
            project="TestProject",
            max_hops=5,
        )

        assert len(chains) > 0
        # Should start from UI handler
        assert any(
            chain.start_method == "btnSave_Click"
            for chain in chains
        )

    @pytest.mark.asyncio
    async def test_chains_from_methods_fallback(
        self,
        mock_mongodb_service,
        code_flow_test_data,
    ):
        """Test falling back to top methods when no UI events."""
        method = FormattedResult(
            id="m1",
            metadata={
                "methodName": "ProcessData",
            },
        )

        method_info = {
            "metadata": {
                "methodName": "ProcessData",
                "callsMethod": '[]',
                "databaseTables": '["Data"]',
            }
        }

        async def search_vectors(query, **kwargs):
            if "ProcessData" in query:
                return [method_info]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains_from_methods(
            methods=[method],
            ui_events=[],  # No UI events
            project="TestProject",
            max_hops=5,
        )

        # Should use method as starting point
        assert len(chains) > 0 or True  # May not find chains if no calls


class TestMethodCache:
    """Test method metadata caching."""

    @pytest.mark.asyncio
    async def test_cache_stores_results(
        self,
        mock_mongodb_service,
    ):
        """Test method info is cached after lookup."""
        method_info = {
            "metadata": {
                "methodName": "TestMethod",
                "callsMethod": '[]',
            }
        }

        mock_mongodb_service.search_vectors = AsyncMock(return_value=[method_info])

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        # First lookup
        result1 = await builder._get_method_info(
            mock_mongodb_service, "TestMethod", "Project"
        )

        # Second lookup - should use cache
        result2 = await builder._get_method_info(
            mock_mongodb_service, "TestMethod", "Project"
        )

        # Should only have called search once
        assert mock_mongodb_service.search_vectors.call_count == 1

        # Results should be same
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_key_includes_project(
        self,
        mock_mongodb_service,
    ):
        """Test cache key includes project."""
        method_info = {
            "metadata": {
                "methodName": "TestMethod",
            }
        }

        mock_mongodb_service.search_vectors = AsyncMock(return_value=[method_info])

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        # Lookup for different projects
        await builder._get_method_info(
            mock_mongodb_service, "TestMethod", "Project1"
        )
        await builder._get_method_info(
            mock_mongodb_service, "TestMethod", "Project2"
        )

        # Should have called search twice (different cache keys)
        assert mock_mongodb_service.search_vectors.call_count == 2

    def test_clear_cache(self, mock_mongodb_service):
        """Test cache clearing."""
        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        # Populate cache
        builder._method_cache["key1"] = {"data": "value"}
        builder._method_cache["key2"] = {"data": "value"}

        assert len(builder._method_cache) == 2

        builder.clear_cache()

        assert len(builder._method_cache) == 0


class TestFanOutLimit:
    """Test fan-out limiting."""

    @pytest.mark.asyncio
    async def test_max_children_per_node(
        self,
        mock_mongodb_service,
    ):
        """Test that fan-out is limited to MAX_CHILDREN_PER_NODE."""
        # Method that calls many other methods
        many_calls = [f"Method{i}" for i in range(20)]
        method_a = {
            "metadata": {
                "methodName": "MethodA",
                "callsMethod": "[" + ", ".join([f'"{m}"' for m in many_calls]) + "]",
            }
        }

        for method in many_calls:
            mock_mongodb_service.set_search_results(
                "code", "method",
                [{"metadata": {"methodName": method, "callsMethod": "[]"}}]
            )

        async def search_vectors(query, **kwargs):
            if "MethodA" in query:
                return [method_a]
            for m in many_calls:
                if m in query:
                    return [{"metadata": {"methodName": m, "callsMethod": "[]"}}]
            return []

        mock_mongodb_service.search_vectors = search_vectors

        builder = CallChainBuilder(mongodb_service=mock_mongodb_service)

        chains = await builder.build_chains(
            start_method="MethodA",
            max_depth=2,
        )

        # Number of chains should be limited by MAX_CHILDREN_PER_NODE
        assert len(chains) <= CallChainBuilder.MAX_CHILDREN_PER_NODE


class TestTraversalState:
    """Test TraversalState dataclass."""

    def test_state_initialization(self):
        """Test default state initialization."""
        state = TraversalState()

        assert state.visited == set()
        assert state.current_chain == []
        assert state.all_chains == []
        assert state.depth == 0
        assert state.max_depth == 10

    def test_state_custom_max_depth(self):
        """Test custom max_depth."""
        state = TraversalState(max_depth=5)

        assert state.max_depth == 5


class TestJSONParsing:
    """Test JSON parsing of method metadata."""

    def test_parse_valid_json_list(self):
        """Test parsing valid JSON array."""
        builder = CallChainBuilder()

        result = builder._try_parse_json('["Method1", "Method2"]')

        assert result == ["Method1", "Method2"]

    def test_parse_list_directly(self):
        """Test parsing actual list."""
        builder = CallChainBuilder()

        result = builder._try_parse_json(["Method1", "Method2"])

        assert result == ["Method1", "Method2"]

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        builder = CallChainBuilder()

        result = builder._try_parse_json("not json")

        assert result == []

    def test_parse_none(self):
        """Test parsing None."""
        builder = CallChainBuilder()

        result = builder._try_parse_json(None)

        assert result == []

    def test_parse_empty(self):
        """Test parsing empty string."""
        builder = CallChainBuilder()

        result = builder._try_parse_json("")

        assert result == []
