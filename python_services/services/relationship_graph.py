"""
Relationship Graph - FK relationship graph for automatic JOIN generation.

This module provides utilities to:
- Build a directed graph from foreign key relationships
- Find shortest JOIN paths between tables
- Get related tables within N hops

Usage:
    from services.relationship_graph import RelationshipGraph

    # Build graph from schema context
    graph = RelationshipGraph(schema_list)

    # Find JOIN path between two tables
    join_clauses = graph.find_join_path("CentralTickets", "CentralUsers")

    # Get all related tables within 2 hops
    related = graph.get_related_tables("CentralTickets", depth=2)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RelationshipGraph:
    """
    Build and query FK relationship graph for automatic JOIN generation.

    Uses a simple adjacency list representation instead of networkx
    to avoid external dependencies.
    """

    def __init__(self, schema_context: List[Dict]):
        """
        Initialize the relationship graph from schema context.

        Args:
            schema_context: List of schema dictionaries from MongoDB,
                           each containing 'table_name' and 'foreign_keys'
        """
        # Adjacency list: table_name -> list of (target_table, fk_info)
        self._outgoing: Dict[str, List[Tuple[str, Dict]]] = {}
        # Reverse adjacency: target_table -> list of (source_table, fk_info)
        self._incoming: Dict[str, List[Tuple[str, Dict]]] = {}
        # All known tables
        self._tables: Set[str] = set()

        self._build_graph(schema_context)

    def _build_graph(self, schema_context: List[Dict]) -> None:
        """Build directed graph from FK relationships."""
        for schema in schema_context:
            table_name = schema.get("table_name", "")
            if not table_name:
                continue

            self._tables.add(table_name)

            if table_name not in self._outgoing:
                self._outgoing[table_name] = []
            if table_name not in self._incoming:
                self._incoming[table_name] = []

            for fk in schema.get("foreign_keys", []):
                ref_table = fk.get("references_table", fk.get("to_table"))
                if not ref_table:
                    continue

                self._tables.add(ref_table)

                if ref_table not in self._outgoing:
                    self._outgoing[ref_table] = []
                if ref_table not in self._incoming:
                    self._incoming[ref_table] = []

                fk_info = {
                    "column": fk.get("column", fk.get("from_column", "")),
                    "ref_column": fk.get("references_column", fk.get("to_column", ""))
                }

                self._outgoing[table_name].append((ref_table, fk_info))
                self._incoming[ref_table].append((table_name, fk_info))

        logger.debug(f"Built relationship graph with {len(self._tables)} tables")

    def find_join_path(self, table1: str, table2: str) -> Optional[List[str]]:
        """
        Find shortest path between two tables and return JOIN clauses.

        Args:
            table1: Source table name
            table2: Target table name

        Returns:
            List of JOIN clause strings, or None if no path exists
        """
        if table1 not in self._tables or table2 not in self._tables:
            return None

        if table1 == table2:
            return []

        # BFS to find shortest path (treating graph as undirected)
        path = self._bfs_path(table1, table2)
        if not path:
            return None

        return self._format_join_path(path)

    def _bfs_path(self, start: str, end: str) -> Optional[List[str]]:
        """BFS to find shortest path in undirected graph."""
        from collections import deque

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            # Get all neighbors (both directions)
            neighbors = []
            for target, _ in self._outgoing.get(current, []):
                neighbors.append(target)
            for source, _ in self._incoming.get(current, []):
                neighbors.append(source)

            for neighbor in neighbors:
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def _format_join_path(self, path: List[str]) -> List[str]:
        """Convert path to JOIN clauses."""
        joins = []

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]

            # Check for forward edge (src has FK to dst)
            for target, fk_info in self._outgoing.get(src, []):
                if target == dst:
                    joins.append(
                        f"JOIN {dst} ON {src}.{fk_info['column']} = {dst}.{fk_info['ref_column']}"
                    )
                    break
            else:
                # Check reverse edge (dst has FK to src)
                for source, fk_info in self._incoming.get(src, []):
                    if source == dst:
                        joins.append(
                            f"JOIN {dst} ON {dst}.{fk_info['column']} = {src}.{fk_info['ref_column']}"
                        )
                        break

        return joins

    def get_related_tables(self, table: str, depth: int = 2) -> List[str]:
        """
        Get all tables within N hops of the given table.

        Args:
            table: Starting table name
            depth: Maximum number of hops

        Returns:
            List of related table names (excluding the starting table)
        """
        if table not in self._tables:
            return []

        related: Set[str] = set()
        current_level = {table}

        for _ in range(depth):
            next_level: Set[str] = set()

            for t in current_level:
                # Add outgoing neighbors
                for target, _ in self._outgoing.get(t, []):
                    if target != table:
                        next_level.add(target)

                # Add incoming neighbors
                for source, _ in self._incoming.get(t, []):
                    if source != table:
                        next_level.add(source)

            related.update(next_level)
            current_level = next_level - related  # Only explore new tables

        return list(related)

    def get_join_hint(self, table1: str, table2: str) -> Optional[str]:
        """
        Get a single JOIN hint between two directly connected tables.

        Args:
            table1: First table name
            table2: Second table name

        Returns:
            JOIN clause string or None if tables are not directly connected
        """
        # Check direct FK from table1 to table2
        for target, fk_info in self._outgoing.get(table1, []):
            if target.lower() == table2.lower():
                return f"JOIN {table2} ON {table1}.{fk_info['column']} = {table2}.{fk_info['ref_column']}"

        # Check reverse FK from table2 to table1
        for target, fk_info in self._outgoing.get(table2, []):
            if target.lower() == table1.lower():
                return f"JOIN {table1} ON {table2}.{fk_info['column']} = {table1}.{fk_info['ref_column']}"

        return None

    def get_all_relationships(self) -> List[Dict]:
        """
        Get all FK relationships in the graph.

        Returns:
            List of relationship dictionaries with source, target, column info
        """
        relationships = []

        for source, edges in self._outgoing.items():
            for target, fk_info in edges:
                relationships.append({
                    "source_table": source,
                    "target_table": target,
                    "source_column": fk_info["column"],
                    "target_column": fk_info["ref_column"]
                })

        return relationships

    @property
    def table_count(self) -> int:
        """Get the number of tables in the graph."""
        return len(self._tables)

    @property
    def edge_count(self) -> int:
        """Get the number of FK relationships (edges) in the graph."""
        return sum(len(edges) for edges in self._outgoing.values())


def build_relationship_context(schema_context: List[Dict]) -> str:
    """
    Build LLM-friendly relationship context from schema list.

    Args:
        schema_context: List of schema dictionaries from MongoDB

    Returns:
        Formatted string describing table relationships
    """
    graph = RelationshipGraph(schema_context)
    relationships = graph.get_all_relationships()

    if not relationships:
        return ""

    lines = ["TABLE RELATIONSHIPS:"]

    for rel in relationships:
        lines.append(
            f"- {rel['source_table']}.{rel['source_column']} -> "
            f"{rel['target_table']}.{rel['target_column']}"
        )

    # Add common join patterns
    lines.append("")
    lines.append("COMMON JOIN PATTERNS:")

    # Find tables with multiple relationships
    table_edges = {}
    for rel in relationships:
        src = rel['source_table']
        if src not in table_edges:
            table_edges[src] = []
        table_edges[src].append(rel)

    for table, edges in sorted(table_edges.items(), key=lambda x: -len(x[1])):
        if len(edges) >= 2:
            targets = [e['target_table'] for e in edges[:3]]
            lines.append(f"- {table} joins to: {', '.join(targets)}")

    return "\n".join(lines)
