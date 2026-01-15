"""
Context Window Management Tests for Query/RAG Pipeline.

Tests for context window management including:
- Context truncation when exceeding limits
- Chunk assembly for context
- Surrounding chunk retrieval
- Token counting and budget allocation
- Context prioritization strategies
- Overlap handling

Proper context window management is critical for RAG systems
to maximize relevant information within LLM token limits.
"""

import pytest
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_document_chunk,
    insert_test_documents,
    cleanup_test_documents,
)
from fixtures.llm_fixtures import LocalLLMClient
from utils import assert_llm_response_valid


@dataclass
class ContextChunk:
    """Represents a chunk in the context window."""
    content: str
    token_count: int
    source_id: str
    chunk_index: int
    relevance_score: float
    metadata: Dict[str, Any]


class TokenCounter:
    """
    Simple token counter for testing.

    Uses word-based approximation. In production, use
    tiktoken or model-specific tokenizers.
    """

    def __init__(self, avg_tokens_per_word: float = 1.3):
        """
        Initialize token counter.

        Args:
            avg_tokens_per_word: Average tokens per word (1.3 is typical for English)
        """
        self.avg_tokens_per_word = avg_tokens_per_word

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        words = text.split()
        return int(len(words) * self.avg_tokens_per_word)

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(t) for t in texts]


class ContextWindowManager:
    """
    Manages context window for RAG pipeline.

    Handles:
    - Token budget allocation
    - Chunk selection and truncation
    - Context assembly
    - Overflow handling
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        reserved_output_tokens: int = 512,
        reserved_prompt_tokens: int = 256
    ):
        """
        Initialize context window manager.

        Args:
            max_context_tokens: Maximum tokens for context
            reserved_output_tokens: Tokens reserved for model output
            reserved_prompt_tokens: Tokens reserved for system prompt
        """
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.reserved_prompt_tokens = reserved_prompt_tokens
        self.token_counter = TokenCounter()

        # Available budget for retrieved context
        self.context_budget = (
            max_context_tokens - reserved_output_tokens - reserved_prompt_tokens
        )

    def build_context(
        self,
        chunks: List[Dict[str, Any]],
        content_field: str = "content"
    ) -> Tuple[str, List[ContextChunk]]:
        """
        Build context from chunks within token budget.

        Args:
            chunks: List of chunks sorted by relevance
            content_field: Field containing chunk content

        Returns:
            Tuple of (assembled_context, included_chunks)
        """
        included_chunks = []
        current_tokens = 0
        context_parts = []

        for chunk in chunks:
            content = chunk.get(content_field, "")
            chunk_tokens = self.token_counter.count_tokens(content)

            # Check if chunk fits in budget
            if current_tokens + chunk_tokens > self.context_budget:
                # Try truncating chunk
                remaining_budget = self.context_budget - current_tokens
                if remaining_budget > 50:  # Minimum useful size
                    truncated = self._truncate_to_tokens(content, remaining_budget)
                    if truncated:
                        context_parts.append(truncated)
                        included_chunks.append(ContextChunk(
                            content=truncated,
                            token_count=self.token_counter.count_tokens(truncated),
                            source_id=str(chunk.get("_id", "")),
                            chunk_index=chunk.get("chunk_index", 0),
                            relevance_score=chunk.get("similarity", 0.0),
                            metadata={k: v for k, v in chunk.items()
                                     if k not in [content_field, "_id", "embedding"]}
                        ))
                break
            else:
                context_parts.append(content)
                current_tokens += chunk_tokens
                included_chunks.append(ContextChunk(
                    content=content,
                    token_count=chunk_tokens,
                    source_id=str(chunk.get("_id", "")),
                    chunk_index=chunk.get("chunk_index", 0),
                    relevance_score=chunk.get("similarity", 0.0),
                    metadata={k: v for k, v in chunk.items()
                             if k not in [content_field, "_id", "embedding"]}
                ))

        assembled_context = "\n\n".join(context_parts)
        return assembled_context, included_chunks

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Truncates at sentence boundaries when possible.
        """
        words = text.split()
        target_words = int(max_tokens / self.token_counter.avg_tokens_per_word)

        if len(words) <= target_words:
            return text

        # Truncate at word boundary
        truncated_words = words[:target_words]
        truncated = " ".join(truncated_words)

        # Try to end at sentence boundary
        last_period = truncated.rfind(".")
        if last_period > len(truncated) * 0.5:  # If period is past halfway
            truncated = truncated[:last_period + 1]

        return truncated + "..." if not truncated.endswith(".") else truncated

    def get_remaining_budget(self, current_context: str) -> int:
        """Get remaining token budget after current context."""
        current_tokens = self.token_counter.count_tokens(current_context)
        return max(0, self.context_budget - current_tokens)


class TestContextTruncation:
    """Tests for context truncation when exceeding limits."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_context_truncation_respects_token_limit(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that context is truncated to respect token limits.
        """
        collection = mongodb_database["documents"]

        # Create chunks that exceed context window
        chunks = []
        for i in range(10):
            # Each chunk ~200 words = ~260 tokens
            long_content = " ".join(["word"] * 200)
            doc = create_mock_document_chunk(
                title=f"Long Doc {i}",
                content=long_content,
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["similarity"] = 0.9 - i * 0.05  # Decreasing relevance
            doc["embedding"] = [0.5] * 384
            chunks.append(doc)

        insert_test_documents(collection, chunks, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Build context with limited budget
        manager = ContextWindowManager(
            max_context_tokens=1000,
            reserved_output_tokens=200,
            reserved_prompt_tokens=100
        )  # ~700 tokens for context

        context, included = manager.build_context(results)

        # Verify truncation
        total_tokens = manager.token_counter.count_tokens(context)
        assert total_tokens <= manager.context_budget, (
            f"Context {total_tokens} tokens exceeds budget {manager.context_budget}"
        )

        # Should include fewer than all chunks
        assert len(included) < len(results), (
            "Should truncate to include fewer chunks than available"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_truncation_preserves_most_relevant_chunks(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that truncation keeps the most relevant chunks.
        """
        collection = mongodb_database["documents"]

        # Create chunks with clear relevance ranking
        chunks = [
            {"content": "Most relevant content about safety", "relevance": 0.95},
            {"content": "Second most relevant content", "relevance": 0.85},
            {"content": "Medium relevance content here", "relevance": 0.70},
            {"content": "Low relevance content here", "relevance": 0.50},
            {"content": "Very low relevance content", "relevance": 0.30},
        ]

        stored_chunks = []
        for i, chunk_data in enumerate(chunks):
            doc = create_mock_document_chunk(
                title=f"Relevance Test {i}",
                content=chunk_data["content"],
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["similarity"] = chunk_data["relevance"]
            doc["embedding"] = [0.5] * 384
            stored_chunks.append(doc)

        insert_test_documents(collection, stored_chunks, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Very limited budget
        manager = ContextWindowManager(
            max_context_tokens=200,
            reserved_output_tokens=50,
            reserved_prompt_tokens=50
        )

        context, included = manager.build_context(results)

        # Should include high relevance chunks
        relevance_scores = [c.relevance_score for c in included]
        assert all(score >= 0.5 for score in relevance_scores), (
            "Included chunks should have high relevance scores"
        )

        # First included should be highest relevance
        if included:
            assert included[0].relevance_score >= 0.9, (
                "First chunk should be highest relevance"
            )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestChunkAssembly:
    """Tests for assembling chunks into coherent context."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_chunks_assembled_in_relevance_order(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that chunks are assembled in order of relevance.
        """
        collection = mongodb_database["documents"]

        # Create chunks with identifiable content
        chunks = [
            {"content": "CHUNK_A content", "relevance": 0.5},
            {"content": "CHUNK_B content", "relevance": 0.9},
            {"content": "CHUNK_C content", "relevance": 0.7},
        ]

        stored_chunks = []
        for i, chunk_data in enumerate(chunks):
            doc = create_mock_document_chunk(
                title=f"Assembly Test {i}",
                content=chunk_data["content"],
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["similarity"] = chunk_data["relevance"]
            doc["embedding"] = [0.5] * 384
            stored_chunks.append(doc)

        insert_test_documents(collection, stored_chunks, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        manager = ContextWindowManager(max_context_tokens=2000)
        context, included = manager.build_context(results)

        # CHUNK_B (highest relevance) should appear first
        assert context.index("CHUNK_B") < context.index("CHUNK_C"), (
            "Higher relevance chunk should appear first"
        )
        assert context.index("CHUNK_C") < context.index("CHUNK_A"), (
            "Medium relevance chunk should appear before low"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_chunks_separated_properly(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that chunks are separated with appropriate delimiters.
        """
        collection = mongodb_database["documents"]

        # Create two chunks
        for i in range(2):
            doc = create_mock_document_chunk(
                title=f"Separator Test {i}",
                content=f"Content for chunk {i}.",
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["similarity"] = 0.8
            doc["embedding"] = [0.5] * 384
            collection.insert_one(doc)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        manager = ContextWindowManager()
        context, included = manager.build_context(results)

        # Chunks should be separated by double newline
        assert "\n\n" in context, "Chunks should be separated by blank line"

        # Should not have triple+ newlines
        assert "\n\n\n" not in context, "Should not have excessive whitespace"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSurroundingChunkRetrieval:
    """Tests for retrieving surrounding chunks for better context."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_retrieve_adjacent_chunks(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test retrieving chunks adjacent to matched chunk.

        Adjacent chunks provide better context for understanding.
        """
        collection = mongodb_database["documents"]

        # Create sequential chunks from same document
        parent_doc_id = "parent_123"
        chunk_contents = [
            "Introduction to the process.",
            "Step 1: Prepare the materials.",  # This is the matching chunk
            "Step 2: Execute the process.",
            "Step 3: Verify the results.",
            "Conclusion and summary.",
        ]

        stored_chunks = []
        for i, content in enumerate(chunk_contents):
            doc = create_mock_document_chunk(
                title="Process Guide",
                content=content,
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["parent_document_id"] = parent_doc_id
            doc["total_chunks"] = len(chunk_contents)
            doc["embedding"] = [0.5] * 384
            stored_chunks.append(doc)

        insert_test_documents(collection, stored_chunks, pipeline_config.test_run_id)

        # Simulate matching chunk_index=1
        matching_chunk_index = 1

        # Retrieve adjacent chunks
        adjacent_indices = [
            matching_chunk_index - 1,  # Previous
            matching_chunk_index,       # Matched
            matching_chunk_index + 1,   # Next
        ]

        results = list(
            collection.find({
                "test_run_id": pipeline_config.test_run_id,
                "parent_document_id": parent_doc_id,
                "chunk_index": {"$in": adjacent_indices}
            }).sort("chunk_index", 1)
        )

        # Should have 3 chunks (previous, matched, next)
        assert len(results) == 3, "Should retrieve 3 adjacent chunks"

        # Verify correct chunks retrieved
        chunk_indices = [r["chunk_index"] for r in results]
        assert chunk_indices == [0, 1, 2], "Should have chunks 0, 1, 2"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_handle_edge_chunks(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test handling of first and last chunks (no previous/next).
        """
        collection = mongodb_database["documents"]

        parent_doc_id = "edge_parent"
        chunk_contents = ["First chunk.", "Middle chunk.", "Last chunk."]

        stored_chunks = []
        for i, content in enumerate(chunk_contents):
            doc = create_mock_document_chunk(
                title="Edge Test",
                content=content,
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["parent_document_id"] = parent_doc_id
            doc["total_chunks"] = len(chunk_contents)
            doc["embedding"] = [0.5] * 384
            stored_chunks.append(doc)

        insert_test_documents(collection, stored_chunks, pipeline_config.test_run_id)

        # Test first chunk (no previous)
        matching_index = 0
        adjacent = [max(0, matching_index - 1), matching_index, matching_index + 1]
        adjacent = list(set(adjacent))  # Remove duplicates

        results = list(
            collection.find({
                "test_run_id": pipeline_config.test_run_id,
                "parent_document_id": parent_doc_id,
                "chunk_index": {"$in": adjacent}
            }).sort("chunk_index", 1)
        )

        assert len(results) == 2, "First chunk should have 2 adjacent (self + next)"

        # Test last chunk (no next)
        matching_index = 2
        adjacent = [matching_index - 1, matching_index, min(2, matching_index + 1)]
        adjacent = list(set(adjacent))

        results = list(
            collection.find({
                "test_run_id": pipeline_config.test_run_id,
                "parent_document_id": parent_doc_id,
                "chunk_index": {"$in": adjacent}
            }).sort("chunk_index", 1)
        )

        assert len(results) == 2, "Last chunk should have 2 adjacent (prev + self)"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestTokenCounting:
    """Tests for token counting accuracy."""

    def test_token_counter_basic(self):
        """Test basic token counting."""
        counter = TokenCounter(avg_tokens_per_word=1.3)

        # Simple text
        text = "This is a test sentence."
        tokens = counter.count_tokens(text)
        assert tokens > 0, "Should count some tokens"

        # Longer text should have more tokens
        long_text = " ".join(["word"] * 100)
        long_tokens = counter.count_tokens(long_text)
        assert long_tokens > tokens, "Longer text should have more tokens"

    def test_token_counter_empty_text(self):
        """Test token counting for empty text."""
        counter = TokenCounter()

        assert counter.count_tokens("") == 0
        assert counter.count_tokens("   ") == 0  # Whitespace only

    def test_token_counter_batch(self):
        """Test batch token counting."""
        counter = TokenCounter()

        texts = ["Short.", "Medium length text.", "A longer piece of text here."]
        counts = counter.count_tokens_batch(texts)

        assert len(counts) == 3
        assert counts[0] < counts[1] < counts[2], (
            "Token counts should increase with text length"
        )


class TestContextBudgetAllocation:
    """Tests for context budget allocation strategies."""

    def test_budget_calculation(self):
        """Test budget calculation from total tokens."""
        manager = ContextWindowManager(
            max_context_tokens=4096,
            reserved_output_tokens=512,
            reserved_prompt_tokens=256
        )

        expected_budget = 4096 - 512 - 256  # = 3328
        assert manager.context_budget == expected_budget

    def test_remaining_budget_tracking(self):
        """Test tracking of remaining budget after adding context."""
        manager = ContextWindowManager(
            max_context_tokens=1000,
            reserved_output_tokens=200,
            reserved_prompt_tokens=100
        )

        initial_budget = manager.context_budget  # 700

        # Add some context
        context = " ".join(["word"] * 100)  # ~130 tokens
        remaining = manager.get_remaining_budget(context)

        assert remaining < initial_budget, "Remaining should be less than initial"
        assert remaining > 0, "Should still have some budget"

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_budget_enforced_during_assembly(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that budget is enforced during context assembly.
        """
        collection = mongodb_database["documents"]

        # Create many chunks
        for i in range(20):
            content = " ".join(["content"] * 50)  # ~65 tokens each
            doc = create_mock_document_chunk(
                title=f"Budget Test {i}",
                content=content,
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["similarity"] = 0.8
            doc["embedding"] = [0.5] * 384
            collection.insert_one(doc)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Very limited budget
        manager = ContextWindowManager(
            max_context_tokens=500,
            reserved_output_tokens=100,
            reserved_prompt_tokens=100
        )

        context, included = manager.build_context(results)

        # Verify budget not exceeded
        total_tokens = manager.token_counter.count_tokens(context)
        assert total_tokens <= manager.context_budget, (
            f"Context {total_tokens} exceeds budget {manager.context_budget}"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestOverlapHandling:
    """Tests for handling overlapping content between chunks."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_detect_overlapping_chunks(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test detection of overlapping content between chunks.

        Chunks often have overlap for context continuity.
        """
        collection = mongodb_database["documents"]

        # Create chunks with deliberate overlap
        parent_doc_id = "overlap_parent"
        chunks = [
            {
                "content": "First sentence. Second sentence. Third sentence.",
                "chunk_index": 0,
            },
            {
                "content": "Third sentence. Fourth sentence. Fifth sentence.",
                "chunk_index": 1,
            },
        ]

        stored_chunks = []
        for chunk_data in chunks:
            doc = create_mock_document_chunk(
                title="Overlap Test",
                content=chunk_data["content"],
                chunk_index=chunk_data["chunk_index"],
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["parent_document_id"] = parent_doc_id
            doc["embedding"] = [0.5] * 384
            stored_chunks.append(doc)

        insert_test_documents(collection, stored_chunks, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Detect overlap
        def find_overlap(chunk1: str, chunk2: str) -> str:
            """Find overlapping text between chunks."""
            # Simple approach: find common substrings
            for length in range(min(len(chunk1), len(chunk2)), 0, -1):
                if chunk1[-length:] == chunk2[:length]:
                    return chunk1[-length:]
            return ""

        if len(results) >= 2:
            overlap = find_overlap(results[0]["content"], results[1]["content"])
            assert "Third sentence" in overlap, (
                "Should detect overlapping 'Third sentence'"
            )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    def test_deduplicate_overlap_in_context(self):
        """
        Test deduplication of overlapping content when assembling context.
        """
        # Simulated chunks with overlap
        chunks = [
            {"content": "A B C D E", "chunk_index": 0},
            {"content": "D E F G H", "chunk_index": 1},  # Overlaps with D E
        ]

        def deduplicate_chunks(chunks: List[Dict]) -> str:
            """Remove duplicate content from chunk overlap."""
            if not chunks:
                return ""

            result = chunks[0]["content"]

            for i in range(1, len(chunks)):
                current = chunks[i]["content"]
                prev = chunks[i-1]["content"]

                # Find overlap
                best_overlap = ""
                for length in range(min(len(prev), len(current)), 0, -1):
                    if prev[-length:] == current[:length]:
                        best_overlap = prev[-length:]
                        break

                # Add non-overlapping part
                if best_overlap:
                    result += " " + current[len(best_overlap):].strip()
                else:
                    result += " " + current

            return result

        deduped = deduplicate_chunks(chunks)

        # Should not have duplicate D E
        assert deduped.count("D E") == 1, (
            f"Should have D E only once, got: {deduped}"
        )
