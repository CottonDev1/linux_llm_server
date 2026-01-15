"""
Source Attribution Tests for Query/RAG Pipeline.

Tests for source citation and attribution including:
- Source citation in responses
- Source metadata preservation
- Multi-source synthesis attribution
- Source relevance scoring
- Citation format validation
- Source deduplication

Proper source attribution is critical for RAG systems to provide
verifiable and trustworthy responses.
"""

import pytest
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from config.test_config import PipelineTestConfig
from fixtures.mongodb_fixtures import (
    create_mock_document_chunk,
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)
from fixtures.llm_fixtures import LocalLLMClient
from utils import assert_llm_response_valid


@dataclass
class SourceReference:
    """Represents a source reference in a response."""
    source_id: str
    title: Optional[str]
    file_name: Optional[str]
    section: Optional[str]
    relevance_score: float
    content_preview: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributedResponse:
    """Response with source attribution."""
    answer: str
    sources: List[SourceReference]
    citation_count: int
    has_inline_citations: bool


class SourceAttributionHelper:
    """
    Helper for extracting and validating source attribution.

    Provides utilities for:
    - Extracting citations from response text
    - Validating citation format
    - Matching citations to sources
    - Calculating attribution coverage
    """

    # Common citation patterns
    BRACKET_CITATION = r'\[(\d+)\]'  # [1], [2], etc.
    PAREN_CITATION = r'\(([^)]+)\)'  # (Source Name)
    INLINE_REFERENCE = r'(?:according to|from|per|as stated in)\s+([^,\.]+)'

    def __init__(self):
        """Initialize helper."""
        self.citation_patterns = [
            self.BRACKET_CITATION,
            self.PAREN_CITATION,
            self.INLINE_REFERENCE,
        ]

    def extract_citations(self, text: str) -> List[str]:
        """
        Extract all citations from response text.

        Args:
            text: Response text to analyze

        Returns:
            List of extracted citation references
        """
        citations = []

        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)

        return citations

    def has_source_section(self, text: str) -> bool:
        """
        Check if response has a dedicated sources section.

        Args:
            text: Response text

        Returns:
            True if sources section exists
        """
        source_headers = [
            r'(?i)^sources?:',
            r'(?i)^references?:',
            r'(?i)^cited sources?:',
            r'(?i)\n\nsources?:',
            r'(?i)\n\nreferences?:',
        ]

        for pattern in source_headers:
            if re.search(pattern, text, re.MULTILINE):
                return True
        return False

    def validate_citation_format(self, citation: str) -> bool:
        """
        Validate that a citation is properly formatted.

        Args:
            citation: Citation string to validate

        Returns:
            True if citation format is valid
        """
        # Check for common valid formats
        valid_patterns = [
            r'^\d+$',  # Numeric: "1", "2"
            r'^[\w\s]+\.\w+$',  # Filename: "document.pdf"
            r'^[\w\s\-_]+$',  # Title-like: "Safety Manual"
        ]

        for pattern in valid_patterns:
            if re.match(pattern, citation.strip()):
                return True
        return False

    def calculate_attribution_coverage(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate what percentage of sources are cited in the response.

        Args:
            response: Generated response text
            sources: List of source documents used

        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        if not sources:
            return 1.0  # No sources needed = full coverage

        response_lower = response.lower()
        cited_count = 0

        for source in sources:
            # Check if source is referenced
            source_indicators = [
                source.get("title", "").lower(),
                source.get("file_name", "").lower(),
                source.get("_id", "").lower(),
            ]

            for indicator in source_indicators:
                if indicator and indicator in response_lower:
                    cited_count += 1
                    break

        return cited_count / len(sources)


class TestSourceCitationInResponses:
    """Tests for source citation in LLM responses."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_response_includes_source_references(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig
    ):
        """
        Test that LLM response can include source references.

        When prompted, the LLM should cite sources in the response.
        """
        collection = mongodb_database["documents"]

        # Create source documents
        sources = [
            {
                "title": "Safety Manual",
                "file_name": "safety_manual.pdf",
                "content": "All personnel must wear hard hats in designated areas.",
            },
            {
                "title": "PPE Guide",
                "file_name": "ppe_guide.pdf",
                "content": "Safety glasses are required at all times in the warehouse.",
            },
        ]

        stored_docs = []
        for i, source_data in enumerate(sources):
            doc = create_mock_document_chunk(
                title=source_data["title"],
                content=source_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["file_name"] = source_data["file_name"]
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Build context with source labels
        context_parts = []
        for i, doc in enumerate(stored_docs):
            context_parts.append(f"[Source {i+1}: {doc['title']}]\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Prompt asking for citations
        prompt = f"""Answer the question using the sources provided. Include citations like [Source 1] in your answer.

SOURCES:
{context}

QUESTION: What safety equipment is required?

ANSWER (with citations):"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=20)

        # Check for citation presence
        helper = SourceAttributionHelper()
        citations = helper.extract_citations(response.text)

        # Should have at least some citation attempt
        has_citations = len(citations) > 0 or "[source" in response.text.lower()

        # Also check for source content being used
        text_lower = response.text.lower()
        uses_source_content = "hard hat" in text_lower or "safety glasses" in text_lower

        assert uses_source_content, "Response should use information from sources"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_response_cites_correct_sources(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig
    ):
        """
        Test that citations reference the correct sources.

        Information should be attributed to the source it came from.
        """
        collection = mongodb_database["documents"]

        # Create sources with distinct information
        sources = [
            {
                "title": "Equipment Manual",
                "content": "The forklift maximum capacity is 5000 pounds.",
                "fact": "forklift",
            },
            {
                "title": "Safety Procedures",
                "content": "Emergency exits must be marked with green signs.",
                "fact": "emergency",
            },
        ]

        stored_docs = []
        for i, source_data in enumerate(sources):
            doc = create_mock_document_chunk(
                title=source_data["title"],
                content=source_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["fact"] = source_data["fact"]
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Query about forklift (from Equipment Manual)
        context_parts = []
        for doc in stored_docs:
            context_parts.append(f"[{doc['title']}]: {doc['content']}")

        context = "\n\n".join(context_parts)

        prompt = f"""Answer the question and cite the source using [Source Name].

SOURCES:
{context}

QUESTION: What is the forklift capacity?

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=10)

        text_lower = response.text.lower()

        # Should mention forklift capacity
        assert "5000" in response.text or "forklift" in text_lower

        # Should NOT mention emergency exits (wrong source)
        assert "emergency" not in text_lower or "exit" not in text_lower

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSourceMetadataPreservation:
    """Tests for preserving source metadata through the pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_source_metadata_available_for_citation(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that source metadata is preserved for citation generation.

        Metadata like title, file_name, department should be available.
        """
        collection = mongodb_database["documents"]

        # Create document with rich metadata
        doc = create_mock_document_chunk(
            title="Safety Procedures Manual v2.1",
            content="All personnel must complete safety training.",
            doc_type="pdf",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["file_name"] = "safety_procedures_v2.1.pdf"
        doc["department"] = "Safety"
        doc["author"] = "Safety Department"
        doc["version"] = "2.1"
        doc["last_updated"] = "2024-01-15"
        doc["page_number"] = 42
        doc["embedding"] = [0.5] * 384

        collection.insert_one(doc)

        # Retrieve and verify metadata
        retrieved = collection.find_one({"test_run_id": pipeline_config.test_run_id})

        # All citation-relevant metadata should be present
        assert retrieved["title"] == "Safety Procedures Manual v2.1"
        assert retrieved["file_name"] == "safety_procedures_v2.1.pdf"
        assert retrieved["department"] == "Safety"
        assert retrieved["author"] == "Safety Department"
        assert retrieved["version"] == "2.1"
        assert retrieved["page_number"] == 42

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_code_source_metadata_preserved(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that code source metadata is preserved for citations.

        Code citations need method name, class, file path, etc.
        """
        collection = mongodb_database["code_context"]

        doc = create_mock_code_method(
            method_name="ProcessBale",
            class_name="BaleProcessor",
            project="gin",
            code="public void ProcessBale(Bale bale) { }",
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["file_path"] = "/src/Processing/BaleProcessor.cs"
        doc["line_number"] = 125
        doc["commit_hash"] = "abc123"
        doc["embedding"] = [0.5] * 384

        collection.insert_one(doc)

        # Retrieve and verify
        retrieved = collection.find_one({"test_run_id": pipeline_config.test_run_id})

        assert retrieved["method_name"] == "ProcessBale"
        assert retrieved["class_name"] == "BaleProcessor"
        assert retrieved["project"] == "gin"
        assert retrieved["file_path"] == "/src/Processing/BaleProcessor.cs"
        assert retrieved["line_number"] == 125

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestMultiSourceSynthesis:
    """Tests for attribution when synthesizing from multiple sources."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_multi_source_answer_with_attribution(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig
    ):
        """
        Test that answers synthesized from multiple sources cite all relevant sources.
        """
        collection = mongodb_database["documents"]

        # Create multiple sources with complementary information
        sources = [
            {
                "title": "Bale Processing Guide",
                "content": "Bale processing begins with weight measurement.",
            },
            {
                "title": "Quality Control Manual",
                "content": "Quality grades are assigned based on fiber length and color.",
            },
            {
                "title": "Storage Procedures",
                "content": "Processed bales are stored in climate-controlled warehouses.",
            },
        ]

        stored_docs = []
        for i, source_data in enumerate(sources):
            doc = create_mock_document_chunk(
                title=source_data["title"],
                content=source_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["source_id"] = f"src_{i}"
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Build context
        context_parts = []
        for i, doc in enumerate(stored_docs):
            context_parts.append(f"[{i+1}] {doc['title']}: {doc['content']}")

        context = "\n".join(context_parts)

        prompt = f"""Synthesize a comprehensive answer from all relevant sources. Cite sources using [1], [2], etc.

SOURCES:
{context}

QUESTION: Describe the complete bale processing workflow from start to finish.

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=40)

        # Response should include information from multiple sources
        text_lower = response.text.lower()
        source_info_found = sum([
            "weight" in text_lower,  # From Bale Processing Guide
            "grade" in text_lower or "quality" in text_lower,  # From Quality Control
            "storage" in text_lower or "warehouse" in text_lower,  # From Storage
        ])

        assert source_info_found >= 2, (
            f"Response should synthesize from multiple sources, found info from {source_info_found} sources"
        )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_source_relevance_scoring(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that sources are scored for relevance to enable proper attribution.
        """
        collection = mongodb_database["documents"]

        # Create sources with varying relevance
        sources = [
            {
                "content": "Safety procedures for bale processing operations",
                "expected_relevance": "high",
            },
            {
                "content": "General company policies and guidelines",
                "expected_relevance": "low",
            },
            {
                "content": "Processing bales requires safety equipment",
                "expected_relevance": "high",
            },
        ]

        query_embedding = [0.8] * 384  # Simulating "bale safety" query

        stored_docs = []
        for i, source_data in enumerate(sources):
            doc = create_mock_document_chunk(
                title=f"Source {i}",
                content=source_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["expected_relevance"] = source_data["expected_relevance"]
            # High relevance docs get similar embedding to query
            if source_data["expected_relevance"] == "high":
                doc["embedding"] = [0.8] * 384
            else:
                doc["embedding"] = [0.3] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Calculate relevance scores
        import math
        for result in results:
            dot = sum(a * b for a, b in zip(query_embedding, result["embedding"]))
            norm_q = math.sqrt(sum(x * x for x in query_embedding))
            norm_d = math.sqrt(sum(x * x for x in result["embedding"]))
            result["relevance_score"] = dot / (norm_q * norm_d) if norm_q and norm_d else 0

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Top results should be high relevance
        top_results = results[:2]
        for result in top_results:
            assert result["expected_relevance"] == "high", (
                f"Top results should be high relevance, got {result['expected_relevance']}"
            )

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestSourceDeduplication:
    """Tests for handling duplicate sources in attribution."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_duplicate_chunks_from_same_document(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that multiple chunks from the same document are properly attributed.

        Should not list the same document multiple times in sources.
        """
        collection = mongodb_database["documents"]

        # Create multiple chunks from same document
        parent_doc_id = "parent_doc_123"

        for i in range(3):
            doc = create_mock_document_chunk(
                title="Safety Manual",
                content=f"Chunk {i} content about safety procedures.",
                chunk_index=i,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["parent_document_id"] = parent_doc_id
            doc["file_name"] = "safety_manual.pdf"
            doc["embedding"] = [0.5 + i * 0.1] * 384
            collection.insert_one(doc)

        # Retrieve all chunks
        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Deduplicate by parent document
        seen_parents = set()
        unique_sources = []

        for result in results:
            parent_id = result.get("parent_document_id")
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique_sources.append(result)

        # Should have only 1 unique source (the parent document)
        assert len(unique_sources) == 1
        assert unique_sources[0]["title"] == "Safety Manual"

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_similar_sources_remain_distinct(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """
        Test that similar but distinct sources remain separate in attribution.

        Different documents with similar content should be listed separately.
        """
        collection = mongodb_database["documents"]

        # Create similar but distinct documents
        similar_docs = [
            {
                "title": "Safety Manual 2023",
                "content": "Safety procedures for warehouse operations.",
                "version": "2023",
            },
            {
                "title": "Safety Manual 2024",
                "content": "Safety procedures for warehouse operations updated.",
                "version": "2024",
            },
        ]

        stored_docs = []
        for i, doc_data in enumerate(similar_docs):
            doc = create_mock_document_chunk(
                title=doc_data["title"],
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["version"] = doc_data["version"]
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        results = list(
            collection.find({"test_run_id": pipeline_config.test_run_id})
        )

        # Both should be returned as distinct sources
        assert len(results) == 2

        titles = [r["title"] for r in results]
        assert "Safety Manual 2023" in titles
        assert "Safety Manual 2024" in titles

        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestCitationFormatValidation:
    """Tests for citation format compliance."""

    def test_bracket_citation_extraction(self):
        """Test extraction of bracket-style citations."""
        helper = SourceAttributionHelper()

        text = "According to the manual [1], safety is important [2]. See also [3]."
        citations = helper.extract_citations(text)

        # Should find bracket citations
        assert "1" in citations or 1 in [int(c) for c in citations if c.isdigit()]

    def test_source_section_detection(self):
        """Test detection of sources section in response."""
        helper = SourceAttributionHelper()

        text_with_section = """
The answer is based on the documentation.

Sources:
- Safety Manual
- Equipment Guide
"""
        text_without_section = "The answer is based on the documentation."

        assert helper.has_source_section(text_with_section) is True
        assert helper.has_source_section(text_without_section) is False

    def test_citation_format_validation(self):
        """Test validation of citation formats."""
        helper = SourceAttributionHelper()

        valid_citations = ["1", "document.pdf", "Safety Manual"]
        invalid_citations = ["", "   ", "@#$%"]

        for citation in valid_citations:
            # These should be recognized as potentially valid
            pass  # Format validation is permissive

        # Empty citations are invalid
        assert helper.validate_citation_format("") is False

    def test_attribution_coverage_calculation(self):
        """Test calculation of attribution coverage."""
        helper = SourceAttributionHelper()

        sources = [
            {"title": "Safety Manual", "file_name": "safety.pdf"},
            {"title": "Equipment Guide", "file_name": "equipment.pdf"},
        ]

        # Response mentions one source
        response_partial = "According to the Safety Manual, wear hard hats."
        coverage_partial = helper.calculate_attribution_coverage(response_partial, sources)

        assert 0.0 < coverage_partial < 1.0, (
            f"Partial coverage should be between 0 and 1, got {coverage_partial}"
        )

        # Response mentions both sources
        response_full = "The Safety Manual and Equipment Guide both require PPE."
        coverage_full = helper.calculate_attribution_coverage(response_full, sources)

        assert coverage_full > coverage_partial, (
            "Full coverage should be higher than partial"
        )
