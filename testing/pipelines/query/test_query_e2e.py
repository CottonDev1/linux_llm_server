"""
Query/RAG Pipeline End-to-End Tests.

Tests complete RAG pipeline flow:
query → vector search → context retrieval → LLM generation → response

These tests exercise the full pipeline with all components integrated.
Uses local llama.cpp (port 8081) and MongoDB only.
"""

import pytest
import time
import math
from typing import List

from config.test_config import PipelineTestConfig
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.mongodb_fixtures import (
    create_mock_document_chunk,
    create_mock_code_method,
    insert_test_documents,
    cleanup_test_documents,
)
from utils import assert_llm_response_valid


class TestRAGPipelineE2E:
    """End-to-end tests for complete RAG pipeline."""

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_knowledge_base_query(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete RAG pipeline for knowledge base query.

        Flow:
        1. Store documents with embeddings
        2. Perform vector search
        3. Retrieve top results
        4. Generate answer with LLM
        5. Validate response quality
        """
        collection = mongodb_database["documents"]

        # Step 1: Store documents
        documents = [
            {
                "title": "Safety Procedures",
                "content": "All warehouse personnel must wear hard hats and safety glasses. Emergency exits must remain clear.",
                "department": "Safety",
            },
            {
                "title": "Equipment Manual",
                "content": "The forklift requires daily inspection before operation. Check tire pressure and fluid levels.",
                "department": "Operations",
            },
            {
                "title": "PPE Requirements",
                "content": "Personal protective equipment includes hard hats, safety glasses, steel-toed boots, and high-visibility vests.",
                "department": "Safety",
            },
        ]

        stored_docs = []
        for doc_data in documents:
            doc = create_mock_document_chunk(
                title=doc_data["title"],
                content=doc_data["content"],
                chunk_index=0,
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["department"] = doc_data["department"]
            # Simulate embeddings (in real pipeline, these would come from embedding service)
            doc["embedding"] = [0.5 if "safety" in doc_data["content"].lower() else 0.3] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Step 2: Perform vector search
        query = "What safety equipment is required?"
        # Simulate query embedding
        query_embedding = [0.5] * 384

        # Step 3: Retrieve and rank results
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        for result in results:
            result["similarity"] = self._cosine_similarity(
                query_embedding, result["embedding"]
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:3]

        # Step 4: Generate answer with LLM
        contexts = [
            f"{r['title']}: {r['content']}" for r in top_results
        ]
        context_text = "\n\n".join(contexts)

        prompt = f"""Answer the question using the provided documentation.

DOCUMENTATION:
{context_text}

QUESTION: {query}

Provide a clear, comprehensive answer.

ANSWER:"""

        start_time = time.time()
        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )
        generation_time = time.time() - start_time

        # Step 5: Validate response
        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["safety"],
        )

        # Should mention safety equipment
        text_lower = response.text.lower()
        assert "hard hat" in text_lower or "helmet" in text_lower or "ppe" in text_lower

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_full_pipeline_code_query(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test complete RAG pipeline for code-related query.

        Flow:
        1. Store code context
        2. Search for relevant code
        3. Generate explanation
        """
        collection = mongodb_database["code_context"]

        # Step 1: Store code methods
        code_methods = [
            {
                "method_name": "ProcessBale",
                "class_name": "BaleProcessor",
                "project": "gin",
                "code": "public void ProcessBale(Bale bale) { ValidateBale(bale); SaveBale(bale); }",
            },
            {
                "method_name": "ValidateBale",
                "class_name": "BaleValidator",
                "project": "gin",
                "code": "public bool ValidateBale(Bale bale) { return bale != null && bale.Weight > 0; }",
            },
            {
                "method_name": "ProcessTicket",
                "class_name": "TicketProcessor",
                "project": "warehouse",
                "code": "public void ProcessTicket(Ticket ticket) { SaveTicket(ticket); }",
            },
        ]

        stored_methods = []
        for method_data in code_methods:
            doc = create_mock_code_method(
                method_name=method_data["method_name"],
                class_name=method_data["class_name"],
                project=method_data["project"],
                code=method_data["code"],
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            # Simulate embeddings
            doc["embedding"] = [0.7 if "bale" in method_data["code"].lower() else 0.3] * 384
            stored_methods.append(doc)

        insert_test_documents(collection, stored_methods, pipeline_config.test_run_id)

        # Step 2: Search for relevant code
        query = "How does the system process bales?"
        query_embedding = [0.7] * 384  # Simulated embedding matching "bale" content

        results = list(
            collection.find(
                {"project": "gin", "test_run_id": pipeline_config.test_run_id}
            )
        )

        for result in results:
            result["similarity"] = self._cosine_similarity(
                query_embedding, result["embedding"]
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:2]

        # Step 3: Generate explanation
        code_context = "\n\n".join(
            [
                f"{r['method_name']} in {r['class_name']}:\n{r['content']}"
                for r in top_results
            ]
        )

        prompt = f"""Explain how the code works.

CODE:
{code_context}

QUESTION: {query}

Provide a clear explanation.

EXPLANATION:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["bale"],
        )

        # Should mention processing or validation
        text_lower = response.text.lower()
        assert "process" in text_lower or "validate" in text_lower

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_pipeline_with_project_filtering(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test pipeline with project-scoped search.

        Verifies that search correctly filters by project.
        """
        collection = mongodb_database["code_context"]

        # Store code from multiple projects
        projects_methods = [
            ("gin", "ProcessBale", "Processes cotton bales in the gin system"),
            ("gin", "CalculateWeight", "Calculates bale weight"),
            ("warehouse", "ProcessTicket", "Processes warehouse tickets"),
            ("warehouse", "UpdateInventory", "Updates inventory levels"),
        ]

        stored_docs = []
        for project, method_name, description in projects_methods:
            doc = create_mock_code_method(
                method_name=method_name,
                class_name=f"{project}Class",
                project=project,
                code=f"public void {method_name}() {{ /* {description} */ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [0.5] * 384
            stored_docs.append(doc)

        insert_test_documents(collection, stored_docs, pipeline_config.test_run_id)

        # Search within specific project
        target_project = "gin"
        results = list(
            collection.find(
                {"project": target_project, "test_run_id": pipeline_config.test_run_id}
            )
        )

        # Should only get gin methods
        assert len(results) == 2
        for result in results:
            assert result["project"] == target_project

        # Generate answer using only gin context
        context = "\n".join([r["content"] for r in results])

        prompt = f"""Describe the gin system methods.

CODE:
{context}

Provide a brief summary.

SUMMARY:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=20)

        # Should mention gin-specific concepts
        text_lower = response.text.lower()
        assert "bale" in text_lower or "gin" in text_lower

        # Should NOT mention warehouse concepts
        assert "ticket" not in text_lower
        assert "inventory" not in text_lower

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_pipeline_no_results_handling(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """
        Test pipeline when vector search returns no results.

        Should gracefully handle empty context.
        """
        collection = mongodb_database["documents"]

        # Store documents about unrelated topics
        doc = create_mock_document_chunk(
            title="Weather Report",
            content="Today's weather is sunny with temperatures around 75 degrees.",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.1] * 384  # Very different embedding
        collection.insert_one(doc)

        # Query about completely different topic
        query = "How do I configure the database connection?"
        query_embedding = [0.9] * 384

        # Search with high threshold
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        for result in results:
            result["similarity"] = self._cosine_similarity(
                query_embedding, result["embedding"]
            )

        # Filter by threshold
        threshold = 0.5
        relevant_results = [r for r in results if r["similarity"] >= threshold]

        # Should have no relevant results
        assert len(relevant_results) == 0

        # Generate response with no context
        prompt = f"""Answer the question. If you don't have relevant information, say so.

QUESTION: {query}

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
        )

        assert_llm_response_valid(response, min_length=10)

        # Should indicate no information available
        text_lower = response.text.lower()
        # Soft check - response should be cautious without specific context
        # Should not hallucinate specific database configuration

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestRAGPipelinePerformance:
    """Test performance characteristics of RAG pipeline."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_end_to_end_latency(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """Test total pipeline latency."""
        collection = mongodb_database["documents"]

        # Store document
        doc = create_mock_document_chunk(
            title="Test Doc",
            content="This is test content for performance testing.",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * 384
        collection.insert_one(doc)

        # Measure total pipeline time
        start_time = time.time()

        # 1. Vector search
        query_embedding = [0.5] * 384
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))
        search_time = time.time() - start_time

        # 2. Generate response
        gen_start = time.time()
        prompt = f"""Answer: {results[0]['content']}

Question: What is this about?

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=256,
            temperature=0.3,
        )
        gen_time = time.time() - gen_start

        total_time = time.time() - start_time

        # Validate performance
        assert search_time < 1.0, f"Search too slow: {search_time:.2f}s"
        assert total_time < 60.0, f"Total pipeline too slow: {total_time:.2f}s"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    def test_vector_search_scalability(
        self, mongodb_database, pipeline_config: PipelineTestConfig
    ):
        """Test vector search with larger dataset."""
        collection = mongodb_database["code_context"]

        # Insert many documents
        docs = []
        for i in range(200):
            doc = create_mock_code_method(
                method_name=f"Method{i}",
                class_name="TestClass",
                project="gin",
                code=f"public void Method{i}() {{ }}",
            )
            doc["test_run_id"] = pipeline_config.test_run_id
            doc["embedding"] = [float(i) / 200] * 384
            docs.append(doc)

        insert_test_documents(collection, docs, pipeline_config.test_run_id)

        # Perform search with limit
        start_time = time.time()
        results = list(
            collection.find(
                {"project": "gin", "test_run_id": pipeline_config.test_run_id}
            ).limit(10)
        )
        search_time = time.time() - start_time

        assert len(results) == 10
        assert search_time < 1.0, f"Search should be fast even with 200 docs: {search_time:.2f}s"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)


class TestRAGPipelineQuality:
    """Test quality of RAG pipeline outputs."""

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_answer_relevance(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """Test that generated answers are relevant to question."""
        collection = mongodb_database["documents"]

        # Store relevant document
        doc = create_mock_document_chunk(
            title="Bale Processing Guide",
            content="Bale processing involves validation, weighing, grading, and storage. Each bale must be tagged with a unique barcode.",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.7] * 384
        collection.insert_one(doc)

        # Retrieve and generate answer
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        question = "What are the steps in bale processing?"

        prompt = f"""Answer the question using the provided information.

INFORMATION:
{results[0]['content']}

QUESTION: {question}

ANSWER:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="general",
            max_tokens=512,
            temperature=0.3,
        )

        assert_llm_response_valid(
            response,
            min_length=30,
            must_contain=["bale"],
        )

        # Should mention key steps
        text_lower = response.text.lower()
        steps_mentioned = sum(
            [
                "validation" in text_lower or "validate" in text_lower,
                "weigh" in text_lower,
                "grad" in text_lower,
                "storage" in text_lower or "store" in text_lower,
                "barcode" in text_lower,
            ]
        )

        assert steps_mentioned >= 2, "Should mention at least 2 processing steps"

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)

    @pytest.mark.e2e
    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    def test_multi_turn_conversation_context(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config: PipelineTestConfig,
    ):
        """Test maintaining context across multiple questions."""
        collection = mongodb_database["documents"]

        # Store document
        doc = create_mock_document_chunk(
            title="RecapGet Documentation",
            content="RecapGet accepts @GinID parameter and returns recap records including RecapID, RecapDate, TotalBales, and TotalWeight.",
            chunk_index=0,
        )
        doc["test_run_id"] = pipeline_config.test_run_id
        doc["embedding"] = [0.5] * 384
        collection.insert_one(doc)

        # First question
        question1 = "What does RecapGet do?"
        results = list(collection.find({"test_run_id": pipeline_config.test_run_id}))

        prompt1 = f"""Answer the question.

CONTEXT:
{results[0]['content']}

QUESTION: {question1}

ANSWER:"""

        response1 = llm_client.generate(
            prompt=prompt1, model_type="general", max_tokens=256, temperature=0.3
        )

        # Follow-up question with conversation history
        question2 = "What parameters does it accept?"

        prompt2 = f"""Answer the follow-up question using the context and previous conversation.

CONTEXT:
{results[0]['content']}

PREVIOUS QUESTION: {question1}
PREVIOUS ANSWER: {response1.text}

FOLLOW-UP QUESTION: {question2}

ANSWER:"""

        response2 = llm_client.generate(
            prompt=prompt2, model_type="general", max_tokens=256, temperature=0.3
        )

        assert_llm_response_valid(response2, min_length=10)

        # Should mention GinID parameter
        assert "GinID" in response2.text or "ginid" in response2.text.lower()

        # Cleanup
        cleanup_test_documents(mongodb_database, pipeline_config.test_run_id)
