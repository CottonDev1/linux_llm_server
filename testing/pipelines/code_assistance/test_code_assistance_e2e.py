"""
Code Assistance End-to-End Tests
=================================

Test complete code assistance pipeline: question → context → generate → validate
"""

import pytest
from datetime import datetime
from fixtures.mongodb_fixtures import create_mock_code_method
from fixtures.llm_fixtures import LocalLLMClient
from utils import (
    assert_document_stored,
    assert_llm_response_valid,
    generate_test_id,
    measure_time,
)


class TestCodeAssistanceE2E:
    """End-to-end tests for code assistance pipeline."""

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    async def test_complete_assistance_workflow(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config
    ):
        """Test complete workflow: question → retrieval → generation → storage."""
        methods_collection = mongodb_database["code_methods"]
        interactions_collection = mongodb_database["code_interactions"]

        # Step 1: Set up code context in database
        method_doc = create_mock_code_method(
            method_name="SaveBale",
            class_name="BaleService",
            project="Gin",
            code="public void SaveBale(Bale bale) { _db.Execute(\"INSERT INTO Bales...\", bale); }",
        )
        method_doc.update({
            "purpose_summary": "Saves a bale record to the database",
            "database_tables": ["Bales"],
            "test_run_id": pipeline_config.test_run_id,
        })
        methods_collection.insert_one(method_doc)

        # Step 2: Simulate user question
        user_question = "How do I save a bale in the system?"

        # Step 3: Retrieve relevant methods (simulated - in real pipeline this would be vector search)
        relevant_methods = list(methods_collection.find({
            "class_name": "BaleService",
            "test_run_id": pipeline_config.test_run_id
        }))

        assert len(relevant_methods) > 0

        # Step 4: Build context from retrieved methods
        context = f"""
Relevant Methods:
- {relevant_methods[0]['method_name']}: {relevant_methods[0]['purpose_summary']}
  Database tables: {', '.join(relevant_methods[0]['database_tables'])}
"""

        # Step 5: Generate answer with LLM
        with measure_time("LLM Generation") as timing:
            prompt = f"""Context:
{context}

Question: {user_question}

Answer:"""

            response = llm_client.generate(
                prompt=prompt,
                model_type="code",
                max_tokens=300,
                temperature=0.2,
            )

        # Verify LLM response
        assert_llm_response_valid(response, min_length=30)
        assert timing["elapsed_ms"] < 30000

        # Step 6: Store interaction
        interaction_id = f"test_{generate_test_id()}"
        response_id = f"resp_{generate_test_id()}"

        interaction = {
            "_id": interaction_id,
            "response_id": response_id,
            "query": user_question,
            "answer": response.text[:5000],
            "sources": [method_doc["method_name"]],
            "project": "Gin",
            "model_used": "qwen2.5-coder",
            "retrieval_time_ms": 50,
            "generation_time_ms": timing["elapsed_ms"],
            "total_time_ms": timing["elapsed_ms"] + 50,
            "feedback_received": False,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
            "pipeline": "code_assistance",
        }

        interactions_collection.insert_one(interaction)

        # Step 7: Verify stored
        stored_interaction = assert_document_stored(
            interactions_collection,
            interaction_id,
            expected_fields=["query", "answer", "sources", "model_used"]
        )

        assert stored_interaction["query"] == user_question
        assert len(stored_interaction["answer"]) > 0
        assert "SaveBale" in stored_interaction["sources"]

    @pytest.mark.requires_mongodb
    @pytest.mark.e2e
    def test_feedback_loop(
        self,
        mongodb_database,
        pipeline_config
    ):
        """Test complete feedback loop."""
        interactions_collection = mongodb_database["code_interactions"]
        feedback_collection = mongodb_database["code_feedback"]

        # Create interaction
        response_id = f"resp_{generate_test_id()}"
        interaction = {
            "_id": f"test_{generate_test_id()}",
            "response_id": response_id,
            "query": "Test question",
            "answer": "Test answer",
            "feedback_received": False,
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        interactions_collection.insert_one(interaction)

        # Submit positive feedback
        feedback = {
            "_id": f"test_{generate_test_id()}",
            "feedback_id": f"fb_{generate_test_id()}",
            "response_id": response_id,
            "is_helpful": True,
            "rating": 5,
            "comment": "Great answer!",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
            "created_at": datetime.utcnow(),
        }
        feedback_collection.insert_one(feedback)

        # Update interaction with feedback
        interactions_collection.update_one(
            {"response_id": response_id},
            {"$set": {"feedback_received": True, "feedback_rating": 5}}
        )

        # Verify feedback loop
        updated_interaction = interactions_collection.find_one({"response_id": response_id})
        stored_feedback = feedback_collection.find_one({"response_id": response_id})

        assert updated_interaction["feedback_received"] is True
        assert updated_interaction["feedback_rating"] == 5
        assert stored_feedback["is_helpful"] is True

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    @pytest.mark.slow
    async def test_multi_turn_conversation(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config
    ):
        """Test multi-turn conversation with context."""
        interactions_collection = mongodb_database["code_interactions"]

        # Turn 1: Initial question
        turn1_prompt = "What is the BaleService class?"
        turn1_response = llm_client.generate(
            prompt=f"Question: {turn1_prompt}\nAnswer:",
            model_type="code",
            max_tokens=200,
            temperature=0.2,
        )

        turn1_interaction = {
            "_id": f"test_{generate_test_id()}",
            "query": turn1_prompt,
            "answer": turn1_response.text if turn1_response.success else "Error",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        interactions_collection.insert_one(turn1_interaction)

        # Turn 2: Follow-up question with context
        turn2_prompt = "How do I use it to save a bale?"
        context = f"Previous question: {turn1_prompt}\nPrevious answer: {turn1_response.text[:200]}"

        turn2_response = llm_client.generate(
            prompt=f"{context}\n\nFollow-up question: {turn2_prompt}\nAnswer:",
            model_type="code",
            max_tokens=200,
            temperature=0.2,
        )

        turn2_interaction = {
            "_id": f"test_{generate_test_id()}",
            "query": turn2_prompt,
            "answer": turn2_response.text if turn2_response.success else "Error",
            "previous_interaction_id": turn1_interaction["_id"],
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        interactions_collection.insert_one(turn2_interaction)

        # Verify conversation flow
        turn1_stored = interactions_collection.find_one({"_id": turn1_interaction["_id"]})
        turn2_stored = interactions_collection.find_one({"_id": turn2_interaction["_id"]})

        assert turn1_stored is not None
        assert turn2_stored is not None
        assert turn2_stored.get("previous_interaction_id") == turn1_interaction["_id"]
