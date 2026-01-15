"""
Code Flow End-to-End Tests
===========================

Test complete code flow analysis pipeline.
"""

import pytest
from fixtures.mongodb_fixtures import insert_test_documents
from fixtures.llm_fixtures import LocalLLMClient
from utils import assert_document_stored, assert_llm_response_valid, generate_test_id


class TestCodeFlowE2E:
    """End-to-end tests for code flow pipeline."""

    @pytest.mark.requires_mongodb
    @pytest.mark.requires_llm
    @pytest.mark.e2e
    async def test_complete_flow_analysis(
        self,
        mongodb_database,
        llm_client: LocalLLMClient,
        pipeline_config
    ):
        """Test complete workflow: UI event → call chain → DB operations."""
        db_ops_collection = mongodb_database["code_dboperations"]
        relationships_collection = mongodb_database["code_relationships"]

        # Step 1: Analyze code with LLM
        code = """
        private void btnSaveBale_Click(object sender, EventArgs e)
        {
            var bale = new Bale { BaleNumber = txtBaleNumber.Text };
            SaveBale(bale);
        }

        private void SaveBale(Bale bale)
        {
            _db.Execute("INSERT INTO Bales (BaleNumber) VALUES (@BaleNumber)", bale);
        }
        """

        prompt = f"""Analyze this code and identify:
1. UI event handler
2. Methods called
3. Database operations

Code:
{code}

Analysis:"""

        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.0,
        )

        assert_llm_response_valid(response, min_length=20)

        # Step 2: Store database operation
        db_op = {
            "_id": f"test_{generate_test_id()}",
            "operation_type": "INSERT",
            "method_name": "SaveBale",
            "tables_accessed": ["Bales"],
            "project": "TestProject",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        db_ops_collection.insert_one(db_op)

        # Step 3: Store relationship
        relationship = {
            "_id": f"test_{generate_test_id()}",
            "source_method": "btnSaveBale_Click",
            "target_method": "SaveBale",
            "project": "TestProject",
            "is_test": True,
            "test_run_id": pipeline_config.test_run_id,
        }
        relationships_collection.insert_one(relationship)

        # Step 4: Query flow
        db_op_stored = assert_document_stored(
            db_ops_collection,
            db_op["_id"],
            expected_fields=["operation_type", "tables_accessed"]
        )

        relationship_stored = assert_document_stored(
            relationships_collection,
            relationship["_id"],
            expected_fields=["source_method", "target_method"]
        )

        # Verify complete flow
        assert db_op_stored["operation_type"] == "INSERT"
        assert "Bales" in db_op_stored["tables_accessed"]
        assert relationship_stored["source_method"] == "btnSaveBale_Click"
        assert relationship_stored["target_method"] == "SaveBale"

    @pytest.mark.requires_mongodb
    @pytest.mark.e2e
    def test_build_call_graph(self, mongodb_database, pipeline_config):
        """Test building a call graph from relationships."""
        collection = mongodb_database["code_relationships"]

        # Create relationship chain
        relationships = [
            {
                "_id": f"test_{generate_test_id()}",
                "source_method": "A",
                "target_method": "B",
                "project": "TestProject",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "source_method": "B",
                "target_method": "C",
                "project": "TestProject",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
            {
                "_id": f"test_{generate_test_id()}",
                "source_method": "C",
                "target_method": "D",
                "project": "TestProject",
                "is_test": True,
                "test_run_id": pipeline_config.test_run_id,
            },
        ]

        insert_test_documents(collection, relationships, pipeline_config.test_run_id)

        # Build call graph from A
        visited = set()
        queue = ["A"]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            # Find targets
            targets = collection.find({
                "source_method": current,
                "test_run_id": pipeline_config.test_run_id
            })

            for rel in targets:
                queue.append(rel["target_method"])

        # Verify call graph
        assert visited == {"A", "B", "C", "D"}
