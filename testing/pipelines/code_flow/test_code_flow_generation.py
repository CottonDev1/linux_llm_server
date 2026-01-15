"""
Code Flow Generation Tests
===========================

Test code flow analysis using local llama.cpp (port 8082).
Includes token usage tracking for all LLM operations.
"""

import pytest
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.shared_fixtures import TokenAssertions
from utils import assert_llm_response_valid


class TestCodeFlowGeneration:
    """Test code flow generation using local LLM."""

    @pytest.mark.requires_llm
    def test_analyze_execution_path(self, llm_client: LocalLLMClient, pipeline_config,
                                     token_assertions: TokenAssertions):
        """Test analyzing execution path through code."""
        code = """
        private void btnSaveBale_Click(object sender, EventArgs e)
        {
            if (ValidateBale())
            {
                var bale = CreateBaleFromForm();
                SaveBale(bale);
                RefreshGrid();
            }
        }
        """

        prompt = f"""Describe the execution flow of this event handler:

{code}

Execution flow:"""

        max_tokens = 300
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        assert_llm_response_valid(response, min_length=20)
        assert any(word in response.text.lower() for word in ["validate", "save", "refresh"])

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=30, max_tokens=500)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_identify_database_operations_in_flow(self, llm_client: LocalLLMClient, pipeline_config,
                                                   token_assertions: TokenAssertions):
        """Test identifying database operations in execution flow."""
        code = """
        public void ProcessOrder(Order order)
        {
            using (var transaction = _db.BeginTransaction())
            {
                _db.Execute("INSERT INTO Orders (OrderNumber) VALUES (@OrderNumber)", order);
                _db.Execute("UPDATE Inventory SET Quantity = Quantity - @Qty WHERE ItemID = @ItemID", order);
                transaction.Commit();
            }
        }
        """

        prompt = f"""List all database operations in this code with their types:

{code}

Database operations:"""

        max_tokens = 200
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        assert_llm_response_valid(response, min_length=10)
        text = response.text.upper()
        assert "INSERT" in text
        assert "UPDATE" in text

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_trace_call_chain(self, llm_client: LocalLLMClient, pipeline_config,
                               token_assertions: TokenAssertions):
        """Test tracing method call chain."""
        code = """
        // UI Event
        btnProcess_Click → ProcessData()

        // ProcessData
        ProcessData() → ValidateData() → SaveData() → NotifyUser()
        """

        prompt = f"""Describe the call chain and data flow:

{code}

Analysis:"""

        max_tokens = 300
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.1,
        )

        assert_llm_response_valid(response, min_length=20)

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=20, max_tokens=500)
        token_assertions.assert_max_tokens_respected(response, max_tokens)
