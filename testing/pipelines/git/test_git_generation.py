"""
Git Analysis Generation Tests
==============================

Test code analysis generation using local llama.cpp (port 8082).

Tests cover:
- Method purpose summary generation
- Code explanation generation
- Relationship extraction (calls, called by)
- Database operation detection
- Token usage tracking
"""

import pytest
from typing import Dict, Any

from config.test_config import get_test_config
from fixtures.llm_fixtures import LocalLLMClient
from fixtures.shared_fixtures import TokenAssertions
from utils import (
    assert_llm_response_valid,
    generate_test_id,
)


class TestGitGeneration:
    """Test git analysis generation using local LLM."""

    @pytest.mark.requires_llm
    def test_generate_method_purpose(self, llm_client: LocalLLMClient, pipeline_config,
                                      token_assertions: TokenAssertions):
        """Test generating a purpose summary for a code method."""
        # Sample C# method code
        code = """
        public User GetUserById(int userId)
        {
            var query = "SELECT * FROM Users WHERE UserID = @UserId";
            using (var conn = new SqlConnection(_connectionString))
            {
                return conn.QuerySingleOrDefault<User>(query, new { UserId = userId });
            }
        }
        """

        # Prompt for purpose generation
        prompt = f"""Analyze this C# method and provide a concise purpose summary (1-2 sentences):

{code}

Purpose:"""

        # Generate purpose
        max_tokens = 150
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=10,
            max_length=500,
            must_contain=["user"],
        )

        # Verify it describes the method purpose
        assert any(word in response.text.lower() for word in ["retrieve", "get", "fetch", "query"])

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=30, max_tokens=400)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_generate_code_explanation(self, llm_client: LocalLLMClient, pipeline_config,
                                        token_assertions: TokenAssertions):
        """Test generating a detailed code explanation."""
        code = """
        public async Task<bool> ProcessOrderAsync(Order order)
        {
            // Validate order
            if (!ValidateOrder(order))
                return false;

            // Calculate totals
            var total = CalculateOrderTotal(order);
            order.Total = total;

            // Save to database
            using (var transaction = await _db.BeginTransactionAsync())
            {
                try
                {
                    await _db.Orders.AddAsync(order);
                    await _db.SaveChangesAsync();
                    await transaction.CommitAsync();
                    return true;
                }
                catch
                {
                    await transaction.RollbackAsync();
                    return false;
                }
            }
        }
        """

        prompt = f"""Explain what this C# method does in detail:

{code}

Explanation:"""

        # Generate explanation
        max_tokens = 512
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.2,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=50,
            must_contain=["order", "transaction"],
        )

        # Verify key concepts are explained
        text_lower = response.text.lower()
        assert any(word in text_lower for word in ["validate", "validation"])
        assert any(word in text_lower for word in ["database", "transaction"])

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_tokens_in_range(response, min_tokens=80, max_tokens=800)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_extract_method_calls(self, llm_client: LocalLLMClient, pipeline_config,
                                   token_assertions: TokenAssertions):
        """Test extracting method calls from code."""
        code = """
        public void ProcessUser(int userId)
        {
            var user = GetUserById(userId);
            if (user != null)
            {
                ValidateUser(user);
                UpdateUserStatus(user);
                SendNotification(user.Email);
            }
        }
        """

        prompt = f"""List all method calls in this code (one per line):

{code}

Method calls:"""

        # Generate method calls list
        max_tokens = 200
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=10,
        )

        # Verify method calls are identified
        text = response.text
        assert "GetUserById" in text
        assert "ValidateUser" in text
        assert "UpdateUserStatus" in text
        assert "SendNotification" in text

        # Token assertions
        token_assertions.assert_tokens_captured(response)
        token_assertions.assert_max_tokens_respected(response, max_tokens)

    @pytest.mark.requires_llm
    def test_identify_database_operations(self, llm_client: LocalLLMClient, pipeline_config):
        """Test identifying database operations in code."""
        code = """
        public void UpdateUserProfile(int userId, string newEmail)
        {
            var sql = "UPDATE Users SET Email = @Email WHERE UserID = @UserId";
            using (var conn = new SqlConnection(_connectionString))
            {
                conn.Execute(sql, new { Email = newEmail, UserId = userId });
            }

            var auditSql = "INSERT INTO UserAudit (UserID, Action, Timestamp) VALUES (@UserId, 'EmailUpdate', GETDATE())";
            using (var conn = new SqlConnection(_connectionString))
            {
                conn.Execute(auditSql, new { UserId = userId });
            }
        }
        """

        prompt = f"""Identify all database operations in this code. For each operation, list:
- Operation type (SELECT, INSERT, UPDATE, DELETE)
- Tables affected

Code:
{code}

Database operations:"""

        # Generate analysis
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.0,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=20,
        )

        # Verify operations identified
        text = response.text
        assert any(word in text.upper() for word in ["UPDATE", "INSERT"])
        assert "Users" in text
        assert "UserAudit" in text

    @pytest.mark.requires_llm
    def test_summarize_class_purpose(self, llm_client: LocalLLMClient, pipeline_config):
        """Test generating a class-level purpose summary."""
        class_code = """
        public class UserService
        {
            private readonly IUserRepository _repository;
            private readonly IEmailService _emailService;

            public UserService(IUserRepository repository, IEmailService emailService)
            {
                _repository = repository;
                _emailService = emailService;
            }

            public User GetUserById(int userId) { /* ... */ }
            public void CreateUser(User user) { /* ... */ }
            public void UpdateUser(User user) { /* ... */ }
            public void DeleteUser(int userId) { /* ... */ }
            public void SendWelcomeEmail(User user) { /* ... */ }
        }
        """

        prompt = f"""Summarize the purpose of this class in 1-2 sentences:

{class_code}

Purpose:"""

        # Generate summary
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=150,
            temperature=0.1,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=20,
            max_length=300,
            must_contain=["user"],
        )

        # Verify it describes CRUD operations
        text_lower = response.text.lower()
        assert any(word in text_lower for word in ["manage", "service", "operations", "crud"])

    @pytest.mark.requires_llm
    def test_detect_async_methods(self, llm_client: LocalLLMClient, pipeline_config):
        """Test detecting async methods and their patterns."""
        code = """
        public async Task<List<User>> GetActiveUsersAsync()
        {
            var users = await _repository.GetAllAsync();
            return users.Where(u => u.IsActive).ToList();
        }

        public Task<User> GetUserByIdAsync(int userId)
        {
            return _repository.GetByIdAsync(userId);
        }
        """

        prompt = f"""List all async methods in this code and explain their async behavior:

{code}

Analysis:"""

        # Generate analysis
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.0,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=20,
        )

        # Verify async methods identified
        text = response.text
        assert "GetActiveUsersAsync" in text or "GetActiveUsers" in text
        assert "GetUserByIdAsync" in text or "GetUserById" in text

    @pytest.mark.requires_llm
    def test_analyze_error_handling(self, llm_client: LocalLLMClient, pipeline_config):
        """Test analyzing error handling patterns."""
        code = """
        public bool SaveUser(User user)
        {
            try
            {
                ValidateUser(user);
                _repository.Save(user);
                return true;
            }
            catch (ValidationException ex)
            {
                _logger.LogWarning("Validation failed: {Message}", ex.Message);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save user");
                throw;
            }
        }
        """

        prompt = f"""Describe the error handling strategy in this code:

{code}

Error handling:"""

        # Generate analysis
        response = llm_client.generate(
            prompt=prompt,
            model_type="code",
            max_tokens=300,
            temperature=0.1,
        )

        # Verify response
        assert_llm_response_valid(
            response,
            min_length=20,
        )

        # Verify error handling concepts mentioned
        text_lower = response.text.lower()
        assert any(word in text_lower for word in ["exception", "catch", "error", "try"])

    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_batch_generation_performance(self, llm_client: LocalLLMClient, pipeline_config):
        """Test performance of generating multiple summaries."""
        import time

        methods = [
            "public void Method1() { /* simple method */ }",
            "public int Method2(string param) { return param.Length; }",
            "public async Task Method3() { await Task.Delay(100); }",
        ]

        start_time = time.time()
        responses = []

        for method_code in methods:
            prompt = f"Summarize this method in one sentence:\n{method_code}\nSummary:"
            response = llm_client.generate(
                prompt=prompt,
                model_type="code",
                max_tokens=50,
                temperature=0.0,
            )
            responses.append(response)

        elapsed = time.time() - start_time

        # Verify all succeeded
        assert all(r.success for r in responses)
        assert len(responses) == 3

        # Performance check (should complete in reasonable time)
        assert elapsed < 60.0, f"Batch generation took too long: {elapsed}s"

    @pytest.mark.requires_llm
    def test_llm_health_check(self, llm_client: LocalLLMClient, test_config):
        """Test that code LLM endpoint is healthy."""
        health = llm_client.health_check("code")

        assert health["healthy"] is True
        assert "endpoint" in health
        assert test_config.llm.code_endpoint in health["endpoint"]
