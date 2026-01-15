using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace RoslynCodeAnalyzer.Services
{
    /// <summary>
    /// Client for generating text embeddings using the Python embedding service.
    /// The Python service runs at http://localhost:3030 and uses sentence-transformers.
    /// </summary>
    public class EmbeddingClient : IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;
        private bool _isAvailable;
        private bool _disposed;

        public EmbeddingClient(string baseUrl = "http://localhost:3030")
        {
            _baseUrl = baseUrl.TrimEnd('/');
            _httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(60)
            };
        }

        /// <summary>
        /// Check if the embedding service is available.
        /// </summary>
        public async Task<bool> CheckAvailabilityAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/health");
                _isAvailable = response.IsSuccessStatusCode;
                return _isAvailable;
            }
            catch
            {
                _isAvailable = false;
                return false;
            }
        }

        /// <summary>
        /// Generate an embedding vector for the given text.
        /// </summary>
        /// <param name="text">Text to embed</param>
        /// <returns>Embedding vector as list of floats, or null if service unavailable</returns>
        public async Task<List<float>?> GenerateEmbeddingAsync(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return null;

            try
            {
                // The Python service expects a POST to /embeddings with JSON body
                var requestBody = new { text = text };
                var json = JsonConvert.SerializeObject(requestBody);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_baseUrl}/embeddings", content);

                if (!response.IsSuccessStatusCode)
                {
                    Console.Error.WriteLine($"Embedding service returned {response.StatusCode}");
                    return null;
                }

                var responseJson = await response.Content.ReadAsStringAsync();
                var result = JsonConvert.DeserializeObject<EmbeddingResponse>(responseJson);

                return result?.Embedding;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error generating embedding: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Generate embeddings for multiple texts in batch (more efficient).
        /// </summary>
        public async Task<List<List<float>?>> GenerateEmbeddingsBatchAsync(List<string> texts)
        {
            var results = new List<List<float>?>();

            try
            {
                // The Python service expects a POST to /embeddings/batch with JSON body
                var requestBody = new { texts = texts };
                var json = JsonConvert.SerializeObject(requestBody);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_baseUrl}/embeddings/batch", content);

                if (!response.IsSuccessStatusCode)
                {
                    // Fall back to individual requests
                    Console.WriteLine("Batch embedding failed, falling back to individual requests");
                    foreach (var text in texts)
                    {
                        results.Add(await GenerateEmbeddingAsync(text));
                    }
                    return results;
                }

                var responseJson = await response.Content.ReadAsStringAsync();
                var result = JsonConvert.DeserializeObject<BatchEmbeddingResponse>(responseJson);

                if (result?.Embeddings != null)
                {
                    return result.Embeddings;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error generating batch embeddings: {ex.Message}");
            }

            // Return list of nulls if failed
            for (int i = 0; i < texts.Count; i++)
            {
                results.Add(null);
            }
            return results;
        }

        /// <summary>
        /// Generate embedding using a simpler direct API call that matches the Python service.
        /// Falls back to calling the store_code_context endpoint which handles embedding internally.
        /// </summary>
        public async Task<StoreResult?> StoreWithEmbeddingAsync(
            string documentId,
            string content,
            Dictionary<string, object>? metadata = null)
        {
            try
            {
                var requestBody = new
                {
                    document_id = documentId,
                    content = content,
                    metadata = metadata
                };

                var json = JsonConvert.SerializeObject(requestBody);
                var httpContent = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_baseUrl}/code-context", httpContent);

                if (!response.IsSuccessStatusCode)
                {
                    Console.Error.WriteLine($"Store with embedding failed: {response.StatusCode}");
                    return null;
                }

                var responseJson = await response.Content.ReadAsStringAsync();
                return JsonConvert.DeserializeObject<StoreResult>(responseJson);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error storing with embedding: {ex.Message}");
                return null;
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _httpClient?.Dispose();
                _disposed = true;
            }
        }

        // Response DTOs
        private class EmbeddingResponse
        {
            [JsonProperty("embedding")]
            public List<float>? Embedding { get; set; }

            [JsonProperty("success")]
            public bool Success { get; set; }
        }

        private class BatchEmbeddingResponse
        {
            [JsonProperty("embeddings")]
            public List<List<float>?>? Embeddings { get; set; }

            [JsonProperty("success")]
            public bool Success { get; set; }
        }
    }

    public class StoreResult
    {
        [JsonProperty("success")]
        public bool Success { get; set; }

        [JsonProperty("document_id")]
        public string? DocumentId { get; set; }

        [JsonProperty("message")]
        public string? Message { get; set; }
    }
}
