using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using MongoDB.Driver;
using RoslynCodeAnalyzer.Models;

namespace RoslynCodeAnalyzer.Services
{
    /// <summary>
    /// Service for storing Roslyn code analysis results in MongoDB with vector embeddings.
    ///
    /// Collections:
    /// - code_classes: C# classes with inheritance and member information
    /// - code_methods: Methods with signatures, complexity, SQL operations
    /// - code_callgraph: Call relationships between methods
    /// - code_eventhandlers: Event handler mappings (UI to code)
    /// - code_dboperations: Database operations (SQL commands, stored procedures)
    ///
    /// All entities are stored with vector embeddings for semantic search.
    /// </summary>
    public class MongoDBService : IDisposable
    {
        private readonly MongoClient _client;
        private readonly IMongoDatabase _database;
        private readonly EmbeddingClient _embeddingClient;
        private readonly bool _generateEmbeddings;
        private bool _disposed;

        // Collection names (must match Python service)
        private const string COLLECTION_CLASSES = "code_classes";
        private const string COLLECTION_METHODS = "code_methods";
        private const string COLLECTION_CALLGRAPH = "code_callgraph";
        private const string COLLECTION_EVENTHANDLERS = "code_eventhandlers";
        private const string COLLECTION_DBOPERATIONS = "code_dboperations";

        public MongoDBService(MongoDBConfig? config = null, bool generateEmbeddings = true)
        {
            config ??= new MongoDBConfig();
            _generateEmbeddings = generateEmbeddings;

            _client = new MongoClient(config.ConnectionString);
            _database = _client.GetDatabase(config.DatabaseName);
            _embeddingClient = new EmbeddingClient(config.EmbeddingServiceUrl);
        }

        /// <summary>
        /// Initialize the service and create necessary indexes.
        /// </summary>
        public async Task InitializeAsync()
        {
            Console.WriteLine("Initializing MongoDB connection...");

            // Test connection
            try
            {
                var pingCommand = new MongoDB.Bson.BsonDocument("ping", 1);
                await _database.RunCommandAsync<MongoDB.Bson.BsonDocument>(pingCommand);
                Console.WriteLine("Connected to MongoDB successfully");
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to connect to MongoDB: {ex.Message}", ex);
            }

            // Create indexes
            await CreateIndexesAsync();

            // Check embedding service
            if (_generateEmbeddings)
            {
                var embeddingAvailable = await _embeddingClient.CheckAvailabilityAsync();
                if (embeddingAvailable)
                {
                    Console.WriteLine("Embedding service available");
                }
                else
                {
                    Console.WriteLine("Warning: Embedding service not available. Documents will be stored without vectors.");
                }
            }
        }

        private async Task CreateIndexesAsync()
        {
            // Classes collection indexes
            var classesCollection = _database.GetCollection<CodeClassDocument>(COLLECTION_CLASSES);
            await classesCollection.Indexes.CreateManyAsync(new[]
            {
                new CreateIndexModel<CodeClassDocument>(Builders<CodeClassDocument>.IndexKeys.Ascending(x => x.Id), new CreateIndexOptions { Unique = true }),
                new CreateIndexModel<CodeClassDocument>(Builders<CodeClassDocument>.IndexKeys.Ascending(x => x.Project)),
                new CreateIndexModel<CodeClassDocument>(Builders<CodeClassDocument>.IndexKeys.Ascending(x => x.Namespace)),
                new CreateIndexModel<CodeClassDocument>(Builders<CodeClassDocument>.IndexKeys.Ascending(x => x.ClassName)),
                new CreateIndexModel<CodeClassDocument>(Builders<CodeClassDocument>.IndexKeys.Ascending(x => x.FilePath))
            });

            // Methods collection indexes
            var methodsCollection = _database.GetCollection<CodeMethodDocument>(COLLECTION_METHODS);
            await methodsCollection.Indexes.CreateManyAsync(new[]
            {
                new CreateIndexModel<CodeMethodDocument>(Builders<CodeMethodDocument>.IndexKeys.Ascending(x => x.Id), new CreateIndexOptions { Unique = true }),
                new CreateIndexModel<CodeMethodDocument>(Builders<CodeMethodDocument>.IndexKeys.Ascending(x => x.Project)),
                new CreateIndexModel<CodeMethodDocument>(Builders<CodeMethodDocument>.IndexKeys.Ascending(x => x.Namespace)),
                new CreateIndexModel<CodeMethodDocument>(Builders<CodeMethodDocument>.IndexKeys.Ascending(x => x.ClassName)),
                new CreateIndexModel<CodeMethodDocument>(Builders<CodeMethodDocument>.IndexKeys.Ascending(x => x.MethodName)),
                new CreateIndexModel<CodeMethodDocument>(Builders<CodeMethodDocument>.IndexKeys.Ascending(x => x.HasSqlOperations))
            });

            // Call graph indexes
            var callGraphCollection = _database.GetCollection<CodeCallGraphDocument>(COLLECTION_CALLGRAPH);
            await callGraphCollection.Indexes.CreateManyAsync(new[]
            {
                new CreateIndexModel<CodeCallGraphDocument>(Builders<CodeCallGraphDocument>.IndexKeys.Ascending(x => x.Id), new CreateIndexOptions { Unique = true }),
                new CreateIndexModel<CodeCallGraphDocument>(Builders<CodeCallGraphDocument>.IndexKeys.Ascending(x => x.CallerClass)),
                new CreateIndexModel<CodeCallGraphDocument>(Builders<CodeCallGraphDocument>.IndexKeys.Ascending(x => x.CallerMethod)),
                new CreateIndexModel<CodeCallGraphDocument>(Builders<CodeCallGraphDocument>.IndexKeys.Ascending(x => x.CalleeClass)),
                new CreateIndexModel<CodeCallGraphDocument>(Builders<CodeCallGraphDocument>.IndexKeys.Ascending(x => x.CalleeMethod)),
                new CreateIndexModel<CodeCallGraphDocument>(Builders<CodeCallGraphDocument>.IndexKeys.Ascending(x => x.IsSqlOperation))
            });

            // Event handlers indexes
            var eventHandlersCollection = _database.GetCollection<CodeEventHandlerDocument>(COLLECTION_EVENTHANDLERS);
            await eventHandlersCollection.Indexes.CreateManyAsync(new[]
            {
                new CreateIndexModel<CodeEventHandlerDocument>(Builders<CodeEventHandlerDocument>.IndexKeys.Ascending(x => x.Id), new CreateIndexOptions { Unique = true }),
                new CreateIndexModel<CodeEventHandlerDocument>(Builders<CodeEventHandlerDocument>.IndexKeys.Ascending(x => x.Project)),
                new CreateIndexModel<CodeEventHandlerDocument>(Builders<CodeEventHandlerDocument>.IndexKeys.Ascending(x => x.EventName)),
                new CreateIndexModel<CodeEventHandlerDocument>(Builders<CodeEventHandlerDocument>.IndexKeys.Ascending(x => x.HandlerMethod))
            });

            // DB operations indexes
            var dbOpsCollection = _database.GetCollection<CodeDbOperationDocument>(COLLECTION_DBOPERATIONS);
            await dbOpsCollection.Indexes.CreateManyAsync(new[]
            {
                new CreateIndexModel<CodeDbOperationDocument>(Builders<CodeDbOperationDocument>.IndexKeys.Ascending(x => x.Id), new CreateIndexOptions { Unique = true }),
                new CreateIndexModel<CodeDbOperationDocument>(Builders<CodeDbOperationDocument>.IndexKeys.Ascending(x => x.Project)),
                new CreateIndexModel<CodeDbOperationDocument>(Builders<CodeDbOperationDocument>.IndexKeys.Ascending(x => x.OperationType)),
                new CreateIndexModel<CodeDbOperationDocument>(Builders<CodeDbOperationDocument>.IndexKeys.Ascending(x => x.StoredProcedure))
            });

            Console.WriteLine("MongoDB indexes created");
        }

        /// <summary>
        /// Store complete analysis results in MongoDB.
        /// </summary>
        public async Task<StorageStats> StoreAnalysisAsync(AnalysisResult result, string projectName)
        {
            var stats = new StorageStats();
            var stopwatch = Stopwatch.StartNew();

            Console.WriteLine($"\nStoring analysis in MongoDB for project: {projectName}");

            try
            {
                // Store classes
                foreach (var cls in result.Classes)
                {
                    try
                    {
                        await StoreClassAsync(cls, projectName);
                        stats.ClassesStored++;
                    }
                    catch (Exception ex)
                    {
                        stats.Errors++;
                        stats.ErrorMessages.Add($"Class {cls.ClassName}: {ex.Message}");
                    }
                }
                Console.WriteLine($"  Classes stored: {stats.ClassesStored}");

                // Store methods
                foreach (var method in result.Methods)
                {
                    try
                    {
                        await StoreMethodAsync(method, projectName);
                        stats.MethodsStored++;
                    }
                    catch (Exception ex)
                    {
                        stats.Errors++;
                        stats.ErrorMessages.Add($"Method {method.MethodName}: {ex.Message}");
                    }
                }
                Console.WriteLine($"  Methods stored: {stats.MethodsStored}");

                // Store call graph
                foreach (var call in result.CallGraph)
                {
                    try
                    {
                        await StoreCallGraphEdgeAsync(call, projectName);
                        stats.CallGraphStored++;
                    }
                    catch (Exception ex)
                    {
                        stats.Errors++;
                        stats.ErrorMessages.Add($"Call graph edge: {ex.Message}");
                    }
                }
                Console.WriteLine($"  Call graph edges stored: {stats.CallGraphStored}");

                // Store event handlers
                foreach (var handler in result.EventHandlers)
                {
                    try
                    {
                        await StoreEventHandlerAsync(handler, projectName);
                        stats.EventHandlersStored++;
                    }
                    catch (Exception ex)
                    {
                        stats.Errors++;
                        stats.ErrorMessages.Add($"Event handler: {ex.Message}");
                    }
                }
                Console.WriteLine($"  Event handlers stored: {stats.EventHandlersStored}");

                // Store database operations
                foreach (var dbOp in result.DatabaseOperations)
                {
                    try
                    {
                        await StoreDbOperationAsync(dbOp, projectName);
                        stats.DbOperationsStored++;
                    }
                    catch (Exception ex)
                    {
                        stats.Errors++;
                        stats.ErrorMessages.Add($"DB operation: {ex.Message}");
                    }
                }
                Console.WriteLine($"  Database operations stored: {stats.DbOperationsStored}");
            }
            finally
            {
                stopwatch.Stop();
                stats.Duration = stopwatch.Elapsed;
            }

            Console.WriteLine($"\nStorage complete in {stats.Duration.TotalSeconds:F2} seconds");
            if (stats.Errors > 0)
            {
                Console.WriteLine($"Errors: {stats.Errors}");
            }

            return stats;
        }

        private async Task StoreClassAsync(ClassInfo cls, string project)
        {
            var collection = _database.GetCollection<CodeClassDocument>(COLLECTION_CLASSES);

            // Generate ID
            var id = $"{project}:{cls.Namespace}.{cls.ClassName}";

            // Create embedding text
            var embeddingText = FormatClassEmbeddingText(cls);

            // Compute content hash
            var hashContent = $"{cls.ClassName}:{cls.Namespace}:{cls.BaseClass}:{string.Join(",", cls.Interfaces)}:{string.Join(",", cls.Methods)}";
            var contentHash = ComputeHash(hashContent);

            // Generate embedding if enabled
            List<float>? embedding = null;
            if (_generateEmbeddings)
            {
                embedding = await _embeddingClient.GenerateEmbeddingAsync(embeddingText);
            }

            var document = new CodeClassDocument
            {
                Id = id,
                Project = project,
                Namespace = cls.Namespace ?? "",
                ClassName = cls.ClassName ?? "",
                BaseClass = cls.BaseClass,
                Interfaces = cls.Interfaces ?? new List<string>(),
                Methods = cls.Methods ?? new List<string>(),
                Properties = cls.Properties ?? new List<string>(),
                Fields = cls.Fields ?? new List<string>(),
                IsStatic = cls.IsStatic,
                IsAbstract = cls.IsAbstract,
                IsSealed = cls.IsSealed,
                Accessibility = cls.Accessibility,
                Summary = cls.Summary,
                FilePath = cls.FilePath ?? "",
                LineNumber = cls.LineNumber,
                EmbeddingText = embeddingText,
                ContentHash = contentHash,
                UpdatedAt = DateTime.UtcNow,
                Vector = embedding
            };

            // Upsert
            await collection.ReplaceOneAsync(
                Builders<CodeClassDocument>.Filter.Eq(x => x.Id, id),
                document,
                new ReplaceOptions { IsUpsert = true });
        }

        private async Task StoreMethodAsync(MethodInfo method, string project)
        {
            var collection = _database.GetCollection<CodeMethodDocument>(COLLECTION_METHODS);

            // Generate ID
            var id = $"{project}:{method.Namespace}.{method.ClassName}.{method.MethodName}";

            // Create embedding text
            var embeddingText = FormatMethodEmbeddingText(method);

            // Compute content hash
            var hashContent = $"{method.MethodName}:{method.ReturnType}:{string.Join(",", method.Parameters.Select(p => p.Type))}";
            var contentHash = ComputeHash(hashContent);

            // Generate embedding if enabled
            List<float>? embedding = null;
            if (_generateEmbeddings)
            {
                embedding = await _embeddingClient.GenerateEmbeddingAsync(embeddingText);
            }

            var document = new CodeMethodDocument
            {
                Id = id,
                Project = project,
                Namespace = method.Namespace ?? "",
                ClassName = method.ClassName ?? "",
                MethodName = method.MethodName ?? "",
                ReturnType = method.ReturnType ?? "void",
                Parameters = method.Parameters?.Select(p => new ParameterDocument
                {
                    Name = p.Name ?? "",
                    Type = p.Type ?? "",
                    HasDefaultValue = p.HasDefaultValue,
                    DefaultValue = p.DefaultValue,
                    IsRef = p.IsRef,
                    IsOut = p.IsOut
                }).ToList() ?? new List<ParameterDocument>(),
                IsStatic = method.IsStatic,
                IsAsync = method.IsAsync,
                IsVirtual = method.IsVirtual,
                IsOverride = method.IsOverride,
                Accessibility = method.Accessibility,
                CyclomaticComplexity = method.CyclomaticComplexity,
                LineCount = method.LineCount,
                LineNumber = method.LineNumber,
                Summary = method.Summary,
                SqlCalls = method.SqlCalls?.Select(s => new SqlCallDocument
                {
                    Type = s.Type ?? "",
                    StoredProcedure = s.StoredProcedure,
                    CommandText = s.CommandText,
                    TableName = s.TableName,
                    Line = s.Line,
                    Parameters = s.Parameters ?? new List<string>()
                }).ToList() ?? new List<SqlCallDocument>(),
                HasSqlOperations = method.SqlCalls?.Any() ?? false,
                CallsTo = method.CallsTo?.Select(c => new CallReferenceDocument
                {
                    Project = c.Project,
                    File = c.File,
                    Class = c.Class,
                    Method = c.Method,
                    Line = c.Line,
                    CallType = c.CallType
                }).ToList() ?? new List<CallReferenceDocument>(),
                CalledBy = method.CalledBy?.Select(c => new CallReferenceDocument
                {
                    Project = c.Project,
                    File = c.File,
                    Class = c.Class,
                    Method = c.Method,
                    Line = c.Line,
                    CallType = c.CallType
                }).ToList() ?? new List<CallReferenceDocument>(),
                FilePath = method.FilePath ?? "",
                EmbeddingText = embeddingText,
                ContentHash = contentHash,
                UpdatedAt = DateTime.UtcNow,
                Vector = embedding
            };

            // Upsert
            await collection.ReplaceOneAsync(
                Builders<CodeMethodDocument>.Filter.Eq(x => x.Id, id),
                document,
                new ReplaceOptions { IsUpsert = true });
        }

        private async Task StoreCallGraphEdgeAsync(CallRelationship call, string project)
        {
            var collection = _database.GetCollection<CodeCallGraphDocument>(COLLECTION_CALLGRAPH);

            // Generate ID
            var id = $"{project}:{call.CallerClass}.{call.CallerMethod}->{call.CalleeClass}.{call.CalleeMethod}";

            // Create embedding text
            var embeddingText = FormatCallGraphEmbeddingText(call);

            // Compute content hash
            var hashContent = $"{call.CallerClass}.{call.CallerMethod}->{call.CalleeClass}.{call.CalleeMethod}:{call.CallType}";
            var contentHash = ComputeHash(hashContent);

            // Generate embedding if enabled
            List<float>? embedding = null;
            if (_generateEmbeddings)
            {
                embedding = await _embeddingClient.GenerateEmbeddingAsync(embeddingText);
            }

            var document = new CodeCallGraphDocument
            {
                Id = id,
                Project = project,
                CallerNamespace = call.CallerNamespace,
                CallerClass = call.CallerClass ?? "",
                CallerMethod = call.CallerMethod ?? "",
                CallerFile = call.CallerFilePath,
                CallerLine = call.CallerLineNumber,
                CalleeNamespace = call.CalleeNamespace,
                CalleeClass = call.CalleeClass ?? "",
                CalleeMethod = call.CalleeMethod ?? "",
                CalleeFile = call.CalleeFilePath,
                CallType = call.CallType ?? "Direct",
                IsSqlOperation = call.IsSqlOperation,
                StoredProcedureName = call.StoredProcedureName,
                SqlCommandText = call.SqlCommandText,
                CallSiteLine = call.CallSiteLineNumber,
                EmbeddingText = embeddingText,
                ContentHash = contentHash,
                UpdatedAt = DateTime.UtcNow,
                Vector = embedding
            };

            // Upsert
            await collection.ReplaceOneAsync(
                Builders<CodeCallGraphDocument>.Filter.Eq(x => x.Id, id),
                document,
                new ReplaceOptions { IsUpsert = true });
        }

        private async Task StoreEventHandlerAsync(EventHandlerInfo handler, string project)
        {
            var collection = _database.GetCollection<CodeEventHandlerDocument>(COLLECTION_EVENTHANDLERS);

            // Generate ID
            var id = $"{project}:{handler.HandlerClass}.{handler.HandlerMethod}:{handler.EventName}";

            // Create embedding text
            var embeddingText = FormatEventHandlerEmbeddingText(handler);

            // Compute content hash
            var hashContent = $"{handler.EventName}:{handler.HandlerMethod}:{handler.HandlerClass}:{handler.EventSource}";
            var contentHash = ComputeHash(hashContent);

            // Generate embedding if enabled
            List<float>? embedding = null;
            if (_generateEmbeddings)
            {
                embedding = await _embeddingClient.GenerateEmbeddingAsync(embeddingText);
            }

            var document = new CodeEventHandlerDocument
            {
                Id = id,
                Project = project,
                EventName = handler.EventName ?? "",
                HandlerMethod = handler.HandlerMethod ?? "",
                HandlerClass = handler.HandlerClass ?? "",
                Namespace = handler.HandlerNamespace,
                UIElementType = handler.UIElementType,
                ElementName = handler.EventSource,
                SubscriptionType = handler.SubscriptionType,
                FilePath = handler.FilePath ?? "",
                LineNumber = handler.LineNumber,
                EmbeddingText = embeddingText,
                ContentHash = contentHash,
                UpdatedAt = DateTime.UtcNow,
                Vector = embedding
            };

            // Upsert
            await collection.ReplaceOneAsync(
                Builders<CodeEventHandlerDocument>.Filter.Eq(x => x.Id, id),
                document,
                new ReplaceOptions { IsUpsert = true });
        }

        private async Task StoreDbOperationAsync(DatabaseOperationInfo dbOp, string project)
        {
            var collection = _database.GetCollection<CodeDbOperationDocument>(COLLECTION_DBOPERATIONS);

            // Generate ID
            var id = $"{project}:{dbOp.ClassName}.{dbOp.MethodName}:{dbOp.LineNumber}";

            // Create embedding text
            var embeddingText = FormatDbOperationEmbeddingText(dbOp);

            // Compute content hash
            var hashContent = $"{dbOp.ClassName}.{dbOp.MethodName}:{dbOp.OperationType}:{dbOp.CommandText}";
            var contentHash = ComputeHash(hashContent);

            // Generate embedding if enabled
            List<float>? embedding = null;
            if (_generateEmbeddings)
            {
                embedding = await _embeddingClient.GenerateEmbeddingAsync(embeddingText);
            }

            var document = new CodeDbOperationDocument
            {
                Id = id,
                Project = project,
                ClassName = dbOp.ClassName ?? "",
                MethodName = dbOp.MethodName ?? "",
                OperationType = dbOp.OperationType,
                TableName = dbOp.TableName,
                StoredProcedure = dbOp.CommandType == "StoredProcedure" ? dbOp.CommandText : null,
                CommandText = dbOp.CommandText,
                CommandType = dbOp.CommandType,
                Parameters = dbOp.Parameters?.ToList() ?? new List<string>(),
                FilePath = dbOp.FilePath ?? "",
                LineNumber = dbOp.LineNumber,
                EmbeddingText = embeddingText,
                ContentHash = contentHash,
                UpdatedAt = DateTime.UtcNow,
                Vector = embedding
            };

            // Upsert
            await collection.ReplaceOneAsync(
                Builders<CodeDbOperationDocument>.Filter.Eq(x => x.Id, id),
                document,
                new ReplaceOptions { IsUpsert = true });
        }

        #region Embedding Text Formatters

        private string FormatClassEmbeddingText(ClassInfo cls)
        {
            var parts = new List<string>();

            // Class identification
            parts.Add($"Class: {cls.Namespace}.{cls.ClassName}".TrimStart('.'));

            // Inheritance
            if (!string.IsNullOrEmpty(cls.BaseClass))
                parts.Add($"Inherits from: {cls.BaseClass}");

            // Interfaces
            if (cls.Interfaces?.Any() == true)
                parts.Add($"Implements: {string.Join(", ", cls.Interfaces)}");

            // Summary
            if (!string.IsNullOrEmpty(cls.Summary))
                parts.Add($"Description: {cls.Summary}");

            // Methods overview
            if (cls.Methods?.Any() == true)
                parts.Add($"Methods: {string.Join(", ", cls.Methods.Take(10))}");

            // Properties overview
            if (cls.Properties?.Any() == true)
                parts.Add($"Properties: {string.Join(", ", cls.Properties.Take(10))}");

            return string.Join("\n", parts);
        }

        private string FormatMethodEmbeddingText(MethodInfo method)
        {
            var parts = new List<string>();

            // Method identification
            parts.Add($"Method: {method.ClassName}.{method.MethodName}");

            // Signature
            var paramStr = method.Parameters != null
                ? string.Join(", ", method.Parameters.Select(p => $"{p.Type} {p.Name}"))
                : "";
            parts.Add($"Signature: {method.ReturnType} {method.MethodName}({paramStr})");

            // Summary
            if (!string.IsNullOrEmpty(method.Summary))
                parts.Add($"Description: {method.Summary}");

            // SQL operations
            if (method.SqlCalls?.Any() == true)
            {
                var sqlInfo = new List<string>();
                foreach (var sql in method.SqlCalls)
                {
                    if (!string.IsNullOrEmpty(sql.StoredProcedure))
                        sqlInfo.Add($"calls stored procedure {sql.StoredProcedure}");
                    else if (!string.IsNullOrEmpty(sql.CommandText))
                    {
                        var cmd = sql.CommandText.ToUpper();
                        if (cmd.Contains("SELECT")) sqlInfo.Add("reads from database");
                        else if (cmd.Contains("INSERT")) sqlInfo.Add("inserts into database");
                        else if (cmd.Contains("UPDATE")) sqlInfo.Add("updates database");
                        else if (cmd.Contains("DELETE")) sqlInfo.Add("deletes from database");
                    }
                }
                if (sqlInfo.Any())
                    parts.Add($"Database: {string.Join(", ", sqlInfo)}");
            }

            // Complexity
            if (method.CyclomaticComplexity > 10)
                parts.Add($"Complex logic (complexity: {method.CyclomaticComplexity})");

            return string.Join("\n", parts);
        }

        private string FormatCallGraphEmbeddingText(CallRelationship call)
        {
            var parts = new List<string>();

            var caller = $"{call.CallerClass}.{call.CallerMethod}";
            var callee = $"{call.CalleeClass}.{call.CalleeMethod}";

            parts.Add($"Call relationship: {caller} calls {callee}");

            if (!string.IsNullOrEmpty(call.CallType))
                parts.Add($"Type: {call.CallType}");

            if (call.IsSqlOperation)
            {
                parts.Add("This is a database operation call");
                if (!string.IsNullOrEmpty(call.StoredProcedureName))
                    parts.Add($"Stored procedure: {call.StoredProcedureName}");
            }

            return string.Join("\n", parts);
        }

        private string FormatEventHandlerEmbeddingText(EventHandlerInfo handler)
        {
            var parts = new List<string>();

            parts.Add($"Event handler: {handler.EventName} -> {handler.HandlerMethod}");

            if (!string.IsNullOrEmpty(handler.UIElementType))
                parts.Add($"UI Element: {handler.UIElementType}");

            if (!string.IsNullOrEmpty(handler.EventSource))
                parts.Add($"Element: {handler.EventSource}");

            // Make it searchable
            parts.Add($"When user interacts with {handler.EventSource ?? "control"}, {handler.HandlerMethod} is executed");

            return string.Join("\n", parts);
        }

        private string FormatDbOperationEmbeddingText(DatabaseOperationInfo dbOp)
        {
            var parts = new List<string>();

            parts.Add($"Database operation in {dbOp.ClassName}.{dbOp.MethodName}");

            if (!string.IsNullOrEmpty(dbOp.OperationType))
                parts.Add($"Type: {dbOp.OperationType}");

            if (!string.IsNullOrEmpty(dbOp.TableName))
                parts.Add($"Table: {dbOp.TableName}");

            if (dbOp.CommandType == "StoredProcedure" && !string.IsNullOrEmpty(dbOp.CommandText))
                parts.Add($"Stored Procedure: {dbOp.CommandText}");
            else if (!string.IsNullOrEmpty(dbOp.CommandText))
                parts.Add($"SQL: {dbOp.CommandText.Substring(0, Math.Min(200, dbOp.CommandText.Length))}");

            return string.Join("\n", parts);
        }

        #endregion

        #region Utility Methods

        private string ComputeHash(string content)
        {
            using var sha256 = SHA256.Create();
            var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(content));
            return Convert.ToHexString(bytes).Substring(0, 16).ToLower();
        }

        /// <summary>
        /// Get statistics about the stored data.
        /// </summary>
        public async Task<Dictionary<string, long>> GetStatsAsync()
        {
            var stats = new Dictionary<string, long>();

            stats["classes"] = await _database.GetCollection<CodeClassDocument>(COLLECTION_CLASSES).CountDocumentsAsync(FilterDefinition<CodeClassDocument>.Empty);
            stats["methods"] = await _database.GetCollection<CodeMethodDocument>(COLLECTION_METHODS).CountDocumentsAsync(FilterDefinition<CodeMethodDocument>.Empty);
            stats["callgraph"] = await _database.GetCollection<CodeCallGraphDocument>(COLLECTION_CALLGRAPH).CountDocumentsAsync(FilterDefinition<CodeCallGraphDocument>.Empty);
            stats["eventhandlers"] = await _database.GetCollection<CodeEventHandlerDocument>(COLLECTION_EVENTHANDLERS).CountDocumentsAsync(FilterDefinition<CodeEventHandlerDocument>.Empty);
            stats["dboperations"] = await _database.GetCollection<CodeDbOperationDocument>(COLLECTION_DBOPERATIONS).CountDocumentsAsync(FilterDefinition<CodeDbOperationDocument>.Empty);
            stats["total"] = stats.Values.Sum();

            return stats;
        }

        #endregion

        public void Dispose()
        {
            if (!_disposed)
            {
                _embeddingClient?.Dispose();
                _disposed = true;
            }
        }
    }
}
