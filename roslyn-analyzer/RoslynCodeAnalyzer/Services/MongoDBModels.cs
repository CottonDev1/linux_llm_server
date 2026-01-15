using System;
using System.Collections.Generic;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;

namespace RoslynCodeAnalyzer.Services
{
    /// <summary>
    /// MongoDB document for storing C# class information with vector embeddings.
    /// </summary>
    public class CodeClassDocument
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string? MongoId { get; set; }

        [BsonElement("id")]
        public string Id { get; set; } = string.Empty;

        [BsonElement("project")]
        public string Project { get; set; } = string.Empty;

        [BsonElement("namespace")]
        public string Namespace { get; set; } = string.Empty;

        [BsonElement("class_name")]
        public string ClassName { get; set; } = string.Empty;

        [BsonElement("base_class")]
        public string? BaseClass { get; set; }

        [BsonElement("interfaces")]
        public List<string> Interfaces { get; set; } = new();

        [BsonElement("methods")]
        public List<string> Methods { get; set; } = new();

        [BsonElement("properties")]
        public List<string> Properties { get; set; } = new();

        [BsonElement("fields")]
        public List<string> Fields { get; set; } = new();

        [BsonElement("is_static")]
        public bool IsStatic { get; set; }

        [BsonElement("is_abstract")]
        public bool IsAbstract { get; set; }

        [BsonElement("is_sealed")]
        public bool IsSealed { get; set; }

        [BsonElement("accessibility")]
        public string? Accessibility { get; set; }

        [BsonElement("summary")]
        public string? Summary { get; set; }

        [BsonElement("file_path")]
        public string FilePath { get; set; } = string.Empty;

        [BsonElement("line_number")]
        public int LineNumber { get; set; }

        [BsonElement("embedding_text")]
        public string EmbeddingText { get; set; } = string.Empty;

        [BsonElement("content_hash")]
        public string ContentHash { get; set; } = string.Empty;

        [BsonElement("created_at")]
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("updated_at")]
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("vector")]
        public List<float>? Vector { get; set; }
    }

    /// <summary>
    /// MongoDB document for storing C# method information with vector embeddings.
    /// </summary>
    public class CodeMethodDocument
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string? MongoId { get; set; }

        [BsonElement("id")]
        public string Id { get; set; } = string.Empty;

        [BsonElement("project")]
        public string Project { get; set; } = string.Empty;

        [BsonElement("namespace")]
        public string Namespace { get; set; } = string.Empty;

        [BsonElement("class_name")]
        public string ClassName { get; set; } = string.Empty;

        [BsonElement("method_name")]
        public string MethodName { get; set; } = string.Empty;

        [BsonElement("return_type")]
        public string ReturnType { get; set; } = "void";

        [BsonElement("parameters")]
        public List<ParameterDocument> Parameters { get; set; } = new();

        [BsonElement("is_static")]
        public bool IsStatic { get; set; }

        [BsonElement("is_async")]
        public bool IsAsync { get; set; }

        [BsonElement("is_virtual")]
        public bool IsVirtual { get; set; }

        [BsonElement("is_override")]
        public bool IsOverride { get; set; }

        [BsonElement("accessibility")]
        public string? Accessibility { get; set; }

        [BsonElement("cyclomatic_complexity")]
        public int CyclomaticComplexity { get; set; } = 1;

        [BsonElement("line_count")]
        public int LineCount { get; set; }

        [BsonElement("line_number")]
        public int LineNumber { get; set; }

        [BsonElement("summary")]
        public string? Summary { get; set; }

        [BsonElement("body")]
        public string? Body { get; set; }

        [BsonElement("sql_calls")]
        public List<SqlCallDocument> SqlCalls { get; set; } = new();

        [BsonElement("has_sql_operations")]
        public bool HasSqlOperations { get; set; }

        [BsonElement("calls_to")]
        public List<CallReferenceDocument> CallsTo { get; set; } = new();

        [BsonElement("called_by")]
        public List<CallReferenceDocument> CalledBy { get; set; } = new();

        [BsonElement("file_path")]
        public string FilePath { get; set; } = string.Empty;

        [BsonElement("embedding_text")]
        public string EmbeddingText { get; set; } = string.Empty;

        [BsonElement("content_hash")]
        public string ContentHash { get; set; } = string.Empty;

        [BsonElement("created_at")]
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("updated_at")]
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("vector")]
        public List<float>? Vector { get; set; }
    }

    /// <summary>
    /// MongoDB document for storing call graph edges.
    /// </summary>
    public class CodeCallGraphDocument
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string? MongoId { get; set; }

        [BsonElement("id")]
        public string Id { get; set; } = string.Empty;

        [BsonElement("project")]
        public string Project { get; set; } = string.Empty;

        [BsonElement("caller_namespace")]
        public string? CallerNamespace { get; set; }

        [BsonElement("caller_class")]
        public string CallerClass { get; set; } = string.Empty;

        [BsonElement("caller_method")]
        public string CallerMethod { get; set; } = string.Empty;

        [BsonElement("caller_file")]
        public string? CallerFile { get; set; }

        [BsonElement("caller_line")]
        public int CallerLine { get; set; }

        [BsonElement("callee_namespace")]
        public string? CalleeNamespace { get; set; }

        [BsonElement("callee_class")]
        public string CalleeClass { get; set; } = string.Empty;

        [BsonElement("callee_method")]
        public string CalleeMethod { get; set; } = string.Empty;

        [BsonElement("callee_file")]
        public string? CalleeFile { get; set; }

        [BsonElement("call_type")]
        public string CallType { get; set; } = "Direct";

        [BsonElement("is_sql_operation")]
        public bool IsSqlOperation { get; set; }

        [BsonElement("stored_procedure_name")]
        public string? StoredProcedureName { get; set; }

        [BsonElement("sql_command_text")]
        public string? SqlCommandText { get; set; }

        [BsonElement("call_site_line")]
        public int CallSiteLine { get; set; }

        [BsonElement("embedding_text")]
        public string EmbeddingText { get; set; } = string.Empty;

        [BsonElement("content_hash")]
        public string ContentHash { get; set; } = string.Empty;

        [BsonElement("updated_at")]
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("vector")]
        public List<float>? Vector { get; set; }
    }

    /// <summary>
    /// MongoDB document for storing event handlers.
    /// </summary>
    public class CodeEventHandlerDocument
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string? MongoId { get; set; }

        [BsonElement("id")]
        public string Id { get; set; } = string.Empty;

        [BsonElement("project")]
        public string Project { get; set; } = string.Empty;

        [BsonElement("event_name")]
        public string EventName { get; set; } = string.Empty;

        [BsonElement("handler_method")]
        public string HandlerMethod { get; set; } = string.Empty;

        [BsonElement("handler_class")]
        public string HandlerClass { get; set; } = string.Empty;

        [BsonElement("namespace")]
        public string? Namespace { get; set; }

        [BsonElement("ui_element_type")]
        public string? UIElementType { get; set; }

        [BsonElement("element_name")]
        public string? ElementName { get; set; }

        [BsonElement("subscription_type")]
        public string? SubscriptionType { get; set; }

        [BsonElement("file_path")]
        public string FilePath { get; set; } = string.Empty;

        [BsonElement("line_number")]
        public int LineNumber { get; set; }

        [BsonElement("embedding_text")]
        public string EmbeddingText { get; set; } = string.Empty;

        [BsonElement("content_hash")]
        public string ContentHash { get; set; } = string.Empty;

        [BsonElement("updated_at")]
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("vector")]
        public List<float>? Vector { get; set; }
    }

    /// <summary>
    /// MongoDB document for storing database operations.
    /// </summary>
    public class CodeDbOperationDocument
    {
        [BsonId]
        [BsonRepresentation(BsonType.ObjectId)]
        public string? MongoId { get; set; }

        [BsonElement("id")]
        public string Id { get; set; } = string.Empty;

        [BsonElement("project")]
        public string Project { get; set; } = string.Empty;

        [BsonElement("class_name")]
        public string ClassName { get; set; } = string.Empty;

        [BsonElement("method_name")]
        public string MethodName { get; set; } = string.Empty;

        [BsonElement("operation_type")]
        public string? OperationType { get; set; }

        [BsonElement("table_name")]
        public string? TableName { get; set; }

        [BsonElement("stored_procedure")]
        public string? StoredProcedure { get; set; }

        [BsonElement("command_text")]
        public string? CommandText { get; set; }

        [BsonElement("command_type")]
        public string? CommandType { get; set; }

        [BsonElement("parameters")]
        public List<string> Parameters { get; set; } = new();

        [BsonElement("file_path")]
        public string FilePath { get; set; } = string.Empty;

        [BsonElement("line_number")]
        public int LineNumber { get; set; }

        [BsonElement("embedding_text")]
        public string EmbeddingText { get; set; } = string.Empty;

        [BsonElement("content_hash")]
        public string ContentHash { get; set; } = string.Empty;

        [BsonElement("updated_at")]
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        [BsonElement("vector")]
        public List<float>? Vector { get; set; }
    }

    /// <summary>
    /// Sub-document for method parameters.
    /// </summary>
    public class ParameterDocument
    {
        [BsonElement("name")]
        public string Name { get; set; } = string.Empty;

        [BsonElement("type")]
        public string Type { get; set; } = string.Empty;

        [BsonElement("has_default_value")]
        public bool HasDefaultValue { get; set; }

        [BsonElement("default_value")]
        public string? DefaultValue { get; set; }

        [BsonElement("is_ref")]
        public bool IsRef { get; set; }

        [BsonElement("is_out")]
        public bool IsOut { get; set; }
    }

    /// <summary>
    /// Sub-document for SQL calls within a method.
    /// </summary>
    public class SqlCallDocument
    {
        [BsonElement("type")]
        public string Type { get; set; } = string.Empty;

        [BsonElement("stored_procedure")]
        public string? StoredProcedure { get; set; }

        [BsonElement("command_text")]
        public string? CommandText { get; set; }

        [BsonElement("table_name")]
        public string? TableName { get; set; }

        [BsonElement("line")]
        public int Line { get; set; }

        [BsonElement("parameters")]
        public List<string> Parameters { get; set; } = new();
    }

    /// <summary>
    /// Sub-document for call references (callers/callees).
    /// </summary>
    public class CallReferenceDocument
    {
        [BsonElement("project")]
        public string? Project { get; set; }

        [BsonElement("file")]
        public string? File { get; set; }

        [BsonElement("class")]
        public string? Class { get; set; }

        [BsonElement("method")]
        public string? Method { get; set; }

        [BsonElement("line")]
        public int Line { get; set; }

        [BsonElement("call_type")]
        public string? CallType { get; set; }
    }

    /// <summary>
    /// Configuration for MongoDB connection.
    /// </summary>
    public class MongoDBConfig
    {
        public string ConnectionString { get; set; } = Environment.GetEnvironmentVariable("MONGODB_URI") ?? "mongodb://localhost:27019";
        public string DatabaseName { get; set; } = "rag_server";
        public string EmbeddingServiceUrl { get; set; } = "http://localhost:3030";
    }

    /// <summary>
    /// Statistics about the MongoDB storage operation.
    /// </summary>
    public class StorageStats
    {
        public int ClassesStored { get; set; }
        public int MethodsStored { get; set; }
        public int CallGraphStored { get; set; }
        public int EventHandlersStored { get; set; }
        public int DbOperationsStored { get; set; }
        public int EmbeddingsGenerated { get; set; }
        public int Errors { get; set; }
        public List<string> ErrorMessages { get; set; } = new();
        public TimeSpan Duration { get; set; }
    }
}
