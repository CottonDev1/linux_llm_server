using System;

namespace RoslynCodeAnalyzer.Models
{
    public class AnalyzerOptions
    {
        public string InputPath { get; set; }
        public string OutputPath { get; set; }
        public string Mode { get; set; }  // "file", "directory", "filelist", "solution", "project"
        public string ProjectName { get; set; }  // Project name for context tracking
        public bool Verbose { get; set; }
        public bool IncludePrivateMembers { get; set; }
        public string FileListPath { get; set; }  // Path to text file containing list of files

        // MongoDB output options
        public OutputType Output { get; set; } = OutputType.Json;  // "json" or "mongodb"
        public string MongoConnectionString { get; set; } = Environment.GetEnvironmentVariable("MONGODB_URI") ?? "mongodb://localhost:27019";
        public string MongoDatabaseName { get; set; } = "rag_server";
        public string EmbeddingServiceUrl { get; set; } = "http://localhost:3030";
        public bool GenerateEmbeddings { get; set; } = true;  // Generate vector embeddings for semantic search
    }

    public enum OutputType
    {
        Json,
        MongoDB,
        Both  // Write to both JSON and MongoDB
    }
}