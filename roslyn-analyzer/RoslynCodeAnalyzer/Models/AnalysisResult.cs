using System;
using System.Collections.Generic;

namespace RoslynCodeAnalyzer.Models
{
    public class AnalysisResult
    {
        public string AnalyzedPath { get; set; }
        public string AnalysisMode { get; set; }
        public DateTime Timestamp { get; set; }
        public List<ClassInfo> Classes { get; set; } = new List<ClassInfo>();
        public List<MethodInfo> Methods { get; set; } = new List<MethodInfo>();
        public List<CallRelationship> CallGraph { get; set; } = new List<CallRelationship>();
        public List<EventHandlerInfo> EventHandlers { get; set; } = new List<EventHandlerInfo>();
        public List<DatabaseOperationInfo> DatabaseOperations { get; set; } = new List<DatabaseOperationInfo>();
        public List<AnalysisError> Errors { get; set; } = new List<AnalysisError>();
    }
}
