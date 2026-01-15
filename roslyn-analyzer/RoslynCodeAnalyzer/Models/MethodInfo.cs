using System.Collections.Generic;

namespace RoslynCodeAnalyzer.Models
{
    public class MethodInfo
    {
        public string ProjectName { get; set; }
        public string MethodName { get; set; }
        public string ClassName { get; set; }
        public string Namespace { get; set; }
        public string FullName { get; set; }
        public string FilePath { get; set; }
        public int LineNumber { get; set; }
        public int LineCount { get; set; }
        public string Accessibility { get; set; }
        public bool IsStatic { get; set; }
        public bool IsAsync { get; set; }
        public bool IsVirtual { get; set; }
        public bool IsOverride { get; set; }
        public string ReturnType { get; set; }
        public List<ParameterInfo> Parameters { get; set; } = new List<ParameterInfo>();
        public string Summary { get; set; }
        public int CyclomaticComplexity { get; set; }
        public List<string> LocalVariables { get; set; } = new List<string>();

        // Call tracking
        public List<CallReference> CalledBy { get; set; } = new List<CallReference>();
        public List<CallReference> CallsTo { get; set; } = new List<CallReference>();

        // SQL operations
        public List<SqlCall> SqlCalls { get; set; } = new List<SqlCall>();
    }

    public class CallReference
    {
        public string Project { get; set; }
        public string File { get; set; }
        public string Class { get; set; }
        public string Method { get; set; }
        public int Line { get; set; }
        public string CallType { get; set; }  // "Direct", "Delegate", "Virtual", "DataLayer", etc.
    }

    public class SqlCall
    {
        public string Type { get; set; }  // "SqlCommand", "StoredProcedure", "DataLayer"
        public string StoredProcedure { get; set; }
        public string CommandText { get; set; }
        public string TableName { get; set; }
        public int Line { get; set; }
        public List<string> Parameters { get; set; } = new List<string>();
    }

    public class ParameterInfo
    {
        public string Name { get; set; }
        public string Type { get; set; }
        public bool HasDefaultValue { get; set; }
        public string DefaultValue { get; set; }
        public bool IsRef { get; set; }
        public bool IsOut { get; set; }
    }
}
