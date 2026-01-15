namespace RoslynCodeAnalyzer.Models
{
    public class CallRelationship
    {
        // Caller information
        public string CallerProject { get; set; }
        public string CallerMethod { get; set; }
        public string CallerClass { get; set; }
        public string CallerNamespace { get; set; }
        public string CallerFilePath { get; set; }
        public int CallerLineNumber { get; set; }

        // Callee information
        public string CalleeProject { get; set; }
        public string CalleeMethod { get; set; }
        public string CalleeClass { get; set; }
        public string CalleeNamespace { get; set; }
        public string CalleeFilePath { get; set; }
        public int CalleeLineNumber { get; set; }  // Line where callee is defined

        // Call information
        public string CallType { get; set; }  // "Direct", "Delegate", "Event", "Virtual", "DataLayer", "StoredProcedure"
        public int CallSiteLineNumber { get; set; }  // Line where the call occurs
        public int CallCount { get; set; }  // How many times called within the method

        // SQL-specific information
        public bool IsSqlOperation { get; set; }
        public string SqlCommandText { get; set; }
        public string StoredProcedureName { get; set; }
    }
}
