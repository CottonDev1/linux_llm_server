namespace RoslynCodeAnalyzer.Models
{
    public class DatabaseOperationInfo
    {
        public string OperationType { get; set; }  // "SqlCommand", "StoredProcedure", "DataAdapter"
        public string MethodName { get; set; }
        public string ClassName { get; set; }
        public string Namespace { get; set; }
        public string FilePath { get; set; }
        public int LineNumber { get; set; }
        public string CommandText { get; set; }  // SQL query or stored proc name
        public string CommandType { get; set; }  // "Text", "StoredProcedure", "TableDirect"
        public string TableName { get; set; }  // If detectable from query
        public string[] Parameters { get; set; }  // Parameter names
    }
}
