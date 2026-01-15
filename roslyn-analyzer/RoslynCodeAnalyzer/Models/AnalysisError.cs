namespace RoslynCodeAnalyzer.Models
{
    public class AnalysisError
    {
        public string FilePath { get; set; }
        public string Message { get; set; }
        public string StackTrace { get; set; }
    }
}
