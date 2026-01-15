using System.Collections.Generic;

namespace RoslynCodeAnalyzer.Models
{
    public class ClassInfo
    {
        public string ClassName { get; set; }
        public string Namespace { get; set; }
        public string FullName { get; set; }
        public string FilePath { get; set; }
        public int LineNumber { get; set; }
        public string Accessibility { get; set; }
        public bool IsStatic { get; set; }
        public bool IsAbstract { get; set; }
        public bool IsSealed { get; set; }
        public string BaseClass { get; set; }
        public List<string> Interfaces { get; set; } = new List<string>();
        public List<string> Fields { get; set; } = new List<string>();
        public List<string> Properties { get; set; } = new List<string>();
        public List<string> Methods { get; set; } = new List<string>();
        public string Summary { get; set; }
    }
}
