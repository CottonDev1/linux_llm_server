namespace RoslynCodeAnalyzer.Models
{
    public class EventHandlerInfo
    {
        public string EventName { get; set; }
        public string HandlerMethod { get; set; }
        public string HandlerClass { get; set; }
        public string HandlerNamespace { get; set; }
        public string FilePath { get; set; }
        public int LineNumber { get; set; }
        public string EventSource { get; set; }  // UI element or class that raises event
        public string SubscriptionType { get; set; }  // "+=", "XAML", "Constructor"
        public string UIElementType { get; set; }  // "Button", "TextBox", etc. (if WPF)
    }
}
