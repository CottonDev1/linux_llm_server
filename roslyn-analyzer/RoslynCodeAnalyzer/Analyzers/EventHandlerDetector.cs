using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using RoslynCodeAnalyzer.Models;

namespace RoslynCodeAnalyzer.Analyzers
{
    /// <summary>
    /// Detects event handler subscriptions and registrations, particularly for WPF/XAML applications.
    /// Identifies both explicit event subscriptions (+=) and implicit handler methods.
    /// </summary>
    public class EventHandlerDetector
    {
        /// <summary>
        /// Detects all event handler subscriptions and handler methods in the syntax tree.
        /// </summary>
        /// <param name="root">Root of the syntax tree to analyze</param>
        /// <param name="semanticModel">Semantic model for symbol resolution</param>
        /// <param name="filePath">Path to the file being analyzed</param>
        /// <returns>List of event handler information found in the file</returns>
        public List<EventHandlerInfo> DetectEventHandlers(SyntaxNode root, SemanticModel semanticModel, string filePath)
        {
            var eventHandlers = new List<EventHandlerInfo>();

            // 1. Find explicit event subscriptions using += operator
            var addAssignments = root.DescendantNodes()
                .OfType<AssignmentExpressionSyntax>()
                .Where(a => a.Kind() == SyntaxKind.AddAssignmentExpression);

            foreach (var assignment in addAssignments)
            {
                var eventInfo = AnalyzeEventSubscription(assignment, semanticModel, filePath);
                if (eventInfo != null)
                {
                    eventHandlers.Add(eventInfo);
                }
            }

            // 2. Find methods that look like event handlers (EventArgs pattern)
            var methods = root.DescendantNodes().OfType<MethodDeclarationSyntax>();

            foreach (var method in methods)
            {
                var handlerInfo = AnalyzeMethodAsEventHandler(method, semanticModel, filePath);
                if (handlerInfo != null)
                {
                    eventHandlers.Add(handlerInfo);
                }
            }

            return eventHandlers;
        }

        /// <summary>
        /// Analyzes an add-assignment expression (+=) to extract event subscription details.
        /// </summary>
        private EventHandlerInfo AnalyzeEventSubscription(
            AssignmentExpressionSyntax assignment,
            SemanticModel semanticModel,
            string filePath)
        {
            // Left side should be the event being subscribed to
            var eventExpression = assignment.Left;
            var handlerExpression = assignment.Right;

            // Get symbol information for the event
            var eventSymbol = semanticModel.GetSymbolInfo(eventExpression).Symbol as IEventSymbol;
            var leftSymbol = semanticModel.GetSymbolInfo(eventExpression).Symbol;

            // If not an event, it might be a delegate field or property
            string eventName = null;
            string eventSource = null;
            string uiElementType = null;

            if (eventSymbol != null)
            {
                eventName = eventSymbol.Name;
                eventSource = GetEventSource(eventExpression, semanticModel);
                uiElementType = eventSymbol.ContainingType?.Name;
            }
            else if (leftSymbol != null)
            {
                // Might be a delegate field or property
                eventName = leftSymbol.Name;
                eventSource = GetEventSource(eventExpression, semanticModel);
                uiElementType = leftSymbol.ContainingType?.Name;
            }
            else
            {
                // Fallback: use expression text
                eventName = eventExpression.ToString();
                eventSource = "Unknown";
            }

            // Get handler method information
            var handlerSymbol = semanticModel.GetSymbolInfo(handlerExpression).Symbol as IMethodSymbol;
            string handlerMethod = handlerSymbol?.Name ?? handlerExpression.ToString();
            string handlerClass = handlerSymbol?.ContainingType?.Name ?? GetContainingClass(assignment);
            string handlerNamespace = handlerSymbol?.ContainingNamespace?.ToDisplayString() ?? GetContainingNamespace(assignment);

            var lineNumber = assignment.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            return new EventHandlerInfo
            {
                EventName = eventName,
                HandlerMethod = handlerMethod,
                HandlerClass = handlerClass,
                HandlerNamespace = handlerNamespace,
                FilePath = filePath,
                LineNumber = lineNumber,
                EventSource = eventSource,
                SubscriptionType = "+=",
                UIElementType = uiElementType
            };
        }

        /// <summary>
        /// Analyzes a method to determine if it appears to be an event handler.
        /// Event handlers typically have (object sender, EventArgs e) signature.
        /// </summary>
        private EventHandlerInfo AnalyzeMethodAsEventHandler(
            MethodDeclarationSyntax method,
            SemanticModel semanticModel,
            string filePath)
        {
            // Check if method signature matches event handler pattern
            if (!IsEventHandlerSignature(method, semanticModel))
            {
                return null;
            }

            var methodSymbol = semanticModel.GetDeclaredSymbol(method);
            if (methodSymbol == null)
                return null;

            var lineNumber = method.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            // Try to infer event name from method name (common patterns)
            string eventName = InferEventNameFromMethodName(method.Identifier.Text);

            return new EventHandlerInfo
            {
                EventName = eventName ?? "Unknown",
                HandlerMethod = method.Identifier.Text,
                HandlerClass = methodSymbol.ContainingType?.Name ?? "Unknown",
                HandlerNamespace = methodSymbol.ContainingNamespace?.ToDisplayString() ?? "Unknown",
                FilePath = filePath,
                LineNumber = lineNumber,
                EventSource = "Inferred",
                SubscriptionType = "Method",
                UIElementType = InferUIElementType(method.Identifier.Text)
            };
        }

        /// <summary>
        /// Checks if a method has the standard event handler signature:
        /// void MethodName(object sender, EventArgs e) or similar.
        /// </summary>
        private bool IsEventHandlerSignature(MethodDeclarationSyntax method, SemanticModel semanticModel)
        {
            // Must have exactly 2 parameters
            if (method.ParameterList.Parameters.Count != 2)
                return false;

            // Must return void
            if (method.ReturnType.ToString() != "void")
                return false;

            // Get parameter symbols
            var param1 = method.ParameterList.Parameters[0];
            var param2 = method.ParameterList.Parameters[1];

            var param1Symbol = semanticModel.GetDeclaredSymbol(param1);
            var param2Symbol = semanticModel.GetDeclaredSymbol(param2);

            // First parameter should be 'object' or similar
            var param1Type = param1Symbol?.Type;
            if (param1Type == null)
                return false;

            // Second parameter should be EventArgs or derived type
            var param2Type = param2Symbol?.Type;
            if (param2Type == null)
                return false;

            // Check if second parameter is EventArgs or derived from it
            bool isEventArgsType = param2Type.Name.Contains("EventArgs") ||
                                   InheritsFromEventArgs(param2Type);

            return isEventArgsType;
        }

        /// <summary>
        /// Checks if a type inherits from EventArgs.
        /// </summary>
        private bool InheritsFromEventArgs(ITypeSymbol type)
        {
            var current = type.BaseType;
            while (current != null)
            {
                if (current.Name == "EventArgs")
                    return true;
                current = current.BaseType;
            }
            return false;
        }

        /// <summary>
        /// Extracts the event source (e.g., "btnSubmit", "txtName") from the event expression.
        /// </summary>
        private string GetEventSource(ExpressionSyntax eventExpression, SemanticModel semanticModel)
        {
            if (eventExpression is MemberAccessExpressionSyntax memberAccess)
            {
                // Get the object/instance the event is on (e.g., "btnSubmit" in "btnSubmit.Click")
                var instanceExpression = memberAccess.Expression;
                return instanceExpression.ToString();
            }

            return eventExpression.ToString();
        }

        /// <summary>
        /// Gets the containing class name from a syntax node.
        /// </summary>
        private string GetContainingClass(SyntaxNode node)
        {
            var classDecl = node.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault();
            return classDecl?.Identifier.Text ?? "Unknown";
        }

        /// <summary>
        /// Gets the containing namespace from a syntax node.
        /// </summary>
        private string GetContainingNamespace(SyntaxNode node)
        {
            var namespaceDecl = node.Ancestors().OfType<NamespaceDeclarationSyntax>().FirstOrDefault();
            if (namespaceDecl != null)
                return namespaceDecl.Name.ToString();

            var fileScopedNamespace = node.Ancestors().OfType<FileScopedNamespaceDeclarationSyntax>().FirstOrDefault();
            if (fileScopedNamespace != null)
                return fileScopedNamespace.Name.ToString();

            return "Global";
        }

        /// <summary>
        /// Infers the event name from the method name using common naming patterns.
        /// Examples: "btnSubmit_Click" -> "Click", "OnDataLoaded" -> "DataLoaded"
        /// </summary>
        private string InferEventNameFromMethodName(string methodName)
        {
            // Pattern 1: controlName_EventName (e.g., "btnSubmit_Click")
            var underscoreIndex = methodName.IndexOf('_');
            if (underscoreIndex > 0 && underscoreIndex < methodName.Length - 1)
            {
                return methodName.Substring(underscoreIndex + 1);
            }

            // Pattern 2: OnEventName (e.g., "OnLoaded", "OnClick")
            if (methodName.StartsWith("On") && methodName.Length > 2)
            {
                return methodName.Substring(2);
            }

            // Pattern 3: HandleEventName (e.g., "HandleClick")
            if (methodName.StartsWith("Handle") && methodName.Length > 6)
            {
                return methodName.Substring(6);
            }

            return null;
        }

        /// <summary>
        /// Infers the UI element type from the method name.
        /// Examples: "btnSubmit_Click" -> "Button", "txtName_TextChanged" -> "TextBox"
        /// </summary>
        private string InferUIElementType(string methodName)
        {
            var underscoreIndex = methodName.IndexOf('_');
            if (underscoreIndex <= 0)
                return null;

            var controlName = methodName.Substring(0, underscoreIndex);

            // Common WPF/WinForms prefixes
            if (controlName.StartsWith("btn"))
                return "Button";
            if (controlName.StartsWith("txt"))
                return "TextBox";
            if (controlName.StartsWith("lbl"))
                return "Label";
            if (controlName.StartsWith("cmb") || controlName.StartsWith("cbo"))
                return "ComboBox";
            if (controlName.StartsWith("chk"))
                return "CheckBox";
            if (controlName.StartsWith("lst"))
                return "ListBox";
            if (controlName.StartsWith("dgv") || controlName.StartsWith("grid"))
                return "DataGrid";
            if (controlName.StartsWith("mnu"))
                return "Menu";

            return null;
        }
    }
}
