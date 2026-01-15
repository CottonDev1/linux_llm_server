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
    /// Enhanced call graph builder that tracks bidirectional relationships,
    /// DataLayer calls, and SQL operations with detailed line number tracking.
    /// </summary>
    public class CallGraphBuilder
    {
        private readonly string _projectName;
        private readonly Dictionary<string, List<IMethodSymbol>> _allMethods;

        public CallGraphBuilder(string projectName = null)
        {
            _projectName = projectName ?? "Unknown";
            _allMethods = new Dictionary<string, List<IMethodSymbol>>();
        }

        /// <summary>
        /// Builds an enhanced call graph with bidirectional tracking and SQL detection.
        /// </summary>
        public List<CallRelationship> BuildCallGraph(
            SyntaxNode root,
            SemanticModel semanticModel,
            string filePath,
            Compilation compilation = null)
        {
            var callRelationships = new List<CallRelationship>();

            // Find all method declarations to analyze their bodies
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();

            foreach (var methodDecl in methodDeclarations)
            {
                var callerSymbol = semanticModel.GetDeclaredSymbol(methodDecl);
                if (callerSymbol == null)
                    continue;

                var callerInfo = GetMethodInfo(callerSymbol, methodDecl, filePath);

                // Find all invocations within this method
                var invocations = methodDecl.DescendantNodes().OfType<InvocationExpressionSyntax>();

                foreach (var invocation in invocations)
                {
                    var relationship = AnalyzeInvocation(
                        invocation,
                        callerInfo,
                        semanticModel,
                        filePath,
                        compilation);

                    if (relationship != null)
                    {
                        callRelationships.Add(relationship);
                    }
                }

                // Also find object creations (for SqlCommand, etc.)
                var objectCreations = methodDecl.DescendantNodes().OfType<ObjectCreationExpressionSyntax>();
                foreach (var creation in objectCreations)
                {
                    var sqlRelationship = AnalyzeSqlObjectCreation(
                        creation,
                        callerInfo,
                        semanticModel,
                        filePath);

                    if (sqlRelationship != null)
                    {
                        callRelationships.Add(sqlRelationship);
                    }
                }
            }

            return callRelationships;
        }

        private CallRelationship AnalyzeInvocation(
            InvocationExpressionSyntax invocation,
            (string MethodName, string ClassName, string Namespace, int LineNumber) callerInfo,
            SemanticModel semanticModel,
            string filePath,
            Compilation compilation)
        {
            var callSiteLine = invocation.GetLocation().GetLineSpan().StartLinePosition.Line + 1;
            var symbolInfo = semanticModel.GetSymbolInfo(invocation);

            // Try to resolve the called method symbol
            var calleeSymbol = symbolInfo.Symbol as IMethodSymbol;
            if (calleeSymbol == null && symbolInfo.CandidateSymbols.Length > 0)
            {
                calleeSymbol = symbolInfo.CandidateSymbols.FirstOrDefault() as IMethodSymbol;
            }

            if (calleeSymbol != null)
            {
                var calleeProject = ExtractProjectName(calleeSymbol);
                var calleeFilePath = GetFilePathFromSymbol(calleeSymbol);
                var calleeLineNumber = GetLineNumberFromSymbol(calleeSymbol);

                // Detect DataLayer calls
                var isDataLayerCall = IsDataLayerMethod(calleeSymbol);
                var callType = DetermineCallType(invocation, calleeSymbol, semanticModel, isDataLayerCall);

                // Detect stored procedure calls
                string storedProcName = null;
                string sqlCommand = null;
                if (isDataLayerCall)
                {
                    (storedProcName, sqlCommand) = ExtractSqlInfoFromInvocation(invocation, semanticModel);
                }

                return new CallRelationship
                {
                    // Caller information
                    CallerProject = _projectName,
                    CallerMethod = callerInfo.MethodName,
                    CallerClass = callerInfo.ClassName,
                    CallerNamespace = callerInfo.Namespace,
                    CallerFilePath = filePath,
                    CallerLineNumber = callerInfo.LineNumber,

                    // Callee information
                    CalleeProject = calleeProject,
                    CalleeMethod = calleeSymbol.Name,
                    CalleeClass = calleeSymbol.ContainingType?.Name ?? "Unknown",
                    CalleeNamespace = calleeSymbol.ContainingNamespace?.ToDisplayString() ?? "Unknown",
                    CalleeFilePath = calleeFilePath,
                    CalleeLineNumber = calleeLineNumber,

                    // Call information
                    CallType = callType,
                    CallSiteLineNumber = callSiteLine,
                    CallCount = 1,

                    // SQL-specific
                    IsSqlOperation = isDataLayerCall || !string.IsNullOrEmpty(storedProcName),
                    StoredProcedureName = storedProcName,
                    SqlCommandText = sqlCommand
                };
            }
            else
            {
                // Unresolved symbol - might be dynamic or external
                var (extractedClass, extractedMethod) = GetInvocationTargetInfo(invocation);

                // Check if it looks like a DataLayer call pattern
                var isLikelyDataLayer = extractedClass.Contains("DataLayer") ||
                                       extractedClass.Contains("Repository") ||
                                       extractedClass.Contains("Dal") ||
                                       extractedMethod.Contains("DataLayer");

                return new CallRelationship
                {
                    CallerProject = _projectName,
                    CallerMethod = callerInfo.MethodName,
                    CallerClass = callerInfo.ClassName,
                    CallerNamespace = callerInfo.Namespace,
                    CallerFilePath = filePath,
                    CallerLineNumber = callerInfo.LineNumber,

                    CalleeProject = "External",
                    CalleeMethod = extractedMethod,                    // Now just "AddWithValue" instead of "cmd.Parameters.AddWithValue"
                    CalleeClass = extractedClass,                      // Now "Parameters" or the actual receiver name
                    CalleeNamespace = "Unresolved",
                    CalleeFilePath = "External/Unknown",
                    CalleeLineNumber = 0,

                    CallType = isLikelyDataLayer ? "DataLayer" : "Unresolved",
                    CallSiteLineNumber = callSiteLine,
                    CallCount = 1,

                    IsSqlOperation = isLikelyDataLayer
                };
            }
        }

        private CallRelationship AnalyzeSqlObjectCreation(
            ObjectCreationExpressionSyntax creation,
            (string MethodName, string ClassName, string Namespace, int LineNumber) callerInfo,
            SemanticModel semanticModel,
            string filePath)
        {
            var typeInfo = semanticModel.GetTypeInfo(creation);
            var typeName = typeInfo.Type?.Name;

            // Check if it's a SQL-related object
            var sqlTypes = new HashSet<string> {
                "SqlCommand", "SqlDataAdapter", "SqlConnection",
                "OleDbCommand", "OleDbDataAdapter", "OleDbConnection"
            };

            if (typeName == null || !sqlTypes.Contains(typeName))
                return null;

            var callSiteLine = creation.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            // Extract command text if available
            string commandText = null;
            if (creation.ArgumentList?.Arguments.Count > 0)
            {
                commandText = ExtractStringLiteral(creation.ArgumentList.Arguments[0].Expression, semanticModel);
            }

            // Try to detect if it's a stored procedure
            string storedProcName = null;
            if (commandText != null)
            {
                // Simple heuristic: if it doesn't contain SQL keywords, it's likely a stored proc name
                var sqlKeywords = new[] { "SELECT", "INSERT", "UPDATE", "DELETE", "FROM", "WHERE" };
                if (!sqlKeywords.Any(kw => commandText.ToUpper().Contains(kw)))
                {
                    storedProcName = commandText.Trim();
                }
            }

            return new CallRelationship
            {
                CallerProject = _projectName,
                CallerMethod = callerInfo.MethodName,
                CallerClass = callerInfo.ClassName,
                CallerNamespace = callerInfo.Namespace,
                CallerFilePath = filePath,
                CallerLineNumber = callerInfo.LineNumber,

                CalleeProject = "SQL",
                CalleeMethod = storedProcName ?? "SqlCommand",
                CalleeClass = typeName,
                CalleeNamespace = "System.Data.SqlClient",
                CalleeFilePath = "SQL Operation",
                CalleeLineNumber = 0,

                CallType = "SqlCommand",
                CallSiteLineNumber = callSiteLine,
                CallCount = 1,

                IsSqlOperation = true,
                StoredProcedureName = storedProcName,
                SqlCommandText = commandText
            };
        }

        private bool IsDataLayerMethod(IMethodSymbol method)
        {
            if (method == null) return false;

            // Check class name patterns
            var className = method.ContainingType?.Name ?? "";
            var dataLayerPatterns = new[] {
                "DataLayer", "Repository", "Dal", "DAO", "DataAccess",
                "DbContext", "Context", "Store", "Service"
            };

            if (dataLayerPatterns.Any(pattern => className.Contains(pattern, StringComparison.OrdinalIgnoreCase)))
                return true;

            // Check method name patterns
            var methodPatterns = new[] {
                "Execute", "Query", "GetData", "SaveData", "Insert", "Update", "Delete",
                "Load", "Fetch", "Store", "Persist", "Retrieve"
            };

            return methodPatterns.Any(pattern =>
                method.Name.Contains(pattern, StringComparison.OrdinalIgnoreCase));
        }

        private (string storedProc, string sqlCommand) ExtractSqlInfoFromInvocation(
            InvocationExpressionSyntax invocation,
            SemanticModel semanticModel)
        {
            string storedProc = null;
            string sqlCommand = null;

            // Check arguments for string literals that might be SQL or stored proc names
            if (invocation.ArgumentList != null)
            {
                foreach (var arg in invocation.ArgumentList.Arguments)
                {
                    var literal = ExtractStringLiteral(arg.Expression, semanticModel);
                    if (!string.IsNullOrEmpty(literal))
                    {
                        // Simple heuristic: check if it looks like SQL or a stored proc name
                        var upperLiteral = literal.ToUpper();
                        if (upperLiteral.Contains("SELECT") || upperLiteral.Contains("INSERT") ||
                            upperLiteral.Contains("UPDATE") || upperLiteral.Contains("DELETE"))
                        {
                            sqlCommand = literal;
                        }
                        else if (literal.StartsWith("usp_") || literal.StartsWith("sp_") ||
                                !literal.Contains(" "))
                        {
                            storedProc = literal;
                        }
                    }
                }
            }

            return (storedProc, sqlCommand);
        }

        private string ExtractStringLiteral(ExpressionSyntax expression, SemanticModel semanticModel)
        {
            // Handle direct string literals
            if (expression is LiteralExpressionSyntax literal &&
                literal.IsKind(SyntaxKind.StringLiteralExpression))
            {
                return literal.Token.ValueText;
            }

            // Handle string concatenation
            if (expression is BinaryExpressionSyntax binary &&
                binary.IsKind(SyntaxKind.AddExpression))
            {
                var left = ExtractStringLiteral(binary.Left, semanticModel);
                var right = ExtractStringLiteral(binary.Right, semanticModel);
                if (left != null && right != null)
                    return left + right;
            }

            // Handle interpolated strings
            if (expression is InterpolatedStringExpressionSyntax interpolated)
            {
                return interpolated.ToString();
            }

            // Try to resolve constant values
            var constantValue = semanticModel.GetConstantValue(expression);
            if (constantValue.HasValue && constantValue.Value is string str)
            {
                return str;
            }

            return null;
        }

        private string DetermineCallType(
            InvocationExpressionSyntax invocation,
            IMethodSymbol calleeSymbol,
            SemanticModel semanticModel,
            bool isDataLayerCall)
        {
            if (isDataLayerCall)
                return "DataLayer";

            // Check if it's a delegate invocation
            var expression = invocation.Expression;
            var typeInfo = semanticModel.GetTypeInfo(expression);

            if (typeInfo.Type?.TypeKind == TypeKind.Delegate)
                return "Delegate";

            // Check if the method is virtual/abstract/override
            if (calleeSymbol.IsVirtual || calleeSymbol.IsAbstract || calleeSymbol.IsOverride)
                return "Virtual";

            // Check if it's an event invocation
            var symbolInfo = semanticModel.GetSymbolInfo(expression);
            if (symbolInfo.Symbol?.Kind == SymbolKind.Event)
                return "Event";

            // Default is direct method call
            return "Direct";
        }

        private string ExtractProjectName(ISymbol symbol)
        {
            // Try to extract project name from assembly name
            var assemblyName = symbol.ContainingAssembly?.Name;
            if (!string.IsNullOrEmpty(assemblyName))
            {
                // Common project name patterns
                if (assemblyName.Contains("Gin")) return "Gin";
                if (assemblyName.Contains("Warehouse")) return "Warehouse";
                if (assemblyName.Contains("Marketing")) return "Marketing";
                if (assemblyName.Contains("EWR") && assemblyName.Contains("Library")) return "EWR Library";
                if (assemblyName.Contains("Provider")) return "Provider";

                return assemblyName;
            }

            return _projectName ?? "Unknown";
        }

        private (string MethodName, string ClassName, string Namespace, int LineNumber) GetMethodInfo(
            IMethodSymbol methodSymbol,
            MethodDeclarationSyntax syntax,
            string filePath)
        {
            var lineNumber = syntax.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            return (
                MethodName: methodSymbol.Name,
                ClassName: methodSymbol.ContainingType?.Name ?? "Unknown",
                Namespace: methodSymbol.ContainingNamespace?.ToDisplayString() ?? "Unknown",
                LineNumber: lineNumber
            );
        }

        private string GetFilePathFromSymbol(ISymbol symbol)
        {
            var locations = symbol.Locations;
            if (locations.Length > 0 && locations[0].IsInSource)
            {
                return locations[0].SourceTree?.FilePath ?? "Unknown";
            }

            // Symbol is in metadata (compiled assembly)
            return $"External:{symbol.ContainingAssembly?.Name ?? "Unknown"}";
        }

        private int GetLineNumberFromSymbol(ISymbol symbol)
        {
            var locations = symbol.Locations;
            if (locations.Length > 0 && locations[0].IsInSource)
            {
                return locations[0].GetLineSpan().StartLinePosition.Line + 1;
            }

            return 0;
        }

        /// <summary>
        /// Extract class name and method name separately from an invocation expression.
        /// Returns a tuple of (ClassName, MethodName) instead of a combined string.
        /// </summary>
        private (string ClassName, string MethodName) GetInvocationTargetInfo(InvocationExpressionSyntax invocation)
        {
            var expression = invocation.Expression;

            switch (expression)
            {
                case MemberAccessExpressionSyntax memberAccess:
                    // Extract just the method name (the Name part of member access)
                    var methodName = memberAccess.Name.Identifier.Text;

                    // Extract the receiver/class name
                    var receiver = memberAccess.Expression;
                    string className;

                    if (receiver is IdentifierNameSyntax identifier)
                    {
                        // Simple case: variable.Method() - use variable name as class hint
                        className = identifier.Identifier.Text;
                    }
                    else if (receiver is MemberAccessExpressionSyntax nestedAccess)
                    {
                        // Nested case: obj.Property.Method() - use last identifier before method
                        className = nestedAccess.Name.Identifier.Text;
                    }
                    else if (receiver is ThisExpressionSyntax)
                    {
                        className = "this";
                    }
                    else if (receiver is BaseExpressionSyntax)
                    {
                        className = "base";
                    }
                    else
                    {
                        // Fall back to expression string for complex cases
                        className = receiver.ToString();
                    }

                    return (className, methodName);

                case IdentifierNameSyntax simpleIdentifier:
                    // Direct method call: Method()
                    return ("", simpleIdentifier.Identifier.Text);

                case GenericNameSyntax genericName:
                    // Generic method call: Method<T>()
                    return ("", genericName.Identifier.Text);

                default:
                    return ("Unknown", expression.ToString());
            }
        }
    }
}