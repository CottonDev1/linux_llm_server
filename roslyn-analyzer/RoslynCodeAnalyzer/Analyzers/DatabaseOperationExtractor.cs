using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using RoslynCodeAnalyzer.Models;

namespace RoslynCodeAnalyzer.Analyzers
{
    /// <summary>
    /// Extracts database operations including SQL commands, stored procedure calls,
    /// and data adapter usage. Analyzes ADO.NET patterns commonly used in enterprise applications.
    /// </summary>
    public class DatabaseOperationExtractor
    {
        private static readonly HashSet<string> SqlCommandTypes = new HashSet<string>
        {
            "SqlCommand", "OleDbCommand", "OdbcCommand", "MySqlCommand",
            "NpgsqlCommand", "SqliteCommand"
        };

        private static readonly HashSet<string> SqlAdapterTypes = new HashSet<string>
        {
            "SqlDataAdapter", "OleDbDataAdapter", "OdbcDataAdapter",
            "MySqlDataAdapter", "NpgsqlDataAdapter"
        };

        /// <summary>
        /// Extracts all database operations from the syntax tree.
        /// </summary>
        /// <param name="root">Root of the syntax tree to analyze</param>
        /// <param name="semanticModel">Semantic model for symbol resolution</param>
        /// <param name="filePath">Path to the file being analyzed</param>
        /// <returns>List of database operations found in the file</returns>
        public List<DatabaseOperationInfo> ExtractDatabaseOperations(SyntaxNode root, SemanticModel semanticModel, string filePath)
        {
            var dbOperations = new List<DatabaseOperationInfo>();

            // 1. Find SqlCommand/OleDbCommand object creations
            var objectCreations = root.DescendantNodes().OfType<ObjectCreationExpressionSyntax>();
            foreach (var creation in objectCreations)
            {
                var dbOp = AnalyzeObjectCreation(creation, semanticModel, filePath);
                if (dbOp != null)
                {
                    dbOperations.Add(dbOp);
                }
            }

            // 2. Find CommandText property assignments
            var assignments = root.DescendantNodes().OfType<AssignmentExpressionSyntax>();
            foreach (var assignment in assignments)
            {
                var dbOp = AnalyzeCommandTextAssignment(assignment, semanticModel, filePath);
                if (dbOp != null)
                {
                    dbOperations.Add(dbOp);
                }
            }

            // 3. Find SqlDataAdapter creations
            foreach (var creation in objectCreations)
            {
                var dbOp = AnalyzeDataAdapterCreation(creation, semanticModel, filePath);
                if (dbOp != null)
                {
                    dbOperations.Add(dbOp);
                }
            }

            return dbOperations;
        }

        /// <summary>
        /// Analyzes object creation expressions to detect SqlCommand instantiation.
        /// </summary>
        private DatabaseOperationInfo AnalyzeObjectCreation(
            ObjectCreationExpressionSyntax creation,
            SemanticModel semanticModel,
            string filePath)
        {
            var typeInfo = semanticModel.GetTypeInfo(creation);
            var typeName = typeInfo.Type?.Name;

            if (typeName == null || !SqlCommandTypes.Contains(typeName))
                return null;

            // Extract command text from constructor arguments
            string commandText = null;
            string commandType = "Text"; // Default

            if (creation.ArgumentList?.Arguments.Count > 0)
            {
                var firstArg = creation.ArgumentList.Arguments[0];
                commandText = ExtractStringLiteral(firstArg.Expression, semanticModel);
            }

            // Get containing method information
            var methodContext = GetContainingMethodContext(creation, semanticModel);

            // Extract parameters from surrounding code
            var parameters = ExtractParametersNearCommand(creation, semanticModel);

            // Detect command type (Text, StoredProcedure, TableDirect)
            commandType = DetectCommandType(creation, semanticModel);

            // Try to detect table name from SQL
            string tableName = null;
            if (commandText != null)
            {
                tableName = ExtractTableNameFromSql(commandText, commandType);
            }

            var lineNumber = creation.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            return new DatabaseOperationInfo
            {
                OperationType = typeName,
                MethodName = methodContext.MethodName,
                ClassName = methodContext.ClassName,
                Namespace = methodContext.Namespace,
                FilePath = filePath,
                LineNumber = lineNumber,
                CommandText = commandText ?? "Unknown",
                CommandType = commandType,
                TableName = tableName,
                Parameters = parameters.ToArray()
            };
        }

        /// <summary>
        /// Analyzes assignments to CommandText property.
        /// </summary>
        private DatabaseOperationInfo AnalyzeCommandTextAssignment(
            AssignmentExpressionSyntax assignment,
            SemanticModel semanticModel,
            string filePath)
        {
            // Check if left side is CommandText property
            if (!(assignment.Left is MemberAccessExpressionSyntax memberAccess))
                return null;

            if (memberAccess.Name.Identifier.Text != "CommandText")
                return null;

            // Verify the object is a SQL command type
            var objectSymbol = semanticModel.GetSymbolInfo(memberAccess.Expression).Symbol;
            var objectType = objectSymbol?.GetType()?.Name ??
                            semanticModel.GetTypeInfo(memberAccess.Expression).Type?.Name;

            // Extract command text
            var commandText = ExtractStringLiteral(assignment.Right, semanticModel);

            var methodContext = GetContainingMethodContext(assignment, semanticModel);
            var parameters = ExtractParametersNearCommand(assignment, semanticModel);
            var commandType = "Text"; // Default, could be overridden

            string tableName = null;
            if (commandText != null)
            {
                tableName = ExtractTableNameFromSql(commandText, commandType);
            }

            var lineNumber = assignment.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            return new DatabaseOperationInfo
            {
                OperationType = "SqlCommand",
                MethodName = methodContext.MethodName,
                ClassName = methodContext.ClassName,
                Namespace = methodContext.Namespace,
                FilePath = filePath,
                LineNumber = lineNumber,
                CommandText = commandText ?? "Unknown",
                CommandType = commandType,
                TableName = tableName,
                Parameters = parameters.ToArray()
            };
        }

        /// <summary>
        /// Analyzes SqlDataAdapter object creation.
        /// </summary>
        private DatabaseOperationInfo AnalyzeDataAdapterCreation(
            ObjectCreationExpressionSyntax creation,
            SemanticModel semanticModel,
            string filePath)
        {
            var typeInfo = semanticModel.GetTypeInfo(creation);
            var typeName = typeInfo.Type?.Name;

            if (typeName == null || !SqlAdapterTypes.Contains(typeName))
                return null;

            string commandText = null;

            // DataAdapters often have SQL in first constructor argument
            if (creation.ArgumentList?.Arguments.Count > 0)
            {
                var firstArg = creation.ArgumentList.Arguments[0];
                commandText = ExtractStringLiteral(firstArg.Expression, semanticModel);
            }

            var methodContext = GetContainingMethodContext(creation, semanticModel);
            var parameters = ExtractParametersNearCommand(creation, semanticModel);

            string tableName = null;
            if (commandText != null)
            {
                tableName = ExtractTableNameFromSql(commandText, "Text");
            }

            var lineNumber = creation.GetLocation().GetLineSpan().StartLinePosition.Line + 1;

            return new DatabaseOperationInfo
            {
                OperationType = typeName,
                MethodName = methodContext.MethodName,
                ClassName = methodContext.ClassName,
                Namespace = methodContext.Namespace,
                FilePath = filePath,
                LineNumber = lineNumber,
                CommandText = commandText ?? "Unknown",
                CommandType = "Text",
                TableName = tableName,
                Parameters = parameters.ToArray()
            };
        }

        /// <summary>
        /// Extracts string literal value from an expression (handles concatenation, verbatim strings, etc.)
        /// </summary>
        private string ExtractStringLiteral(ExpressionSyntax expression, SemanticModel semanticModel)
        {
            // Handle direct string literals
            if (expression is LiteralExpressionSyntax literal && literal.IsKind(SyntaxKind.StringLiteralExpression))
            {
                return literal.Token.ValueText;
            }

            // Handle string concatenation with + operator
            if (expression is BinaryExpressionSyntax binary && binary.IsKind(SyntaxKind.AddExpression))
            {
                var left = ExtractStringLiteral(binary.Left, semanticModel);
                var right = ExtractStringLiteral(binary.Right, semanticModel);
                if (left != null && right != null)
                    return left + right;
            }

            // Handle interpolated strings
            if (expression is InterpolatedStringExpressionSyntax interpolated)
            {
                return interpolated.ToString(); // Return the template
            }

            // Handle variable references (try to resolve constant values)
            var constantValue = semanticModel.GetConstantValue(expression);
            if (constantValue.HasValue && constantValue.Value is string str)
            {
                return str;
            }

            return null;
        }

        /// <summary>
        /// Gets information about the containing method context.
        /// </summary>
        private (string MethodName, string ClassName, string Namespace) GetContainingMethodContext(
            SyntaxNode node,
            SemanticModel semanticModel)
        {
            var methodDecl = node.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();
            if (methodDecl != null)
            {
                var methodSymbol = semanticModel.GetDeclaredSymbol(methodDecl);
                return (
                    MethodName: methodSymbol?.Name ?? methodDecl.Identifier.Text,
                    ClassName: methodSymbol?.ContainingType?.Name ?? "Unknown",
                    Namespace: methodSymbol?.ContainingNamespace?.ToDisplayString() ?? "Unknown"
                );
            }

            // Might be in a property, constructor, etc.
            var classDecl = node.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault();
            var namespaceDecl = node.Ancestors().OfType<NamespaceDeclarationSyntax>().FirstOrDefault();

            return (
                MethodName: "Unknown",
                ClassName: classDecl?.Identifier.Text ?? "Unknown",
                Namespace: namespaceDecl?.Name.ToString() ?? "Unknown"
            );
        }

        /// <summary>
        /// Extracts parameter names from nearby SqlParameter or parameter.Add() calls.
        /// </summary>
        private List<string> ExtractParametersNearCommand(SyntaxNode commandNode, SemanticModel semanticModel)
        {
            var parameters = new List<string>();

            // Look in the containing method for parameter additions
            var containingMethod = commandNode.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();
            if (containingMethod == null)
                return parameters;

            // Find invocations that add parameters
            var invocations = containingMethod.DescendantNodes().OfType<InvocationExpressionSyntax>();

            foreach (var invocation in invocations)
            {
                // Check for Parameters.Add() or Parameters.AddWithValue() patterns
                if (invocation.Expression is MemberAccessExpressionSyntax memberAccess)
                {
                    var methodName = memberAccess.Name.Identifier.Text;
                    if (methodName == "Add" || methodName == "AddWithValue")
                    {
                        // Check if it's on a Parameters collection
                        var objectExpr = memberAccess.Expression;
                        if (objectExpr is MemberAccessExpressionSyntax innerMember &&
                            innerMember.Name.Identifier.Text == "Parameters")
                        {
                            // Extract parameter name from first argument
                            if (invocation.ArgumentList.Arguments.Count > 0)
                            {
                                var paramName = ExtractStringLiteral(
                                    invocation.ArgumentList.Arguments[0].Expression,
                                    semanticModel);
                                if (paramName != null)
                                {
                                    parameters.Add(paramName);
                                }
                            }
                        }
                    }
                }

                // Check for SqlParameter object creation
                var objectCreations = invocation.DescendantNodes().OfType<ObjectCreationExpressionSyntax>();
                foreach (var creation in objectCreations)
                {
                    var type = semanticModel.GetTypeInfo(creation).Type;
                    if (type?.Name.Contains("Parameter") == true)
                    {
                        if (creation.ArgumentList?.Arguments.Count > 0)
                        {
                            var paramName = ExtractStringLiteral(
                                creation.ArgumentList.Arguments[0].Expression,
                                semanticModel);
                            if (paramName != null)
                            {
                                parameters.Add(paramName);
                            }
                        }
                    }
                }
            }

            return parameters.Distinct().ToList();
        }

        /// <summary>
        /// Detects the CommandType (Text, StoredProcedure, TableDirect) by looking for
        /// assignments to the CommandType property.
        /// </summary>
        private string DetectCommandType(SyntaxNode commandNode, SemanticModel semanticModel)
        {
            // Look for CommandType assignment in the same method
            var containingMethod = commandNode.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();
            if (containingMethod == null)
                return "Text";

            var assignments = containingMethod.DescendantNodes().OfType<AssignmentExpressionSyntax>();

            foreach (var assignment in assignments)
            {
                if (assignment.Left is MemberAccessExpressionSyntax memberAccess &&
                    memberAccess.Name.Identifier.Text == "CommandType")
                {
                    // Get the value being assigned
                    if (assignment.Right is MemberAccessExpressionSyntax valueAccess)
                    {
                        return valueAccess.Name.Identifier.Text; // "StoredProcedure", "Text", etc.
                    }
                }
            }

            return "Text"; // Default
        }

        /// <summary>
        /// Attempts to extract table name from SQL query text using simple pattern matching.
        /// </summary>
        private string ExtractTableNameFromSql(string sql, string commandType)
        {
            if (string.IsNullOrWhiteSpace(sql))
                return null;

            // If it's a stored procedure, the entire text is likely the proc name
            if (commandType == "StoredProcedure")
            {
                return sql.Trim();
            }

            // Simple regex patterns for common SQL operations
            var patterns = new[]
            {
                @"FROM\s+(\[?\w+\]?\.?\[?\w+\]?)",           // SELECT ... FROM TableName
                @"INTO\s+(\[?\w+\]?\.?\[?\w+\]?)",           // INSERT INTO TableName
                @"UPDATE\s+(\[?\w+\]?\.?\[?\w+\]?)",         // UPDATE TableName
                @"DELETE\s+FROM\s+(\[?\w+\]?\.?\[?\w+\]?)",  // DELETE FROM TableName
                @"JOIN\s+(\[?\w+\]?\.?\[?\w+\]?)",           // JOIN TableName
            };

            foreach (var pattern in patterns)
            {
                var match = Regex.Match(sql, pattern, RegexOptions.IgnoreCase);
                if (match.Success && match.Groups.Count > 1)
                {
                    return match.Groups[1].Value.Trim('[', ']');
                }
            }

            return null;
        }
    }
}
