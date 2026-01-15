using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using RoslynCodeAnalyzer.Models;

namespace RoslynCodeAnalyzer.Analyzers
{
    public class MethodExtractor
    {
        public List<MethodInfo> ExtractMethods(SyntaxNode root, SemanticModel semanticModel, string filePath)
        {
            var methods = new List<MethodInfo>();

            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();

            foreach (var methodDecl in methodDeclarations)
            {
                var methodSymbol = semanticModel.GetDeclaredSymbol(methodDecl) as IMethodSymbol;
                if (methodSymbol == null) continue;

                var containingClass = methodSymbol.ContainingType;
                var lineSpan = methodDecl.GetLocation().GetLineSpan();

                var methodInfo = new MethodInfo
                {
                    MethodName = methodDecl.Identifier.Text,
                    ClassName = containingClass?.Name ?? "",
                    Namespace = containingClass?.ContainingNamespace?.ToDisplayString() ?? "",
                    FullName = methodSymbol.ToDisplayString(),
                    FilePath = filePath,
                    LineNumber = lineSpan.StartLinePosition.Line + 1,
                    LineCount = lineSpan.EndLinePosition.Line - lineSpan.StartLinePosition.Line + 1,
                    Accessibility = methodSymbol.DeclaredAccessibility.ToString(),
                    IsStatic = methodSymbol.IsStatic,
                    IsAsync = methodSymbol.IsAsync,
                    IsVirtual = methodSymbol.IsVirtual,
                    IsOverride = methodSymbol.IsOverride,
                    ReturnType = methodSymbol.ReturnType.ToDisplayString()
                };

                // Extract parameters
                foreach (var param in methodSymbol.Parameters)
                {
                    methodInfo.Parameters.Add(new ParameterInfo
                    {
                        Name = param.Name,
                        Type = param.Type.ToDisplayString(),
                        HasDefaultValue = param.HasExplicitDefaultValue,
                        DefaultValue = param.HasExplicitDefaultValue ? param.ExplicitDefaultValue?.ToString() : null,
                        IsRef = param.RefKind == RefKind.Ref,
                        IsOut = param.RefKind == RefKind.Out
                    });
                }

                // Extract local variables
                var localDeclarations = methodDecl.DescendantNodes().OfType<LocalDeclarationStatementSyntax>();
                foreach (var localDecl in localDeclarations)
                {
                    foreach (var variable in localDecl.Declaration.Variables)
                    {
                        methodInfo.LocalVariables.Add($"{localDecl.Declaration.Type} {variable.Identifier.Text}");
                    }
                }

                // Calculate cyclomatic complexity
                methodInfo.CyclomaticComplexity = CalculateCyclomaticComplexity(methodDecl);

                // Extract XML documentation summary
                var trivia = methodDecl.GetLeadingTrivia();
                var xmlTrivia = trivia.FirstOrDefault(t => t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) ||
                                                           t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia));
                if (xmlTrivia != default(SyntaxTrivia))
                {
                    methodInfo.Summary = ExtractSummaryFromXml(xmlTrivia.ToString());
                }

                methods.Add(methodInfo);
            }

            return methods;
        }

        private int CalculateCyclomaticComplexity(MethodDeclarationSyntax methodDecl)
        {
            // Start with base complexity of 1
            int complexity = 1;

            // Count decision points
            var descendants = methodDecl.DescendantNodes();

            // If statements
            complexity += descendants.OfType<IfStatementSyntax>().Count();

            // While loops
            complexity += descendants.OfType<WhileStatementSyntax>().Count();

            // For loops
            complexity += descendants.OfType<ForStatementSyntax>().Count();

            // Foreach loops
            complexity += descendants.OfType<ForEachStatementSyntax>().Count();

            // Switch cases (each case adds 1)
            complexity += descendants.OfType<SwitchSectionSyntax>().Count();

            // Catch blocks
            complexity += descendants.OfType<CatchClauseSyntax>().Count();

            // Conditional expressions (ternary)
            complexity += descendants.OfType<ConditionalExpressionSyntax>().Count();

            // Null-coalescing operators
            complexity += descendants.OfType<BinaryExpressionSyntax>()
                .Count(b => b.IsKind(SyntaxKind.CoalesceExpression));

            // Logical AND and OR in conditions
            complexity += descendants.OfType<BinaryExpressionSyntax>()
                .Count(b => b.IsKind(SyntaxKind.LogicalAndExpression) ||
                           b.IsKind(SyntaxKind.LogicalOrExpression));

            return complexity;
        }

        private string ExtractSummaryFromXml(string xml)
        {
            var startTag = "<summary>";
            var endTag = "</summary>";
            var startIndex = xml.IndexOf(startTag);
            var endIndex = xml.IndexOf(endTag);

            if (startIndex >= 0 && endIndex > startIndex)
            {
                var summary = xml.Substring(startIndex + startTag.Length, endIndex - startIndex - startTag.Length);
                return summary.Replace("///", "").Trim();
            }

            return "";
        }
    }
}
