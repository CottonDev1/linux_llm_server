using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using RoslynCodeAnalyzer.Models;

namespace RoslynCodeAnalyzer.Analyzers
{
    public class ClassExtractor
    {
        public List<ClassInfo> ExtractClasses(SyntaxNode root, SemanticModel semanticModel, string filePath)
        {
            var classes = new List<ClassInfo>();

            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();

            foreach (var classDecl in classDeclarations)
            {
                var classSymbol = semanticModel.GetDeclaredSymbol(classDecl) as INamedTypeSymbol;
                if (classSymbol == null) continue;

                var classInfo = new ClassInfo
                {
                    ClassName = classDecl.Identifier.Text,
                    Namespace = classSymbol.ContainingNamespace?.ToDisplayString() ?? "",
                    FullName = classSymbol.ToDisplayString(),
                    FilePath = filePath,
                    LineNumber = classDecl.GetLocation().GetLineSpan().StartLinePosition.Line + 1,
                    Accessibility = classSymbol.DeclaredAccessibility.ToString(),
                    IsStatic = classSymbol.IsStatic,
                    IsAbstract = classSymbol.IsAbstract,
                    IsSealed = classSymbol.IsSealed,
                    BaseClass = classSymbol.BaseType?.ToDisplayString() ?? "object"
                };

                // Extract interfaces
                foreach (var iface in classSymbol.Interfaces)
                {
                    classInfo.Interfaces.Add(iface.ToDisplayString());
                }

                // Extract fields
                foreach (var member in classDecl.Members.OfType<FieldDeclarationSyntax>())
                {
                    foreach (var variable in member.Declaration.Variables)
                    {
                        classInfo.Fields.Add($"{member.Declaration.Type} {variable.Identifier.Text}");
                    }
                }

                // Extract properties
                foreach (var member in classDecl.Members.OfType<PropertyDeclarationSyntax>())
                {
                    classInfo.Properties.Add($"{member.Type} {member.Identifier.Text}");
                }

                // Extract method names (just names for class info)
                foreach (var member in classDecl.Members.OfType<MethodDeclarationSyntax>())
                {
                    classInfo.Methods.Add(member.Identifier.Text);
                }

                // Extract XML documentation summary if available
                var trivia = classDecl.GetLeadingTrivia();
                var xmlTrivia = trivia.FirstOrDefault(t => t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) ||
                                                           t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia));
                if (xmlTrivia != default(SyntaxTrivia))
                {
                    classInfo.Summary = ExtractSummaryFromXml(xmlTrivia.ToString());
                }

                classes.Add(classInfo);
            }

            return classes;
        }

        private string ExtractSummaryFromXml(string xml)
        {
            // Simple extraction of <summary> content
            var startTag = "<summary>";
            var endTag = "</summary>";
            var startIndex = xml.IndexOf(startTag);
            var endIndex = xml.IndexOf(endTag);

            if (startIndex >= 0 && endIndex > startIndex)
            {
                var summary = xml.Substring(startIndex + startTag.Length, endIndex - startIndex - startTag.Length);
                // Remove /// and extra whitespace
                return summary.Replace("///", "").Trim();
            }

            return "";
        }
    }
}
