using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Newtonsoft.Json;
using RoslynCodeAnalyzer.Analyzers;
using RoslynCodeAnalyzer.Models;
using RoslynCodeAnalyzer.Services;

namespace RoslynCodeAnalyzer
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            try
            {
                var options = ParseArguments(args);
                if (options == null)
                {
                    PrintUsage();
                    return 1;
                }

                Console.WriteLine($"Enhanced Roslyn Code Analyzer v2.1 (MongoDB Support)");
                Console.WriteLine($"Analyzing: {options.InputPath}");
                Console.WriteLine($"Output Type: {options.Output}");
                if (options.Output == OutputType.Json || options.Output == OutputType.Both)
                    Console.WriteLine($"JSON Output: {options.OutputPath}");
                if (options.Output == OutputType.MongoDB || options.Output == OutputType.Both)
                    Console.WriteLine($"MongoDB: {options.MongoConnectionString}/{options.MongoDatabaseName}");
                Console.WriteLine($"Mode: {options.Mode}");
                Console.WriteLine($"Project: {options.ProjectName}");
                Console.WriteLine();

                var result = await AnalyzeCodeAsync(options);

                // Post-process to build bidirectional relationships
                BuildBidirectionalRelationships(result);

                // Write output based on output type
                if (options.Output == OutputType.Json || options.Output == OutputType.Both)
                {
                    // Write output to JSON file
                    var json = JsonConvert.SerializeObject(result, Formatting.Indented,
                        new JsonSerializerSettings { NullValueHandling = NullValueHandling.Ignore });
                    await File.WriteAllTextAsync(options.OutputPath, json);
                    Console.WriteLine($"\nJSON results written to: {options.OutputPath}");
                }

                if (options.Output == OutputType.MongoDB || options.Output == OutputType.Both)
                {
                    // Store in MongoDB
                    await StoreInMongoDBAsync(result, options);
                }

                Console.WriteLine($"\nAnalysis complete!");
                Console.WriteLine($"Classes analyzed: {result.Classes.Count}");
                Console.WriteLine($"Methods analyzed: {result.Methods.Count}");
                Console.WriteLine($"Call relationships: {result.CallGraph.Count}");
                Console.WriteLine($"  - DataLayer calls: {result.CallGraph.Count(c => c.CallType == "DataLayer")}");
                Console.WriteLine($"  - SQL operations: {result.CallGraph.Count(c => c.IsSqlOperation)}");
                Console.WriteLine($"Event handlers: {result.EventHandlers.Count}");
                Console.WriteLine($"Database operations: {result.DatabaseOperations.Count}");

                // Show caller/callee summary
                var methodsWithCallers = result.Methods.Count(m => m.CalledBy.Any());
                var methodsWithCallees = result.Methods.Count(m => m.CallsTo.Any());
                Console.WriteLine($"\nCall tracking:");
                Console.WriteLine($"  - Methods with callers tracked: {methodsWithCallers}");
                Console.WriteLine($"  - Methods with callees tracked: {methodsWithCallees}");

                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error: {ex.Message}");
                Console.Error.WriteLine(ex.StackTrace);
                return 1;
            }
        }

        static async Task StoreInMongoDBAsync(AnalysisResult result, AnalyzerOptions options)
        {
            Console.WriteLine($"\nStoring results in MongoDB...");

            var config = new MongoDBConfig
            {
                ConnectionString = options.MongoConnectionString,
                DatabaseName = options.MongoDatabaseName,
                EmbeddingServiceUrl = options.EmbeddingServiceUrl
            };

            using var mongoService = new MongoDBService(config, options.GenerateEmbeddings);

            try
            {
                await mongoService.InitializeAsync();

                var stats = await mongoService.StoreAnalysisAsync(result, options.ProjectName);

                Console.WriteLine($"\nMongoDB storage complete:");
                Console.WriteLine($"  - Classes stored: {stats.ClassesStored}");
                Console.WriteLine($"  - Methods stored: {stats.MethodsStored}");
                Console.WriteLine($"  - Call graph edges stored: {stats.CallGraphStored}");
                Console.WriteLine($"  - Event handlers stored: {stats.EventHandlersStored}");
                Console.WriteLine($"  - DB operations stored: {stats.DbOperationsStored}");
                Console.WriteLine($"  - Duration: {stats.Duration.TotalSeconds:F2} seconds");

                if (stats.Errors > 0)
                {
                    Console.WriteLine($"  - Errors: {stats.Errors}");
                    foreach (var error in stats.ErrorMessages.Take(5))
                    {
                        Console.WriteLine($"    - {error}");
                    }
                    if (stats.ErrorMessages.Count > 5)
                    {
                        Console.WriteLine($"    ... and {stats.ErrorMessages.Count - 5} more errors");
                    }
                }

                // Show collection stats
                var collectionStats = await mongoService.GetStatsAsync();
                Console.WriteLine($"\nTotal documents in MongoDB:");
                foreach (var stat in collectionStats)
                {
                    Console.WriteLine($"  - {stat.Key}: {stat.Value}");
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"MongoDB storage failed: {ex.Message}");
                throw;
            }
        }

        static async Task<AnalysisResult> AnalyzeCodeAsync(AnalyzerOptions options)
        {
            var result = new AnalysisResult
            {
                AnalyzedPath = options.InputPath,
                AnalysisMode = options.Mode,
                Timestamp = DateTime.UtcNow
            };

            // Try to create a proper compilation if analyzing a solution or project
            Compilation compilation = null;
            Dictionary<string, SemanticModel> semanticModels = new Dictionary<string, SemanticModel>();

            if (options.Mode == "solution" || options.Mode == "project")
            {
                compilation = await CreateCompilationAsync(options);
                if (compilation != null)
                {
                    Console.WriteLine($"Created compilation with {compilation.SyntaxTrees.Count()} syntax trees");
                    foreach (var tree in compilation.SyntaxTrees)
                    {
                        semanticModels[tree.FilePath] = compilation.GetSemanticModel(tree);
                    }
                }
            }

            // Get all C# files to analyze
            var files = GetCSharpFiles(options.InputPath, options.Mode);
            Console.WriteLine($"Found {files.Count} C# files to analyze");

            foreach (var filePath in files)
            {
                Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");
                await AnalyzeFileAsync(filePath, result, options, compilation, semanticModels);
            }

            return result;
        }

        static async Task AnalyzeFileAsync(
            string filePath,
            AnalysisResult result,
            AnalyzerOptions options,
            Compilation compilation,
            Dictionary<string, SemanticModel> semanticModels)
        {
            try
            {
                var code = await File.ReadAllTextAsync(filePath);
                var tree = CSharpSyntaxTree.ParseText(code, path: filePath);
                var root = await tree.GetRootAsync();

                // Use existing semantic model if available, otherwise create a basic one
                SemanticModel semanticModel;
                if (semanticModels.ContainsKey(filePath))
                {
                    semanticModel = semanticModels[filePath];
                }
                else if (compilation != null)
                {
                    // Add this tree to the compilation
                    compilation = compilation.AddSyntaxTrees(tree);
                    semanticModel = compilation.GetSemanticModel(tree);
                }
                else
                {
                    // Create a basic compilation for standalone analysis
                    var basicCompilation = CreateBasicCompilation(tree);
                    semanticModel = basicCompilation.GetSemanticModel(tree);
                }

                // Extract classes
                var classExtractor = new ClassExtractor();
                var classes = classExtractor.ExtractClasses(root, semanticModel, filePath);
                result.Classes.AddRange(classes);

                // Extract methods with project name
                var methodExtractor = new MethodExtractor();
                var methods = methodExtractor.ExtractMethods(root, semanticModel, filePath);

                // Add project name to methods
                foreach (var method in methods)
                {
                    method.ProjectName = options.ProjectName;
                }
                result.Methods.AddRange(methods);

                // Build enhanced call graph with project context
                var callGraphBuilder = new CallGraphBuilder(options.ProjectName);
                var callRelationships = callGraphBuilder.BuildCallGraph(root, semanticModel, filePath, compilation);
                result.CallGraph.AddRange(callRelationships);

                // Detect event handlers
                var eventHandlerDetector = new EventHandlerDetector();
                var eventHandlers = eventHandlerDetector.DetectEventHandlers(root, semanticModel, filePath);
                result.EventHandlers.AddRange(eventHandlers);

                // Extract database operations with enhanced DataLayer detection
                var dbExtractor = new DatabaseOperationExtractor();
                var dbOperations = dbExtractor.ExtractDatabaseOperations(root, semanticModel, filePath);
                result.DatabaseOperations.AddRange(dbOperations);

                // Extract SQL operations from methods and add to method info
                ExtractSqlCallsFromMethods(root, semanticModel, filePath, methods);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  Error processing {Path.GetFileName(filePath)}: {ex.Message}");
                result.Errors.Add(new AnalysisError
                {
                    FilePath = filePath,
                    Message = ex.Message,
                    StackTrace = ex.StackTrace
                });
            }
        }

        static void ExtractSqlCallsFromMethods(
            SyntaxNode root,
            SemanticModel semanticModel,
            string filePath,
            List<MethodInfo> methods)
        {
            // Find SQL operations within methods and add them to the method's SqlCalls
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();

            foreach (var methodDecl in methodDeclarations)
            {
                var methodSymbol = semanticModel.GetDeclaredSymbol(methodDecl);
                if (methodSymbol == null) continue;

                var methodInfo = methods.FirstOrDefault(m =>
                    m.MethodName == methodSymbol.Name &&
                    m.ClassName == methodSymbol.ContainingType?.Name);

                if (methodInfo == null) continue;

                // Find SQL object creations
                var sqlCreations = methodDecl.DescendantNodes()
                    .OfType<ObjectCreationExpressionSyntax>()
                    .Where(c =>
                    {
                        var typeInfo = semanticModel.GetTypeInfo(c);
                        var typeName = typeInfo.Type?.Name;
                        return typeName != null && (typeName.Contains("SqlCommand") ||
                                                   typeName.Contains("SqlDataAdapter"));
                    });

                foreach (var creation in sqlCreations)
                {
                    var line = creation.GetLocation().GetLineSpan().StartLinePosition.Line + 1;
                    var commandText = ExtractCommandText(creation, semanticModel);

                    var sqlCall = new SqlCall
                    {
                        Type = "SqlCommand",
                        Line = line,
                        CommandText = commandText
                    };

                    // Try to detect stored procedure
                    if (!string.IsNullOrEmpty(commandText))
                    {
                        var sqlKeywords = new[] { "SELECT", "INSERT", "UPDATE", "DELETE", "FROM" };
                        if (!sqlKeywords.Any(kw => commandText.ToUpper().Contains(kw)))
                        {
                            sqlCall.StoredProcedure = commandText;
                            sqlCall.Type = "StoredProcedure";
                        }
                    }

                    methodInfo.SqlCalls.Add(sqlCall);
                }
            }
        }

        static string ExtractCommandText(ObjectCreationExpressionSyntax creation, SemanticModel semanticModel)
        {
            if (creation.ArgumentList?.Arguments.Count > 0)
            {
                var firstArg = creation.ArgumentList.Arguments[0].Expression;

                // Handle string literals
                if (firstArg is LiteralExpressionSyntax literal &&
                    literal.IsKind(SyntaxKind.StringLiteralExpression))
                {
                    return literal.Token.ValueText;
                }

                // Try to resolve constants
                var constantValue = semanticModel.GetConstantValue(firstArg);
                if (constantValue.HasValue && constantValue.Value is string str)
                {
                    return str;
                }
            }

            return null;
        }

        static void BuildBidirectionalRelationships(AnalysisResult result)
        {
            // Build a method lookup dictionary for quick access
            var methodLookup = new Dictionary<string, MethodInfo>();
            foreach (var method in result.Methods)
            {
                var key = $"{method.Namespace}.{method.ClassName}.{method.MethodName}";
                methodLookup[key] = method;
            }

            // Process call graph to populate CalledBy and CallsTo
            foreach (var call in result.CallGraph)
            {
                // Find caller method
                var callerKey = $"{call.CallerNamespace}.{call.CallerClass}.{call.CallerMethod}";
                if (methodLookup.TryGetValue(callerKey, out var callerMethod))
                {
                    // Add to CallsTo list
                    callerMethod.CallsTo.Add(new CallReference
                    {
                        Project = call.CalleeProject,
                        File = call.CalleeFilePath,
                        Class = call.CalleeClass,
                        Method = call.CalleeMethod,
                        Line = call.CallSiteLineNumber,
                        CallType = call.CallType
                    });

                    // If it's a SQL operation, add to SqlCalls
                    if (call.IsSqlOperation)
                    {
                        callerMethod.SqlCalls.Add(new SqlCall
                        {
                            Type = call.CallType,
                            StoredProcedure = call.StoredProcedureName,
                            CommandText = call.SqlCommandText,
                            Line = call.CallSiteLineNumber
                        });
                    }
                }

                // Find callee method (only if it's in our analyzed code)
                if (!call.CalleeFilePath.StartsWith("External"))
                {
                    var calleeKey = $"{call.CalleeNamespace}.{call.CalleeClass}.{call.CalleeMethod}";
                    if (methodLookup.TryGetValue(calleeKey, out var calleeMethod))
                    {
                        // Add to CalledBy list
                        calleeMethod.CalledBy.Add(new CallReference
                        {
                            Project = call.CallerProject,
                            File = call.CallerFilePath,
                            Class = call.CallerClass,
                            Method = call.CallerMethod,
                            Line = call.CallerLineNumber,
                            CallType = call.CallType
                        });
                    }
                }
            }
        }

        static async Task<Compilation> CreateCompilationAsync(AnalyzerOptions options)
        {
            // Simplified version - MSBuild workspace requires complex setup
            // For now, return null to use basic compilation
            // This can be enhanced later with proper MSBuild support
            return null;
        }

        static Compilation CreateBasicCompilation(SyntaxTree tree)
        {
            // Create a basic compilation with common references
            var references = new[]
            {
                MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(Console).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(System.Linq.Enumerable).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(System.Text.RegularExpressions.Regex).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(System.IO.File).Assembly.Location),
                MetadataReference.CreateFromFile(typeof(System.Collections.Generic.List<>).Assembly.Location)
            };

            return CSharpCompilation.Create("TempCompilation")
                .AddReferences(references)
                .AddSyntaxTrees(tree);
        }

        static List<string> GetCSharpFiles(string path, string mode)
        {
            var files = new List<string>();

            if (mode == "file")
            {
                if (File.Exists(path) && path.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
                {
                    files.Add(path);
                }
            }
            else if (mode == "directory" || mode == "solution" || mode == "project")
            {
                // For solution/project modes, analyze the directory containing them
                var directory = mode == "directory" ? path : Path.GetDirectoryName(path);
                if (Directory.Exists(directory))
                {
                    files.AddRange(Directory.GetFiles(directory, "*.cs", SearchOption.AllDirectories)
                        .Where(f => !f.Contains("\\obj\\") && !f.Contains("/obj/"))
                        .Where(f => !f.Contains("\\bin\\") && !f.Contains("/bin/"))
                        .Where(f => !f.EndsWith(".g.cs"))
                        .Where(f => !f.EndsWith(".designer.cs")));
                }
            }
            else if (mode == "filelist")
            {
                if (File.Exists(path))
                {
                    var fileList = File.ReadAllLines(path)
                        .Where(line => !string.IsNullOrWhiteSpace(line))
                        .Where(line => !line.TrimStart().StartsWith("#"))
                        .Select(line => line.Trim())
                        .Where(filePath => File.Exists(filePath) && filePath.EndsWith(".cs", StringComparison.OrdinalIgnoreCase))
                        .ToList();

                    files.AddRange(fileList);
                }
            }

            return files;
        }

        static AnalyzerOptions ParseArguments(string[] args)
        {
            if (args.Length < 2)
                return null;

            var options = new AnalyzerOptions
            {
                InputPath = args[0],
                OutputPath = args[1],
                Mode = "file",
                ProjectName = "Unknown",
                Output = OutputType.Json
            };

            // Parse optional arguments
            for (int i = 2; i < args.Length; i++)
            {
                var arg = args[i].ToLower();
                if (arg == "--project" && i + 1 < args.Length)
                {
                    options.ProjectName = args[++i];
                }
                else if (arg == "--filelist")
                {
                    options.Mode = "filelist";
                }
                else if (arg == "--verbose" || arg == "-v")
                {
                    options.Verbose = true;
                }
                else if (arg == "--include-private")
                {
                    options.IncludePrivateMembers = true;
                }
                else if (arg == "--output" && i + 1 < args.Length)
                {
                    var outputType = args[++i].ToLower();
                    options.Output = outputType switch
                    {
                        "mongodb" => OutputType.MongoDB,
                        "both" => OutputType.Both,
                        _ => OutputType.Json
                    };
                }
                else if (arg == "--mongo-connection" && i + 1 < args.Length)
                {
                    options.MongoConnectionString = args[++i];
                }
                else if (arg == "--mongo-database" && i + 1 < args.Length)
                {
                    options.MongoDatabaseName = args[++i];
                }
                else if (arg == "--embedding-service" && i + 1 < args.Length)
                {
                    options.EmbeddingServiceUrl = args[++i];
                }
                else if (arg == "--no-embeddings")
                {
                    options.GenerateEmbeddings = false;
                }
            }

            // Auto-detect mode and project name
            if (options.Mode == "file")
            {
                if (options.InputPath.EndsWith(".sln", StringComparison.OrdinalIgnoreCase))
                {
                    options.Mode = "solution";
                    if (options.ProjectName == "Unknown")
                    {
                        options.ProjectName = Path.GetFileNameWithoutExtension(options.InputPath);
                    }
                }
                else if (options.InputPath.EndsWith(".csproj", StringComparison.OrdinalIgnoreCase))
                {
                    options.Mode = "project";
                    if (options.ProjectName == "Unknown")
                    {
                        options.ProjectName = Path.GetFileNameWithoutExtension(options.InputPath);
                    }
                }
                else if (Directory.Exists(options.InputPath))
                {
                    options.Mode = "directory";
                    // Try to detect project name from directory
                    if (options.ProjectName == "Unknown")
                    {
                        var dirName = Path.GetFileName(options.InputPath.TrimEnd(Path.DirectorySeparatorChar));
                        options.ProjectName = dirName;
                    }
                }
                else if (!File.Exists(options.InputPath) || !options.InputPath.EndsWith(".cs"))
                {
                    Console.Error.WriteLine($"Error: Invalid input path: {options.InputPath}");
                    return null;
                }
            }

            // Try to detect project name from path if still unknown
            if (options.ProjectName == "Unknown")
            {
                var pathLower = options.InputPath.ToLower();
                if (pathLower.Contains("gin")) options.ProjectName = "Gin";
                else if (pathLower.Contains("warehouse")) options.ProjectName = "Warehouse";
                else if (pathLower.Contains("marketing")) options.ProjectName = "Marketing";
                else if (pathLower.Contains("ewr") && pathLower.Contains("library")) options.ProjectName = "EWR Library";
                else if (pathLower.Contains("provider")) options.ProjectName = "Provider";
            }

            return options;
        }

        static void PrintUsage()
        {
            Console.WriteLine("Enhanced Roslyn Code Analyzer v2.1 with MongoDB Support");
            Console.WriteLine();
            Console.WriteLine("Usage: RoslynCodeAnalyzer <input-path> <output-path> [options]");
            Console.WriteLine();
            Console.WriteLine("Arguments:");
            Console.WriteLine("  <input-path>         C# file, directory, solution (.sln), or project (.csproj)");
            Console.WriteLine("  <output-path>        Path where JSON output will be written (required for json/both output)");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --project <name>     Specify project name (auto-detected from path if not provided)");
            Console.WriteLine("  --filelist           Treat input-path as a text file containing list of .cs files");
            Console.WriteLine("  --verbose, -v        Enable verbose output");
            Console.WriteLine("  --include-private    Include private members in analysis");
            Console.WriteLine();
            Console.WriteLine("MongoDB Output Options:");
            Console.WriteLine("  --output <type>      Output type: json (default), mongodb, or both");
            Console.WriteLine("  --mongo-connection   MongoDB connection string (default: $MONGODB_URI or mongodb://localhost:27019)");
            Console.WriteLine("  --mongo-database     MongoDB database name (default: rag_server)");
            Console.WriteLine("  --embedding-service  Python embedding service URL (default: http://localhost:3030)");
            Console.WriteLine("  --no-embeddings      Skip generating vector embeddings");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  # JSON output (default)");
            Console.WriteLine("  RoslynCodeAnalyzer MyFile.cs output.json");
            Console.WriteLine();
            Console.WriteLine("  # MongoDB output with embeddings");
            Console.WriteLine("  RoslynCodeAnalyzer /path/to/project output.json --output mongodb --project \"Gin\"");
            Console.WriteLine();
            Console.WriteLine("  # Both JSON and MongoDB output");
            Console.WriteLine("  RoslynCodeAnalyzer MySolution.sln output.json --output both --project \"Warehouse\"");
            Console.WriteLine();
            Console.WriteLine("  # MongoDB with custom connection");
            Console.WriteLine("  RoslynCodeAnalyzer MyProject.csproj out.json --output mongodb --mongo-connection \"mongodb://localhost:27017\"");
            Console.WriteLine();
            Console.WriteLine("  # MongoDB without embeddings (faster, but no semantic search)");
            Console.WriteLine("  RoslynCodeAnalyzer /path/to/code out.json --output mongodb --no-embeddings");
            Console.WriteLine();
            Console.WriteLine("Enhanced features:");
            Console.WriteLine("  - Tracks who calls each method (CalledBy)");
            Console.WriteLine("  - Tracks what each method calls (CallsTo)");
            Console.WriteLine("  - Detects DataLayer method calls");
            Console.WriteLine("  - Identifies SQL operations and stored procedures");
            Console.WriteLine("  - Cross-project call tracking");
            Console.WriteLine("  - Vector embeddings for semantic code search (MongoDB mode)");
            Console.WriteLine();
            Console.WriteLine("MongoDB Collections Created:");
            Console.WriteLine("  - code_classes:       C# classes with inheritance info");
            Console.WriteLine("  - code_methods:       Methods with signatures and SQL operations");
            Console.WriteLine("  - code_callgraph:     Call relationships between methods");
            Console.WriteLine("  - code_eventhandlers: Event handler mappings (UI to code)");
            Console.WriteLine("  - code_dboperations:  Database operations (SQL commands)");
        }
    }
}