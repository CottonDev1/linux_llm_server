"""
EWR Code Agent CLI
==================

Command-line interface for the Code Agent.

Usage:
    ewr-code-agent analyze <file>
    ewr-code-agent generate <prompt> [--language=python]
    ewr-code-agent explain <file>
    ewr-code-agent scan <directory>
    ewr-code-agent search <pattern> [--path=.]
    ewr-code-agent interactive
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path

from ewr_agent_core import load_config

from .agent import CodeAgent


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ewr-code-agent",
        description="EWR Code Agent - Code analysis and generation"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="LLM model to use"
    )
    parser.add_argument(
        "--backend",
        default="llamacpp",
        choices=["llamacpp", "openai", "anthropic"],
        help="LLM backend"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a code file")
    analyze_parser.add_argument("file", help="File to analyze")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate code")
    generate_parser.add_argument("prompt", help="Description of code to generate")
    generate_parser.add_argument("--language", "-l", default="python", help="Target language")
    generate_parser.add_argument("--context", "-c", nargs="*", help="Context files")

    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain code")
    explain_parser.add_argument("file", help="File to explain")
    explain_parser.add_argument("--detail", "-d", default="medium",
                                choices=["low", "medium", "high"])

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan project structure")
    scan_parser.add_argument("directory", nargs="?", default=".", help="Directory to scan")
    scan_parser.add_argument("--max-depth", type=int, default=10, help="Maximum depth")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for code patterns")
    search_parser.add_argument("pattern", help="Pattern to search for")
    search_parser.add_argument("--path", "-p", default=".", help="Directory to search")
    search_parser.add_argument("--files", "-f", help="File pattern (e.g., *.py)")
    search_parser.add_argument("--regex", "-r", action="store_true", help="Use regex")

    # Review command
    review_parser = subparsers.add_parser("review", help="Review code")
    review_parser.add_argument("file", help="File to review")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")

    return parser


async def run_analyze(agent: CodeAgent, args, output_json: bool):
    """Run analyze command."""
    result = await agent.analyze_file(args.file)
    if output_json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(f"\n=== Analysis: {args.file} ===")
        print(f"Language: {result.language}")
        print(f"Lines: {result.line_count}")
        print(f"Functions: {len(result.functions)}")
        print(f"Classes: {len(result.classes)}")
        print(f"Imports: {len(result.imports)}")
        print(f"Complexity Score: {result.complexity_score:.1f}")
        if result.summary:
            print(f"\nSummary: {result.summary}")
        if result.functions:
            print("\nFunctions:")
            for func in result.functions[:10]:
                print(f"  - {func.signature} (line {func.start_line})")
        if result.classes:
            print("\nClasses:")
            for cls in result.classes[:10]:
                print(f"  - {cls.name} (line {cls.start_line})")


async def run_generate(agent: CodeAgent, args, output_json: bool):
    """Run generate command."""
    result = await agent.generate_code(
        prompt=args.prompt,
        language=args.language,
        context_files=args.context or []
    )
    if output_json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(f"\n=== Generated {args.language} code ===\n")
        print(result.code)


async def run_explain(agent: CodeAgent, args, output_json: bool):
    """Run explain command."""
    code = await agent.read_file(args.file)
    file_info = await agent.get_file_info(args.file)
    explanation = await agent.explain_code(
        code=code,
        language=file_info.language,
        detail_level=args.detail
    )
    if output_json:
        print(json.dumps({"explanation": explanation}))
    else:
        print(f"\n=== Explanation: {args.file} ===\n")
        print(explanation)


async def run_scan(agent: CodeAgent, args, output_json: bool):
    """Run scan command."""
    result = await agent.scan_project(args.directory, max_depth=args.max_depth)
    if output_json:
        # Limit output for JSON
        result.source_files = result.source_files[:100]
        result.directories = result.directories[:50]
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(f"\n=== Project: {result.name} ===")
        print(f"Path: {result.root_path}")
        print(f"Total Files: {result.total_files}")
        print(f"Total Lines: {result.total_lines}")
        print(f"Total Size: {result.total_size_bytes / 1024:.1f} KB")
        if result.languages:
            print("\nLanguages:")
            for lang, count in sorted(result.languages.items(), key=lambda x: -x[1]):
                print(f"  {lang}: {count} files")
        if result.entry_points:
            print("\nEntry Points:")
            for ep in result.entry_points[:5]:
                print(f"  - {ep}")
        if result.config_files:
            print("\nConfig Files:")
            for cf in result.config_files[:5]:
                print(f"  - {cf}")


async def run_search(agent: CodeAgent, args, output_json: bool):
    """Run search command."""
    results = await agent.search_code(
        pattern=args.pattern,
        path=args.path,
        file_pattern=args.files,
        regex=args.regex
    )
    if output_json:
        print(json.dumps([r.model_dump() for r in results], indent=2))
    else:
        print(f"\n=== Search: '{args.pattern}' ===")
        print(f"Found {len(results)} matches\n")
        for result in results[:20]:
            print(f"{result.file_path}:{result.line_number}")
            print(f"  {result.line_content.strip()}")
            print()


async def run_review(agent: CodeAgent, args, output_json: bool):
    """Run review command."""
    result = await agent.review_code(file_path=args.file)
    if output_json:
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(f"\n=== Code Review: {args.file} ===")
        print(f"Score: {result.overall_score}/10")
        print(f"\n{result.summary}")


async def run_interactive(agent: CodeAgent):
    """Run interactive mode."""
    print("\n=== EWR Code Agent Interactive Mode ===")
    print("Commands: analyze, generate, explain, scan, search, review, quit")
    print()

    while True:
        try:
            line = input("code> ").strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "analyze" and arg:
                result = await agent.analyze_file(arg)
                print(f"Functions: {len(result.functions)}, Classes: {len(result.classes)}")
                if result.summary:
                    print(f"Summary: {result.summary}")
            elif cmd == "generate" and arg:
                result = await agent.generate_code(arg)
                print(result.code)
            elif cmd == "explain" and arg:
                code = await agent.read_file(arg)
                explanation = await agent.explain_code(code)
                print(explanation)
            elif cmd == "scan":
                result = await agent.scan_project(arg or ".")
                print(f"Files: {result.total_files}, Lines: {result.total_lines}")
            elif cmd == "search" and arg:
                results = await agent.search_code(arg)
                for r in results[:10]:
                    print(f"{r.file_path}:{r.line_number}: {r.line_content.strip()}")
            elif cmd == "review" and arg:
                result = await agent.review_code(file_path=arg)
                print(f"Score: {result.overall_score}/10")
                print(result.summary)
            else:
                print(f"Unknown command or missing argument: {cmd}")

        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


async def async_main(args):
    """Async main function."""
    # Load config
    config = load_config(
        llm_backend=args.backend,
        llm_model=args.model,
        name="code-agent-cli"
    )

    # Create agent
    agent = CodeAgent(config=config)
    await agent.start()

    try:
        if args.command == "analyze":
            await run_analyze(agent, args, args.json)
        elif args.command == "generate":
            await run_generate(agent, args, args.json)
        elif args.command == "explain":
            await run_explain(agent, args, args.json)
        elif args.command == "scan":
            await run_scan(agent, args, args.json)
        elif args.command == "search":
            await run_search(agent, args, args.json)
        elif args.command == "review":
            await run_review(agent, args, args.json)
        elif args.command == "interactive":
            await run_interactive(agent)
        else:
            print("No command specified. Use --help for usage.")
    finally:
        await agent.stop()


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
