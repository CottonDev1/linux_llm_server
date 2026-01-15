"""
EWR Code Intelligence Agent CLI
===============================

Command-line interface for the Code Intelligence Agent.

Usage:
    ewr-code-intel analyze <repo_path>
    ewr-code-intel call-graph <entry_point> [--repo=.]
    ewr-code-intel question <question> [--repo=.]
    ewr-code-intel validate <questions_file> [--repo=.]
    ewr-code-intel workflow <question> [--repo=.]
    ewr-code-intel interactive [--repo=.]
"""

import asyncio
import argparse
import sys
import json

from ewr_agent_core import load_config

from .agent import CodeIntelAgent


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ewr-code-intel",
        description="EWR Code Intelligence Agent - Deep code analysis and understanding"
    )

    parser.add_argument("--model", default=None, help="LLM model to use")
    parser.add_argument("--backend", default="llamacpp", help="LLM backend")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a repository")
    analyze_parser.add_argument("repo_path", help="Path to repository")
    analyze_parser.add_argument("--depth", choices=["quick", "standard", "deep"],
                                default="standard", help="Analysis depth")

    # Call graph command
    callgraph_parser = subparsers.add_parser("call-graph", help="Build call graph")
    callgraph_parser.add_argument("entry_point", help="Entry point function/method")
    callgraph_parser.add_argument("--repo", "-r", default=".", help="Repository path")
    callgraph_parser.add_argument("--max-depth", type=int, default=10, help="Max call depth")

    # Question command
    question_parser = subparsers.add_parser("question", help="Ask a question about the code")
    question_parser.add_argument("question", nargs="+", help="Question to ask")
    question_parser.add_argument("--repo", "-r", default=".", help="Repository path")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate knowledge")
    validate_parser.add_argument("questions", nargs="+", help="Questions to test (or file)")
    validate_parser.add_argument("--repo", "-r", default=".", help="Repository path")

    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Explain a workflow")
    workflow_parser.add_argument("question", nargs="+", help="Workflow question")
    workflow_parser.add_argument("--repo", "-r", default=".", help="Repository path")

    # Entry points command
    entry_parser = subparsers.add_parser("entry-points", help="List entry points")
    entry_parser.add_argument("--repo", "-r", default=".", help="Repository path")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--repo", "-r", default=".", help="Repository path")

    return parser


async def run_analyze(agent: CodeIntelAgent, args, output_json: bool):
    """Run analyze command."""
    print(f"Analyzing repository: {args.repo_path}...")
    print("This may take a few minutes for large codebases.\n")

    result = await agent.analyze_repository(
        args.repo_path,
        analyze_depth=args.depth
    )

    if output_json:
        # Limit output for JSON
        output = result.model_dump()
        output["call_graphs"] = {k: "..." for k in output.get("call_graphs", {})}
        print(json.dumps(output, indent=2, default=str))
    else:
        print(f"=== Analysis Complete: {result.repo_name} ===")
        print(f"Status: {result.status.value}")
        print(f"Files: {result.total_files}")
        print(f"Lines: {result.total_lines}")

        print(f"\nLanguages:")
        for lang, count in sorted(result.languages.items(), key=lambda x: -x[1]):
            print(f"  {lang}: {count} files")

        print(f"\nEntry Points ({len(result.entry_points)}):")
        for ep in result.entry_points[:10]:
            route = f" [{ep.route}]" if ep.route else ""
            print(f"  - {ep.entry_type.value}: {ep.name}{route}")

        if result.entry_points[10:]:
            print(f"  ... and {len(result.entry_points) - 10} more")

        print(f"\nKey Classes ({len(result.key_classes)}):")
        for cls in result.key_classes[:10]:
            print(f"  - {cls}")

        print(f"\nDependencies ({len(result.dependencies)}):")
        for dep in result.dependencies[:10]:
            version = f" ({dep.version})" if dep.version else ""
            print(f"  - {dep.name}{version}")

        if result.architecture_notes:
            print(f"\nArchitecture Notes:")
            for note in result.architecture_notes:
                print(f"  - {note}")


async def run_call_graph(agent: CodeIntelAgent, args, output_json: bool):
    """Run call graph command."""
    print(f"Building call graph for: {args.entry_point}")

    graph = await agent.build_call_graph(
        args.entry_point,
        repo_path=args.repo,
        max_depth=args.max_depth
    )

    if output_json:
        print(json.dumps(graph.model_dump(), indent=2, default=str))
    else:
        print(f"\n=== Call Graph ===")
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Edges: {len(graph.edges)}")
        print(f"Total Calls: {graph.total_calls}")

        print(f"\nNodes:")
        for node_id, node in list(graph.nodes.items())[:20]:
            print(f"  - {node.full_name} ({node.node_type})")
            print(f"    File: {node.file_path}:{node.line_number}")

        if len(graph.nodes) > 20:
            print(f"  ... and {len(graph.nodes) - 20} more nodes")


async def run_question(agent: CodeIntelAgent, args, output_json: bool):
    """Run question command."""
    question = " ".join(args.question)
    print(f"Answering: {question}\n")

    answer = await agent.answer_question(question, args.repo)

    if output_json:
        print(json.dumps(answer.model_dump(), indent=2, default=str))
    else:
        print(f"=== Answer ===")
        print(answer.answer)

        print(f"\nConfidence: {answer.confidence:.0%}")

        if answer.relevant_files:
            print(f"\nRelevant Files:")
            for f in answer.relevant_files:
                print(f"  - {f}")

        if answer.follow_up_suggestions:
            print(f"\nFollow-up Suggestions:")
            for s in answer.follow_up_suggestions:
                print(f"  - {s}")


async def run_validate(agent: CodeIntelAgent, args, output_json: bool):
    """Run validate command."""
    # Check if questions is a file
    questions = args.questions
    if len(questions) == 1 and questions[0].endswith(".txt"):
        try:
            with open(questions[0]) as f:
                questions = [line.strip() for line in f if line.strip()]
        except Exception:
            pass

    print(f"Validating with {len(questions)} questions...\n")

    result = await agent.validate_knowledge(questions, args.repo)

    if output_json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(f"=== Validation Results ===")
        print(f"Questions: {result.total_questions}")
        print(f"Correct: {result.correct_answers}")
        print(f"Accuracy: {result.accuracy_score:.0%}")
        print(f"Needs Refinement: {result.needs_refinement}")

        if result.gaps:
            print(f"\nKnowledge Gaps ({len(result.gaps)}):")
            for gap in result.gaps:
                print(f"  - {gap.question}")
                print(f"    Topic: {gap.expected_topic}")
                print(f"    Severity: {gap.severity}")
                if gap.suggested_files:
                    print(f"    Files: {', '.join(gap.suggested_files[:3])}")

        if result.suggestions:
            print(f"\nSuggestions:")
            for s in result.suggestions:
                print(f"  - {s}")


async def run_workflow(agent: CodeIntelAgent, args, output_json: bool):
    """Run workflow command."""
    question = " ".join(args.question)
    print(f"Explaining workflow: {question}\n")

    explanation = await agent.explain_workflow(question, args.repo)

    if output_json:
        print(json.dumps(explanation.model_dump(), indent=2, default=str))
    else:
        print(f"=== Workflow Explanation ===")
        print(f"\nSummary: {explanation.summary}")

        print(f"\nSteps:")
        for step in explanation.steps:
            print(f"\n  Step {step.step_number}: {step.description}")
            if step.file_path:
                print(f"    File: {step.file_path}")
            if step.function_name:
                print(f"    Function: {step.function_name}")

        print(f"\nConfidence: {explanation.confidence:.0%}")


async def run_entry_points(agent: CodeIntelAgent, args, output_json: bool):
    """List entry points."""
    result = await agent.analyze_repository(args.repo, analyze_depth="quick")

    if output_json:
        print(json.dumps([ep.model_dump() for ep in result.entry_points], indent=2, default=str))
    else:
        print(f"=== Entry Points ({len(result.entry_points)}) ===\n")
        for ep in result.entry_points:
            route = f" [{ep.http_method} {ep.route}]" if ep.route else ""
            print(f"  {ep.entry_type.value}: {ep.name}{route}")
            print(f"    File: {ep.file_path}:{ep.line_number}")


async def run_interactive(agent: CodeIntelAgent, args):
    """Run interactive mode."""
    print(f"\n=== EWR Code Intelligence Agent ===")
    print(f"Repository: {args.repo}")
    print("Commands: analyze, question, workflow, validate, entry-points, quit\n")

    # Pre-analyze the repository
    print("Analyzing repository...")
    await agent.analyze_repository(args.repo, analyze_depth="quick")
    print("Ready!\n")

    while True:
        try:
            line = input("intel> ").strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd in ("question", "ask", "q") and arg:
                answer = await agent.answer_question(arg, args.repo)
                print(f"\n{answer.answer}")
                print(f"\n[Confidence: {answer.confidence:.0%}]")
                if answer.relevant_files:
                    print(f"[Files: {', '.join(answer.relevant_files[:3])}]\n")
            elif cmd == "workflow" and arg:
                explanation = await agent.explain_workflow(arg, args.repo)
                print(f"\n{explanation.summary}")
                for step in explanation.steps:
                    print(f"  {step.step_number}. {step.description}")
                print()
            elif cmd == "analyze":
                path = arg or args.repo
                result = await agent.analyze_repository(path)
                print(f"Analyzed: {result.total_files} files, {len(result.entry_points)} entry points")
            elif cmd in ("entry-points", "entries", "ep"):
                result = agent._analysis_cache.get(args.repo)
                if result:
                    for ep in result.entry_points[:10]:
                        print(f"  {ep.entry_type.value}: {ep.name}")
            elif cmd == "validate" and arg:
                questions = [q.strip() for q in arg.split(",")]
                result = await agent.validate_knowledge(questions, args.repo)
                print(f"Accuracy: {result.accuracy_score:.0%}")
                for gap in result.gaps:
                    print(f"  Gap: {gap.question}")
            else:
                print(f"Unknown command: {cmd}")
                print("Try: question <q>, workflow <q>, analyze, entry-points, quit")

        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


async def async_main(args):
    """Async main function."""
    config = load_config(
        name="code-intel-cli",
        llm_backend=args.backend,
        llm_model=args.model,
    )

    agent = CodeIntelAgent(config=config)
    await agent.start()

    try:
        if args.command == "analyze":
            await run_analyze(agent, args, args.json)
        elif args.command == "call-graph":
            await run_call_graph(agent, args, args.json)
        elif args.command == "question":
            await run_question(agent, args, args.json)
        elif args.command == "validate":
            await run_validate(agent, args, args.json)
        elif args.command == "workflow":
            await run_workflow(agent, args, args.json)
        elif args.command == "entry-points":
            await run_entry_points(agent, args, args.json)
        elif args.command == "interactive":
            await run_interactive(agent, args)
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
