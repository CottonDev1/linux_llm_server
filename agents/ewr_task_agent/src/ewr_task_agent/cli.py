"""
EWR Task Agent CLI
==================

Command-line interface for the Task Agent.

Usage:
    ewr-task-agent shell <command>
    ewr-task-agent powershell <command>
    ewr-task-agent git status [--path=.]
    ewr-task-agent git log [--path=.] [--limit=10]
    ewr-task-agent git commit <message> [--path=.] [--all]
    ewr-task-agent interactive
"""

import asyncio
import argparse
import sys
import json

from ewr_agent_core import load_config

from .agent import TaskAgent


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ewr-task-agent",
        description="EWR Task Agent - Shell and Git operations"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Shell command
    shell_parser = subparsers.add_parser("shell", help="Execute shell command")
    shell_parser.add_argument("cmd", nargs="+", help="Command to execute")
    shell_parser.add_argument("--dir", "-d", help="Working directory")
    shell_parser.add_argument("--timeout", "-t", type=int, default=300, help="Timeout")

    # PowerShell command
    ps_parser = subparsers.add_parser("powershell", help="Execute PowerShell command")
    ps_parser.add_argument("cmd", nargs="+", help="Command to execute")
    ps_parser.add_argument("--dir", "-d", help="Working directory")
    ps_parser.add_argument("--timeout", "-t", type=int, default=300, help="Timeout")

    # Git commands
    git_parser = subparsers.add_parser("git", help="Git operations")
    git_subparsers = git_parser.add_subparsers(dest="git_command")

    # git status
    status_parser = git_subparsers.add_parser("status", help="Show repository status")
    status_parser.add_argument("--path", "-p", default=".", help="Repository path")

    # git log
    log_parser = git_subparsers.add_parser("log", help="Show commit history")
    log_parser.add_argument("--path", "-p", default=".", help="Repository path")
    log_parser.add_argument("--limit", "-n", type=int, default=10, help="Number of commits")

    # git commit
    commit_parser = git_subparsers.add_parser("commit", help="Create a commit")
    commit_parser.add_argument("message", help="Commit message")
    commit_parser.add_argument("--path", "-p", default=".", help="Repository path")
    commit_parser.add_argument("--all", "-a", action="store_true", help="Stage all files")
    commit_parser.add_argument("--files", "-f", nargs="*", help="Specific files to commit")

    # git branches
    branch_parser = git_subparsers.add_parser("branches", help="List branches")
    branch_parser.add_argument("--path", "-p", default=".", help="Repository path")

    # git diff
    diff_parser = git_subparsers.add_parser("diff", help="Show changes")
    diff_parser.add_argument("--path", "-p", default=".", help="Repository path")
    diff_parser.add_argument("--staged", "-s", action="store_true", help="Show staged changes")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive mode")

    return parser


async def run_shell(agent: TaskAgent, args, output_json: bool):
    """Run shell command."""
    command = " ".join(args.cmd)
    result = await agent.execute_shell(
        command=command,
        working_dir=args.dir,
        timeout=args.timeout
    )

    if output_json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if not result.success:
            sys.exit(result.exit_code or 1)


async def run_powershell(agent: TaskAgent, args, output_json: bool):
    """Run PowerShell command."""
    command = " ".join(args.cmd)
    result = await agent.execute_powershell(
        command=command,
        working_dir=args.dir,
        timeout=args.timeout
    )

    if output_json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if not result.success:
            sys.exit(result.exit_code or 1)


async def run_git(agent: TaskAgent, args, output_json: bool):
    """Run git command."""
    if args.git_command == "status":
        result = await agent.git_status(args.path)
        if output_json:
            print(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            print(f"Branch: {result.branch}")
            if result.is_clean:
                print("Working directory clean")
            else:
                if result.staged_files:
                    print("\nStaged changes:")
                    for f in result.staged_files:
                        print(f"  {f.status.value}: {f.path}")
                if result.modified_files:
                    print("\nModified files:")
                    for f in result.modified_files:
                        print(f"  {f.status.value}: {f.path}")
                if result.untracked_files:
                    print("\nUntracked files:")
                    for f in result.untracked_files:
                        print(f"  {f}")

            if result.ahead or result.behind:
                print(f"\nAhead: {result.ahead}, Behind: {result.behind}")

    elif args.git_command == "log":
        commits = await agent.git_log(args.path, limit=args.limit)
        if output_json:
            print(json.dumps([c.model_dump() for c in commits], indent=2, default=str))
        else:
            for commit in commits:
                print(f"{commit.short_hash} - {commit.message}")
                print(f"  Author: {commit.author} <{commit.author_email}>")
                print(f"  Date: {commit.date}")
                print()

    elif args.git_command == "commit":
        commit = await agent.git_commit(
            repo_path=args.path,
            message=args.message,
            files=args.files,
            all_files=args.all
        )
        if output_json:
            print(json.dumps(commit.model_dump() if commit else None, indent=2, default=str))
        else:
            if commit:
                print(f"Created commit: {commit.short_hash}")
                print(f"  Message: {commit.message}")
            else:
                print("Commit failed")
                sys.exit(1)

    elif args.git_command == "branches":
        branches = await agent.git_branches(args.path)
        if output_json:
            print(json.dumps([b.model_dump() for b in branches], indent=2, default=str))
        else:
            for branch in branches:
                marker = "* " if branch.is_current else "  "
                tracking = f" -> {branch.tracking}" if branch.tracking else ""
                print(f"{marker}{branch.name}{tracking}")

    elif args.git_command == "diff":
        diffs = await agent.git_diff(args.path, staged=args.staged)
        if output_json:
            print(json.dumps([d.model_dump() for d in diffs], indent=2, default=str))
        else:
            for diff in diffs:
                print(f"{diff.file_path}: +{diff.additions} -{diff.deletions}")

    else:
        print(f"Unknown git command: {args.git_command}")
        sys.exit(1)


async def run_interactive(agent: TaskAgent):
    """Run interactive mode."""
    print("\n=== EWR Task Agent Interactive Mode ===")
    print("Commands: shell, powershell, git, quit")
    print()

    while True:
        try:
            line = input("task> ").strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd in ("shell", "sh", "bash") and arg:
                result = await agent.execute_shell(arg)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                print(f"[Exit code: {result.exit_code}]")
            elif cmd in ("powershell", "ps") and arg:
                result = await agent.execute_powershell(arg)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
                print(f"[Exit code: {result.exit_code}]")
            elif cmd == "git":
                git_parts = arg.split(maxsplit=1)
                git_cmd = git_parts[0] if git_parts else "status"
                git_arg = git_parts[1] if len(git_parts) > 1 else "."

                if git_cmd == "status":
                    status = await agent.git_status(git_arg)
                    print(f"Branch: {status.branch}, Clean: {status.is_clean}")
                    if not status.is_clean:
                        print(f"Modified: {len(status.modified_files)}, Staged: {len(status.staged_files)}")
                elif git_cmd == "log":
                    commits = await agent.git_log(git_arg, limit=5)
                    for c in commits:
                        print(f"{c.short_hash} - {c.message}")
                elif git_cmd == "branches":
                    branches = await agent.git_branches(git_arg)
                    for b in branches:
                        marker = "* " if b.is_current else "  "
                        print(f"{marker}{b.name}")
                else:
                    print(f"Unknown git command: {git_cmd}")
            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


async def async_main(args):
    """Async main function."""
    config = load_config(name="task-agent-cli")
    agent = TaskAgent(config=config)
    await agent.start()

    try:
        if args.command == "shell":
            await run_shell(agent, args, args.json)
        elif args.command == "powershell":
            await run_powershell(agent, args, args.json)
        elif args.command == "git":
            await run_git(agent, args, args.json)
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
