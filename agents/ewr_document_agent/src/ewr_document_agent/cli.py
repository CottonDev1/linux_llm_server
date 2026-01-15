"""
EWR Document Agent CLI
======================

Command-line interface for the Document Processing Agent.

Usage:
    ewr-document process <file_path> [--strategy=recursive]
    ewr-document folder <folder_path> [--pattern=*.*] [--recursive]
    ewr-document search <query> [--top-k=5]
    ewr-document stats
    ewr-document interactive
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path

from ewr_agent_core import load_config

from .agent import DocumentAgent
from .models import ChunkingConfig, ChunkingStrategy, VectorStoreConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="ewr-document",
        description="EWR Document Agent - Document processing, chunking, and retrieval"
    )

    parser.add_argument("--model", default=None, help="Embedding model to use")
    parser.add_argument("--backend", default="llamacpp", help="LLM backend")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--mongodb-uri", default="mongodb://localhost:27017", help="MongoDB connection URI")
    parser.add_argument("--database", default="EWRAI", help="MongoDB database name")
    parser.add_argument("--collection", default="documents", help="MongoDB collection name")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("file_path", help="Path to document")
    process_parser.add_argument("--strategy", choices=[s.value for s in ChunkingStrategy],
                                default="recursive", help="Chunking strategy")
    process_parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size")
    process_parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap")

    # Folder command
    folder_parser = subparsers.add_parser("folder", help="Process a folder of documents")
    folder_parser.add_argument("folder_path", help="Path to folder")
    folder_parser.add_argument("--pattern", default="*.*", help="File pattern")
    folder_parser.add_argument("--recursive", "-r", action="store_true", help="Search recursively")
    folder_parser.add_argument("--strategy", choices=[s.value for s in ChunkingStrategy],
                               default="recursive", help="Chunking strategy")
    folder_parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--folder", "-f", help="Initial folder to process")

    return parser


async def run_process(agent: DocumentAgent, args, output_json: bool):
    """Run process command."""
    print(f"Processing document: {args.file_path}...")

    config = ChunkingConfig(
        strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
    )

    result = await agent.process_document(args.file_path, chunking_config=config)

    if output_json:
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(f"\n=== Processing Complete ===")
        print(f"Status: {result.status.value}")
        print(f"Document Type: {result.document_type.value}")
        print(f"Total Chunks: {result.total_chunks}")
        print(f"Chunks Embedded: {result.chunks_embedded}")
        print(f"Processing Time: {result.processing_time_ms}ms")

        if result.metadata:
            print(f"\nMetadata:")
            print(f"  Title: {result.metadata.title}")
            print(f"  Word Count: {result.metadata.word_count}")
            if result.metadata.page_count:
                print(f"  Pages: {result.metadata.page_count}")

        if result.error:
            print(f"\nError: {result.error}")

        if result.warnings:
            print(f"\nWarnings:")
            for w in result.warnings:
                print(f"  - {w}")


async def run_folder(agent: DocumentAgent, args, output_json: bool):
    """Run folder command."""
    print(f"Processing folder: {args.folder_path}...")
    print(f"Pattern: {args.pattern}, Recursive: {args.recursive}")

    config = ChunkingConfig(
        strategy=ChunkingStrategy(args.strategy),
        chunk_size=args.chunk_size,
    )

    results = await agent.process_folder(
        args.folder_path,
        pattern=args.pattern,
        recursive=args.recursive,
        chunking_config=config,
    )

    if output_json:
        print(json.dumps([r.model_dump() for r in results], indent=2, default=str))
    else:
        print(f"\n=== Folder Processing Complete ===")
        print(f"Total Documents: {len(results)}")

        completed = [r for r in results if r.status.value == "completed"]
        failed = [r for r in results if r.status.value == "failed"]

        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")

        total_chunks = sum(r.total_chunks for r in completed)
        total_embedded = sum(r.chunks_embedded for r in completed)
        print(f"Total Chunks: {total_chunks}")
        print(f"Total Embedded: {total_embedded}")

        if completed:
            print(f"\nSuccessful:")
            for r in completed[:10]:
                print(f"  - {Path(r.file_path).name}: {r.total_chunks} chunks")
            if len(completed) > 10:
                print(f"  ... and {len(completed) - 10} more")

        if failed:
            print(f"\nFailed:")
            for r in failed[:5]:
                print(f"  - {Path(r.file_path).name}: {r.error}")


async def run_search(agent: DocumentAgent, args, output_json: bool):
    """Run search command."""
    query = " ".join(args.query)
    print(f"Searching for: {query}\n")

    results = await agent.search(
        query,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    if output_json:
        output = []
        for r in results:
            output.append({
                "score": r.score,
                "document": r.document_path,
                "content": r.chunk.content[:200] + "..." if len(r.chunk.content) > 200 else r.chunk.content,
                "chunk_index": r.chunk.chunk_index,
            })
        print(json.dumps(output, indent=2))
    else:
        print(f"=== Search Results ({len(results)}) ===\n")

        for i, result in enumerate(results):
            print(f"[{i+1}] Score: {result.score:.3f}")
            print(f"    Source: {result.document_path}")
            print(f"    Chunk: {result.chunk.chunk_index}")

            # Show preview
            content = result.chunk.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    Preview: {content[:100]}...")
            print()


async def run_stats(agent: DocumentAgent, args, output_json: bool):
    """Run stats command."""
    stats = await agent.get_stats()

    if output_json:
        print(json.dumps(stats, indent=2))
    else:
        print("=== Document Agent Statistics ===\n")
        print(f"Cached Documents: {stats['cached_documents']}")
        print(f"Cached Chunks: {stats['cached_chunks']}")
        print(f"Vector Store: {stats['vector_store']}")
        print(f"Vector Store Connected: {stats['vector_store_connected']}")
        print(f"Chunking Strategy: {stats['chunking_strategy']}")
        print(f"Chunk Size: {stats['chunk_size']}")


async def run_interactive(agent: DocumentAgent, args):
    """Run interactive mode."""
    print("\n=== EWR Document Agent Interactive Mode ===")
    print("Commands: process <file>, folder <path>, search <query>, stats, quit\n")

    # Process initial folder if specified
    if args.folder:
        print(f"Processing initial folder: {args.folder}")
        results = await agent.process_folder(args.folder, recursive=True)
        completed = len([r for r in results if r.status.value == "completed"])
        print(f"Processed {completed} documents.\n")

    while True:
        try:
            line = input("doc> ").strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "process" and arg:
                result = await agent.process_document(arg)
                print(f"Status: {result.status.value}, Chunks: {result.total_chunks}")
            elif cmd == "folder" and arg:
                folder_parts = arg.split()
                folder_path = folder_parts[0]
                recursive = "-r" in folder_parts or "--recursive" in folder_parts
                results = await agent.process_folder(folder_path, recursive=recursive)
                completed = len([r for r in results if r.status.value == "completed"])
                print(f"Processed {completed}/{len(results)} documents")
            elif cmd == "search" and arg:
                results = await agent.search(arg, top_k=5)
                print(f"\nFound {len(results)} results:")
                for i, r in enumerate(results):
                    print(f"  [{i+1}] {r.score:.3f} - {r.chunk.content[:80]}...")
            elif cmd == "stats":
                stats = await agent.get_stats()
                print(f"Documents: {stats['cached_documents']}, Chunks: {stats['cached_chunks']}")
            elif cmd == "help":
                print("Commands:")
                print("  process <file>    - Process a document")
                print("  folder <path> [-r] - Process folder")
                print("  search <query>    - Search documents")
                print("  stats             - Show statistics")
                print("  quit              - Exit")
            else:
                print(f"Unknown command: {cmd}. Type 'help' for commands.")

        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

    print("Goodbye!")


async def async_main(args):
    """Async main function."""
    config = load_config(
        name="document-cli",
        llm_backend=args.backend,
        llm_model=args.model or "nomic-embed-text",
    )

    vector_config = VectorStoreConfig(
        store_type="mongodb",
        connection_string=args.mongodb_uri,
        database_name=args.database,
        collection_name=args.collection,
    )

    agent = DocumentAgent(config=config, vector_config=vector_config)
    await agent.start()

    try:
        if args.command == "process":
            await run_process(agent, args, args.json)
        elif args.command == "folder":
            await run_folder(agent, args, args.json)
        elif args.command == "search":
            await run_search(agent, args, args.json)
        elif args.command == "stats":
            await run_stats(agent, args, args.json)
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
