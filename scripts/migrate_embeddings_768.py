"""
Embedding Migration Script - 384 to 768 dimensions
Regenerates all vector embeddings using nomic-embed-text-v1.5 via llama.cpp

Run from project root:
    python scripts/migrate_embeddings_768.py

Or with venv:
    python_services/venv/Scripts/python.exe scripts/migrate_embeddings_768.py
"""
import asyncio
import sys
import os
import time

# Add python_services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_services'))

from pymongo import MongoClient
import aiohttp
from aiohttp import ClientTimeout

# Configuration - Using IP to avoid DNS timeout issues
MONGODB_URI = "mongodb://10.101.20.29:27017/?directConnection=true"
DATABASE = "rag_server"
EMBEDDING_URL = "http://10.101.20.29:8083/embedding"
BATCH_SIZE = 50  # Process documents in batches
TEXT_LIMIT = 4000  # Limit text length to avoid "input too large" errors


def format_time(seconds: float) -> str:
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_progress(current: int, total: int, start_time: float, prefix: str = ""):
    """Print a progress bar with ETA"""
    if total == 0:
        return

    elapsed = time.time() - start_time
    percent = current / total
    bar_width = 30
    filled = int(bar_width * percent)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Calculate ETA
    if current > 0:
        rate = current / elapsed
        remaining = (total - current) / rate if rate > 0 else 0
        eta_str = format_time(remaining)
    else:
        eta_str = "calculating..."

    # Build progress line
    progress_line = f"\r  {prefix}[{bar}] {current}/{total} ({percent*100:.1f}%) | Elapsed: {format_time(elapsed)} | ETA: {eta_str}    "

    print(progress_line, end="", flush=True)

# Collection configs: collection_name -> text_field to embed
# documents uses 'content', others use 'embedding_text'
COLLECTIONS = {
    "documents": "content",
    "code_classes": "embedding_text",
    "code_context": "embedding_text",
    "code_methods": "embedding_text",
    "sql_stored_procedures": "embedding_text",
    "sql_examples": "embedding_text",
    "sql_schema_context": "embedding_text",
}


async def get_embedding(session: aiohttp.ClientSession, text: str) -> list:
    """Get embedding from llama.cpp server"""
    async with session.post(EMBEDDING_URL, json={"content": text}) as response:
        if response.status != 200:
            error = await response.text()
            raise RuntimeError(f"Embedding failed: {error}")
        data = await response.json()
        # llama.cpp format: [{"index": 0, "embedding": [[...floats...]]}]
        return data[0]["embedding"][0]


def get_text_for_embedding(doc: dict, field: str) -> str:
    """Extract text from document field for embedding"""
    value = doc.get(field)
    if value and isinstance(value, str):
        return value
    # Fallback to common fields
    for fallback in ["embedding_text", "content", "text", "description"]:
        value = doc.get(fallback)
        if value and isinstance(value, str):
            return value
    return None


async def migrate_collection(collection_name: str, text_field: str):
    """Migrate a single collection to 768-dim embeddings"""
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE]
    collection = db[collection_name]

    # Count documents with vectors
    total = collection.count_documents({"vector": {"$exists": True}})
    if total == 0:
        print(f"  {collection_name}: No documents with vectors, skipping")
        return 0, 0

    print(f"\n  {collection_name}: Migrating {total} documents...")

    updated = 0
    errors = 0
    start_time = time.time()

    async with aiohttp.ClientSession(timeout=ClientTimeout(total=60)) as session:
        cursor = collection.find({"vector": {"$exists": True}}, no_cursor_timeout=True)
        batch = []
        processed = 0

        for doc in cursor:
            text = get_text_for_embedding(doc, text_field)

            if not text:
                errors += 1
                processed += 1
                continue

            batch.append((doc["_id"], text))

            if len(batch) >= BATCH_SIZE:
                # Process batch
                for doc_id, doc_text in batch:
                    try:
                        embedding = await get_embedding(session, doc_text[:TEXT_LIMIT])
                        collection.update_one(
                            {"_id": doc_id},
                            {"$set": {"vector": embedding}}
                        )
                        updated += 1
                    except Exception as e:
                        print(f"\n    Error on {doc_id}: {e}")
                        errors += 1

                    processed += 1
                    print_progress(processed, total, start_time)

                batch = []

        # Process remaining batch
        for doc_id, doc_text in batch:
            try:
                embedding = await get_embedding(session, doc_text[:TEXT_LIMIT])
                collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"vector": embedding}}
                )
                updated += 1
            except Exception as e:
                print(f"\n    Error on {doc_id}: {e}")
                errors += 1

            processed += 1
            print_progress(processed, total, start_time)

        cursor.close()

    client.close()

    # Final stats for this collection
    elapsed = time.time() - start_time
    rate = updated / elapsed if elapsed > 0 else 0
    print(f"\n  {collection_name}: Done - {updated} updated, {errors} errors in {format_time(elapsed)} ({rate:.1f} docs/sec)")

    return updated, errors


async def main():
    print("=" * 60)
    print("Embedding Migration: 384 -> 768 dimensions")
    print("=" * 60)
    print(f"MongoDB: {MONGODB_URI}/{DATABASE}")
    print(f"Embedding Service: {EMBEDDING_URL}")
    print()

    # Test embedding service
    print("Testing embedding service...")
    async with aiohttp.ClientSession(timeout=ClientTimeout(total=60)) as session:
        try:
            embedding = await get_embedding(session, "test")
            print(f"  OK - Embedding dimensions: {len(embedding)}")
            if len(embedding) != 768:
                print(f"  WARNING: Expected 768 dimensions, got {len(embedding)}")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            print("  Make sure the embedding service is running on EWRSPT-AI:8083")
            return

    print()
    print("Starting migration...")

    overall_start = time.time()
    total_updated = 0
    total_errors = 0
    collection_count = len(COLLECTIONS)

    for idx, (collection_name, text_field) in enumerate(COLLECTIONS.items(), 1):
        print(f"\n[{idx}/{collection_count}] Processing {collection_name}...")
        try:
            updated, errors = await migrate_collection(collection_name, text_field)
            total_updated += updated
            total_errors += errors
        except Exception as e:
            print(f"\n  {collection_name}: FAILED - {e}")

    # Final summary
    overall_elapsed = time.time() - overall_start
    overall_rate = total_updated / overall_elapsed if overall_elapsed > 0 else 0

    print()
    print("=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print(f"  Total documents updated: {total_updated}")
    print(f"  Total errors:            {total_errors}")
    print(f"  Total time:              {format_time(overall_elapsed)}")
    print(f"  Average rate:            {overall_rate:.1f} docs/sec")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
