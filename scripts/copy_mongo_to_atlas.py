"""
Copy data from Native MongoDB (27017) to Atlas Local (27018)
Keeps source data intact as backup.

Run:
    python_services/venv/Scripts/python.exe scripts/copy_mongo_to_atlas.py
"""
from pymongo import MongoClient

# Source: Native MongoDB on port 27017
SOURCE_URI = "mongodb://10.101.20.29:27017/?directConnection=true"

# Destination: Atlas Local on port 27018
DEST_URI = "mongodb://10.101.20.29:27018/?directConnection=true"

DATABASE = "rag_server"


def copy_collection(source_db, dest_db, coll_name):
    """Copy all documents from source to destination collection"""
    source_coll = source_db[coll_name]
    dest_coll = dest_db[coll_name]

    count = source_coll.count_documents({})
    if count == 0:
        print(f"  {coll_name}: empty, skipping")
        return 0

    print(f"  {coll_name}: copying {count} documents...")

    # Clear destination first
    dest_coll.delete_many({})

    # Copy in batches
    batch_size = 1000
    copied = 0

    cursor = source_coll.find({})
    batch = []

    for doc in cursor:
        batch.append(doc)
        if len(batch) >= batch_size:
            dest_coll.insert_many(batch)
            copied += len(batch)
            print(f"    {copied}/{count} ({100*copied//count}%)")
            batch = []

    # Insert remaining
    if batch:
        dest_coll.insert_many(batch)
        copied += len(batch)

    print(f"  {coll_name}: done ({copied} documents)")
    return copied


def main():
    print("=" * 60)
    print("Copy MongoDB Data: Native (27017) -> Atlas Local (27018)")
    print("=" * 60)
    print(f"Source: {SOURCE_URI}")
    print(f"Destination: {DEST_URI}")
    print()

    # Connect to both
    source_client = MongoClient(SOURCE_URI, serverSelectionTimeoutMS=5000)
    dest_client = MongoClient(DEST_URI, serverSelectionTimeoutMS=5000)

    # Test connections
    print("Testing connections...")
    try:
        source_client.admin.command('ping')
        print("  Source (27017): OK")
    except Exception as e:
        print(f"  Source (27017): FAILED - {e}")
        return

    try:
        dest_client.admin.command('ping')
        print("  Destination (27018): OK")
    except Exception as e:
        print(f"  Destination (27018): FAILED - {e}")
        return

    source_db = source_client[DATABASE]
    dest_db = dest_client[DATABASE]

    # Get collections from source
    collections = source_db.list_collection_names()
    print(f"\nFound {len(collections)} collections in source")
    print()

    print("Copying collections...")
    total_copied = 0

    for coll_name in sorted(collections):
        try:
            copied = copy_collection(source_db, dest_db, coll_name)
            total_copied += copied
        except Exception as e:
            print(f"  {coll_name}: ERROR - {e}")

    print()
    print("=" * 60)
    print(f"Copy complete. Total documents copied: {total_copied}")
    print("Source data preserved as backup.")
    print("=" * 60)

    source_client.close()
    dest_client.close()


if __name__ == "__main__":
    main()
