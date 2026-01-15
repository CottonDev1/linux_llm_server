#!/usr/bin/env python3
"""
Update a single summary in MongoDB.
Usage: python update_summary.py <type> <id> <summary_json>
  type: 'schema' or 'procedure'
  id: MongoDB _id as string
  summary_json: JSON string with summary, purpose/operations, keywords
"""
import asyncio
import json
import sys
from datetime import datetime

sys.path.insert(0, '.')
import config
config.MONGODB_URI = 'mongodb://EWRSPT-AI:27018/?directConnection=true&serverSelectionTimeoutMS=30000&connectTimeoutMS=10000'

async def update(record_type: str, record_id: str, summary_data: dict):
    from mongodb import get_mongodb_service
    from config import COLLECTION_SQL_SCHEMA_CONTEXT, COLLECTION_SQL_STORED_PROCEDURES
    from bson import ObjectId

    mongo = get_mongodb_service()
    await mongo.initialize()

    if record_type == 'schema':
        collection = mongo.db[COLLECTION_SQL_SCHEMA_CONTEXT]
        update_data = {
            "summary": summary_data.get('summary', ''),
            "purpose": summary_data.get('purpose', ''),
            "keywords": summary_data.get('keywords', []),
            "summary_generated_at": datetime.utcnow()
        }
    else:
        collection = mongo.db[COLLECTION_SQL_STORED_PROCEDURES]
        update_data = {
            "summary": summary_data.get('summary', ''),
            "operations": summary_data.get('operations', []),
            "keywords": summary_data.get('keywords', []),
            "summary_generated_at": datetime.utcnow()
        }

    result = await collection.update_one(
        {"_id": ObjectId(record_id)},
        {"$set": update_data}
    )

    if result.modified_count > 0:
        print(f"Updated {record_type} {record_id}")
    else:
        print(f"No update for {record_type} {record_id}")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python update_summary.py <schema|procedure> <id> '<json>'")
        sys.exit(1)

    record_type = sys.argv[1]
    record_id = sys.argv[2]
    summary_json = sys.argv[3]

    summary_data = json.loads(summary_json)
    asyncio.run(update(record_type, record_id, summary_data))
