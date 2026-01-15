"""Verify vector dimensions after migration"""
from pymongo import MongoClient

MONGODB_URI = "mongodb://10.101.20.29:27017/?directConnection=true"
DATABASE = "rag_server"

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
db = client[DATABASE]

collections = ['documents', 'code_classes', 'code_methods', 'sql_stored_procedures', 'sql_schema_context']

print("Verifying vector dimensions:")
print("-" * 50)

for coll_name in collections:
    coll = db[coll_name]

    # Count documents by vector size
    pipeline = [
        {'$match': {'vector': {'$exists': True}}},
        {'$project': {'dim': {'$size': '$vector'}}},
        {'$group': {'_id': '$dim', 'count': {'$sum': 1}}}
    ]

    results = list(coll.aggregate(pipeline))

    if results:
        dims_str = ", ".join([f"{r['_id']} dims: {r['count']}" for r in sorted(results, key=lambda x: x['_id'])])
        print(f"{coll_name}: {dims_str}")
    else:
        print(f"{coll_name}: no vectors")

client.close()
