"""
Extract views with columns from multiple databases and store in MongoDB.
Supports NCSQLTEST and EWRSQLTEST servers.
"""
import pymssql
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
MONGO_HOST = 'localhost'
MONGO_PORT = 27017
MONGO_DB = 'rag_server'

# Database configurations
DATABASES = [
    {
        'server': 'NCSQLTEST',
        'database': 'EWR.Gin.Entity',
        'mongo_db_name': 'ewr_gin_entity',
        'user': 'EWRUser',
        'password': '66a3904d69'
    },
    {
        'server': 'NCSQLTEST',
        'database': 'EWR.Marketing.Entity',
        'mongo_db_name': 'ewr_marketing_entity',
        'user': 'EWRUser',
        'password': '66a3904d69'
    },
    {
        'server': 'NCSQLTEST',
        'database': 'EWR.Warehouse.Entity',
        'mongo_db_name': 'ewr_warehouse_entity',
        'user': 'EWRUser',
        'password': '66a3904d69'
    }
]

def get_mongo_connection():
    client = MongoClient(f'mongodb://{MONGO_HOST}:{MONGO_PORT}/', directConnection=True)
    return client[MONGO_DB]

def extract_views_with_columns(db_config, mongo_db):
    """Extract all views with their columns from a database."""
    server = db_config['server']
    database = db_config['database']
    mongo_db_name = db_config['mongo_db_name']

    print(f"\n=== Extracting Views from {server}/{database} ===")

    try:
        sql_conn = pymssql.connect(
            server=server,
            database=database,
            user=db_config['user'],
            password=db_config['password'],
            port='1433'
        )
    except Exception as e:
        print(f"  Connection failed: {e}")
        return 0

    cursor = sql_conn.cursor(as_dict=True)

    # Get all views with their columns
    query = """
    SELECT
        SCHEMA_NAME(v.schema_id) as schema_name,
        v.name as view_name,
        c.name as column_name,
        t.name as data_type,
        c.max_length,
        c.is_nullable,
        c.column_id as ordinal_position,
        m.definition as view_definition
    FROM sys.views v
    JOIN sys.columns c ON v.object_id = c.object_id
    JOIN sys.types t ON c.user_type_id = t.user_type_id
    JOIN sys.sql_modules m ON v.object_id = m.object_id
    ORDER BY schema_name, view_name, c.column_id
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        print(f"  No views found")
        sql_conn.close()
        return 0

    # Group by view
    views = {}
    for row in rows:
        key = f"{row['schema_name']}.{row['view_name']}"
        if key not in views:
            views[key] = {
                'id': f"{mongo_db_name}_view_{row['schema_name']}_{row['view_name']}",
                'database': mongo_db_name,
                'server': server,
                'table_name': key,
                'schema_name': row['schema_name'],
                'object_type': 'view',
                'columns': [],
                'definition': row['view_definition'],
                'primary_keys': [],
                'foreign_keys': [],
                'extracted_at': datetime.utcnow()
            }

        # Determine type string
        data_type = row['data_type']
        max_length = row['max_length']
        if data_type in ('varchar', 'nvarchar', 'char', 'nchar'):
            if max_length == -1:
                type_str = f"{data_type}(max)"
            elif data_type.startswith('n'):
                type_str = f"{data_type}({max_length // 2})"
            else:
                type_str = f"{data_type}({max_length})"
        else:
            type_str = data_type

        views[key]['columns'].append({
            'name': row['column_name'],
            'type': type_str,
            'data_type': row['data_type'],
            'max_length': row['max_length'],
            'nullable': row['is_nullable'],
            'ordinal': row['ordinal_position']
        })

    print(f"  Found {len(views)} views with columns")

    # Insert into sql_schema_context
    schema_coll = mongo_db['sql_schema_context']
    deleted = schema_coll.delete_many({'database': mongo_db_name, 'object_type': 'view'})
    print(f"  Deleted {deleted.deleted_count} existing views from schema context")

    if views:
        result = schema_coll.insert_many(list(views.values()))
        print(f"  Inserted {len(result.inserted_ids)} views into sql_schema_context")

    # Also insert/update sql_views collection
    views_coll = mongo_db['sql_views']
    deleted_views = views_coll.delete_many({'database': mongo_db_name})
    print(f"  Deleted {deleted_views.deleted_count} existing views from sql_views")

    views_for_collection = []
    for view_key, view_data in views.items():
        views_for_collection.append({
            'id': view_data['id'],
            'database': mongo_db_name,
            'server': server,
            'view_name': view_key,
            'schema_name': view_data['schema_name'],
            'definition': view_data['definition'],
            'columns': view_data['columns'],
            'column_count': len(view_data['columns']),
            'extracted_at': datetime.utcnow()
        })

    if views_for_collection:
        views_coll.insert_many(views_for_collection)
        print(f"  Inserted {len(views_for_collection)} views into sql_views")

    sql_conn.close()
    return len(views)

def main():
    print("=" * 60)
    print("Multi-Database Views Extraction")
    print("=" * 60)

    mongo_db = get_mongo_connection()

    total_views = 0
    for db_config in DATABASES:
        count = extract_views_with_columns(db_config, mongo_db)
        total_views += count

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total views extracted: {total_views}")
    print("=" * 60)

if __name__ == '__main__':
    main()
