"""
Extract EWR database views with columns from EWRSQLTEST
and store them in MongoDB sql_schema_context collection for SQL pipeline use.
"""
import pymssql
from pymongo import MongoClient
from datetime import datetime

# SQL Server connection
SQL_SERVER = 'EWRSQLTEST'
SQL_DATABASE = 'EWR'
SQL_USER = 'EWR\\chad.walker'
SQL_PASSWORD = '6454@@Christina'

# MongoDB connection
MONGO_HOST = 'EWRSPT-AI'
MONGO_PORT = 27018
MONGO_DB = 'rag_server'

def get_sql_connection():
    return pymssql.connect(
        server=SQL_SERVER,
        database=SQL_DATABASE,
        user=SQL_USER,
        password=SQL_PASSWORD,
        port='1433'
    )

def get_mongo_connection():
    client = MongoClient(f'mongodb://{MONGO_HOST}:{MONGO_PORT}/', directConnection=True)
    return client[MONGO_DB]

def extract_views_with_columns(sql_conn, mongo_db):
    """Extract all views with their columns for SQL pipeline use."""
    print("\n=== Extracting Views with Columns ===")
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

    # Group by view
    views = {}
    for row in rows:
        key = f"{row['schema_name']}.{row['view_name']}"
        if key not in views:
            views[key] = {
                'id': f"ewr_view_{row['schema_name']}_{row['view_name']}",
                'database': 'ewr',
                'server': SQL_SERVER,
                'table_name': key,  # Use table_name for compatibility with schema service
                'schema_name': row['schema_name'],
                'object_type': 'view',  # Mark as view
                'columns': [],
                'definition': row['view_definition'],
                'primary_keys': [],  # Views don't have PKs
                'foreign_keys': [],  # Views don't have FKs
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

    # Insert into sql_schema_context collection (alongside tables)
    collection = mongo_db['sql_schema_context']

    # Delete existing EWR views from schema context
    deleted = collection.delete_many({'database': 'ewr', 'object_type': 'view'})
    print(f"  Deleted {deleted.deleted_count} existing EWR views from schema context")

    if views:
        result = collection.insert_many(list(views.values()))
        print(f"  Inserted {len(result.inserted_ids)} views into sql_schema_context")

    # Also update the sql_views collection with column info
    views_collection = mongo_db['sql_views']
    for view_key, view_data in views.items():
        views_collection.update_one(
            {'database': 'ewr', 'view_name': view_key},
            {'$set': {
                'columns': view_data['columns'],
                'column_count': len(view_data['columns'])
            }}
        )
    print(f"  Updated {len(views)} views in sql_views collection with columns")

    return len(views)

def main():
    print("=" * 60)
    print("EWR Views with Columns Extraction from EWRSQLTEST")
    print("=" * 60)
    print(f"SQL Server: {SQL_SERVER}/{SQL_DATABASE}")
    print(f"MongoDB: {MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}")

    # Connect to SQL Server
    print("\nConnecting to SQL Server...")
    sql_conn = get_sql_connection()
    print("  Connected!")

    # Connect to MongoDB
    print("Connecting to MongoDB...")
    mongo_db = get_mongo_connection()
    print("  Connected!")

    # Extract views with columns
    view_count = extract_views_with_columns(sql_conn, mongo_db)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Views extracted with columns: {view_count}")
    print(f"  Added to: sql_schema_context (for SQL pipeline)")
    print(f"  Updated: sql_views (with column info)")
    print("=" * 60)

    sql_conn.close()

if __name__ == '__main__':
    main()
