"""
Extract EWR database schemas, views, and stored procedures from EWRSQLTEST
and store them in MongoDB.
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

def extract_table_schemas(sql_conn, mongo_db):
    """Extract all table schemas with columns."""
    print("\n=== Extracting Table Schemas ===")
    cursor = sql_conn.cursor(as_dict=True)

    # Get all tables with their columns
    query = """
    SELECT
        t.TABLE_SCHEMA as schema_name,
        t.TABLE_NAME as table_name,
        c.COLUMN_NAME as column_name,
        c.DATA_TYPE as data_type,
        c.CHARACTER_MAXIMUM_LENGTH as max_length,
        c.IS_NULLABLE as is_nullable,
        c.COLUMN_DEFAULT as column_default,
        c.ORDINAL_POSITION as ordinal_position
    FROM INFORMATION_SCHEMA.TABLES t
    JOIN INFORMATION_SCHEMA.COLUMNS c
        ON t.TABLE_NAME = c.TABLE_NAME AND t.TABLE_SCHEMA = c.TABLE_SCHEMA
    WHERE t.TABLE_TYPE = 'BASE TABLE'
    ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    # Group by table
    tables = {}
    for row in rows:
        key = f"{row['schema_name']}.{row['table_name']}"
        if key not in tables:
            tables[key] = {
                'id': f"ewr_{row['schema_name']}_{row['table_name']}",
                'database': 'ewr',
                'server': SQL_SERVER,
                'table_name': key,
                'schema_name': row['schema_name'],
                'columns': [],
                'extracted_at': datetime.utcnow()
            }
        tables[key]['columns'].append({
            'name': row['column_name'],
            'data_type': row['data_type'],
            'max_length': row['max_length'],
            'is_nullable': row['is_nullable'],
            'default': row['column_default'],
            'ordinal': row['ordinal_position']
        })

    # Clear existing EWR schemas and insert new ones
    collection = mongo_db['sql_schema_context']
    deleted = collection.delete_many({'database': 'ewr'})
    print(f"  Deleted {deleted.deleted_count} existing EWR schemas")

    if tables:
        result = collection.insert_many(list(tables.values()))
        print(f"  Inserted {len(result.inserted_ids)} table schemas")

    return len(tables)

def extract_views(sql_conn, mongo_db):
    """Extract all views with their definitions."""
    print("\n=== Extracting Views ===")
    cursor = sql_conn.cursor(as_dict=True)

    # Get all views with their definitions
    query = """
    SELECT
        SCHEMA_NAME(v.schema_id) as schema_name,
        v.name as view_name,
        m.definition as view_definition,
        v.create_date,
        v.modify_date
    FROM sys.views v
    JOIN sys.sql_modules m ON v.object_id = m.object_id
    ORDER BY schema_name, view_name
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    views = []
    for row in rows:
        views.append({
            'id': f"ewr_view_{row['schema_name']}_{row['view_name']}",
            'database': 'ewr',
            'server': SQL_SERVER,
            'view_name': f"{row['schema_name']}.{row['view_name']}",
            'schema_name': row['schema_name'],
            'definition': row['view_definition'],
            'create_date': row['create_date'],
            'modify_date': row['modify_date'],
            'extracted_at': datetime.utcnow()
        })

    # Create sql_views collection if needed and insert
    collection = mongo_db['sql_views']
    deleted = collection.delete_many({'database': 'ewr'})
    print(f"  Deleted {deleted.deleted_count} existing EWR views")

    if views:
        result = collection.insert_many(views)
        print(f"  Inserted {len(result.inserted_ids)} views")

    return len(views)

def extract_stored_procedures(sql_conn, mongo_db):
    """Extract all stored procedures with their definitions."""
    print("\n=== Extracting Stored Procedures ===")
    cursor = sql_conn.cursor(as_dict=True)

    # Get all stored procedures with their definitions
    query = """
    SELECT
        SCHEMA_NAME(p.schema_id) as schema_name,
        p.name as procedure_name,
        m.definition as procedure_definition,
        p.create_date,
        p.modify_date
    FROM sys.procedures p
    JOIN sys.sql_modules m ON p.object_id = m.object_id
    ORDER BY schema_name, procedure_name
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    procedures = []
    for row in rows:
        procedures.append({
            'id': f"ewr_proc_{row['schema_name']}_{row['procedure_name']}",
            'database': 'ewr',
            'server': SQL_SERVER,
            'procedure_name': f"{row['schema_name']}.{row['procedure_name']}",
            'name': row['procedure_name'],  # Short name for compatibility
            'schema_name': row['schema_name'],
            'definition': row['procedure_definition'],
            'create_date': row['create_date'],
            'modify_date': row['modify_date'],
            'extracted_at': datetime.utcnow()
        })

    # Clear existing EWR procedures and insert new ones
    collection = mongo_db['sql_stored_procedures']
    deleted = collection.delete_many({'database': 'ewr'})
    print(f"  Deleted {deleted.deleted_count} existing EWR procedures")

    if procedures:
        result = collection.insert_many(procedures)
        print(f"  Inserted {len(result.inserted_ids)} stored procedures")

    return len(procedures)

def main():
    print("=" * 60)
    print("EWR Database Extraction from EWRSQLTEST")
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

    # Extract all data
    table_count = extract_table_schemas(sql_conn, mongo_db)
    view_count = extract_views(sql_conn, mongo_db)
    proc_count = extract_stored_procedures(sql_conn, mongo_db)

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Tables:            {table_count}")
    print(f"  Views:             {view_count}")
    print(f"  Stored Procedures: {proc_count}")
    print("=" * 60)

    sql_conn.close()

if __name__ == '__main__':
    main()
