# GenerateEmbeddings.ps1
# Generates embeddings for all tables in sql_schema_context that are missing them
# Run this on EWRSPT-AI where the embedding service is running

$pythonScript = @'
import asyncio
import aiohttp
from pymongo import MongoClient

async def generate_all_embeddings():
    client = MongoClient('mongodb://localhost:27018', directConnection=True)
    db = client['rag_server']

    # Find all tables without embeddings
    tables = list(db.sql_schema_context.find({
        '$or': [
            {'embedding': {'$exists': False}},
            {'embedding': []},
            {'embedding': None}
        ]
    }))

    print(f'Found {len(tables)} tables without embeddings')

    if len(tables) == 0:
        print('All tables already have embeddings!')
        client.close()
        return

    success_count = 0
    error_count = 0

    async with aiohttp.ClientSession() as session:
        for i, table in enumerate(tables):
            table_name = table.get('table_name', '')
            db_name = table.get('database', '')
            summary = table.get('summary', '')
            columns = table.get('columns', [])
            col_names = ', '.join([c.get('name', c) if isinstance(c, dict) else str(c) for c in columns[:10]])

            # Build embedding text
            embed_text = f'{table_name}: {summary}. Columns: {col_names}'

            try:
                async with session.post('http://localhost:8002/embed', json={'text': embed_text}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        db.sql_schema_context.update_one(
                            {'_id': table['_id']},
                            {'$set': {'embedding': data['embedding']}}
                        )
                        success_count += 1
                    else:
                        print(f'HTTP {resp.status} for {table_name}')
                        error_count += 1
            except Exception as e:
                print(f'Error on {db_name}.{table_name}: {e}')
                error_count += 1

            # Progress update every 50 tables
            if (i + 1) % 50 == 0:
                print(f'Progress: {i + 1}/{len(tables)} ({success_count} success, {error_count} errors)')

    print(f'\nComplete!')
    print(f'  Success: {success_count}')
    print(f'  Errors: {error_count}')
    print(f'  Total: {len(tables)}')

    client.close()

if __name__ == '__main__':
    asyncio.run(generate_all_embeddings())
'@

# Check if embedding service is running
Write-Host "Checking embedding service..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8002/health" -Method Get -TimeoutSec 5
    Write-Host "Embedding service is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Embedding service not running on port 8002" -ForegroundColor Red
    Write-Host "Start it with: cd C:\projects\embedding_service && python main.py" -ForegroundColor Yellow
    exit 1
}

# Run the Python script
Write-Host "`nGenerating embeddings for all tables..." -ForegroundColor Cyan
$pythonScript | python -
