# Claude Code Instructions - Scripts

## MongoDB Configuration

**MongoDB runs on localhost:27017**

When running scripts, use the correct MongoDB URI:

```bash
export MONGODB_URI="mongodb://localhost:27017/?directConnection=true"
export MONGODB_DATABASE="rag_server"
```

Do NOT use:
- Port 27018 (old Atlas Local)
- Port 27019 (incorrect)
- EWRSPT-AI hostname (use localhost)

## Script Execution

Many scripts in this directory have hardcoded incorrect ports. Before running, either:

1. Set environment variables (preferred):
   ```bash
   MONGODB_URI="mongodb://localhost:27017/?directConnection=true" python script.py
   ```

2. Or edit the script to use the correct port

## Known Issues

Scripts with wrong MongoDB ports that need fixing:
- `copy_mongo_to_atlas.py` - references 27018
- `prefect_pipelines/test_flows/` - references 27018
- `MongoAtlas/` scripts - various port issues
