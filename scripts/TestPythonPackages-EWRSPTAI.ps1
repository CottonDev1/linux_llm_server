# Test if required Python packages are installed on EWRSPT-AI
$code = @"
try:
    import fastapi
    import uvicorn
    import httpx
    print("All packages installed")
except ImportError as e:
    print(f"Missing: {e}")
"@

& "C:\Python312\python.exe" -c $code
