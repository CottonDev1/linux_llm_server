try:
    import fastapi
    import uvicorn
    import httpx
    print("All packages installed")
except ImportError as e:
    print(f"Missing: {e}")
