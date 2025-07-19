#!/usr/bin/env python
"""
Start Script for Multi-Agent Finance Assistant
This script handles initialization and startup of all required services.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Ensure we're in the correct directory
project_root = Path(__file__).parent.absolute()
os.chdir(project_root)

# Load environment variables - Updated to use your .env file
load_dotenv(dotenv_path="config/.env")
load_dotenv(dotenv_path="config/secrets.env")  # Fallback

def check_dependencies():
    """Check for required dependencies and install them if missing."""
    try:
        print("Checking for required dependencies...")
        import importlib.util        # List of critical dependencies to check
        dependencies = [
            {"name": "newspaper", "package": "newspaper3k", "optional": True},
            {"name": "sec_edgar_downloader", "package": "sec-edgar-downloader", "optional": True},
            {"name": "fastapi", "package": "fastapi", "optional": False},
            {"name": "streamlit", "package": "streamlit", "optional": False},
            {"name": "sentence_transformers", "package": "sentence-transformers", "optional": True},
            {"name": "faiss", "package": "faiss-cpu", "optional": True},
            {"name": "langchain_community", "package": "langchain-community", "optional": True},
            {"name": "whisper", "package": "openai-whisper", "optional": True},
            {"name": "TTS", "package": "TTS", "optional": True},
            {"name": "sounddevice", "package": "sounddevice", "optional": True}
        ]
        
        for dep in dependencies:
            if importlib.util.find_spec(dep["name"]) is None:
                if dep["optional"]:
                    print(f"Optional dependency {dep['name']} not found. Will use fallback implementation.")
                else:
                    print(f"Required dependency {dep['name']} not found. Installing {dep['package']}...")
                    import subprocess
                    subprocess.check_call([sys.executable, "-m", "pip", "install", dep["package"]])
    except Exception as e:
        print(f"Warning: Dependency check failed: {str(e)}")
        print("Continuing anyway, but some features may not work.")

def check_env_file():
    """Check if environment file exists and create it if not."""
    # Check for your existing .env file first
    env_file = project_root / "config" / ".env"
    secrets_file = project_root / "config" / "secrets.env"
    example_file = project_root / "config" / "secrets.env.example"
    
    if env_file.exists():
        print("✅ Environment file found: config/.env")
        return
    elif secrets_file.exists():
        print("✅ Environment file found: config/secrets.env")
        return
    else:
        print("Environment file not found. Creating from example...")
        if example_file.exists():
            with open(example_file, 'r') as src, open(secrets_file, 'w') as dest:
                dest.write(src.read())
            print("Created environment file. Please edit 'config/secrets.env' with your API keys.")
        else:
            print("Error: Example environment file not found.")
            sys.exit(1)

def create_vector_store():
    """Initialize vector store if it doesn't exist."""
    vector_store_path = project_root / "data" / "vector_store"
    if not vector_store_path.exists():
        print("Creating vector store directory...")
        vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Create a basic initial document for the vector store
        try:
            print("Initializing vector store with basic financial data...")
            # We'll do this directly from the server
            print("Vector store path created. Will be initialized on first use.")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {str(e)}")
            print("The application will continue but RAG functionality may be limited.")

def start_services(fastapi_only=False):
    """Start FastAPI and Streamlit servers."""
    # Ensure the server is accessible from other machines
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8000"))
    
    if fastapi_only:
        print(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(
            "orchestrator.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info"
        )
    else:
        print("Starting both FastAPI and Streamlit servers...")
        
        # Start FastAPI in a separate process
        fastapi_cmd = ["python", "-m", "uvicorn", "orchestrator.main:app", "--reload", "--port", "8000"]
        fastapi_process = subprocess.Popen(fastapi_cmd)
        
        # Give FastAPI time to start
        print("Waiting for FastAPI server to initialize...")
        time.sleep(5)
        
        # Start Streamlit in the foreground
        print("Starting Streamlit frontend...")
        streamlit_cmd = ["streamlit", "run", "streamlit_app/app.py"]
        subprocess.run(streamlit_cmd)
        
        # If Streamlit is closed, also terminate FastAPI
        print("Shutting down servers...")
        fastapi_process.terminate()

if __name__ == "__main__":
    print("==== Multi-Agent Finance Assistant ====")
    check_env_file()
    check_dependencies()
    create_vector_store()
    
    # Check if we should only start FastAPI (for Docker environments)
    fastapi_only = "--fastapi-only" in sys.argv
    start_services(fastapi_only)