import subprocess
import time
import requests
import os

def ensure_qdrant_running():
    """Ensure Qdrant server is running, install if needed, and start it if not running."""
    try:
        # First check if Docker is installed
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            print("✅ Docker is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Docker is required but not installed.")
            print("Please install Docker from https://docs.docker.com/desktop/install/mac-install/")
            raise Exception("Docker is required but not installed")

        # Check if Qdrant container exists and is running
        try:
            response = requests.get("http://localhost:6333/dashboard/")
            if response.status_code == 200:
                print("✅ Qdrant server is already running")
                return
        except requests.exceptions.ConnectionError:
            print("🚀 Starting Qdrant server...")
            try:
                # Check if container exists
                result = subprocess.run(["docker", "ps", "-a", "--filter", "name=qdrant", "--format", "{{.Names}}"], capture_output=True, text=True)
                if "qdrant" not in result.stdout:
                    print("📦 Creating Qdrant container...")
                    subprocess.run([
                        "docker", "run", "-d",
                        "--name", "qdrant",
                        "-p", "6333:6333",
                        "-p", "6334:6334",
                        "-v", "qdrant_storage:/qdrant/storage:consistent",
                        "qdrant/qdrant"
                    ], check=True)
                else:
                    print("🔄 Starting existing Qdrant container...")
                    subprocess.run(["docker", "start", "qdrant"], check=True)
                
                # Wait for server to start
                print("⏳ Waiting for Qdrant server to start...")
                max_attempts = 10
                for i in range(max_attempts):
                    try:
                        requests.get("http://localhost:6333/dashboard/")
                        print("✅ Qdrant server started successfully")
                        return
                    except requests.exceptions.ConnectionError:
                        if i < max_attempts - 1:
                            time.sleep(2)
                            print(f"   Attempt {i + 1}/{max_attempts}...")
                        else:
                            raise Exception("Failed to start Qdrant server after multiple attempts")
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to start Qdrant container: {str(e)}")
                
    except Exception as e:
        print(f"❌ Error with Qdrant setup: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the function
    ensure_qdrant_running()