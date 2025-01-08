import subprocess
import sys

def main():
    """Post-installation script to run playwright install and verify OpenCV and Tavily."""
    try:
        # Install playwright browsers
        subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        print("Successfully installed playwright browsers")
        
        # Verify OpenCV installation
        try:
            import cv2
            print(f"Successfully verified OpenCV installation (version {cv2.__version__})")
        except ImportError as e:
            print(f"Warning: Error importing OpenCV: {e}", file=sys.stderr)
            
        # Verify Tavily installation
        try:
            import tavily
            print(f"Successfully verified Tavily installation (version {tavily.__version__})")
            print("Note: Remember to set your TAVILY_API_KEY environment variable")
        except ImportError as e:
            print(f"Warning: Error importing Tavily: {e}", file=sys.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during post-install: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 