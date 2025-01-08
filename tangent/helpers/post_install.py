import subprocess
import sys

def main():
    """Post-installation script to run playwright install and verify OpenCV."""
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
            
    except subprocess.CalledProcessError as e:
        print(f"Error during post-install: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 