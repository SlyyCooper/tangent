import subprocess
import sys

def main():
    """Post-installation script to run playwright install."""
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install"])
        print("Successfully installed playwright browsers")
    except subprocess.CalledProcessError as e:
        print(f"Error installing playwright browsers: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 