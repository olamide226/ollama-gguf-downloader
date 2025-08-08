#!/usr/bin/env python3
"""
Test runner script for ollama-gguf-downloader
"""
import subprocess
import sys
import os


def run_tests():
    """Run the test suite."""
    print("🧪 Running unit tests...")

    # Run tests with coverage
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "test_download_gguf.py",
                "--cov=download_gguf",
                "--cov-report=term-missing",
                "--cov-report=html",
                "-v",
            ],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install test dependencies:")
        print("   pip install -r requirements-dev.txt")
        return False


def main():
    """Main function."""
    if not run_tests():
        sys.exit(1)


if __name__ == "__main__":
    main()
