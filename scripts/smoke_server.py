"""Verify a temporary local Streamlit server and stop it."""
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    log_path = root / "artifacts" / "server-smoke.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.address=127.0.0.1", f"--server.port={port}", "--browser.gatherUsageStats=false"], cwd=root, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited: see {log_path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                        if response.status == 200 and response.read() == b"ok":
                            print("Streamlit HTTP health: OK")
                            return
                except OSError:
                    pass
                time.sleep(0.25)
            raise TimeoutError(f"Server health timeout: see {log_path}")
        finally:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
