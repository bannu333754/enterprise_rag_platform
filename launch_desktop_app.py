import time
import subprocess
import urllib.request
import sys

print("=" * 60)
print("Starting Enterprise RAG Backend Engine...")
print("=" * 60)

server_proc = subprocess.Popen([
    sys.executable, "-m", "uvicorn", 
    "day75_unified_production_app:app", 
    "--host", "127.0.0.1", 
    "--port", "8000",
    "--log-level", "warning"
])

url = "http://127.0.0.1:8000"
ready = False

print("Loading embeddings and vector store into memory...")
for _ in range(25):
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            if resp.status == 200:
                ready = True
                break
    except Exception:
        time.sleep(1)

if ready:
    print("[READY] Server is active! Launching Desktop App Window...")
    subprocess.run([
        "msedge.exe", 
        f"--app={url}", 
        "--window-size=1200,820"
    ])
    server_proc.terminate()
else:
    print("[ERROR] Server failed to start in time.")
    server_proc.terminate()
