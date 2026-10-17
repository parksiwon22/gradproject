"""Local health check; always terminates its temporary Streamlit server."""
import socket
import subprocess
import sys
import time
import urllib.request
from v11.store import ROOT
from v19.common import HOME

def main():
    with socket.socket() as sock: sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    with (HOME/'server_smoke.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-m','streamlit','run','app_v19.py','--server.headless=true','--server.address=127.0.0.1',f'--server.port={port}','--browser.gatherUsageStats=false'],cwd=ROOT,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+40
            while time.monotonic()<deadline:
                if process.poll() is not None: raise RuntimeError('server exited')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health',timeout=1) as response:
                        if response.status==200 and response.read()==b'ok': print('V19 Streamlit HTTP health: OK'); return
                except OSError: pass
                time.sleep(.25)
            raise TimeoutError('V19 health timeout')
        finally: process.terminate(); process.wait(timeout=10)

if __name__=='__main__': main()

