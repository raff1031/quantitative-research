"""Launcher — esegue run_pipeline.py dalla directory corrente."""
import subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__))
dest = here + " v2"
os.chdir(dest)
r = subprocess.run(
    [sys.executable, "run_pipeline.py", "--equity", "10000"],
    capture_output=False, text=True, encoding="utf-8", errors="replace"
)
sys.exit(r.returncode)
