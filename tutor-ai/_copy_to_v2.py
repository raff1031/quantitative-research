"""Copia i 3 file mancanti nella directory v2."""
import shutil, os

here = os.path.dirname(os.path.abspath(__file__))
dest = here + " v2"

for f in ["equity_optimizer.py", "tws_executor.py", "generate_signals.py"]:
    src = os.path.join(here, f)
    dst = os.path.join(dest, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied {f} ({os.path.getsize(dst)} bytes)")
    else:
        print(f"  MISSING {f}")
print("Done")
