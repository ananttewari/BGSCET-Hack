
import agno.tools.yfinance
import shutil

src = agno.tools.yfinance.__file__
dst = "yfinance_source_dump.py"

print(f"Copying {src} to {dst}")
shutil.copy(src, dst)
