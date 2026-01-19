
import agno.tools.yfinance
import os

path = agno.tools.yfinance.__file__
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if "self.stock_price" in line or "self.company_info" in line or "kwargs.get" in line or "def " in line:
            print(line.strip())
