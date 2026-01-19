
import inspect
from agno.tools.yfinance import YFinanceTools

sig = inspect.signature(YFinanceTools.__init__)
with open("yfinance_args.txt", "w") as f:
    for name, param in sig.parameters.items():
        f.write(f"{name}: {param.default}\n")
