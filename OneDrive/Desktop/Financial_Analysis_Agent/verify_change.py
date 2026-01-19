
from my_os import stock_agent
from agno.tools.yfinance import YFinanceTools

print(f"Agent Name: {stock_agent.name}")
tools = stock_agent.tools
print(f"Number of tools: {len(tools)}")
if tools:
    t = tools[0]
    if isinstance(t, YFinanceTools):
        print("Tool is YFinanceTools")
        # Check internal flags if accessible, or just the repr/dict
        print(f"Tool config: {t.__dict__}")
