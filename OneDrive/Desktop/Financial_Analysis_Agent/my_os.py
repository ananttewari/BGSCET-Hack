# my_os.py

import os
import sys
from textwrap import dedent
from dotenv import load_dotenv
import logging
import datetime

# --- Load Environment Variables ---
load_dotenv()

# PDF ingestion mode: 'auto' (try unstructured then pypdf) or 'pypdf' (force pypdf-only)
PDF_INGEST_MODE = os.getenv("PDF_INGEST_MODE", "auto").lower()

# Logging: default to WARNING to avoid noisy output; override with LOG_LEVEL env var
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING), format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Today's date (module-level so agents can reference it) ---
TODAY_DATE = datetime.date.today().isoformat()

# Check if GROQ_API_KEY is set
if not os.getenv("GROQ_API_KEY"):
    logger.error("GROQ_API_KEY environment variable not set.")
    logger.error("Please create a .env file and add your key.")
    sys.exit(1)

# --- Imports for Agents ---
from agno.agent import Agent
from agno.os import AgentOS
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.yfinance import YFinanceTools
from sentence_transformers import SentenceTransformer
from typing import Union, List, Tuple, Optional

# --- NEW Imports for Agno 2.x Knowledge ---
from agno.vectordb.chroma import ChromaDb 
import chromadb
from vector_db import setup_vector_db, ingest_pdf_to_collection
# pypdf and chunking live in `vector_db` now; keep my_os lightweight
from agno.knowledge.knowledge import Knowledge
# ========================================

# Helper to choose Groq models via env vars so we can update IDs without code edits
def make_groq_model(env_var: str, default: str, allow_function_calls: Optional[bool] = None) -> Groq:
    # Allow a single override for all Groq models via `GROQ_DEFAULT_MODEL` env var.
    global_default = os.getenv("GROQ_DEFAULT_MODEL")
    if global_default:
        default_to_use = global_default
    else:
        default_to_use = default

    model_id = os.getenv(env_var, default_to_use)
    logger.info(f"Using model for {env_var or 'model'}: {model_id}")

    # Try to construct the Groq model while optionally controlling provider-side
    # function/tool call behavior by attempting common constructor keyword names.
    tried_kwargs = [
        "allow_function_calls",
        "function_calling",
        "allow_tool_calls",
        "disable_functions",
        "functions_enabled",
    ]

    # If caller explicitly requests allow/disallow, try to honor it first.
    if allow_function_calls is not None:
        for kw in tried_kwargs:
            try:
                logger.debug(f"Attempting to construct Groq with {kw}={allow_function_calls}")
                return Groq(id=model_id, **{kw: allow_function_calls})
            except TypeError:
                continue
        try:
            return Groq(id=model_id)
        except Exception:
            logger.warning(f"Failed to construct Groq model '{model_id}' with explicit function-call flag; falling back to default model '{default}'.")
            return Groq(id=default)

    # Default behavior: attempt to disable function/tool calls to be conservative
    for kw in tried_kwargs:
        try:
            logger.debug(f"Attempting to construct Groq with {kw}=False")
            return Groq(id=model_id, **{kw: False})
        except TypeError:
            continue

    # Final fallback: normal construction
    try:
        return Groq(id=model_id)
    except Exception:
        logger.warning(f"Failed to construct Groq model '{model_id}', falling back to '{default}'.")
        try:
            return Groq(id=default)
        except Exception:
            raise

# ==============================================================================
# AGENT 1: WEB RESEARCH AGENT
# ==============================================================================
logger.info("Initializing Agent 1: Web Research Agent...")
research_agent = Agent(
    name="ResearchAgent",
    # Allow provider-side function/tool calls for the ResearchAgent so it can
    # use the provided web tools (DuckDuckGo, Newspaper4k).
    model=make_groq_model("RESEARCH_MODEL", "llama-3.1-8b-instant", allow_function_calls=True),
    tools=[DuckDuckGoTools()],
    description=dedent("""\
        You are an elite research analyst in the financial services domain.
        Your expertise encompasses:
        - Deep investigative financial research and analysis
        - Fact-checking and source verification
    """),
     instructions=dedent(f"""\
          You must follow these rules:
          - Today's date is {TODAY_DATE}.
          - When you write the final report, you MUST include a "Published:" line using this date.

          Important: use the provided web search tools when you need live or up-to-date information.
          The available tools are `DuckDuckGoTools` and `Newspaper4kTools`.
          When you need to find sources, DO NOT hallucinate — call the appropriate tool, inspect its output, and cite the exact URL or source id.

          1. Research Phase
              - Use the web search tools to find 5 authoritative sources on the topic.
              - Prioritize recent publications and expert opinions; when in doubt, prefer primary sources.
          2. Analysis Phase
              - For each factual claim you intend to make, first search and then cite the supporting source (tool output).
              - Cross-reference facts across multiple sources and note any contradictions.
          3. Writing Phase
              - Craft an attention-grabbing headline
              - Structure content in Financial Report style
     """),
     expected_output=dedent("""\
          # {Compelling Headline}
          ## Executive Summary
          {Concise overview of key findings and significance}
          ... (rest of the report structure) ...
          ---\
          Research conducted by Financial Agent
          Published: [The date you were given in your instructions]
     """),
    markdown=True,
)

# ==============================================================================
# AGENT 2: RAG AGENT (DOCUMENT Q&A) - UPDATED FOR AGNO 2.x
# ==============================================================================
logger.info("Initializing Agent 2: RAG Agent (Agno 2.x)...")

# --- Embedding Model Class (Helper for RAG) ---
class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.dimensions = 384
        logger.info("✅ Embedding model (paraphrase-MiniLM-L6-v2) initialized!")

    def get_embedding_and_usage(self, text: Union[str, List[str]]) -> Tuple[Union[List[List[float]], List[float]], dict]:
        if isinstance(text, str):
            embedding = self.model.encode(text)
            embedding_list = embedding.tolist()
            usage = {"prompt_tokens": len(text.split()), "total_tokens": len(text.split())}
            return embedding_list, usage
        else:
            embeddings = self.model.encode(text)
            embedding_list = embeddings.tolist()
            total_tokens = sum(len(t.split()) for t in text)
            usage = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
            return embedding_list, usage

    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            return self.model.encode(text).tolist()
        return self.model.encode(text).tolist()

try:
    # 1. Initialize the Embedder
    embedder = EmbeddingModel()

    # 2. Create or get the persistent ChromaDB client and adapter (lightweight)
    chroma_client, vector_db, RECREATE_COLLECTION, collection_size = setup_vector_db(
        name="apple_docs", path="./chroma_db", embedder=embedder, logger=logger
    )

    # Wrap the vector DB adapter in an agno Knowledge object so AgentOS can discover DBs safely
    try:
        knowledge_obj = Knowledge(name="apple_docs", vector_db=vector_db)
    except Exception as e:
        logger.error(f"Failed to construct Knowledge object: {e}")
        knowledge_obj = None

    # 3. Ingestion guard: Skip ingestion if the collection already contains vectors unless forced
    FORCE_INGEST = os.getenv("FORCE_INGEST", "false").lower() in ("1", "true", "yes")

    # Check existing collection size to avoid re-ingesting repeatedly
    try:
        coll = chroma_client.get_collection(name="apple_docs")
        collection_size = None
        if hasattr(coll, "count") and callable(getattr(coll, "count")):
            try:
                collection_size = coll.count()
            except Exception:
                collection_size = None
        if collection_size is None and hasattr(coll, "get") and callable(getattr(coll, "get")):
            try:
                res = coll.get(include=["ids"]) if True else coll.get()
                ids = res.get("ids") if isinstance(res, dict) else None
                if ids is not None:
                    collection_size = len(ids)
            except Exception:
                collection_size = None
    except Exception:
        collection_size = None

    if collection_size and collection_size > 0 and not FORCE_INGEST and not RECREATE_COLLECTION:
        SKIP_INGEST = True
        logger.info(f"Collection 'apple_docs' already has {collection_size} items; skipping ingest (set FORCE_INGEST=1 to override).")
    else:
        SKIP_INGEST = os.getenv("SKIP_INGEST", "false").lower() in ("1", "true", "yes")

    # 4. Construct the RAG Agent (knowledge-backed, tool-less to avoid unintended web calls)
    rag_agent = Agent(
        name="RAGAgent",
        knowledge=knowledge_obj,
        search_knowledge=True,
        model=make_groq_model("RAG_MODEL", "llama-3.1-8b-instant"),
        tools=[],
        instructions=["Answer the user's question based ONLY on the provided context."]
    )
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG Agent: {e}")
    rag_agent = None

# ==============================================================================
# AGENT 3: STOCK MARKET ANALYSIS AGENT
# ==============================================================================
logger.info("Initializing Agent 3: Stock Analysis Agent...")
# Initialize YFinanceTools instance first
yf_tools = YFinanceTools()

stock_agent = Agent(
    name="StockAgent",
    model=make_groq_model("STOCK_MODEL", "llama-3.1-8b-instant", allow_function_calls=True),
    # StockAgent is allowed to use Yahoo Finance tools for live market data (YFinanceTools).
    # Restricted to basic tools: price and company info only.
    # Note: We pass the specific bound methods from the toolkit instance.
    tools=[yf_tools.get_current_stock_price, yf_tools.get_company_info],
    instructions=dedent("""\
        You are a seasoned credit rating analyst with deep expertise in market analysis! 📊
        
        You have access to limited tools:
        1. `get_current_stock_price`: Use this to get the latest price.
        2. `get_company_info`: Use this to get 52-week high/low and other basic info.

        Follow these steps for a brief financial report:
        1. Market Overview
           - Latest stock price
           - 52-week high and low
        2. Company Profile
           - Brief description of the company
           - Key sector/industry info
    """),
    markdown=True,
)

# ==============================================================================
# AGENT 4: EVALUATOR AGENT (LLM-as-a-judge)
# ==============================================================================
logger.info("Initializing Agent 4: Evaluator Agent...")
evaluator_agent = Agent(
    name="EvaluatorAgent",
    model=make_groq_model("EVALUATOR_MODEL", "llama-3.1-8b-instant"),
    description=dedent("""\
        You are an expert RAG system evaluator with deep expertise in:
        - Information retrieval quality assessment
        - Response accuracy evaluation
        - Source attribution verification
    """),
    instructions=dedent("""\
        Evaluate the RAG system output based on these key metrics:
        1. Faithfulness (1-5):
           - How accurately does the response reflect the source documents?
           - Are there any hallucinations or incorrect statements?
        2. Context Relevance (1-5):
           - Are the retrieved passages relevant to the query?
        3. Answer Completeness (1-5):
           - Does the response fully address the query?
        4. Response Coherence (1-5):
           - Is the response well-structured and easy to understand?
        Provide specific examples and explanations for each score.
    """),
    expected_output=dedent("""\
        # RAG Evaluation Report
        ## Overview
        Query: {query}
        ## Metric Scores
        ### Faithfulness: {score}/5
        - Justification:
        ### Context Relevance: {score}/5
        - Justification:
        ### Answer Completeness: {score}/5
        - Justification:
        ### Response Coherence: {score}/5
        - Justification:
        ## Overall Score: {total}/20
        ## Summary
        {final_assessment}
    """),
    markdown=True,
)

# ==============================================================================
# BUNDLE ALL AGENTS INTO THE AGENT-OS
# ==============================================================================
logger.info("Bundling all agents into AgentOS...")

all_agents = [research_agent, stock_agent, evaluator_agent]
if rag_agent:
    all_agents.append(rag_agent)
else:
    logger.warning("RAG Agent was not initialized and will not be available.")

agent_os = AgentOS(
    id="finance-agent-os",
    description="A set of Financial AI Agents",
    agents=all_agents,
)

# Get the FastAPI app
app = agent_os.get_app()

# Log a concise status summary for all agents so the operator can see updates
def _agent_summary(agents: List[Agent]):
    rows = []
    for a in agents:
        try:
            name = a.name or "<unnamed>"
            model_id = getattr(a.model, 'id', getattr(a.model, '__class__', None))
            tools = len(a.tools) if a.tools else 0
            has_knowledge = bool(a.knowledge)
            search_knowledge = bool(getattr(a, 'search_knowledge', False))
            rows.append(f"{name}: model={model_id} tools={tools} knowledge={has_knowledge} search_knowledge={search_knowledge}")
        except Exception as e:
            rows.append(f"{getattr(a,'name','<error>')}: error summarizing agent: {e}")

    logger.info("Agent summary:\n" + "\n".join(rows))

_agent_summary(all_agents)

# Log which tools are registered for the ResearchAgent to aid debugging
try:
    if research_agent and getattr(research_agent, 'tools', None):
        tool_names = [t.__class__.__name__ for t in research_agent.tools]
        logger.info(f"ResearchAgent tools registered: {tool_names}")
    else:
        logger.info("ResearchAgent has no registered tools.")
except Exception:
    logger.debug("Unable to enumerate ResearchAgent tools.")

# ==============================================================================
# RUN THE SERVER
# ==============================================================================
if __name__ == "__main__":
    logger.info("\n🚀 Starting Agno AgentOS Server...")
    logger.info("   All agents are being loaded.")
    logger.info("   Connect at http://localhost:7777")
    
    # This runs the server
    agent_os.serve(
        app="my_os:app",
        host="0.0.0.0",
        port=7777,
        reload=False
    )