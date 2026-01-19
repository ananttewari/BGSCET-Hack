# Internship Project Details: Financial Analysis Agent

## Project Title
**Financial Analysis Agent: A Multi-Agent AI System for Real-Time Market Research and Document Analysis**

## Brief Summary
This project involves the development of a **Financial Analysis Agent**, a sophisticated multi-agent AI system designed to automate and enhance financial research. The system orchestrates multiple specialized AI agents—including a **Web Researcher** for live market news, a **Stock Analyst** for quantitative data, and a **RAG (Retrieval-Augmented Generation) Agent** for internal document analysis—to utilize a "Team of Experts" approach. 

Built on the **Agno (AgentOS)** framework and powered by **Groq's high-performance Llama 3 models**, the application delivers real-time stock insights, deep-dive research reports, and accurate Q&A from financial documents. A key differentiator is its inclusion of an **Evaluator Agent**, which uses an "LLM-as-a-judge" mechanism to rigorously score the system's outputs for accuracy and faithfulness, ensuring high-reliability responses suitable for financial contexts.

## Technical Details

### 1. Technology Stack
*   **Core Framework:** Agno (AgentOS) - A lightweight framework for building and serving agentic workflows.
*   **LLM Provider:** Groq - Leveraging Llama 3.1 models (`llama-3.1-70b-versatile` for reasoning, `llama-3.1-8b-instant` for speed) for ultra-fast inference.
*   **Backend:** FastAPI - Serves the AgentOS as a RESTful API.
*   **Vector Database:** ChromaDB (Persistent) - Stores document embeddings for the RAG pipeline.
*   **Embeddings:** `sentence-transformers/paraphrase-MiniLM-L6-v2` - Used for semantic search and retrieval.
*   **Data Ingestion:** Custom pipeline using `unstructured` (partitioning), `pypdf` (fallback parsing), and `chonkie` (smart chunking).
*   **Tools & Integrations:**
    *   **DuckDuckGo & Newspaper4k:** For real-time web research and article extraction.
    *   **Yahoo Finance (YFinanceTools):** For live stock prices, fundamentals, and analyst ratings.
    *   **Python:** Core programming language.

### 2. System Architecture & Workflow
The system follows a modular "Agentic Orchestration" pattern where a central Coordinator (AgentOS) manages four distinct agents:

1.  **Web Research Agent:**
    *   **Role:** Investigates market trends and news.
    *   **Workflow:** Performs queries using DuckDuckGo, verifies sources, cross-references facts, and synthesizes findings into a professional financial report with citations.
2.  **Stock Market Analysis Agent:**
    *   **Role:** Quantitative analyst.
    *   **Workflow:** Fetches real-time market data (P/E ratio, Market Cap, EPS, 52-week high/low) using Yahoo Finance and provides a fundamental analysis breakdown.
3.  **RAG (Knowledge) Agent:**
    *   **Role:** Document expert.
    *   **Workflow:** Answers user queries based *only* on ingested private documents (PDFs) stored in the local Vector DB (Chroma), ensuring data privacy and context-specific answers.
4.  **Evaluator Agent:**
    *   **Role:** Quality Assurance.
    *   **Workflow:** Acts as a critic, evaluating the RAG Agent's responses on four metrics: **Faithfulness**, **Context Relevance**, **Answer Completeness**, and **Response Coherence** (scored 1-5). This ensures the "LLM doesn't hallucinate."

### 3. Key Features
*   **Multi-Agent Coordination:** Seamlessly routes tasks to the correct expert (e.g., asking about "Apple's stock price" triggers the Stock Agent, while "What does the internal Q3 report say?" triggers the RAG Agent).
*   **Retrieval-Augmented Generation (RAG):** Implements a robust "Chat with your Data" feature that ingests complex PDF financial reports, chunks them intelligently, and retrieves relevant context for accurate answering.
*   **Self-Correction & Evaluation:** The system doesn't just output text; it grades itself, providing a confidence score and justification for its answers.
*   **Real-Time Data Access:** Unlike static models, this agent has live access to the internet and stock markets, making it suitable for time-sensitive financial decision-making.

## Key Learnings
*   **Agentic Workflows:** Learned how to design and orchestrate autonomous agents that can plan, execute tools, and collaborate.
*   **Vector Databases & RAG:** Gained deep understanding of embedding models, vector storage (ChromaDB), and semantic search optimization.
*   **LLM Integration:** Experience with integrating high-performance LLMs (Llama 3 via Groq) and handling context windows/prompt engineering.
*   **System Evaluation:** Implemented quantitative evaluation metrics for Generative AI, moving beyond simple "does it look good" testing to rigorous "faithfulness" scoring.
