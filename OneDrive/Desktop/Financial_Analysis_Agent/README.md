# Financial Analysis Agent 🤖📊

A sophisticated multi-agent AI system for real-time financial market research and document analysis, built with **Agno (AgentOS)** and powered by **Groq's Llama 3** models.

## 🎯 Overview

The Financial Analysis Agent orchestrates multiple specialized AI agents in a "Team of Experts" approach to deliver:
- **Real-time stock market analysis** with live data from Yahoo Finance
- **Web-based research** with automated fact-checking and source verification
- **Document Q&A** using RAG (Retrieval-Augmented Generation) for private PDFs
- **Quality assurance** via an LLM-as-a-judge evaluation mechanism

## ✨ Key Features

### Multi-Agent Architecture
1. **Web Research Agent** - Investigates market trends, verifies sources, and generates professional financial reports
2. **Stock Market Analysis Agent** - Fetches real-time market data (P/E ratio, Market Cap, EPS, 52-week ranges)
3. **RAG (Knowledge) Agent** - Answers queries based on ingested private documents with semantic search
4. **Evaluator Agent** - Quality assurance that scores responses on Faithfulness, Context Relevance, Completeness, and Coherence

### Technical Highlights
- ⚡ **Ultra-fast inference** with Groq's optimized Llama 3.1 models
- 🔍 **Semantic search** using sentence-transformers embeddings
- 📚 **Vector database** with ChromaDB for persistent knowledge storage
- 🌐 **Live data access** via DuckDuckGo, Newspaper4k, and Yahoo Finance
- 🛡️ **Self-evaluation** mechanism to prevent hallucinations

## 🛠️ Technology Stack

- **Framework**: Agno (AgentOS)
- **LLM Provider**: Groq (Llama 3.1 70B & 8B)
- **Backend**: FastAPI
- **Vector DB**: ChromaDB
- **Embeddings**: sentence-transformers (paraphrase-MiniLM-L6-v2)
- **Tools**: DuckDuckGo, Newspaper4k, YFinance
- **Data Processing**: unstructured, pypdf, chonkie

## 📋 Prerequisites

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com))

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Financial_Analysis_Agent.git
cd Financial_Analysis_Agent
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional configurations
PDF_INGEST_MODE=auto
LOG_LEVEL=WARNING
GROQ_DEFAULT_MODEL=llama-3.1-70b-versatile
FORCE_INGEST=false
SKIP_INGEST=false
RECREATE_COLLECTION=false
```

## 📖 Usage

### 1. Ingest Documents (Optional - for RAG Agent)

Run the ingestion script to load PDFs into the vector database:
```bash
python ingest_runner.py
```

By default, it ingests Apple's Environmental Progress Report. To ingest your own PDF:
```env
# Add to .env
PDF_URL=https://example.com/your-document.pdf
# OR
PDF_LOCAL_PATH=/path/to/local/document.pdf
```

### 2. Start the Agent Server

```bash
python my_os.py
```

The server will start at `http://localhost:7777`

### 3. Interact with Agents

Access the AgentOS interface at `http://localhost:7777` and interact with:
- **ResearchAgent** - For market research and news analysis
- **StockAgent** - For real-time stock data and analysis
- **RAGAgent** - For document-based Q&A
- **EvaluatorAgent** - For quality assessment of RAG responses

## 🎓 Key Learnings

- **Agentic Workflows**: Designing autonomous agents that plan, execute tools, and collaborate
- **Vector Databases & RAG**: Embedding models, vector storage, and semantic search optimization
- **LLM Integration**: High-performance LLM integration with Groq's infrastructure
- **System Evaluation**: Quantitative evaluation metrics for Generative AI outputs

## 📁 Project Structure

```
Financial_Analysis_Agent/
├── my_os.py                          # Main AgentOS application
├── vector_db.py                      # Vector database and RAG utilities
├── ingest_runner.py                  # PDF ingestion script
├── internship_project_details.md    # Project documentation
├── .env                              # Environment variables (DO NOT COMMIT)
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── chroma_db/                        # Vector database storage (generated)
└── venv/                             # Virtual environment (generated)
```

## 🔒 Security Notes

- **Never commit your `.env` file** - It contains sensitive API keys
- The `.gitignore` file is configured to exclude `.env` automatically
- Keep your Groq API key secure and rotate it if exposed

## 🤝 Contributing

This is an internship project. Feel free to fork and experiment!

## 📄 License

This project was developed as part of an internship program.

## 🙏 Acknowledgments

- Built with [Agno (AgentOS)](https://github.com/agno-agi/agno)
- Powered by [Groq](https://groq.com) and Meta's Llama 3.1 models
- Uses various open-source tools and libraries

---

**Developed by**: Anant Tewari  
**Project Type**: Internship Project - Financial AI Agent System
