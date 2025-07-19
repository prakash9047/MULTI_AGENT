# Multi-Agent Finance Assistant

A sophisticated multi-source, multi-agent finance assistant that delivers spoken market briefs via a Streamlit app with advanced RAG capabilities.


## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI       │    │   Vector Store  │
│   (Frontend)    │◄──►│   Orchestrator  │◄──►│   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐        ┌──────▼──────┐        ┌────▼────┐
   │ API     │        │ Scraping    │        │ Voice   │
   │ Agent   │        │ Agent       │        │ Agent   │
   └─────────┘        └─────────────┘        └─────────┘
        │                     │                     │
   ┌────▼────┐        ┌──────▼──────┐        ┌────▼────┐
   │Retriever│        │ Analysis    │        │Language │
   │Agent    │        │ Agent       │        │Agent    │
   └─────────┘        └─────────────┘        └─────────┘
```

## ✨ Key Features

- **🎙️ Voice Interface**: Spoken market briefs with real Whisper STT and Coqui TTS
- **🔍 RAG System**: Advanced retrieval-augmented generation with FAISS vector store
- **📊 Multi-Source Data**: Integrates Alpha Vantage, Finnhub, Twelve Data and more
- **🧠 AI-Powered Analysis**: Financial insights using GROQ LLM
- **🛡️ Robust Architecture**: Multi-agent system with fallbacks and error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- API keys for financial data sources
- GROQ API key for LLM capabilities

### Installation

```bash
# Clone the repository
git clone https://github.com/prakash9047/MULTI_AGENT.git
cd MULTI_AGENT

# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/secrets.env.example config/secrets.env
# Edit secrets.env with your API keys
```

### Running the Application

```bash
# Start the backend FastAPI server
python start_server.py

# In a new terminal, start the Streamlit frontend
streamlit run streamlit_app/app.py
```

### Using Docker (Optional)

```bash
# Build and run with docker-compose
docker-compose up --build

# Access the app at http://localhost:8501
```

## 🤖 Agent Architecture

### API Agent
- **Purpose**: Fetches real-time financial data from multiple sources
- **Features**: Smart API waterfall strategy with fallbacks
- **Sources**: Alpha Vantage, Finnhub, Twelve Data, Yahoo Finance (backup)

### Scraping Agent
- **Purpose**: Extracts structured data from financial websites
- **Features**: SEC filings analysis, earnings report processing
- **Technologies**: BeautifulSoup, Scrapy

### Retriever Agent
- **Purpose**: Semantic search against financial knowledge base
- **Features**: FAISS vector store, sentence-transformers embeddings
- **Capabilities**: Context retrieval for RAG

### Analysis Agent
- **Purpose**: Performs financial analysis on market data
- **Features**: Risk assessment, technical indicators, sentiment analysis
- **Technologies**: LangGraph for complex workflows

### Language Agent
- **Purpose**: Generates natural language responses and reports
- **Features**: RAG integration, narrative generation
- **Technologies**: LangChain, GROQ LLM

### Voice Agent
- **Purpose**: Provides voice interface for the system
- **Features**: Speech-to-text and text-to-speech capabilities
- **Technologies**: Whisper STT, Coqui TTS

## 📊 API Integration

The system integrates with multiple financial data sources:

### Working APIs
- **Alpha Vantage**: Real-time stock data, technical indicators
- **Finnhub**: Company fundamentals, financial news
- **Twelve Data**: Multi-asset coverage (stocks, forex, crypto)
- **GROQ**: Advanced AI analysis and LLM capabilities

### Smart API Waterfall Strategy
```
Alpha Vantage → Finnhub → Twelve Data → Yahoo Finance → Mock Data
```

## 💡 Advanced Features

### RAG Pipeline
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: HuggingFace sentence-transformers
- **Context Integration**: Blends retrieved knowledge with LLM outputs
- **Confidence Scoring**: Ranks retrieved results by relevance

### Real Voice Interface
- **STT**: OpenAI Whisper model for accurate speech recognition
- **TTS**: Coqui TTS for natural-sounding voice synthesis
- **Audio Processing**: Real-time streaming capabilities

### Error Handling
- **Status Tracking**: Each component maintains status attributes
- **Graceful Fallbacks**: System degrades gracefully when components fail
- **Mock Data**: Provides sensible defaults when APIs are unavailable


```

## 🛠️ Technology Stack

- **Backend**: FastAPI, LangChain, CrewAI, LangGraph
- **Frontend**: Streamlit
- **Database**: FAISS vector store
- **AI Models**: GROQ LLM, Whisper, Coqui TTS
- **Embeddings**: HuggingFace sentence-transformers


## Acknowledgements

- Financial data provided by Alpha Vantage, Finnhub, and Twelve Data
- AI capabilities powered by GROQ
- Voice technologies: Whisper (OpenAI) and Coqui TTS
