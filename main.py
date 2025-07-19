import dotenv
dotenv.load_dotenv(dotenv_path="config/secrets.env")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from agents.api_agent import APIAgent
from agents.scraping_agent import ScrapingAgent
from agents.retriever_agent import RetrieverAgent
from agents.analysis_agent import AnalysisAgent
from agents.language_agent import LanguageAgent
from agents.voice_agent import VoiceAgent
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Finance Assistant",
    description="A sophisticated multi-agent finance assistant with voice capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize all agents
print("Initializing agents...")
try:
    api_agent = APIAgent()
    print("✅ API Agent initialized")
except Exception as e:
    print(f"❌ API Agent failed: {e}")
    api_agent = None

try:
    scraping_agent = ScrapingAgent()
    print("✅ Scraping Agent initialized")
except Exception as e:
    print(f"❌ Scraping Agent failed: {e}")
    scraping_agent = None

try:
    retriever_agent = RetrieverAgent()
    print("✅ Retriever Agent initialized")
except Exception as e:
    print(f"❌ Retriever Agent failed: {e}")
    retriever_agent = None

try:
    analysis_agent = AnalysisAgent()
    print("✅ Analysis Agent initialized")
except Exception as e:
    print(f"❌ Analysis Agent failed: {e}")
    analysis_agent = None

try:
    language_agent = LanguageAgent()
    print("✅ Language Agent initialized")
except Exception as e:
    print(f"❌ Language Agent failed: {e}")
    language_agent = None

try:
    voice_agent = VoiceAgent()
    print("✅ Voice Agent initialized")
except Exception as e:
    print(f"❌ Voice Agent failed: {e}")
    voice_agent = None

# Pydantic models for API requests
class Query(BaseModel):
    text: str
    use_voice: bool = False

class PortfolioQuery(BaseModel):
    symbols: List[str]
    use_voice: bool = False

class VoiceQuery(BaseModel):
    duration: int = 5  # seconds to record
    use_mock: bool = True  # Use mock data for development

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with welcome message."""
    return {
        "message": "Welcome to Multi-Agent Finance Assistant",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "query": "/query",
            "market_brief": "/market-brief",
            "voice_query": "/voice-query"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    agent_status = {
        "api_agent": api_agent is not None,
        "scraping_agent": scraping_agent is not None,
        "retriever_agent": retriever_agent is not None,
        "analysis_agent": analysis_agent is not None,
        "language_agent": language_agent is not None,
        "voice_agent": voice_agent is not None
    }
    
    return {
        "status": "healthy",
        "agents": agent_status,
        "working_agents": sum(agent_status.values()),
        "total_agents": len(agent_status)
    }

@app.post("/query")
async def process_query(query: Query):
    """Process a general query using the multi-agent system."""
    try:
        # Handle voice input if requested
        if query.use_voice and voice_agent:
            try:
                stt_result = voice_agent.speech_to_text(mock=True)  # Use mock for development
                if "error" in stt_result:
                    print(f"STT error: {stt_result['error']}")
                    # Continue with text query
                else:
                    query.text = stt_result.get("text", query.text)
            except Exception as e:
                print(f"Voice processing error: {str(e)}")
                # Continue with text query

        # Ensure we have a query
        if not query.text:
            query.text = "Tell me about recent market conditions"

        # Get relevant context from retriever agent
        context = {"results": [], "query": query.text}
        if retriever_agent:
            try:
                context = retriever_agent.get_relevant_context(query.text)
                if "error" in context:
                    print(f"Retriever error: {context['error']}")
                    context = {"results": [], "query": query.text}
            except Exception as e:
                print(f"Context retrieval error: {str(e)}")
                context = {"results": [], "query": query.text}

        # Generate response using language agent
        response = {"summary": "I'm processing your query about financial markets."}
        if language_agent:
            try:
                response = language_agent.generate_summary(context, query.text)
                if "error" in response:
                    print(f"Language generation error: {response['error']}")
                    response = {"summary": "I apologize, but I'm having trouble processing your query at the moment."}
            except Exception as e:
                print(f"Summary generation error: {str(e)}")
                response = {"summary": "I apologize, but I'm having trouble processing your query at the moment."}

        # Convert to speech if needed
        if query.use_voice and voice_agent:
            try:
                tts_result = voice_agent.text_to_speech(response.get("summary", ""))
                if "error" not in tts_result:
                    response["audio_file"] = tts_result.get("audio_file")
            except Exception as e:
                print(f"TTS error: {str(e)}")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/market-brief")
async def get_market_brief(query: PortfolioQuery):
    """Generate a comprehensive market brief for the given portfolio."""
    try:
        if not query.symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")

        # Get market data using API agent
        market_data = {}
        if api_agent:
            try:
                market_data = await api_agent.get_market_brief(query.symbols)
                if "error" in market_data and len(market_data) == 1:
                    # Only if it's a global error
                    print(f"API Agent error: {market_data['error']}")
            except Exception as e:
                print(f"API Agent error: {str(e)}")

        # Get company data using scraping agent
        company_data = {}
        if scraping_agent:
            for symbol in query.symbols:
                try:
                    data = await scraping_agent.get_company_data(symbol)
                    if isinstance(data, dict) and "error" not in data:
                        company_data[symbol] = data
                except Exception as e:
                    print(f"Error getting company data for {symbol}: {str(e)}")

        # Create fallback data if needed
        if not market_data:
            market_data = {symbol: {
                "stock": {
                    "symbol": symbol,
                    "current_price": 100.0,
                    "change": 2.5,
                    "volume": 1000000,
                    "note": "Using mock data for development"
                },
                "earnings": {
                    "symbol": symbol,
                    "last_earnings": {
                        "Revenue": 10000000000,
                        "Earnings": 2000000000,
                        "note": "Using mock data for development"
                    }
                }
            } for symbol in query.symbols}

        # Analyze data using analysis agent
        analysis = {"portfolio": "Portfolio analysis", "earnings": "Earnings analysis"}
        if analysis_agent:
            try:
                analysis = analysis_agent.generate_market_brief(market_data, company_data)
            except Exception as e:
                print(f"Analysis agent error: {str(e)}")
                analysis = {
                    "portfolio": f"Analysis of portfolio containing {', '.join(query.symbols)}.",
                    "earnings": f"Recent earnings analysis for {', '.join(query.symbols)}."
                }

        # Generate comprehensive brief using language agent
        brief = {"brief": f"Market brief for {', '.join(query.symbols)} shows mixed performance."}
        if language_agent:
            try:
                brief = language_agent.generate_market_brief(
                    analysis.get("portfolio", "No portfolio analysis available"),
                    analysis.get("earnings", "No earnings analysis available")
                )
                if "error" in brief:
                    print(f"Language agent error: {brief['error']}")
                    brief = {"brief": f"Market brief for {', '.join(query.symbols)} shows mixed performance."}
            except Exception as e:
                print(f"Brief generation error: {str(e)}")
                brief = {"brief": f"Market brief for {', '.join(query.symbols)} shows mixed performance."}

        # Convert to speech if needed
        if query.use_voice and voice_agent:
            try:
                tts_result = voice_agent.text_to_speech(brief.get("brief", ""))
                if "error" not in tts_result:
                    brief["audio_file"] = tts_result.get("audio_file")
            except Exception as e:
                print(f"TTS error: {str(e)}")

        # Add market data to response
        brief["market_data"] = market_data
        brief["analysis"] = analysis

        return brief

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/voice-query")
async def process_voice_query(query: VoiceQuery):
    """Process a voice query."""
    try:
        if not voice_agent:
            raise HTTPException(status_code=503, detail="Voice agent not available")

        # Record and process speech
        stt_result = voice_agent.speech_to_text(mock=query.use_mock)
        if "error" in stt_result:
            raise HTTPException(status_code=400, detail=stt_result["error"])

        # Process the text query
        text_query = Query(text=stt_result["text"], use_voice=True)
        return await process_query(text_query)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/agents/status")
async def get_agents_status():
    """Get detailed status of all agents."""
    status = {}
    
    # Test each agent
    if api_agent:
        try:
            test_data = api_agent.market_data.get_stock_data("AAPL")
            status["api_agent"] = {"status": "healthy", "test": "passed"}
        except Exception as e:
            status["api_agent"] = {"status": "error", "error": str(e)}
    else:
        status["api_agent"] = {"status": "not_initialized"}

    if language_agent:
        try:
            test_summary = language_agent.generate_summary({"test": "data"}, "test query")
            status["language_agent"] = {"status": "healthy", "test": "passed"}
        except Exception as e:
            status["language_agent"] = {"status": "error", "error": str(e)}
    else:
        status["language_agent"] = {"status": "not_initialized"}

    if voice_agent:
        try:
            test_tts = voice_agent.text_to_speech("test")
            status["voice_agent"] = {"status": "healthy", "test": "passed"}
        except Exception as e:
            status["voice_agent"] = {"status": "error", "error": str(e)}
    else:
        status["voice_agent"] = {"status": "not_initialized"}

    if retriever_agent:
        try:
            test_search = retriever_agent.search_documents("test query")
            status["retriever_agent"] = {"status": "healthy", "test": "passed"}
        except Exception as e:
            status["retriever_agent"] = {"status": "error", "error": str(e)}
    else:
        status["retriever_agent"] = {"status": "not_initialized"}

    if analysis_agent:
        status["analysis_agent"] = {"status": "healthy"}
    else:
        status["analysis_agent"] = {"status": "not_initialized"}

    if scraping_agent:
        status["scraping_agent"] = {"status": "healthy"}
    else:
        status["scraping_agent"] = {"status": "not_initialized"}

    return status

# Run the application
if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Finance Assistant...")
    print("📡 FastAPI server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 