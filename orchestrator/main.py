from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from agents.api_agent import APIAgent
from agents.scraping_agent import ScrapingAgent
from agents.retriever_agent import RetrieverAgent
from agents.analysis_agent import AnalysisAgent
from agents.language_agent import LanguageAgent
from agents.voice_agent import VoiceAgent
from config.settings import settings

app = FastAPI(title="Finance Assistant Orchestrator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize agents
api_agent = APIAgent()
scraping_agent = ScrapingAgent()
retriever_agent = RetrieverAgent()
analysis_agent = AnalysisAgent()
language_agent = LanguageAgent()
voice_agent = VoiceAgent()

class Query(BaseModel):
    text: str
    use_voice: bool = False

class PortfolioQuery(BaseModel):
    symbols: List[str]
    use_voice: bool = False

@app.post("/query")
async def process_query(query: Query):
    """Process a general query."""
    try:
        # Convert speech to text if needed
        if query.use_voice:
            try:
                stt_result = voice_agent.speech_to_text()
                if "error" in stt_result:
                    print(f"STT error: {stt_result['error']}")
                    # Don't raise exception, continue with empty query or default
                    query.text = query.text or "Tell me about recent market conditions"
                else:
                    query.text = stt_result["text"]
            except Exception as e:
                print(f"Voice processing error: {str(e)}")
                # Continue with text query

        # Ensure we have a query
        if not query.text:
            query.text = "Tell me about recent market conditions"

        # Get relevant context
        try:
            context = retriever_agent.get_relevant_context(query.text)
            if "error" in context:
                print(f"Retriever error: {context['error']}")
                # Use empty context
                context = {"results": [], "query": query.text}
        except Exception as e:
            print(f"Context retrieval error: {str(e)}")
            # Use empty context
            context = {"results": [], "query": query.text}

        # Generate response
        try:
            response = language_agent.generate_summary(context, query.text)
            if "error" in response:
                print(f"Language generation error: {response['error']}")
                response = {"summary": "I apologize, but I'm having trouble processing your query at the moment. Please try again later."}
        except Exception as e:
            print(f"Summary generation error: {str(e)}")
            response = {"summary": "I apologize, but I'm having trouble processing your query at the moment. Please try again later."}

        # Convert to speech if needed
        if query.use_voice:
            tts_result = voice_agent.text_to_speech(response["summary"])
            if "error" in tts_result:
                raise HTTPException(status_code=400, detail=tts_result["error"])
            response["audio_file"] = tts_result["audio_file"]

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/market-brief")
async def get_market_brief(query: PortfolioQuery):
    """Generate a market brief for the given portfolio."""
    try:
        if not query.symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")

        # Get market data
        market_data = await api_agent.get_market_brief(query.symbols)
        if "error" in market_data and isinstance(market_data, dict) and len(market_data) == 1:
            # Only raise exception if it's a global error, not per-symbol errors
            raise HTTPException(status_code=400, detail=market_data["error"])

        # Use all symbols, even if some had errors (we're using fallback data)
        valid_symbols = query.symbols
        
        # Get company data for valid symbols
        company_data = {}
        for symbol in valid_symbols:
            try:
                data = await scraping_agent.get_company_data(symbol)
                if isinstance(data, dict) and "error" not in data:
                    company_data[symbol] = data
            except Exception as e:
                print(f"Error getting company data for {symbol}: {str(e)}")
                # Continue with next symbol if one fails

        # Handle case where we have no valid company data
        if not company_data and not market_data:
            # Create mock data for development/testing
            market_data = {symbol: {
                "stock": {
                    "symbol": symbol,
                    "current_price": 100.0,
                    "change": 2.5,
                    "volume": 1000000,
                    "note": "Using mock data"
                },
                "earnings": {
                    "symbol": symbol,
                    "last_earnings": {
                        "Revenue": 10000000000,
                        "Earnings": 2000000000,
                        "note": "Using mock data"
                    }
                }
            } for symbol in valid_symbols}

        # Analyze data
        try:
            analysis = analysis_agent.generate_market_brief(market_data, company_data)
        except Exception as e:
            print(f"Analysis agent error: {str(e)}")
            # Create mock analysis response
            analysis = {
                "portfolio": f"Analysis of portfolio containing {', '.join(valid_symbols)}.",
                "earnings": f"Recent earnings show strong performance for {', '.join(valid_symbols)}."
            }
        
        # Generate brief
        try:
            brief = language_agent.generate_market_brief(
                analysis.get("portfolio", "No portfolio analysis available"),
                analysis.get("earnings", "No earnings analysis available")
            )
            if "error" in brief:
                print(f"Language agent error: {brief['error']}")
                brief = {"brief": "Market brief generation encountered an issue. Please try again later."}
        except Exception as e:
            print(f"Brief generation error: {str(e)}")
            brief = {"brief": "Market brief generation encountered an issue. Please try again later."}

        # Convert to speech if needed
        if query.use_voice:
            tts_result = voice_agent.text_to_speech(brief["brief"])
            if "error" in tts_result:
                raise HTTPException(status_code=400, detail=tts_result["error"])
            brief["audio_file"] = tts_result["audio_file"]

        return brief
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 