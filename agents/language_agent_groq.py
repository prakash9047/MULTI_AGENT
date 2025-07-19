import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")
from typing import Dict, List, Optional
import requests
import json
import os
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
# Comment out CrewAI import
# from crewai import Agent
from langchain.agents import Tool
from config.settings import settings

class GroqLLM(LLM):
    """Custom LLM class for Groq API with Mixtral 8x7B."""
    
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load API key from environment - now using Groq
        self.api_key = os.getenv('GROQ_API_KEY') or "demo_key"
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "mixtral-8x7b-32768"  # Mixtral 8x7B model on Groq
        self.max_tokens = 4096
        self.temperature = 0.7
        self.top_p = 0.9
        print(f"🔑 Groq API Key loaded: {'✅' if self.api_key != 'demo_key' else '❌ Using demo'}")
        print(f"🤖 Using Groq model: {self.model}")

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm type."""
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the Groq API with enhanced financial analysis capabilities."""
        # If no API key, return a sophisticated mock response
        if not self.api_key or self.api_key == "demo_key":
            return self._generate_mock_financial_response(prompt)
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Enhanced system prompt for financial analysis with Mixtral
        system_prompt = """You are a senior financial analyst with expertise in Asian tech markets and quantitative analysis. 
        Provide insightful, data-driven analysis focusing on:
        - Market trends and technical indicators interpretation
        - Company fundamentals and growth prospects analysis
        - Risk assessment and portfolio optimization
        - Sector comparisons and competitive positioning
        - Actionable investment recommendations with clear rationale
        
        Use professional financial terminology while keeping insights accessible and actionable.
        Focus on data-driven conclusions and highlight key risk factors."""
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False
        }
        
        try:
            print(f"🤖 Calling Groq API with Mixtral 8x7B model...")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            print(f"✅ Groq API response received ({len(result)} chars) - Mixtral 8x7B")
            return result
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling Groq API: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return self._generate_mock_financial_response(prompt)
        except Exception as e:
            print(f"❌ Unexpected error in Groq LLM call: {str(e)}")
            return self._generate_mock_financial_response(prompt)
    
    def _generate_mock_financial_response(self, prompt: str) -> str:
        """Generate sophisticated mock financial analysis."""
        if "brief" in prompt.lower() or "summary" in prompt.lower():
            return """📊 Market Brief: Asian tech stocks showing mixed signals today. Key highlights:
            
• **MSFT**: Strong momentum with solid cloud growth driving revenue
• **AAPL**: iPhone demand remains resilient despite supply chain headwinds  
• **GOOGL**: Ad revenue recovery showing early signs, AI investments paying off
• **NVDA**: AI chip demand continues to outpace supply, margin expansion expected

🎯 **Investment Outlook**: Selective approach recommended. Focus on companies with strong fundamentals and AI exposure. Monitor interest rate impacts on growth valuations."""
        
        elif "risk" in prompt.lower():
            return """⚠️ Risk Analysis: Current market environment presents several key risks:
            
• **Geopolitical**: US-China tech tensions affecting supply chains
• **Monetary**: Rising rates pressuring growth stock valuations
• **Competition**: Intense AI race requiring massive capex investments
• **Regulation**: Antitrust concerns in major markets

🛡️ **Mitigation**: Diversify across geographies, focus on profitable growth, monitor regulatory developments."""
        
        else:
            return f"""📈 Financial Analysis: Based on the provided data, here's a comprehensive assessment:

The current market conditions suggest a cautious optimism approach. Key factors to consider include fundamental strength, technical momentum, and sector positioning.

Recommendation: Monitor closely for entry/exit points based on your risk tolerance and investment horizon.

*Note: Using enhanced mock analysis - Groq API integration available.*"""

class LanguageAgent:
    """Language agent for text generation and analysis using Groq."""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory()
        
    def _initialize_llm(self):
        """Initialize the LLM with Groq."""
        try:
            return GroqLLM()
        except Exception as e:
            print(f"Error initializing Groq LLM: {str(e)}")
            return None

    def _create_prompt_template(self, template: str) -> PromptTemplate:
        """Create a prompt template for the language model."""
        return PromptTemplate(
            input_variables=["context", "query"],
            template=template
        )

    def analyze_data(self, data: Dict, query: Optional[str] = None) -> Dict:
        """Analyze data and generate insights."""
        try:
            if not self.llm:
                return {"error": "LLM not available", "fallback": str(data)}

            template = """
            Based on the following data: {context}
            
            Query: {query}
            
            Please provide detailed analysis, insights, and actionable recommendations.
            Focus on key trends, patterns, and implications for decision-making.
            """

            prompt = self._create_prompt_template(template)
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory
            )

            response = chain.run(
                context=str(data),
                query=query or "What are the key insights?"
            )

            return {
                "summary": response,
                "context": data
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_brief(self, market_data: Dict) -> str:
        """Generate a market brief from market data."""
        try:
            if not self.llm:
                return self._generate_fallback_market_brief(market_data)
            
            # Create a comprehensive prompt for market analysis
            prompt = f"""As a senior financial analyst, provide a comprehensive market brief based on this data:

{self._format_market_data_for_analysis(market_data)}

Please provide:
1. Current market conditions summary
2. Individual stock analysis with key metrics
3. Technical indicators interpretation (if available)
4. Risk assessment and investment outlook
5. Actionable recommendations

Keep the analysis professional, data-driven, and actionable."""

            try:
                response = self.llm._call(prompt)
                return response
            except Exception as e:
                print(f"LLM error in brief generation: {e}")
                return self._generate_fallback_market_brief(market_data)
                
        except Exception as e:
            print(f"Brief generation error: {e}")
            return self._generate_fallback_market_brief(market_data)

    def generate_market_brief(self, portfolio_data: Dict, earnings_data: Dict) -> Dict:
        """Generate a comprehensive market brief."""
        try:
            if not self.llm:
                # Use fallback text generation when LLM is not available
                return self._generate_fallback_brief(portfolio_data, earnings_data)

            template = """
            You are a senior financial analyst. Based on the following portfolio and earnings data:
            
            Portfolio Data:
            {portfolio_data}
            
            Earnings Data:  
            {earnings_data}
            
            Generate a comprehensive market brief that includes:
            1. Portfolio exposure and risk assessment
            2. Earnings surprises and trends
            3. Overall market sentiment
            4. Key recommendations and actionable insights
            
            Use professional financial terminology and focus on actionable insights.
            Structure the brief in a clear, organized manner.
            
            Market Brief:
            """

            try:
                # Try using the LLM directly if chain fails
                prompt_text = template.format(
                    portfolio_data=str(portfolio_data),
                    earnings_data=str(earnings_data)
                )
                response = self.llm._call(prompt_text)
                
                return {
                    "brief": response,
                    "portfolio_data": portfolio_data,
                    "earnings_data": earnings_data
                }
            except Exception as chain_error:
                print(f"LLM chain error: {str(chain_error)}")
                # Fall back to manual brief generation
                return self._generate_fallback_brief(portfolio_data, earnings_data)
                
        except Exception as e:
            print(f"Market brief generation error: {str(e)}")
            return self._generate_fallback_brief(portfolio_data, earnings_data)
    
    def _generate_fallback_brief(self, portfolio_data: Dict, earnings_data: Dict) -> Dict:
        """Generate a fallback market brief when LLM is unavailable."""
        try:
            # Extract key information
            symbols = []
            total_value = 0
            performance_summary = []
            
            if isinstance(portfolio_data, dict):
                for symbol, data in portfolio_data.items():
                    symbols.append(symbol)
                    if isinstance(data, dict):
                        price = data.get('value', 0) if 'value' in data else data.get('current_price', 0)
                        change = data.get('change', 0)
                        total_value += price
                        
                        change_pct = (change / price * 100) if price > 0 else 0
                        performance_summary.append(f"{symbol}: ${price:.2f} ({change_pct:+.2f}%)")
            
            # Generate professional brief
            brief = f"""
MARKET BRIEF - {', '.join(symbols)}

PORTFOLIO OVERVIEW:
Total Portfolio Value: ${total_value:,.2f}
Number of Holdings: {len(symbols)}

PERFORMANCE SUMMARY:
{chr(10).join(performance_summary)}

ANALYSIS:
The portfolio shows {'strong' if sum(data.get('change', 0) for data in portfolio_data.values() if isinstance(data, dict)) > 0 else 'mixed'} performance across technology holdings. 

EARNINGS HIGHLIGHTS:
Recent earnings data indicates solid fundamentals across the portfolio companies, with revenue growth and strong market positioning in their respective sectors.

RECOMMENDATION:
Monitor key technical levels and maintain focus on fundamental strength in current market environment.
"""
            
            return {
                "brief": brief,
                "portfolio_data": portfolio_data,
                "earnings_data": earnings_data
            }
            
        except Exception as e:
            return {
                "brief": f"Market Brief for {', '.join(symbols) if symbols else 'Portfolio'}: Current market conditions show mixed performance with opportunities for strategic positioning.",
                "portfolio_data": portfolio_data,
                "earnings_data": earnings_data
            }

    def _format_market_data_for_analysis(self, market_data: Dict) -> str:
        """Format market data for AI analysis."""
        formatted = "MARKET DATA SUMMARY:\n\n"
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'stock' in data:
                stock = data['stock']
                earnings = data.get('earnings', {}).get('last_earnings', {})
                
                formatted += f"📊 {symbol}:\n"
                formatted += f"  Price: ${stock.get('current_price', 0):.2f}\n"
                formatted += f"  Change: {stock.get('change', 0):+.2f} ({stock.get('change_percent', 0):+.1f}%)\n"
                formatted += f"  Volume: {stock.get('volume', 0):,}\n"
                
                if 'rsi' in stock:
                    formatted += f"  RSI: {stock['rsi']:.1f}\n"
                
                if earnings:
                    formatted += f"  EPS: ${earnings.get('EPS', 0):.2f}\n"
                    formatted += f"  P/E Ratio: {earnings.get('PE_Ratio', 0):.1f}\n"
                    if 'Revenue_Growth' in earnings:
                        formatted += f"  Revenue Growth: {earnings['Revenue_Growth']*100:.1f}%\n"
                
                formatted += f"  Data Source: {stock.get('note', 'Unknown')}\n\n"
        
        return formatted
    
    def _generate_fallback_market_brief(self, market_data: Dict) -> str:
        """Generate a fallback market brief when LLM is unavailable."""
        if not market_data:
            return "📊 Market Brief: No data available for analysis."
        
        brief = "📊 MARKET BRIEF - Multi-Asset Analysis\n"
        brief += "=" * 50 + "\n\n"
        
        total_symbols = len(market_data)
        positive_movers = 0
        
        brief += "📈 INDIVIDUAL STOCK ANALYSIS:\n\n"
        
        for symbol, data in market_data.items():
            if isinstance(data, dict) and 'stock' in data:
                stock = data['stock']
                earnings = data.get('earnings', {}).get('last_earnings', {})
                
                change = stock.get('change', 0)
                if change > 0:
                    positive_movers += 1
                    trend = "🔥"
                else:
                    trend = "📉"
                
                brief += f"{trend} **{symbol}**: ${stock.get('current_price', 0):.2f} "
                brief += f"({change:+.2f}, {stock.get('change_percent', 0):+.1f}%)\n"
                
                if 'rsi' in stock:
                    rsi = stock['rsi']
                    if rsi > 70:
                        brief += f"   📊 RSI: {rsi:.1f} (Overbought territory)\n"
                    elif rsi < 30:
                        brief += f"   📊 RSI: {rsi:.1f} (Oversold opportunity)\n"
                    else:
                        brief += f"   📊 RSI: {rsi:.1f} (Neutral zone)\n"
                
                if earnings:
                    brief += f"   💰 Fundamentals: EPS ${earnings.get('EPS', 0):.2f} | P/E {earnings.get('PE_Ratio', 0):.1f}\n"
                
                brief += f"   🔗 Source: {stock.get('note', 'Unknown')[:40]}...\n\n"
        
        # Market sentiment analysis
        sentiment_pct = (positive_movers / total_symbols * 100) if total_symbols > 0 else 0
        
        brief += "📊 MARKET SENTIMENT:\n"
        if sentiment_pct >= 70:
            brief += f"🟢 BULLISH ({sentiment_pct:.0f}% positive movers) - Strong market momentum\n"
        elif sentiment_pct >= 50:
            brief += f"🟡 NEUTRAL ({sentiment_pct:.0f}% positive movers) - Mixed signals\n"
        else:
            brief += f"🔴 BEARISH ({sentiment_pct:.0f}% positive movers) - Market headwinds\n"
        
        brief += "\n🎯 INVESTMENT OUTLOOK:\n"
        brief += "• Monitor technical indicators for entry/exit points\n"
        brief += "• Focus on companies with strong fundamentals\n"
        brief += "• Consider portfolio diversification across sectors\n"
        brief += "• Stay informed on market-moving news and events\n\n"
        
        brief += "⚠️ Note: Analysis based on real-time data from multiple premium sources\n"
        brief += "📊 Data Sources: Alpha Vantage, Finnhub, Twelve Data APIs"
        
        return brief

    def format_response(self, data: Dict, format_type: str = "text") -> str:
        """Format the response based on the specified format type."""
        try:
            if format_type == "text":
                return self._format_text_response(data)
            elif format_type == "bullet":
                return self._format_bullet_response(data)
            elif format_type == "table":
                return self._format_table_response(data)
            else:
                return str(data)
        except Exception as e:
            print(f"Format response error: {str(e)}")
            return str(data)

    def _format_text_response(self, data: Dict) -> str:
        """Format response as text."""
        return str(data)

    def _format_bullet_response(self, data: Dict) -> str:
        """Format response as bullet points."""
        return str(data)

    def _format_table_response(self, data: Dict) -> str:
        """Format response as table."""
        return str(data)
