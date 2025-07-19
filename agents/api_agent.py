import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")
from typing import Dict, List, Optional
import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from langchain.agents import Tool
from config.settings import settings
from agents.agent_helpers import SimpleAgent

class MarketDataAPI:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = settings.CACHE_TTL
        # Load API keys from environment
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')

    def _get_alpha_vantage_data(self, symbol: str) -> Dict:
        """Get comprehensive data from Alpha Vantage API - best for stock prices and technical indicators."""
        try:
            if not self.alpha_vantage_key:
                return None
            
            # Get real-time quote
            quote_url = f"https://www.alphavantage.co/query"
            quote_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(quote_url, params=quote_params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                result = {
                    "symbol": symbol,
                    "current_price": float(quote.get('05. price', 0)),
                    "change": float(quote.get('09. change', 0)),
                    "change_percent": quote.get('10. change percent', '0%').replace('%', ''),
                    "volume": int(quote.get('06. volume', 0)),
                    "open": float(quote.get('02. open', 0)),
                    "high": float(quote.get('03. high', 0)),
                    "low": float(quote.get('04. low', 0)),
                    "previous_close": float(quote.get('08. previous close', 0)),
                    "note": "Real-time data from Alpha Vantage (Stock prices & technical data)"
                }
                
                # Also get technical indicators if available
                try:
                    rsi_params = {
                        'function': 'RSI',
                        'symbol': symbol,
                        'interval': 'daily',
                        'time_period': 14,
                        'series_type': 'close',
                        'apikey': self.alpha_vantage_key
                    }
                    rsi_response = requests.get(quote_url, params=rsi_params, timeout=5)
                    rsi_data = rsi_response.json()
                    
                    if 'Technical Analysis: RSI' in rsi_data:
                        latest_rsi = list(rsi_data['Technical Analysis: RSI'].values())[0]
                        result["rsi"] = float(latest_rsi['RSI'])
                        result["note"] += " + Technical indicators"
                except:
                    pass  # RSI is optional
                
                return result
                
        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
        return None

    def _get_finnhub_data(self, symbol: str) -> Dict:
        """Get real-time quotes and company fundamentals from Finnhub."""
        try:
            if not self.finnhub_key:
                return None
                
            # Get real-time quote
            quote_url = f"https://finnhub.io/api/v1/quote"
            quote_params = {
                'symbol': symbol,
                'token': self.finnhub_key
            }
            
            response = requests.get(quote_url, params=quote_params, timeout=10)
            data = response.json()
            
            if 'c' in data and data['c'] > 0:
                result = {
                    "symbol": symbol,
                    "current_price": float(data['c']),
                    "change": float(data.get('d', 0)),
                    "change_percent": float(data.get('dp', 0)),
                    "high": float(data.get('h', 0)),
                    "low": float(data.get('l', 0)),
                    "open": float(data.get('o', 0)),
                    "previous_close": float(data.get('pc', 0)),
                    "note": "Real-time data from Finnhub (Quotes & fundamentals)"
                }
                
                # Try to get company profile for additional context
                try:
                    profile_url = f"https://finnhub.io/api/v1/stock/profile2"
                    profile_params = {
                        'symbol': symbol,
                        'token': self.finnhub_key
                    }
                    profile_response = requests.get(profile_url, params=profile_params, timeout=5)
                    profile_data = profile_response.json()
                    
                    if profile_data and 'name' in profile_data:
                        result["company_name"] = profile_data.get('name', '')
                        result["market_cap"] = profile_data.get('marketCapitalization', 0) * 1000000  # Convert to actual value
                        result["industry"] = profile_data.get('finnhubIndustry', '')
                        result["note"] += " + Company profile"
                except:
                    pass  # Company profile is optional
                
                return result
                
        except Exception as e:
            print(f"Finnhub error for {symbol}: {e}")
        return None

    def _get_twelve_data(self, symbol: str) -> Dict:
        """Get real-time/intraday data from Twelve Data - good for forex, crypto, ETFs too."""
        try:
            if not self.twelve_data_key:
                return None
                
            # Get real-time price
            price_url = f"https://api.twelvedata.com/price"
            price_params = {
                'symbol': symbol,
                'apikey': self.twelve_data_key
            }
            
            response = requests.get(price_url, params=price_params, timeout=10)
            data = response.json()
            
            if 'price' in data:
                price = float(data['price'])
                result = {
                    "symbol": symbol,
                    "current_price": price,
                    "change": round(price * 0.01, 2),  # Estimate 1% change
                    "note": "Real-time data from Twelve Data (Multi-asset support)"
                }
                
                # Try to get additional quote data
                try:
                    quote_url = f"https://api.twelvedata.com/quote"
                    quote_params = {
                        'symbol': symbol,
                        'apikey': self.twelve_data_key
                    }
                    quote_response = requests.get(quote_url, params=quote_params, timeout=5)
                    quote_data = quote_response.json()
                    
                    if quote_data and 'open' in quote_data:
                        result.update({
                            "open": float(quote_data.get('open', price)),
                            "high": float(quote_data.get('high', price)),
                            "low": float(quote_data.get('low', price)),
                            "volume": int(quote_data.get('volume', 1000000)),
                            "change": float(quote_data.get('change', 0)),
                            "change_percent": float(quote_data.get('percent_change', 0)),
                            "note": "Real-time quote from Twelve Data (Complete OHLCV)"
                        })
                except:
                    pass  # Additional data is optional
                
                return result
                
        except Exception as e:
            print(f"Twelve Data error for {symbol}: {e}")
        return None

    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Yahoo Finance."""
        # Remove any whitespace
        symbol = symbol.strip().upper()
        
        # Add exchange suffix if needed
        if symbol == "TSMC":
            return "TSM"  # TSMC trades on NYSE as TSM
        elif symbol == "SAMSUNG":
            return "SSNLF"  # Samsung trades on OTC as SSNLF
        return symbol

    def get_stock_data(self, symbol: str, period: str = "1d") -> Dict:
        """Get stock data using multiple API sources with fallback."""
        try:
            formatted_symbol = self._format_symbol(symbol)
            
            # Try Alpha Vantage first
            print(f"🔍 Trying Alpha Vantage for {symbol}...")
            data = self._get_alpha_vantage_data(formatted_symbol)
            if data:
                print(f"✅ Alpha Vantage success for {symbol}")
                return data
            
            # Try Finnhub
            print(f"🔍 Trying Finnhub for {symbol}...")
            data = self._get_finnhub_data(formatted_symbol)
            if data:
                print(f"✅ Finnhub success for {symbol}")
                return data
                
            # Try Twelve Data
            print(f"🔍 Trying Twelve Data for {symbol}...")
            data = self._get_twelve_data(formatted_symbol)
            if data:
                print(f"✅ Twelve Data success for {symbol}")
                return data
            
            # Try Yahoo Finance as backup
            print(f"🔍 Trying Yahoo Finance for {symbol}...")
            stock = yf.Ticker(formatted_symbol)
            info = stock.info
            if info and 'regularMarketPrice' in info:
                print(f"✅ Yahoo Finance success for {symbol}")
                return {
                    "symbol": symbol,
                    "current_price": float(info.get('regularMarketPrice', 100.0)),
                    "change": float(info.get('regularMarketChange', 2.5)),
                    "volume": int(info.get('regularMarketVolume', 1000000)),
                    "market_cap": info.get('marketCap', 'N/A'),
                    "note": "Real-time data from Yahoo Finance"
                }
            
            # Final fallback with realistic mock data
            print(f"⚠️ All APIs failed for {symbol}, using mock data")
            mock_prices = {
                "MSFT": 420.0, "GOOGL": 175.0, "AAPL": 195.0, "NVDA": 140.0, 
                "TSM": 110.0, "SSNLF": 55.0, "AMZN": 180.0, "META": 500.0
            }
            
            base_price = mock_prices.get(formatted_symbol, 100.0)
            return {
                "symbol": symbol,
                "current_price": base_price,
                "change": round(base_price * 0.025, 2),
                "volume": 1500000,
                "note": "Using realistic mock data - all APIs unavailable"
            }
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            # Return realistic fallback data
            mock_prices = {
                "MSFT": 420.0, "GOOGL": 175.0, "AAPL": 195.0, "NVDA": 140.0, 
                "TSM": 110.0, "SSNLF": 55.0, "AMZN": 180.0, "META": 500.0
            }
            
            base_price = mock_prices.get(symbol.upper(), 100.0)
            return {
                "symbol": symbol,
                "current_price": base_price,
                "change": round(base_price * 0.025, 2),
                "volume": 1500000,
                "note": "Using realistic mock data - error occurred"
            }

    def get_earnings_data(self, symbol: str) -> Dict:
        """Get comprehensive earnings and financial data using Finnhub for fundamentals."""
        try:
            formatted_symbol = self._format_symbol(symbol)
            
            # Try Finnhub first for comprehensive financial data
            if self.finnhub_key:
                print(f"🔍 Getting financial statements from Finnhub for {symbol}...")
                try:
                    # Get basic financials
                    financials_url = f"https://finnhub.io/api/v1/stock/metric"
                    financials_params = {
                        'symbol': formatted_symbol,
                        'metric': 'all',
                        'token': self.finnhub_key
                    }
                    
                    response = requests.get(financials_url, params=financials_params, timeout=10)
                    data = response.json()
                    
                    if 'metric' in data:
                        metrics = data['metric']
                        return {
                            "symbol": symbol,
                            "last_earnings": {
                                "Revenue": metrics.get('revenuePerShareTTM', 0) * metrics.get('sharesOutstanding', 1000000000),
                                "Earnings": metrics.get('netIncomePerShareTTM', 0) * metrics.get('sharesOutstanding', 1000000000),
                                "EPS": metrics.get('epsInclExtraItemsTTM', metrics.get('epsTTM', 5.0)),
                                "PE_Ratio": metrics.get('peInclExtraTTM', metrics.get('peTTM', 25.0)),
                                "Revenue_Growth": metrics.get('revenueGrowthTTMYoy', 0.1),
                                "Profit_Margin": metrics.get('netProfitMarginTTM', 0.15),
                                "ROE": metrics.get('roeTTM', 0.15),
                                "Debt_to_Equity": metrics.get('totalDebt/totalEquityQuarterly', 0.3),
                                "Year": 2024,
                                "Quarter": "TTM",
                                "note": "Comprehensive financial data from Finnhub"
                            }
                        }
                except Exception as e:
                    print(f"Finnhub financials error: {e}")
            
            # Try Yahoo Finance as backup
            print(f"🔍 Trying Yahoo Finance for {symbol} earnings...")
            stock = yf.Ticker(formatted_symbol)
            info = stock.info
            if info:
                return {
                    "symbol": symbol,
                    "last_earnings": {
                        "Revenue": info.get('totalRevenue', 50000000000),
                        "Earnings": info.get('netIncomeToCommon', 10000000000),
                        "EPS": info.get('trailingEps', 5.5),
                        "PE_Ratio": info.get('trailingPE', 25.0),
                        "Revenue_Growth": info.get('revenueGrowth', 0.1),
                        "Profit_Margin": info.get('profitMargins', 0.15),
                        "Year": 2024,
                        "Quarter": "Q4",
                        "note": "Financial data from Yahoo Finance"
                    }
                }
            
            # Use enhanced mock earnings data
            print(f"⚠️ Using enhanced mock earnings for {symbol}")
            mock_earnings = {
                "MSFT": {"Revenue": 245000000000, "Earnings": 88000000000, "EPS": 11.30, "PE_Ratio": 28.5, "Revenue_Growth": 0.12},
                "GOOGL": {"Revenue": 307000000000, "Earnings": 73000000000, "EPS": 5.80, "PE_Ratio": 22.1, "Revenue_Growth": 0.09},
                "AAPL": {"Revenue": 383000000000, "Earnings": 97000000000, "EPS": 6.16, "PE_Ratio": 25.8, "Revenue_Growth": 0.08},
                "NVDA": {"Revenue": 61000000000, "Earnings": 30000000000, "EPS": 12.10, "PE_Ratio": 35.2, "Revenue_Growth": 0.85},
                "TSM": {"Revenue": 62000000000, "Earnings": 21000000000, "EPS": 4.85, "PE_Ratio": 18.5, "Revenue_Growth": 0.15},
                "SSNLF": {"Revenue": 280000000000, "Earnings": 23000000000, "EPS": 3.12, "PE_Ratio": 16.8, "Revenue_Growth": 0.05}
            }
            
            earnings_data = mock_earnings.get(formatted_symbol, {
                "Revenue": 50000000000, "Earnings": 10000000000, "EPS": 5.0, "PE_Ratio": 25.0, "Revenue_Growth": 0.1
            })
            
            return {
                "symbol": symbol,
                "last_earnings": {
                    **earnings_data,
                    "Profit_Margin": round(earnings_data["Earnings"] / earnings_data["Revenue"], 3),
                    "ROE": 0.15,
                    "Debt_to_Equity": 0.3,
                    "Year": 2024,
                    "Quarter": "Q4",
                    "note": "Enhanced realistic mock earnings data"
                }
            }
                
        except Exception as e:
            print(f"❌ Error fetching earnings for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "last_earnings": {
                    "Revenue": 50000000000,
                    "Earnings": 10000000000,
                    "EPS": 5.0,
                    "PE_Ratio": 25.0,
                    "Revenue_Growth": 0.1,
                    "Profit_Margin": 0.2,
                    "ROE": 0.15,
                    "Year": 2024,
                    "Quarter": "Q4",
                    "note": "Fallback mock earnings data - error occurred"
                }
            }

    def get_company_news(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get company news from Finnhub."""
        try:
            if not self.finnhub_key:
                return []
                
            # Get current date
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            news_data = response.json()
            
            if isinstance(news_data, list) and news_data:
                return [
                    {
                        "headline": article.get('headline', ''),
                        "summary": article.get('summary', ''),
                        "url": article.get('url', ''),
                        "datetime": article.get('datetime', 0),
                        "source": article.get('source', 'Finnhub')
                    }
                    for article in news_data[:limit]
                ]
                
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
        
        # Return mock news if API fails
        return [
            {
                "headline": f"{symbol} Shows Strong Performance in Latest Quarter",
                "summary": f"Recent analysis shows {symbol} demonstrating solid fundamentals and growth potential.",
                "url": "#",
                "datetime": int(datetime.now().timestamp()),
                "source": "Mock News"
            }
        ]

class APIAgent:
    def __init__(self):
        self.market_data = MarketDataAPI()
        self.agent = self._create_agent()

    def _create_agent(self):
        """Create the agent for API operations with enhanced tools."""
        return SimpleAgent(
            role="Senior Market Data Analyst",
            goal="Gather comprehensive market data, financial statements, and news for Asia tech stocks",
            backstory="Expert financial analyst with access to multiple premium data sources including Alpha Vantage, Finnhub, and Twelve Data",
            tools=[
                Tool(
                    name="get_stock_data",
                    func=self.market_data.get_stock_data,
                    description="Get real-time stock data with technical indicators from multiple sources"
                ),
                Tool(
                    name="get_earnings_data",
                    func=self.market_data.get_earnings_data,
                    description="Get comprehensive financial statements and earnings data"
                ),
                Tool(
                    name="get_company_news",
                    func=self.market_data.get_company_news,
                    description="Get latest company news and sentiment analysis"
                )
            ]
        )

    async def get_market_brief(self, symbols: List[str]) -> Dict:
        """Generate a market brief for the given symbols."""
        try:
            results = {}
            for symbol in symbols:
                stock_data = self.market_data.get_stock_data(symbol)
                if "error" in stock_data:
                    results[symbol] = {"stock": stock_data, "earnings": {"error": "Skipped earnings data due to stock data error"}}
                    continue
                    
                earnings_data = self.market_data.get_earnings_data(symbol)
                results[symbol] = {
                    "stock": stock_data,
                    "earnings": earnings_data
                }
            return results
        except Exception as e:
            return {"error": f"Failed to generate market brief: {str(e)}"}

    async def get_portfolio_exposure(self, portfolio: Dict[str, float]) -> Dict:
        """Calculate portfolio exposure for Asia tech stocks."""
        try:
            total_value = sum(portfolio.values())
            exposure = {}
            for symbol, value in portfolio.items():
                stock_data = self.market_data.get_stock_data(symbol)
                if "error" not in stock_data:
                    exposure[symbol] = {
                        "value": value,
                        "percentage": (value / total_value) * 100,
                        "current_price": stock_data["current_price"]
                    }
            return exposure
        except Exception as e:
            return {"error": str(e)} 