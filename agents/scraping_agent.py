import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")
from typing import Dict, List, Optional
import requests
import warnings
from bs4 import BeautifulSoup

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*pydantic.*")

# Handle newspaper import with a fallback
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    # Removed warning print - will show in agent status instead

# Handle SEC Edgar downloader with a fallback
try:
    from sec_edgar_downloader import Downloader
    SEC_EDGAR_AVAILABLE = True
except ImportError:
    SEC_EDGAR_AVAILABLE = False
    # Removed warning print - will show in agent status instead

# Simplified web scraping without LangChain dependencies
WEBLOADER_AVAILABLE = False  # Use our own implementation instead

# Custom WebBaseLoader replacement - no LangChain dependencies
class WebBaseLoader:
    """Custom web content loader without LangChain dependencies."""
    def __init__(self, web_paths=None):
        self.web_paths = web_paths or []
        
    def load(self):
        """Load content from web URLs using requests and BeautifulSoup."""
        documents = []
        for url in self.web_paths:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                documents.append({
                    "page_content": text[:5000],  # Limit content size
                    "metadata": {"source": url}
                })
            except Exception as e:
                documents.append({
                    "page_content": f"Error loading content: {str(e)}",
                    "metadata": {"source": url, "error": True}
                })
        return documents

from agents.agent_helpers import SimpleAgent

# Simple tool class to replace LangChain Tool
class Tool:
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func
        self.description = description

class ScrapingAgent:
    def __init__(self):
        self.agent = self._create_agent()
        # Initialize SEC downloader only if available
        self.downloader = None
        if SEC_EDGAR_AVAILABLE:
            self.downloader = Downloader("finance-assistant", "your-email@example.com")
        
        # Agent status for debugging
        self.status = {
            "newspaper3k": NEWSPAPER_AVAILABLE,
            "sec_edgar": SEC_EDGAR_AVAILABLE,
            "web_scraping": True  # Always available with our custom implementation
        }

    def _create_agent(self):
        """Create a simplified agent for scraping operations."""
        return SimpleAgent(
            role="Financial Data Scraper",
            goal="Gather financial data from various web sources",
            backstory="Expert in web scraping and data extraction from financial websites",
            tools=[
                Tool(
                    name="scrape_news",
                    func=self.scrape_news,
                    description="Scrape news articles from financial websites"
                ),
                Tool(
                    name="get_sec_filings",
                    func=self.get_sec_filings,
                    description="Download SEC filings for a given company"
                )
            ]
        )

    def scrape_news(self, url: str) -> Dict:
        """Scrape news articles from financial websites."""
        try:
            if not NEWSPAPER_AVAILABLE:
                # Enhanced fallback implementation using BeautifulSoup
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Enhanced extraction of title and text
                title = soup.title.string if soup.title else "No title found"
                
                # Try to find article content in common containers
                content_selectors = ['article', '.article-content', '.content', '.post-content', 'main']
                article_content = None
                
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        article_content = elements[0]
                        break
                
                if article_content:
                    paragraphs = article_content.find_all('p')
                else:
                    paragraphs = soup.find_all('p')
                
                text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                
                return {
                    "title": title.strip(),
                    "text": text[:3000] + ("..." if len(text) > 3000 else ""),
                    "summary": text[:500] + ("..." if len(text) > 500 else ""),
                    "keywords": self._extract_keywords(text),
                    "publish_date": None,
                    "source": url,
                    "method": "BeautifulSoup fallback"
                }
            else:
                # Use newspaper3k when available
                article = Article(url)
                article.download()
                article.parse()
                article.nlp()
                
                return {
                    "title": article.title,
                    "text": article.text,
                    "summary": article.summary,
                    "keywords": article.keywords,
                    "publish_date": article.publish_date,
                    "source": url,
                    "method": "newspaper3k"
                }
        except Exception as e:
            return {
                "error": str(e), 
                "url": url,
                "title": "Error loading article",
                "text": f"Failed to load content from {url}: {str(e)}",
                "method": "error"
            }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text."""
        if not text:
            return []
        
        # Simple keyword extraction
        financial_terms = [
            'revenue', 'profit', 'earnings', 'stock', 'market', 'investment',
            'financial', 'growth', 'analysis', 'report', 'quarterly', 'annual',
            'CEO', 'company', 'business', 'industry', 'sector', 'performance'
        ]
        
        words = text.lower().split()
        keywords = [term for term in financial_terms if term in words]
        return keywords[:10]  # Return top 10 keywords

    def scrape_multiple_sources(self, urls: List[str]) -> List[Dict]:
        """Scrape multiple URLs and return combined results."""
        results = []
        for url in urls:
            try:
                result = self.scrape_news(url)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "url": url,
                    "method": "error"
                })
        return results

    def get_financial_news_urls(self, ticker: str) -> List[str]:
        """Get relevant financial news URLs for a ticker."""
        # Return some common financial news sources
        base_urls = [
            f"https://finance.yahoo.com/quote/{ticker}/news",
            f"https://www.marketwatch.com/investing/stock/{ticker}",
            f"https://seekingalpha.com/symbol/{ticker}/news"
        ]
        return base_urls
    def get_sec_filings(self, ticker: str, filing_type: str = "10-K") -> Dict:
        """Download SEC filings for a given company."""
        try:
            if not SEC_EDGAR_AVAILABLE or self.downloader is None:
                # Return enhanced mock data if SEC downloader is not available
                return {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "status": "mock_data",
                    "note": "SEC Edgar downloader not available, using mock data",
                    "mock_filing": {
                        "company": self._get_company_name(ticker),
                        "filing_summary": f"Mock {filing_type} filing for {ticker}",
                        "key_metrics": {
                            "total_revenue": "$50.2B",
                            "net_income": "$12.3B",
                            "total_assets": "$180.5B",
                            "shareholders_equity": "$95.7B"
                        },
                        "business_description": f"{ticker} is a leading company in its sector with strong financial performance.",
                        "risk_factors": ["Market competition", "Regulatory changes", "Economic conditions"]
                    }
                }
            
            self.downloader.get(filing_type, ticker, limit=1)
            return {
                "ticker": ticker,
                "filing_type": filing_type,
                "status": "success",
                "note": "Real SEC filing downloaded successfully"
            }
        except Exception as e:
            return {
                "error": str(e), 
                "ticker": ticker, 
                "filing_type": filing_type,
                "status": "error"
            }

    def scrape_macrotrends(self, ticker: str) -> Dict:
        """Scrape data from Macrotrends.net."""
        try:
            url = f"https://www.macrotrends.net/stocks/charts/{ticker}/stock-price-history"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant data (customize based on website structure)
            data = {
                "ticker": ticker,
                "price_history": self._extract_price_history(soup),
                "financial_metrics": self._extract_financial_metrics(soup)
            }
            return data
        except Exception as e:
            return {"error": str(e)}

    def _extract_price_history(self, soup: BeautifulSoup) -> Dict:
        """Extract price history from Macrotrends page."""
        # Implement specific extraction logic
        return {}

    def _extract_financial_metrics(self, soup: BeautifulSoup) -> Dict:
        """Extract financial metrics from Macrotrends page."""
        # Implement specific extraction logic
        return {}

    async def get_company_data(self, ticker: str) -> Dict:
        """Get comprehensive company data from various sources."""
        try:
            # Enhanced company data with more realistic information
            company_info = {
                "company": ticker,
                "name": self._get_company_name(ticker),
                "description": self._get_company_description(ticker),
                "sector": self._get_company_sector(ticker),
                "market_cap": self._get_market_cap(ticker),
                "headquarters": self._get_headquarters(ticker),
                "founded": self._get_founded_year(ticker),
                "employees": self._get_employee_count(ticker),
                "recent_news": f"Recent positive developments in {ticker}'s business strategy and market expansion.",
                "financial_highlights": {
                    "revenue_growth": self._get_revenue_growth(ticker),
                    "profit_margin": self._get_profit_margin(ticker),
                    "pe_ratio": self._get_pe_ratio(ticker),
                    "recent_milestone": f"{ticker} continues to show strong performance in key markets."
                },
                "key_executives": self._get_key_executives(ticker),
                "competitive_advantages": self._get_competitive_advantages(ticker),
                "data_source": "Enhanced mock data with realistic company information"
            }
            
            return company_info
        except Exception as e:
            print(f"Error getting company data for {ticker}: {str(e)}")
            # Return minimal data on error
            return {
                "company": ticker,
                "name": ticker,
                "description": "Company information temporarily unavailable.",
                "sector": "Technology",
                "error": str(e)
            }
            
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker symbol."""
        name_map = {
            "TSMC": "Taiwan Semiconductor Manufacturing Company",
            "SAMSUNG": "Samsung Electronics Co., Ltd.",
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms, Inc.",
            "NVDA": "NVIDIA Corporation",
            "TSLA": "Tesla, Inc.",
            "NFLX": "Netflix, Inc.",
            "AMD": "Advanced Micro Devices, Inc.",
            "INTC": "Intel Corporation"
        }
        return name_map.get(ticker.upper(), f"{ticker} Corporation")
        
    def _get_company_sector(self, ticker: str) -> str:
        """Get company sector from ticker symbol."""
        sector_map = {
            "TSMC": "Semiconductor Manufacturing",
            "SAMSUNG": "Consumer Electronics",
            "AAPL": "Consumer Electronics",
            "MSFT": "Software",
            "GOOGL": "Internet Services",
            "AMZN": "E-commerce",
            "META": "Social Media",
            "NVDA": "Semiconductors",
            "TSLA": "Electric Vehicles",
            "NFLX": "Streaming Entertainment",
            "AMD": "Semiconductors",
            "INTC": "Semiconductors"
        }
        return sector_map.get(ticker.upper(), "Technology")

    def _get_company_description(self, ticker: str) -> str:
        """Get company description."""
        descriptions = {
            "AAPL": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
            "MSFT": "Microsoft Corporation develops, licenses, and supports software, services, devices, and solutions worldwide.",
            "GOOGL": "Alphabet Inc. provides search, advertising, operating systems, cloud computing, and related services and products worldwide.",
            "AMZN": "Amazon.com Inc. engages in the retail sale of consumer products and subscriptions through online and physical stores worldwide.",
            "NVDA": "NVIDIA Corporation provides graphics, and compute and networking solutions in the United States, Taiwan, China, and internationally."
        }
        return descriptions.get(ticker.upper(), f"A leading company in the {self._get_company_sector(ticker)} sector.")

    def _get_market_cap(self, ticker: str) -> str:
        """Get market capitalization."""
        market_caps = {
            "AAPL": "$2.8T", "MSFT": "$2.5T", "GOOGL": "$1.6T", 
            "AMZN": "$1.4T", "NVDA": "$1.2T", "META": "$800B",
            "TSLA": "$700B", "NFLX": "$180B"
        }
        return market_caps.get(ticker.upper(), "$50B+")

    def _get_headquarters(self, ticker: str) -> str:
        """Get company headquarters."""
        headquarters = {
            "AAPL": "Cupertino, California", "MSFT": "Redmond, Washington",
            "GOOGL": "Mountain View, California", "AMZN": "Seattle, Washington",
            "NVDA": "Santa Clara, California", "META": "Menlo Park, California",
            "TSLA": "Austin, Texas", "NFLX": "Los Gatos, California"
        }
        return headquarters.get(ticker.upper(), "United States")

    def _get_founded_year(self, ticker: str) -> str:
        """Get company founding year."""
        founded = {
            "AAPL": "1976", "MSFT": "1975", "GOOGL": "1998",
            "AMZN": "1994", "NVDA": "1993", "META": "2004",
            "TSLA": "2003", "NFLX": "1997"
        }
        return founded.get(ticker.upper(), "1990s")

    def _get_employee_count(self, ticker: str) -> str:
        """Get employee count."""
        employees = {
            "AAPL": "164,000", "MSFT": "221,000", "GOOGL": "174,000",
            "AMZN": "1,541,000", "NVDA": "26,000", "META": "87,000",
            "TSLA": "128,000", "NFLX": "12,800"
        }
        return employees.get(ticker.upper(), "50,000+")

    def _get_revenue_growth(self, ticker: str) -> str:
        """Get revenue growth rate."""
        growth = {
            "AAPL": "8%", "MSFT": "12%", "GOOGL": "15%",
            "AMZN": "9%", "NVDA": "61%", "META": "23%",
            "TSLA": "51%", "NFLX": "6%"
        }
        return growth.get(ticker.upper(), "10%")

    def _get_profit_margin(self, ticker: str) -> str:
        """Get profit margin."""
        margins = {
            "AAPL": "25%", "MSFT": "37%", "GOOGL": "21%",
            "AMZN": "6%", "NVDA": "32%", "META": "29%",
            "TSLA": "8%", "NFLX": "18%"
        }
        return margins.get(ticker.upper(), "15%")

    def _get_pe_ratio(self, ticker: str) -> str:
        """Get P/E ratio."""
        pe_ratios = {
            "AAPL": "28", "MSFT": "32", "GOOGL": "22",
            "AMZN": "45", "NVDA": "65", "META": "24",
            "TSLA": "55", "NFLX": "35"
        }
        return pe_ratios.get(ticker.upper(), "25")

    def _get_key_executives(self, ticker: str) -> List[str]:
        """Get key executives."""
        executives = {
            "AAPL": ["Tim Cook (CEO)", "Luca Maestri (CFO)", "Katherine Adams (General Counsel)"],
            "MSFT": ["Satya Nadella (CEO)", "Amy Hood (CFO)", "Brad Smith (President)"],
            "GOOGL": ["Sundar Pichai (CEO)", "Ruth Porat (CFO)", "Kent Walker (General Counsel)"]
        }
        return executives.get(ticker.upper(), ["CEO", "CFO", "CTO"])

    def _get_competitive_advantages(self, ticker: str) -> List[str]:
        """Get competitive advantages."""
        advantages = {
            "AAPL": ["Strong brand loyalty", "Integrated ecosystem", "Premium pricing power"],
            "MSFT": ["Enterprise software dominance", "Cloud infrastructure", "Subscription model"],
            "GOOGL": ["Search market dominance", "Advertising technology", "AI capabilities"]
        }
        return advantages.get(ticker.upper(), ["Market leadership", "Innovation", "Strong financials"])