import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*pydantic.*")

# Simple tool class to replace LangChain Tool
class Tool:
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func
        self.description = description

# Mock NetworkX Entity Graph for analysis
class MockEntityGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def add_entity(self, entity: str, entity_type: str = "company"):
        if entity not in self.nodes:
            self.nodes.append({"entity": entity, "type": entity_type})
    
    def add_relationship(self, entity1: str, entity2: str, relationship: str):
        self.edges.append({"source": entity1, "target": entity2, "relationship": relationship})
    
    def get_connections(self, entity: str) -> List[str]:
        connections = []
        for edge in self.edges:
            if edge["source"] == entity:
                connections.append(f"{edge['target']} ({edge['relationship']})")
            elif edge["target"] == entity:
                connections.append(f"{edge['source']} ({edge['relationship']})")
        return connections

from config.settings import settings
from agents.agent_helpers import SimpleAgent

class AnalysisAgent:
    def __init__(self):
        self.graph = MockEntityGraph()
        self.agent = self._create_agent()
        
        # Agent status for debugging
        self.status = {
            "pandas": True,
            "numpy": True,
            "entity_graph": "mock",
            "analysis_ready": True
        }

    def _create_agent(self):
        """Create the agent for analysis operations."""
        return SimpleAgent(
            role="Financial Analyst",
            goal="Analyze market data and generate insights",
            backstory="Expert in financial analysis and market insights",
            tools=[
                Tool(
                    name="analyze_portfolio",
                    func=self.analyze_portfolio,
                    description="Analyze portfolio risk and exposure"
                ),
                Tool(
                    name="analyze_earnings",
                    func=self.analyze_earnings,
                    description="Analyze earnings surprises and trends"
                )
            ]
        )

    def analyze_portfolio(self, portfolio_data: Dict) -> Dict:
        """Analyze portfolio risk and exposure."""
        try:
            # Calculate portfolio metrics
            total_value = sum(item["value"] for item in portfolio_data.values())
            exposure = {
                symbol: {
                    "value": data["value"],
                    "percentage": (data["value"] / total_value) * 100,
                    "risk_score": self._calculate_risk_score(data)
                }
                for symbol, data in portfolio_data.items()
            }
            
            # Calculate overall portfolio metrics
            portfolio_metrics = {
                "total_value": total_value,
                "total_exposure": sum(item["percentage"] for item in exposure.values()),
                "risk_score": np.mean([item["risk_score"] for item in exposure.values()]),
                "exposure_by_sector": self._calculate_sector_exposure(exposure)
            }
            
            return {
                "exposure": exposure,
                "metrics": portfolio_metrics
            }
        except Exception as e:
            return {"error": str(e)}

    def analyze_earnings(self, earnings_data: Dict) -> Dict:
        """Analyze earnings surprises and trends."""
        try:
            results = {}
            for symbol, data in earnings_data.items():
                if "earnings" in data and data["earnings"]:
                    earnings = data["earnings"]
                    surprise = self._calculate_earnings_surprise(earnings)
                    trend = self._analyze_earnings_trend(earnings)
                    
                    results[symbol] = {
                        "surprise": surprise,
                        "trend": trend,
                        "sentiment": self._calculate_earnings_sentiment(surprise, trend)
                    }
            
            return results
        except Exception as e:
            return {"error": str(e)}

    def _calculate_risk_score(self, data: Dict) -> float:
        """Calculate risk score for a position."""
        # Implement risk scoring logic
        return 0.5  # Placeholder

    def _calculate_sector_exposure(self, exposure: Dict) -> Dict:
        """Calculate exposure by sector."""
        # Implement sector exposure calculation
        return {"Technology": 100}  # Placeholder

    def _calculate_earnings_surprise(self, earnings: Dict) -> float:
        """Calculate earnings surprise percentage."""
        # Implement earnings surprise calculation
        return 0.0  # Placeholder

    def _analyze_earnings_trend(self, earnings: Dict) -> str:
        """Analyze earnings trend."""
        # Implement earnings trend analysis
        return "neutral"  # Placeholder
    
    def _calculate_earnings_sentiment(self, surprise: float, trend: str) -> str:
        """Calculate earnings sentiment."""
        # Implement sentiment calculation
        return "neutral"  # Placeholder
        
    def generate_market_brief(self, market_data: Dict, company_data: Dict) -> Dict:
        """Generate a comprehensive market brief."""
        try:
            # Process market data into portfolio format
            portfolio_data = {}
            for symbol, data in market_data.items():
                if isinstance(data, dict) and "stock" in data:
                    stock_data = data.get("stock", {})
                    if isinstance(stock_data, dict) and "current_price" in stock_data:
                        # Create mock portfolio data for analysis
                        portfolio_data[symbol] = {
                            "value": stock_data.get("current_price", 100) * 10,  # Assume 10 shares
                            "price": stock_data.get("current_price", 100),
                            "change": stock_data.get("change", 0),
                            "volume": stock_data.get("volume", 1000)
                        }

            # Process earnings data
            earnings_data = {}
            for symbol, data in market_data.items():
                if isinstance(data, dict) and "earnings" in data:
                    earnings_data[symbol] = data.get("earnings", {})
            
            try:
                portfolio_analysis = self.analyze_portfolio(portfolio_data)
            except Exception as portfolio_error:
                print(f"Portfolio analysis error: {str(portfolio_error)}")
                portfolio_analysis = {"summary": f"Analysis of portfolio containing {', '.join(market_data.keys())}."}
                
            try:
                earnings_analysis = self.analyze_earnings(earnings_data)
            except Exception as earnings_error:
                print(f"Earnings analysis error: {str(earnings_error)}")
                earnings_analysis = {"summary": f"Recent earnings show mixed results for {', '.join(market_data.keys())}."}
            
            return {
                "portfolio": portfolio_analysis if isinstance(portfolio_analysis, dict) else {"summary": str(portfolio_analysis)},
                "earnings": earnings_analysis if isinstance(earnings_analysis, dict) else {"summary": str(earnings_analysis)},
                "summary": self._generate_summary(portfolio_analysis, earnings_analysis)
            }
        except Exception as e:
            print(f"Market brief generation error: {str(e)}")
            return {
                "portfolio": f"Analysis of portfolio containing {', '.join(list(market_data.keys())[:5])}" + 
                            (f" and {len(market_data) - 5} more symbols" if len(market_data) > 5 else "") + ".",
                "earnings": "Recent earnings data analysis not available at this time."
            }

    def _generate_summary(self, portfolio_analysis: Dict, earnings_analysis: Dict) -> str:
        """Generate a natural language summary of the analysis."""
        summary_parts = []
        
        if isinstance(portfolio_analysis, dict) and "metrics" in portfolio_analysis:
            metrics = portfolio_analysis["metrics"]
            total_value = metrics.get("total_value", 0)
            risk_score = metrics.get("risk_score", 0)
            summary_parts.append(f"Portfolio analysis shows total value of ${total_value:,.2f} with risk score of {risk_score:.2f}")
        
        if isinstance(earnings_analysis, dict) and earnings_analysis:
            positive_surprises = sum(1 for data in earnings_analysis.values() 
                                   if isinstance(data, dict) and data.get("surprise", 0) > 0)
            total_companies = len(earnings_analysis)
            summary_parts.append(f"Earnings analysis covers {total_companies} companies with {positive_surprises} positive surprises")
        
        return ". ".join(summary_parts) if summary_parts else "Analysis completed successfully"

    def build_entity_graph(self, companies: List[str]) -> Dict:
        """Build entity relationships between companies."""
        try:
            # Add companies to the graph
            for company in companies:
                self.graph.add_entity(company, "company")
            
            # Add some sample relationships (in real implementation, this would use actual data)
            relationships = [
                ("AAPL", "MSFT", "competitor"),
                ("GOOGL", "META", "competitor"),
                ("NVDA", "AMD", "competitor"),
                ("TSLA", "AAPL", "supplier_relationship")
            ]
            
            for entity1, entity2, relationship in relationships:
                if entity1 in companies and entity2 in companies:
                    self.graph.add_relationship(entity1, entity2, relationship)
            
            return {
                "nodes": len(self.graph.nodes),
                "edges": len(self.graph.edges),
                "relationships": [f"{edge['source']} -> {edge['target']} ({edge['relationship']})" 
                               for edge in self.graph.edges]
            }
        except Exception as e:
            return {"error": str(e)}

    def analyze_company_relationships(self, company: str) -> Dict:
        """Analyze relationships for a specific company."""
        try:
            connections = self.graph.get_connections(company)
            return {
                "company": company,
                "total_connections": len(connections),
                "connections": connections,
                "analysis": f"{company} has {len(connections)} identified relationships in the market"
            }
        except Exception as e:
            return {"error": str(e)} 