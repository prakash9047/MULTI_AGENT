"""
Embeddings module for initializing the vector store.
This module handles the creation and initialization of the FAISS vector store.
"""
import os
import argparse
from typing import List
import sys

# Add the parent directory to the path so we can import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agents.retriever_agent import RetrieverAgent
except ImportError as e:
    print(f"Error importing RetrieverAgent: {e}")
    print("Make sure all dependencies are installed and the agents module is properly configured")
    sys.exit(1)

def initialize_vector_store():
    """Initialize the vector store with sample financial documents."""
    print("Initializing vector store...")
    
    try:
        # Create retriever agent
        retriever = RetrieverAgent()
        
        # Sample financial documents to populate the vector store
        sample_docs = [
            "Stock markets represent public markets for the issuing, buying and selling of stocks that trade on a stock exchange or over-the-counter. Stocks, also known as equities, represent fractional ownership in a company.",
            "Financial analysis involves examining a company's financial statements to assess its performance and make recommendations about its future. Key financial statements include the income statement, balance sheet, and cash flow statement.",
            "Risk management in finance involves identifying, analyzing, and mitigating potential losses in investments. Common risk management strategies include diversification, hedging, and asset allocation.",
            "Market volatility refers to the degree of variation in trading prices over time. High volatility indicates large price swings, while low volatility suggests more stable prices.",
            "Portfolio management is the art and science of making decisions about investment mix and policy, matching investments to objectives, asset allocation for individuals and institutions.",
            "Technical analysis is a trading discipline employed to evaluate investments and identify trading opportunities by analyzing statistical trends gathered from trading activity.",
            "Fundamental analysis is a method of measuring a security's intrinsic value by examining related economic and financial factors including company financials, industry conditions, and market conditions.",
            "Dividend investing focuses on stocks that pay regular dividends to shareholders. Dividends provide a steady income stream and can be an important component of total return.",
            "Growth investing is an investment strategy that focuses on capital appreciation. Growth investors look for companies that exhibit signs of above-average growth.",
            "Value investing is an investment strategy that involves picking stocks that appear to be trading for less than their intrinsic or book value."
        ]
        
        print(f"Adding {len(sample_docs)} sample documents to vector store...")
        
        # Add documents to the vector store
        result = retriever.add_documents(sample_docs)
        
        if result.get("status") == "success":
            print("✓ Vector store initialized successfully!")
            print(f"Added {len(sample_docs)} documents to the vector store")
        else:
            print(f"⚠ Vector store initialization completed with warnings: {result.get('message', 'Unknown warning')}")
            
    except Exception as e:
        print(f"✗ Error initializing vector store: {str(e)}")
        print("This is likely due to missing optional dependencies (faiss-cpu, sentence-transformers)")
        print("The system will still work with fallback implementations")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Initialize the vector store for the Finance Assistant")
    parser.add_argument("--init", action="store_true", help="Initialize the vector store with sample data")
    
    args = parser.parse_args()
    
    if args.init:
        initialize_vector_store()
    else:
        print("Use --init to initialize the vector store")
        print("Example: python -m data_ingestion.embeddings --init")

if __name__ == "__main__":
    main()
