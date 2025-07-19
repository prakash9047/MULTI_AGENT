import dotenv
dotenv.load_dotenv(dotenv_path="config/secrets.env")
import streamlit as st
import requests
import json
from typing import Dict, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_URL = "http://localhost:8000"

def main():
    st.title("Multi-Agent Finance Assistant")
    st.write("Your AI-powered financial analysis companion")

    # Sidebar
    st.sidebar.title("Settings")
    use_voice = st.sidebar.checkbox("Use Voice Interface", value=False)

    # Main content
    tab1, tab2 = st.tabs(["Market Brief", "General Query"])

    with tab1:
        st.header("Market Brief")
        st.write("Get a comprehensive market brief for your portfolio")

        # Portfolio input
        symbols = st.text_input(
            "Enter stock symbols (comma-separated)",
            "TSMC, SAMSUNG"
        ).split(",")
        symbols = [s.strip() for s in symbols]

        if st.button("Generate Market Brief"):
            with st.spinner("Generating market brief..."):
                try:
                    response = requests.post(
                        f"{API_URL}/market-brief",
                        json={
                            "symbols": symbols,
                            "use_voice": use_voice
                        }
                    )
                    response.raise_for_status()
                    brief = response.json()

                    # Display brief
                    st.subheader("Market Brief")
                    st.write(brief["brief"])

                    # Display portfolio analysis
                    if "portfolio" in brief:
                        st.subheader("Portfolio Analysis")
                        st.write(brief["portfolio"])

                    # Display earnings analysis
                    if "earnings" in brief:
                        st.subheader("Earnings Analysis")
                        st.write(brief["earnings"])

                    # Play audio if available
                    if use_voice and "audio_file" in brief:
                        st.audio(brief["audio_file"])

                except Exception as e:
                    st.error(f"Error generating market brief: {str(e)}")

    with tab2:
        st.header("General Query")
        st.write("Ask any financial question")

        # Query input
        query = st.text_input("Enter your query")

        if st.button("Submit Query"):
            with st.spinner("Processing query..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "text": query,
                            "use_voice": use_voice
                        }
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Display response
                    st.subheader("Response")
                    st.write(result["summary"])

                    # Play audio if available
                    if use_voice and "audio_file" in result:
                        st.audio(result["audio_file"])

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main() 