
"""
Agent Helper Module
This module provides a substitute for CrewAI agents to avoid dependency issues.
"""

class SimpleAgent:
    """A simple agent class that mimics the interface of CrewAI Agent."""
    
    def __init__(self, role, goal, backstory, tools=None, llm=None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.llm = llm
