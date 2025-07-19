from langgraph.graph import StateGraph, START, END
from typing import Optional, Dict, TypedDict, Union, List
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool, tool

class AgentState(TypedDict):
    messages: List[BaseMessage]
    query: str
    tools: List[Tool]

@tool

