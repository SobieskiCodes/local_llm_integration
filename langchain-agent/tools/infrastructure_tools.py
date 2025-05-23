from crewai.tools import BaseTool
import requests
import json
from typing import Optional

class LlamaIndexRAGTool(BaseTool):
    """Tool for agents to query your existing LlamaIndex RAG system"""
    
    name: str = "LlamaIndex_RAG"
    description: str = (
        "Query the knowledge base using RAG retrieval. "
        "Use this for factual information, document searches, and knowledge base queries. "
        "Input should be a clear, specific question."
    )
    
    def _run(self, query: str) -> str:
        """Execute RAG query against LlamaIndex service"""
        try:
            response = requests.post(
                "http://llamaindex:8081/query",
                json={"input": query},
                timeout=30
            )
            response.raise_for_status()
            result = response.json().get("result", "No result found")
            
            # Add metadata for agent context
            return f"RAG Result: {result}\n[Source: Knowledge Base via LlamaIndex]"
            
        except requests.exceptions.Timeout:
            return "RAG query timed out. Try a more specific question."
        except requests.exceptions.RequestException as e:
            return f"RAG system unavailable: {str(e)}"
        except Exception as e:
            return f"RAG query failed: {str(e)}"

class OllamaReasoningTool(BaseTool):
    """Tool for agents to use Ollama for direct reasoning and analysis"""
    
    name: str = "Ollama_Reasoning"
    description: str = (
        "Use Ollama LLM for reasoning, analysis, and synthesis tasks. "
        "Best for analytical thinking, comparing information, and generating insights. "
        "Can include a system prompt for specialized reasoning."
    )
    
    def _run(self, query: str, system_prompt: str = "") -> str:
        """Execute reasoning query against Ollama"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})
            
            response = requests.post(
                "http://ollama:11434/v1/chat/completions",
                json={
                    "model": "mistral:latest",
                    "messages": messages,
                    "temperature": 0.1,  # Low temperature for consistent reasoning
                    "max_tokens": 1000
                },
                timeout=45
            )
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            return f"Analysis: {result}\n[Source: Ollama Reasoning Engine]"
            
        except requests.exceptions.Timeout:
            return "Reasoning request timed out. Try breaking down the question."
        except requests.exceptions.RequestException as e:
            return f"Ollama unavailable: {str(e)}"
        except Exception as e:
            return f"Reasoning failed: {str(e)}"

class QdrantVectorTool(BaseTool):
    """Tool for direct vector similarity searches (optional advanced feature)"""
    
    name: str = "Vector_Search"
    description: str = (
        "Search for semantically similar content in the vector database. "
        "Use when you need to find related concepts or similar documents. "
        "Currently routes through LlamaIndex for simplicity."
    )
    
    def _run(self, query: str) -> str:
        """Execute vector search (via LlamaIndex for now)"""
        # For now, route through LlamaIndex since it already integrates with Qdrant
        # In a more advanced setup, you could query Qdrant directly
        rag_tool = LlamaIndexRAGTool()
        result = rag_tool._run(f"Find information related to: {query}")
        return result.replace("RAG Result:", "Vector Search Result:")