from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from typing import Dict, Any, List, Optional

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# LangChain for LLM connection
from langchain.llms.ollama import Ollama

# Web scraping tools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import requests

# Database tools
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text

# Vector store for knowledge persistence
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# LLM setup
OLLAMA_URL = os.getenv("OLLAMA_API", "http://ollama:11434")
llm = Ollama(model="mistral:latest", base_url=OLLAMA_URL)

# Qdrant setup
qdrant_client = QdrantClient(host="qdrant", port=6333)
DSCSA_COLLECTION = "dscsa_knowledge"

# Make sure the collection exists
existing_collections = [c.name for c in qdrant_client.get_collections().collections]
if DSCSA_COLLECTION not in existing_collections:
    from qdrant_client.models import VectorParams, Distance
    print(f"Creating collection '{DSCSA_COLLECTION}' in Qdrant...")
    qdrant_client.recreate_collection(
        collection_name=DSCSA_COLLECTION,
        vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
    )

# Vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=DSCSA_COLLECTION)


# ===== TOOL DEFINITIONS =====

class WebScraperTool(BaseTool):
    name: str = "Web Scraper Tool"
    description: str = "Scrapes content from websites using URLs"
    
    def _run(self, url: str) -> str:
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            page_source = driver.page_source
            driver.quit()
            
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract text content, removing script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

class DatabaseQueryTool(BaseTool):
    name: str = "Database Query Tool"
    description: str = "Runs SQL queries against a connected database"
    
    def _run(self, query: str, connection_string: Optional[str] = None) -> str:
        # Default to a demo database if none provided
        if connection_string is None:
            # This would be populated from environment variables in production
            connection_string = os.getenv("DATABASE_URL", "sqlite:///demo.db")
        
        try:
            engine = create_engine(connection_string)
            with engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall())
                if not df.empty:
                    df.columns = result.keys()
                    return df.to_string()
                return "Query executed successfully. No results returned."
        except Exception as e:
            return f"Database query error: {str(e)}"

class DSCSAKnowledgeSearchTool(BaseTool):
    name: str = "DSCSA Knowledge Search Tool"
    description: str = "Searches the DSCSA knowledge base for relevant information"
    
    def _run(self, query: str) -> str:
        try:
            # Using llama_index to query the vector store
            from llama_index.core import Settings
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            Settings.embed_model = OllamaEmbedding(model_name="mistral:latest", base_url=OLLAMA_URL)
            
            index = VectorStoreIndex.from_vector_store(vector_store)
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            
            return str(response)
        except Exception as e:
            return f"Knowledge search error: {str(e)}"

class StoreToKnowledgeBaseTool(BaseTool):
    name: str = "Store To Knowledge Base Tool"
    description: str = "Stores information to the DSCSA knowledge base"
    
    def _run(self, content: str, metadata: Optional[Dict[str, str]] = None) -> str:
        try:
            if metadata is None:
                metadata = {"source": "agent", "type": "dscsa"}
                
            doc = Document(text=content, metadata=metadata)
            
            # Store document in vector database
            from llama_index.core import Settings, StorageContext
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            Settings.embed_model = OllamaEmbedding(model_name="mistral:latest", base_url=OLLAMA_URL)
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex.from_documents([doc], storage_context=storage_context)
            
            return "Successfully stored in knowledge base"
        except Exception as e:
            return f"Error storing to knowledge base: {str(e)}"


# ===== AGENT DEFINITIONS =====

# Database Analytics Agent
db_analytics_agent = Agent(
    role="Database Analytics Expert",
    goal="Analyze databases and provide insightful information based on SQL queries",
    backstory="""You are an expert in database analytics with years of experience in SQL 
    and data analysis. You can understand database schemas and write complex queries 
    to extract valuable insights from any database.""",
    verbose=True,
    allow_delegation=True,
    tools=[DatabaseQueryTool()],
    llm=llm
)

# DSCSA Knowledge Agent
dscsa_agent = Agent(
    role="DSCSA Compliance Expert",
    goal="Gather, organize and provide accurate information about DSCSA compliance requirements",
    backstory="""You are a pharmaceutical compliance expert specializing in the Drug Supply 
    Chain Security Act (DSCSA). You stay updated with the latest regulations and can explain 
    complex compliance requirements in simple terms.""",
    verbose=True,
    allow_delegation=True,
    tools=[WebScraperTool(), DSCSAKnowledgeSearchTool(), StoreToKnowledgeBaseTool()],
    llm=llm
)

# ===== TASK DEFINITIONS =====

def create_web_scraping_task(url: str):
    return Task(
        description=f"Scrape DSCSA compliance information from {url}, extract the most important facts about regulations, requirements, and deadlines, and store this knowledge in our database.",
        expected_output="A comprehensive summary of the DSCSA information found on the website, with key facts organized by category.",
        agent=dscsa_agent
    )

def create_db_analysis_task(query: str, connection_string: Optional[str] = None):
    return Task(
        description=f"Analyze the database using the following query: {query}. Provide insights about the data, identify patterns or anomalies, and suggest any additional queries that might yield more insights.",
        expected_output="A clear analysis of the database query results with actionable insights.",
        agent=db_analytics_agent,
        context={"connection_string": connection_string}
    )

def create_dscsa_knowledge_query_task(query: str):
    return Task(
        description=f"Answer the following question about DSCSA compliance: {query}. Use your knowledge and search tools to provide accurate and up-to-date information.",
        expected_output="A comprehensive answer to the DSCSA compliance question with references to regulations.",
        agent=dscsa_agent
    )

# ===== API SETUP =====

app = FastAPI()

@app.post("/scrape-dscsa")
async def scrape_dscsa(request: Request):
    body = await request.json()
    url = body.get("url")
    
    if not url:
        return JSONResponse(content={"error": "URL is required"}, status_code=400)
    
    task = create_web_scraping_task(url)
    crew = Crew(
        agents=[dscsa_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    
    # Store the result
    store_tool = StoreToKnowledgeBaseTool()
    storage_result = store_tool._run(
        result,
        {"source": url, "type": "dscsa_scraped"}
    )
    
    return {"result": result, "storage": storage_result}

@app.post("/analyze-database")
async def analyze_database(request: Request):
    body = await request.json()
    query = body.get("query")
    connection_string = body.get("connection_string")
    
    if not query:
        return JSONResponse(content={"error": "SQL query is required"}, status_code=400)
    
    task = create_db_analysis_task(query, connection_string)
    crew = Crew(
        agents=[db_analytics_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return {"result": result}

@app.post("/query-dscsa")
async def query_dscsa(request: Request):
    body = await request.json()
    query = body.get("query")
    
    if not query:
        return JSONResponse(content={"error": "Query is required"}, status_code=400)
    
    task = create_dscsa_knowledge_query_task(query)
    crew = Crew(
        agents=[dscsa_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return {"result": result}

@app.post("/multi-agent-query")
async def multi_agent_query(request: Request):
    body = await request.json()
    query = body.get("query")
    
    if not query:
        return JSONResponse(content={"error": "Query is required"}, status_code=400)
    
    # Create a more complex task that might require both agents
    dscsa_task = Task(
        description=f"Research DSCSA compliance related to this query: {query}",
        expected_output="Relevant DSCSA compliance information",
        agent=dscsa_agent
    )
    
    db_task = Task(
        description=f"Analyze database records related to: {query}",
        expected_output="Database analysis relevant to the query",
        agent=db_analytics_agent
    )
    
    crew = Crew(
        agents=[dscsa_agent, db_analytics_agent],
        tasks=[dscsa_task, db_task],
        process=Process.sequential,
        verbose=True
    )
    
    result = crew.kickoff()
    return {"result": result}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agents": ["DSCSA Knowledge Agent", "Database Analytics Agent"],
        "version": "0.1.0"
    }