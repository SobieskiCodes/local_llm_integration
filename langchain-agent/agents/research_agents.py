from crewai import Agent
from langchain_community.chat_models import ChatOpenAI
from tools.infrastructure_tools import LlamaIndexRAGTool, OllamaReasoningTool, QdrantVectorTool

# Initialize LLM for all agents (connected to your Ollama)
def get_llm():
    return ChatOpenAI(
        openai_api_base="http://ollama:11434/v1",
        openai_api_key="not-needed",
        model="mistral:latest",
        temperature=0.1
    )

# Initialize tools
rag_tool = LlamaIndexRAGTool()
reasoning_tool = OllamaReasoningTool()
vector_tool = QdrantVectorTool()

def create_research_planner() -> Agent:
    """Agent responsible for breaking down complex queries into research plans"""
    return Agent(
        role="Research Planner",
        goal="Create systematic research plans by breaking complex queries into focused, answerable questions",
        backstory=(
            "You are an expert research methodologist with extensive experience in "
            "academic and industry research. You excel at decomposing complex problems "
            "into manageable investigation steps and designing research strategies."
        ),
        tools=[reasoning_tool],  # Uses reasoning for planning
        llm=get_llm(),
        verbose=True,
        max_iter=3,  # Limit iterations to prevent loops
        memory=True   # Remember context within the task
    )

def create_data_investigator() -> Agent:
    """Agent responsible for gathering information from various sources"""
    return Agent(
        role="Data Investigator", 
        goal="Thoroughly investigate research questions using all available data sources",
        backstory=(
            "You are a meticulous researcher who knows how to extract relevant "
            "information from knowledge bases, databases, and reasoning systems. "
            "You're skilled at formulating precise queries and interpreting results."
        ),
        tools=[rag_tool, vector_tool, reasoning_tool],  # All tools for comprehensive search
        llm=get_llm(),
        verbose=True,
        max_iter=5,
        memory=True
    )

def create_domain_analyst() -> Agent:
    """Agent responsible for expert analysis and interpretation"""
    return Agent(
        role="Domain Analyst",
        goal="Provide expert-level analysis, identify patterns, and generate insights from research findings",
        backstory=(
            "You are a domain expert with deep analytical skills. You excel at "
            "interpreting research findings, identifying patterns and trends, "
            "connecting disparate information, and providing expert insights."
        ),
        tools=[reasoning_tool],  # Focuses on analysis rather than data gathering
        llm=get_llm(),
        verbose=True,
        max_iter=4,
        memory=True
    )

def create_quality_validator() -> Agent:
    """Agent responsible for fact-checking and quality assurance"""
    return Agent(
        role="Quality Validator",
        goal="Validate research accuracy, identify gaps and inconsistencies, assess confidence levels",
        backstory=(
            "You are a rigorous fact-checker and quality assurance expert. "
            "You have a keen eye for inconsistencies, gaps in reasoning, "
            "and potential biases. You ensure research meets high standards."
        ),
        tools=[rag_tool, reasoning_tool],  # Can verify facts and reason about quality
        llm=get_llm(),
        verbose=True,
        max_iter=3,
        memory=True
    )

def create_report_synthesizer() -> Agent:
    """Agent responsible for creating comprehensive reports"""
    return Agent(
        role="Report Synthesizer",
        goal="Create well-structured, comprehensive reports that synthesize all research findings",
        backstory=(
            "You are an expert technical writer and synthesizer. You excel at "
            "taking complex research findings from multiple sources and creating "
            "clear, well-structured, actionable reports and recommendations."
        ),
        tools=[reasoning_tool],  # Focuses on synthesis and writing
        llm=get_llm(),
        verbose=True,
        max_iter=3,
        memory=True
    )

# Agent factory function for easy instantiation
def create_research_team():
    """Create a complete research team with all specialized agents"""
    return {
        "planner": create_research_planner(),
        "investigator": create_data_investigator(),
        "analyst": create_domain_analyst(), 
        "validator": create_quality_validator(),
        "synthesizer": create_report_synthesizer()
    }