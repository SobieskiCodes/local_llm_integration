from crewai import Task, Crew, Process
from agents.research_agents import create_research_team
from typing import Dict, Any

def create_planning_task(query: str, agent) -> Task:
    """Task for breaking down the research query"""
    return Task(
        description=f"""
        Analyze this research query and create a systematic investigation plan:
        
        Original Query: "{query}"
        
        Your task:
        1. Identify the key aspects that need to be researched
        2. Break down the query into 3-5 specific, focused research questions
        3. For each question, specify the type of information needed
        4. Estimate the complexity level (simple, moderate, complex)
        5. Suggest the best research approach for each question
        
        Output Format:
        Research Plan for: [Original Query]
        
        Research Questions:
        1. [Question 1] - [Approach/Strategy]
        2. [Question 2] - [Approach/Strategy]
        ...
        
        Overall Complexity: [Assessment]
        Recommended Strategy: [High-level approach]
        """,
        agent=agent,
        expected_output="Detailed research plan with specific questions and strategies"
    )

def create_investigation_task(research_plan: str, query: str, agent) -> Task:
    """Task for gathering information based on the research plan"""
    return Task(
        description=f"""
        Execute the research plan by investigating each research question:
        
        Original Query: "{query}"
        Research Plan: {research_plan}
        
        Your task:
        1. For each research question in the plan, gather comprehensive information
        2. Use appropriate tools (RAG search, vector search, reasoning) as needed
        3. Document your findings with source attribution
        4. Note any information gaps or uncertainties
        5. Gather supporting evidence and examples
        
        For each question, provide:
        - Main findings
        - Supporting evidence
        - Sources used
        - Confidence level (High/Medium/Low)
        - Related information discovered
        
        Be thorough but focused on answering the specific research questions.
        """,
        agent=agent,
        expected_output="Comprehensive investigation findings for each research question with sources"
    )

def create_analysis_task(investigation_results: str, query: str, agent) -> Task:
    """Task for expert analysis of the gathered information"""
    return Task(
        description=f"""
        Analyze the investigation findings and provide expert insights:
        
        Original Query: "{query}"
        Investigation Results: {investigation_results}
        
        Your task:
        1. Analyze the findings to identify key patterns and themes
        2. Look for connections between different pieces of information
        3. Identify any contradictions or inconsistencies
        4. Provide expert interpretation and insights
        5. Consider implications and broader context
        6. Highlight the most significant findings
        
        Focus on:
        - What the findings reveal about the original query
        - Key insights that emerge from the analysis
        - Important patterns or trends
        - Implications for practical application
        - Areas requiring further investigation
        """,
        agent=agent,
        expected_output="Expert analysis with insights, patterns, and implications"
    )

def create_validation_task(analysis_results: str, query: str, agent) -> Task:
    """Task for validating research quality and identifying gaps"""
    return Task(
        description=f"""
        Validate the research quality and assess completeness:
        
        Original Query: "{query}"
        Analysis Results: {analysis_results}
        
        Your task:
        1. Fact-check key claims and findings
        2. Assess the completeness of the research
        3. Identify any logical gaps or inconsistencies
        4. Evaluate the reliability of sources and methods
        5. Assign confidence scores to major findings
        6. Suggest improvements or additional research needed
        
        Validation Checklist:
        - Are the findings well-supported by evidence?
        - Are there significant gaps in the research?
        - Do the conclusions follow logically from the findings?
        - What is the overall confidence level?
        - What additional research would strengthen the findings?
        
        Provide specific recommendations for improvement.
        """,
        agent=agent,
        expected_output="Quality validation report with confidence scores and improvement recommendations"
    )

def create_synthesis_task(all_results: Dict[str, str], query: str, agent) -> Task:
    """Task for creating the final comprehensive report"""
    return Task(
        description=f"""
        Create a comprehensive research report synthesizing all findings:
        
        Original Query: "{query}"
        Research Plan: {all_results.get('planning', '')}
        Investigation: {all_results.get('investigation', '')}
        Analysis: {all_results.get('analysis', '')}
        Validation: {all_results.get('validation', '')}
        
        Create a professional research report with:
        
        1. Executive Summary (2-3 paragraphs)
        2. Research Methodology 
        3. Key Findings (organized by theme)
        4. Analysis and Insights
        5. Conclusions and Recommendations
        6. Confidence Assessment
        7. Areas for Further Research
        
        The report should be:
        - Comprehensive yet accessible
        - Well-structured with clear headings
        - Focused on answering the original query
        - Actionable where appropriate
        - Honest about limitations and uncertainties
        
        Use professional formatting and clear language.
        """,
        agent=agent,
        expected_output="Complete professional research report with all sections"
    )

def execute_research_crew(query: str) -> Dict[str, Any]:
    """Execute the complete CrewAI research workflow"""
    
    # Create the research team
    team = create_research_team()
    
    # Phase 1: Planning
    planning_task = create_planning_task(query, team["planner"])
    planning_crew = Crew(
        agents=[team["planner"]],
        tasks=[planning_task],
        process=Process.sequential,
        verbose=True
    )
    
    planning_result = planning_crew.kickoff()
    
    # Phase 2: Investigation  
    investigation_task = create_investigation_task(str(planning_result), query, team["investigator"])
    investigation_crew = Crew(
        agents=[team["investigator"]],
        tasks=[investigation_task], 
        process=Process.sequential,
        verbose=True
    )
    
    investigation_result = investigation_crew.kickoff()
    
    # Phase 3: Analysis
    analysis_task = create_analysis_task(str(investigation_result), query, team["analyst"])
    analysis_crew = Crew(
        agents=[team["analyst"]],
        tasks=[analysis_task],
        process=Process.sequential,
        verbose=True
    )
    
    analysis_result = analysis_crew.kickoff()
    
    # Phase 4: Validation
    validation_task = create_validation_task(str(analysis_result), query, team["validator"])
    validation_crew = Crew(
        agents=[team["validator"]],
        tasks=[validation_task],
        process=Process.sequential,
        verbose=True
    )
    
    validation_result = validation_crew.kickoff()
    
    # Phase 5: Synthesis
    all_results = {
        'planning': str(planning_result),
        'investigation': str(investigation_result),
        'analysis': str(analysis_result),
        'validation': str(validation_result)
    }
    
    synthesis_task = create_synthesis_task(all_results, query, team["synthesizer"])
    synthesis_crew = Crew(
        agents=[team["synthesizer"]],
        tasks=[synthesis_task],
        process=Process.sequential,
        verbose=True
    )
    
    final_report = synthesis_crew.kickoff()
    
    return {
        "query": query,
        "planning": str(planning_result),
        "investigation": str(investigation_result),
        "analysis": str(analysis_result),
        "validation": str(validation_result), 
        "final_report": str(final_report),
        "crew_results": all_results
    }