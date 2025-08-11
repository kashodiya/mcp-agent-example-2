import asyncio
import os
from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MCPApp(name="memory_agent_example")

async def memory_example():
    """
    Example demonstrating how mcp-agent automatically handles memory 
    for follow-up questions through AugmentedLLM's built-in conversation history.
    """
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        
        # Create an agent with access to filesystem and fetch servers
        research_agent = Agent(
            name="research_agent",
            instruction="""You are a helpful research assistant that can:
            - Read files from the filesystem
            - Fetch content from URLs
            - Remember previous conversations to answer follow-up questions
            - Provide detailed analysis and summaries
            """,
            server_names=["fetch", "filesystem"],
        )
        
        async with research_agent:
            # Attach an LLM with built-in memory
            llm = await research_agent.attach_llm(OpenAIAugmentedLLM)
            
            logger.info("Starting conversation with memory-enabled agent...")
            
            # First interaction - initial research request
            print("\n=== Initial Question ===")
            result1 = await llm.generate_str(
                message="""Please fetch the content from https://www.anthropic.com/research/building-effective-agents 
                and give me a summary of the key points about agent design patterns."""
            )
            print(f"Agent Response 1: {result1}")
            
            # Follow-up question 1 - references previous context
            print("\n=== Follow-up Question 1 ===")
            result2 = await llm.generate_str(
                message="Which of those patterns would be best for a customer service chatbot?"
            )
            print(f"Agent Response 2: {result2}")
            
            # Follow-up question 2 - builds on conversation history  
            print("\n=== Follow-up Question 2 ===")
            result3 = await llm.generate_str(
                message="Can you give me a specific example of how to implement that pattern?"
            )
            print(f"Agent Response 3: {result3}")
            
            # Follow-up question 3 - references earlier context
            print("\n=== Follow-up Question 3 ===")
            result4 = await llm.generate_str(
                message="Going back to the original article, what did it say about evaluation methods?"
            )
            print(f"Agent Response 4: {result4}")
            
            # Demonstrate memory by asking about conversation history
            print("\n=== Memory Test ===")
            result5 = await llm.generate_str(
                message="What was the first question I asked you in this conversation?"
            )
            print(f"Agent Response 5: {result5}")

async def advanced_memory_example():
    """
    Advanced example showing how to work with memory across different contexts
    and implement custom memory patterns using external memory systems.
    """
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        
        # Agent with memory server for persistent storage
        memory_agent = Agent(
            name="memory_agent", 
            instruction="""You are an agent with long-term memory capabilities.
            You can store important information for later retrieval and 
            maintain context across multiple conversation sessions.""",
            server_names=["memory", "fetch", "filesystem"],  # Includes memory server
        )
        
        async with memory_agent:
            llm = await memory_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Store some information in memory
            print("\n=== Storing Information ===")
            result1 = await llm.generate_str(
                message="""Please remember that my name is John, I'm working on a Python project 
                called 'AIAssistant', and my preferred coding style is to use type hints and docstrings."""
            )
            print(f"Memory Storage: {result1}")
            
            # Later in the conversation - reference stored memories
            print("\n=== Using Stored Memory ===")
            result2 = await llm.generate_str(
                message="What was my name and what project am I working on?"
            )
            print(f"Memory Recall: {result2}")
            
            # Complex follow-up using both short-term and long-term memory
            print("\n=== Complex Follow-up ===")
            result3 = await llm.generate_str(
                message="Given my coding preferences that you remember, can you help me write a function?"
            )
            print(f"Contextual Response: {result3}")

async def multi_turn_workflow_example():
    """
    Example showing how memory works in complex multi-turn workflows
    where the agent needs to maintain context across multiple steps.
    """
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        
        analyst_agent = Agent(
            name="data_analyst",
            instruction="""You are a data analyst that can read files and analyze data.
            Maintain context from previous analyses to answer follow-up questions.""",
            server_names=["filesystem", "fetch"],
        )
        
        async with analyst_agent:
            llm = await analyst_agent.attach_llm(OpenAIAugmentedLLM)
            
            # Step 1: Load and analyze data
            print("\n=== Data Analysis Step 1 ===")
            result1 = await llm.generate_str(
                message="Please read the file 'sales_data.csv' and give me a summary of the key metrics."
            )
            print(f"Analysis 1: {result1}")
            
            # Step 2: Follow-up analysis based on first results
            print("\n=== Follow-up Analysis ===")
            result2 = await llm.generate_str(
                message="Based on those metrics, which product category performed best?"
            )
            print(f"Analysis 2: {result2}")
            
            # Step 3: Deeper dive using conversation context
            print("\n=== Detailed Investigation ===")
            result3 = await llm.generate_str(
                message="Can you analyze the trend for that top-performing category over the last 6 months?"
            )
            print(f"Analysis 3: {result3}")
            
            # Step 4: Strategic recommendation using full context
            print("\n=== Strategic Recommendation ===")
            result4 = await llm.generate_str(
                message="Given everything we've discussed, what's your recommendation for next quarter?"
            )
            print(f"Recommendation: {result4}")


if __name__ == "__main__":
    print("MCP-Agent Memory Examples")
    print("=" * 50)
    
    # print("\nRunning basic memory example...")
    # asyncio.run(memory_example())
    
    print("\nRunning advance memory example...")
    asyncio.run(advanced_memory_example())

    # print("\nRunning multi turn workflow example...")
    # asyncio.run(multi_turn_workflow_example())
    
