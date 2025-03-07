from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import os

#Get the gemini api key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Create a knowledge source
content = "I am Abdul Qadir. I live in RehmanPura, Pakistan. Currently I am working as a Agentic AI Engineer at Nvidia. I am 24 years old."
string_source = StringKnowledgeSource(
    content=content,
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gemini/gemini-2.0-flash",
          api_key=GEMINI_API_KEY,
          temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About User",
    goal="You know everything about the user.",
    backstory="""You are a master at understanding people and their preferences.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description="Answer the following questions about the user: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[string_source], # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
    embedder={
        "provider":"google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY
        }
    })

def my_knowledge():
    result = crew.kickoff(inputs={"question": "What village does Abdul Qadir live in and how old is he?"})
    print(result)