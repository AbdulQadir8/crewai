from crewai import LLM, Agent, Crew, Process, Task
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
import os


# Create a knowledge source
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

content_source = CrewDoclingSource(
    file_paths=[
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination",
    ],
)

# Create an LLM with a temperature of 0 to ensure deterministic outputs
llm = LLM(model="gemini/gemini-1.5-flash",
          api_key=GEMINI_API_KEY,
          temperature=0)

# Create an agent with the knowledge store
agent = Agent(
    role="About papers",
    goal="You know everything about the papers.",
    backstory="""You are a master at understanding papers and their content.""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description="Answer the following questions about the papers: {question}",
    expected_output="An answer to the question.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential,
    knowledge_sources=[
        content_source
    ],  # Enable knowledge by adding the sources here. You can also add more sources to the sources list.
    embedder={
        "provider":"google",
        "config": {
            "model": "models/text-embedding-004",
            "api_key": GEMINI_API_KEY
        }
    }

)
def docling_knowledge():
    result = crew.kickoff(
        inputs={
            "question": "What is the reward hacking paper about? Be sure to provide sources."
        }
    )