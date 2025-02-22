from crewai import Agent, Task, Crew
from crewai.project import CrewBase, agent, task, crew
from crewai.process import Process


@CrewBase
class TeachingCrew:
    #1. Agent
    # agent_config = "config/agents.yaml"
    # task_config = "config/tasks.yaml"

    def sirzia(self) ->Agent:
        topic = "Software Engineering"  # Define a default topic or make it a parameter
        return Agent(
            role="Sir Zia",
            goal=f"You are a teacher who is teaching a class about {topic}.",
            backstory=f"You have SWE in a class. You will be today teaching {topic}.",
            llm="gemini/gemini-1.5-flash",
            verbose=True
        )
    
    #2. task
    def describe_topic_task(self) -> Task:
        return Task(
            description="We are mentoring to create the best Agentic AI Engineers.",
            expected_output="The sutends will have mastered the topic",
            agent=self.sirzia(),
            verbose=True
        )
    
    def crew(self) -> Crew:
        return Crew(
            agents=[self.sirzia()],
            tasks=[self.describe_topic_task()],  # Fixed method name
            process=Process.sequential,
            verbose=True,
        )
