from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import PDFSearchTool
from litellm import completion
from dotenv import find_dotenv, load_dotenv
import os
_: bool = load_dotenv(find_dotenv())

llm1 = completion(model="gemini/gemini-2.0-flash")

@CrewBase
class LawyerSupportAgentForLegalQueriesCrew():
    """LawyerSupportAgentForLegalQueries crew"""

    @agent
    def legal_query_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_query_analyzer'],
            tools=[],
            llm=llm1
        )

    @agent
    def legal_document_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_document_searcher'],
            tools=[WebsiteSearchTool()],
            llm=llm1
        )

    @agent
    def legal_document_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_document_retriever'],
            tools=[ScrapeWebsiteTool(), PDFSearchTool()],
            llm=llm1
        )

    @agent
    def legal_summary_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_summary_generator'],
            tools=[],
            llm=llm1
        )


    @task
    def analyze_legal_query_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_legal_query_task'],
            tools=[],
        )

    @task
    def search_legal_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_legal_documents_task'],
            tools=[WebsiteSearchTool()],
        )

    @task
    def retrieve_legal_documents_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_legal_documents_task'],
            tools=[ScrapeWebsiteTool(), PDFSearchTool()],
        )

    @task
    def generate_legal_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_legal_summary_task'],
            tools=[],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the LawyerSupportAgentForLegalQueries crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
