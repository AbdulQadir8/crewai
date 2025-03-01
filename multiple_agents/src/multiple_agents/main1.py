from crewai.flow.flow  import Flow, start
from multiple_agents.crews.dev_crew.dev_crew import DevCrew


class DevFlow(Flow):
    @start()
    def run_dev_crew(self):
        output = DevCrew().crew().kickoff(
            inputs={"problem":"Write python code for addition of two numbers"}
        )
        return output.raw
    
def kickoff():
    flow = DevFlow()
    output = flow.kickoff()
    # print(output)