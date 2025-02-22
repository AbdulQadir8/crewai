from crewai.flow.flow import Flow, start, listen
from dotenv import load_dotenv, find_dotenv
import litellm
from pana_flow.crews.teaching_crew.teaching_crew import TeachingCrew
_: bool = load_dotenv(find_dotenv())




class PanaFlow(Flow):
    @start()
    def generate_topic(self) -> str:
        "Generate a topic for a blog post"
        completion = litellm.completion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user",
                      "content": "Generate a topic for a blog post"}],
            )
        self.state["topic"] = completion["choices"][0]["message"]["content"]
        return self.state["topic"]
    
    @listen("generate_topic")
    def generate_content(self) -> str:
        print("Step 2: Generate Content")
        print("In Generate Content\n")
        result = TeachingCrew().crew().kickoff(
            inputs={"topic": self.state["topic"]})
        print(result)


def kickoff():
    flow = PanaFlow()
    result = flow.kickoff()
    print(result)

