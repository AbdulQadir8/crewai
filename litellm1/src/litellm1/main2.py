from crewai.flow import Flow, start, listen
from litellm import completion

class LiteLmmFlow(Flow):

    @start()
    def start_function(self):
        output = completion(
            model="deepseek/deepseek-r1:1.5b",
            messages=[
            {"role":"user",
             "content":"Who is the founder of Pakistan?"}
        ])

        return output["choices"][0]["message"]["content"]
    

def run_litellm_flow():
    obj = LiteLmmFlow()
    result = obj.kickoff()
    print(result)
