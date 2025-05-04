import os
import openai
from datetime import datetime
from forge.core.agent import Agent

openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=openai.api_key)


class NarratorAgent(Agent):
    def __init__(self):
        super().__init__(
            "Narrator", "Summarize the following logs into a developer update:"
        )

    def run(self):
        logs = self._load_logs()
        prompt = self.prompt_template + "\n" + logs

        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": prompt}]
        )
        summary = response.choices[0].message.content
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_path = f"outputs/summary_{timestamp}.md"

        if summary:
            with open(output_path, "w") as f:
                f.write(summary)
            print(f"[Narrator] Summary written to {output_path}")

    def _load_logs(self):
        logs_dir = "context/logs"
        all_logs = []
        for fname in os.listdir(logs_dir):
            with open(os.path.join(logs_dir, fname), "r") as f:
                all_logs.append(f"## {fname}\n" + f.read())
        return "\n\n".join(all_logs)


def run_narrator():
    agent = NarratorAgent()
    agent.run()
