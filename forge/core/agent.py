from abc import ABC, abstractmethod
from forge.core.context import Context


class Agent(ABC):
    def __init__(self, name: str, prompt_template: str):
        self.name = name
        self.prompt_template = prompt_template
        self.context = Context.get_instance()

    @abstractmethod
    def run(self):
        pass
