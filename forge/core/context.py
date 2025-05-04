import json
import os
from typing import Any, Dict


class Context:
    _instance = None
    _file_path = "context/memory.json"

    def __init__(self):
        if os.path.exists(self._file_path):
            with open(self._file_path, "r") as f:
                self.memory: Dict[str, Any] = json.load(f)
        else:
            self.memory = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update(self, key: str, value: Any):
        self.memory[key] = value
        self.save()

    def get(self, key: str, default=None):
        return self.memory.get(key, default)

    def save(self):
        with open(self._file_path, "w") as f:
            json.dump(self.memory, f, indent=2)
