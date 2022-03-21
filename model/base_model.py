"""Abstract base model for UNet Phase Detectors
   Copied class from https://github.com/The-AI-Summer/Deep-Learning-In-Production"""

from abc import ABC, abstractmethod
from utils.config import Config

class BaseModel(abs):
    def __init__(self, cfg):
        self.config = Config.from_json(cfg)

    @abstractmethod
    def load_data():
        pass

    @abstractmethod
    def build():
        pass

    @abstractmethod
    def train():
        pass

    @abstractmethod
    def evaluate():
        pass