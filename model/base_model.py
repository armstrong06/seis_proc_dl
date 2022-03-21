from abc import ABC, abstractmethod

class BaseModel(abs):
    def __init__(self, phase_type, num_channels, num_classes, learning_rate):
        self.phase_type = phase_type
        self.learning_rate = learning_rate
        self.model = self.build(num_classes, num_channels)

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