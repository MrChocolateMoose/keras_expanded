from abc import ABC, abstractmethod
import os.path

class BaseExperiment(ABC):

    base_dir = None
    data_dir = None
    name = None
    named_partitions = None

    def __init__(self, name, base_dir, data_dir, named_partitions):
        self.name = name
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.named_partitions = named_partitions

    def get_base_dir(self):
        return self.base_dir

    def get_experiment_dir(self):
        return os.path.join(self.base_dir, self.name)

    def get_experiment_data_dir(self):
        return os.path.join(self.base_dir, self.name, self.data_dir)

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_model(self, input_shape):
        pass

