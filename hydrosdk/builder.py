from abc import ABC, abstractmethod

from hydrosdk.cluster import Cluster


class AbstractBuilder(ABC):
    @abstractmethod
    def build(self, cluster: Cluster):
        pass
