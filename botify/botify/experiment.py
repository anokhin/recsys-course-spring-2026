from enum import Enum
from typing import List

import mmh3


class Treatment(Enum):
    C = 0
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4
    T5 = 5
    T6 = 6
    T7 = 7
    T8 = 8
    T9 = 9


class Split(Enum):
    HALF_HALF = 2
    THREE_WAY = 3
    FOUR_WAY = 4
    FIVE_WAY = 5
    SEVEN_WAY = 7
    EIGHT_WAY = 8
    NINE_WAY = 9


class Experiment:
    """Assigns a user to one of the treatment buckets of a single A/B test.

    Example::

        experiment = Experiments.HW2
        if experiment.assign(user) == Treatment.C:
            ...
        elif experiment.assign(user) == Treatment.T1:
            ...
    """

    def __init__(self, name: str, split: Split):
        self.name = name
        self.split = split
        self.hash = mmh3.hash(self.name)

    def assign(self, user: int) -> Treatment:
        user_hash = mmh3.hash(str(user), self.hash, False)
        return Treatment(user_hash % self.split.value)

    def __repr__(self):
        return f"{self.name}:{self.split}"


class Experiments:
    """A static container for all the existing experiments."""

    STICKY_ARTIST = Experiment("STICKY_ARTIST", Split.HALF_HALF)
    AA = Experiment("AA", Split.HALF_HALF)
    I2I = Experiment("I2I", Split.THREE_WAY)
    HSTU = Experiment("HSTU", Split.HALF_HALF)
    HW2 = Experiment("HW2", Split.HALF_HALF)

    def __init__(self) -> None:
        self.experiments: List[Experiment] = [Experiments.HW2]
