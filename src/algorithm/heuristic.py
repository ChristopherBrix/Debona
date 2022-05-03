"""
Defines list of possible heuristics.
"""

from enum import Enum


class Heuristic(Enum):
    ARCHITECTURE = 1
    CACHED = 2
    DYNAMIC = 3
