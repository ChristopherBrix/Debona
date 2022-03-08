"""
Status of the verification
"""

from enum import Enum


class Status(Enum):
    """
    For keeping track of the verification _status.
    """

    SAFE = 1
    UNSAFE = 2
    UNDECIDED = 3
    UNDERFLOW = 4
