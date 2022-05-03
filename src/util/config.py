"""
Config file
"""

import logging
from typing import Type

from src.algorithm.heuristic import Heuristic
from src.propagation.abstract_domain_propagation import AbstractDomainPropagation
from src.propagation.deep_poly_propagation import DeepPolyForwardPropagation

LOGS_LEVEL = logging.INFO
logging.basicConfig(level=LOGS_LEVEL)

DOMAIN_PROPAGATION: Type[AbstractDomainPropagation] = DeepPolyForwardPropagation

HEURISTIC: Heuristic = Heuristic.DYNAMIC
