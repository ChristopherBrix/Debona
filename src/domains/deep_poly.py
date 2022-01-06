"""
DeepPoly uses one independent symbolic equation for lower and upper bounds each.
"""

from src.domains.abstract_domains import AbstractDomain


class DeepPoly(AbstractDomain):
    """
    Subclass of AbstractDomain for the DeepPoly algorithm. For now has no special
    functionality but method implementation may move from AbstractDomain to here, if the
    functionality differs between domains.
    """
