"""
The BoundPropagation hides the concrete domain and propagation from the other classes.
"""

from src.domains.abstract_domains import AbstractDomain
from src.domains.deep_poly import DeepPoly
from src.propagation.abstract_domain_propagation import AbstractDomainPropagation
from src.propagation.deep_poly_propagation import DeepPolyForwardPropagation


class BoundPropagation:

    """
    This class has the propagation method (direction) and the domain needed for the
    verification step. A BoundPropagation object is then passed to the worker, solver,
    objective, etc. classes, without them needing to know, which domain and propagation
    method is used.
    """

    def __init__(
        self, domain: AbstractDomain, propagation_method: AbstractDomainPropagation
    ):
        self._domain = domain
        self._propagation_method = propagation_method

        """
        Args:
            domain                          : Domain used in the verification pipeline.
                                              Manages relevant data for the verification
                                              , like bounds, etc.
            propagtion_method               : Has the implementation of the propagation
                                              in the verification step.
        """

    @property
    def propagation_method(self):
        return self._propagation_method

    @property
    def domain(self):
        return self._domain


class ForwardPropagation(BoundPropagation):
    """
    Subclass of BoundPropagation, specifically for forward propagation methods.
    """

    # TODO: In init, check if given propagation method is a forward propagation method


class BackwardPropagation(BoundPropagation):
    """
    Subclass of BoundPropagation, specifically for backward propagation methods.
    """

    # TODO: In init, check if given propagation method is a backward propagation method

    # TODO: Caching, check if bounds need to further propagation or not
    def cache(self):
        pass


bound_propagation = ForwardPropagation(DeepPoly, DeepPolyForwardPropagation)
