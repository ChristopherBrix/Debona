from src.propagation.abstract_domain_propagation import AbstractDomainPropagation
from src.propagation.deep_poly_propagation import (
    DeepPolyBackwardPropagation,
    DeepPolyForwardPropagation,
)


class BoundPropagation:
    def __init__(self, propagation_method: AbstractDomainPropagation):
        self.propagation_method = propagation_method


class ForwardPropagation(BoundPropagation):
    # TODO: differentiate between DeepPoly and Starsets
    def __init__(self):
        super().__init__(DeepPolyForwardPropagation)


class BackwardPropagation(BoundPropagation):
    # TODO: differentiate between DeepPoly and Starsets
    def __init__(self):
        super().__init__(DeepPolyBackwardPropagation)

    # TODO: Caching, check if bounds need to further propagation or not


bound_propagation = ForwardPropagation().propagation_method
