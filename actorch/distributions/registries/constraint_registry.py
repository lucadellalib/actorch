# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Constraint registry."""

from torch.distributions import CatTransform, constraint_registry

from actorch.distributions.constraints import cat


__all__ = []


#####################################################################################################
# Concatenated constraint
#####################################################################################################


@constraint_registry.biject_to.register(cat)
def _biject_to_cat(constraint: "cat") -> "CatTransform":
    return constraint_registry._biject_to_cat(constraint)


@constraint_registry.transform_to.register(cat)
def _transform_to_cat(constraint: "cat") -> "CatTransform":
    return constraint_registry._transform_to_cat(constraint)
