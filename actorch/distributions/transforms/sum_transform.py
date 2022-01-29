# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

"""Sum transform."""

from typing import Any, Tuple

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.constraints import Constraint

from actorch.distributions.transforms.maskable_transform import MaskableTransform
from actorch.registry import register


__all__ = [
    "SumTransform",
]


@register
class SumTransform(MaskableTransform):
    """Sum the input tensor along one of its event dimensions."""

    is_constant_jacobian = False
    """Whether the Jacobian matrix is constant (i.e. the transform is affine)."""

    def __init__(
        self, in_shape: "Tuple[int, ...]", dim: "int" = 0, cache_size: "int" = 0
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        in_shape:
            The input event shape.
        dim:
            The event dimension along which to sum.
        cache_size:
            The cache size. If 0, no caching is done. If 1, the latest
            single value is cached. Only 0 and 1 are supported.

        Raises
        ------
        IndexError
            If `dim` is out of range or not integer.

        """
        if dim < -len(in_shape) or dim >= len(in_shape) or not float(dim).is_integer():
            raise IndexError(
                f"`dim` ({dim}) must be in the integer interval [-{len(in_shape)}, {len(in_shape)})"
            )
        self.in_shape = in_shape
        self.dim = dim = int(dim) % len(in_shape)
        out_shape = list(in_shape)
        del out_shape[dim]
        self._out_shape = tuple(out_shape)
        super().__init__(cache_size)

    # override
    @property
    def domain(self) -> "Constraint":
        return constraints.independent(constraints.real, len(self.in_shape))

    # override
    @property
    def codomain(self) -> "Constraint":
        return constraints.independent(constraints.real, len(self._out_shape))

    # override
    @property
    def transformed_mask(self) -> "Tensor":
        return self.mask.any(dim=self.dim - len(self.in_shape))

    # override
    def with_cache(self, cache_size: "int" = 1) -> "SumTransform":
        if self._cache_size == cache_size:
            return self
        transform = SumTransform(self.in_shape, self.dim, cache_size)
        transform.mask = self.mask
        return transform

    # override
    def _call(self, x: "Tensor") -> "Tensor":
        return x.masked_fill(~self.mask, 0.0).sum(dim=self.dim - len(self.in_shape))

    # override
    def forward_shape(self, shape: "Tuple[int, ...]") -> "Tuple[int, ...]":
        if len(shape) < len(self.in_shape):
            raise ValueError(
                f"The number of input dimensions ({len(shape)}) "
                f"must be at least {len(self.in_shape)}"
            )
        if shape[-len(self.in_shape) :] != self.in_shape:
            raise ValueError(
                f"Input event shape ({shape[-len(self.in_shape):]}) "
                f"must be equal to {self.in_shape}"
            )
        return shape[: -len(self.in_shape)] + self._out_shape

    # override
    def inverse_shape(self, shape: "Tuple[int, ...]") -> "Tuple[int, ...]":
        if len(shape) < len(self._out_shape):
            raise ValueError(
                f"The number of output dimensions ({len(shape)}) "
                f"must be at least {len(self._out_shape)}"
            )
        if shape[len(shape) - len(self._out_shape) :] != self._out_shape:
            raise ValueError(
                f"Output event shape ({shape[len(shape) - len(self._out_shape):]}) "
                f"must be equal to {self._out_shape}"
            )
        return shape[: len(shape) - len(self._out_shape)] + self.in_shape

    def __eq__(self, other: "Any") -> "bool":
        return (
            isinstance(other, SumTransform)
            and (self.in_shape == other.in_shape)
            and (self.dim == other.dim)
            and (self.mask == other.mask).all()
        )

    def __repr__(self) -> "str":
        return (
            f"{self.__class__.__name__}"
            f"(in_shape: {self.in_shape}, "
            f"dim: {self.dim}, "
            f"mask: {self.mask if self.mask.numel() == 1 else self.mask.shape})"
        )
