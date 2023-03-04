# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Conjugate gradient backtracking line search optimizer."""

# Adapted from:
# https://github.com/rlworkgroup/garage/blob/6585ac7a5f0adda75a8deda7c5fac5b92cfc0385/src/garage/torch/optimizers/conjugate_gradient_optimizer.py

import logging
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import torch
from torch import Tensor, optim


__all__ = [
    "CGBLS",
]


_LOGGER = logging.getLogger(__name__)


class CGBLS(optim.Optimizer):
    """Conjugate gradient backtracking line search optimizer.

    Perform constrained optimization through backtracking line search.
    The search direction is computed using the conjugate gradient method,
    which gives `x = A^(-1) g`, where `A` is a second order approximation
    of the constraint and `g` the gradient of the loss function.

    References
    ----------
    .. [1] J. Schulman, S. Levine, P. Moritz, M. Jordan, and P. Abbeel.
           "Trust Region Policy Optimization".
           In: ICML. 2015, pp. 1889-1897.
           URL: https://arxiv.org/abs/1502.05477

    """

    # override
    def __init__(
        self,
        params: "Union[Iterable[Tensor], Iterable[Dict[str, Any]]]",
        max_constraint: "float",
        num_cg_iters: "int" = 10,
        max_backtracks: "int" = 15,
        backtrack_ratio: "float" = 0.8,
        hvp_reg_coeff: "float" = 1e-5,
        accept_violation: "bool" = False,
        epsilon: "float" = 1e-8,
    ) -> "None":
        """Initialize the object.

        Parameters
        ----------
        params:
            The parameters to optimize.
        max_constraint:
            The maximum constraint value.
        num_cg_iters:
            The number of conjugate gradient iterations for `A^(-1) g` computation.
        max_backtracks:
            The maximum number of backtracking line search iterations.
        backtrack_ratio:
            The backtrack ratio for backtracking line search.
        hvp_reg_coeff:
            The regularization coefficient for the Hessian-vector product computation.
        accept_violation:
            True to accept the descent step if it violates the line search
            condition after exhausting all backtracking budgets, False otherwise.
        epsilon:
            The term added to the denominators to improve numerical stability.

        Raises
        ------
        ValueError
            If an invalid argument value is given.

        """
        if num_cg_iters < 1 or not float(num_cg_iters).is_integer():
            raise ValueError(
                f"`num_cg_iters` ({num_cg_iters}) must be in the integer interval [1, inf)"
            )
        if max_backtracks < 1 or not float(max_backtracks).is_integer():
            raise ValueError(
                f"`max_backtracks` ({max_backtracks}) must be in the integer interval [1, inf)"
            )
        if backtrack_ratio <= 0.0 or backtrack_ratio >= 1.0:
            raise ValueError(
                f"`backtrack_ratio` ({backtrack_ratio}) must be in the interval (0, 1)"
            )
        if hvp_reg_coeff <= 0.0:
            raise ValueError(
                f"`hvp_reg_coeff` ({hvp_reg_coeff}) must be in the interval (0, inf)"
            )
        if epsilon <= 0.0:
            raise ValueError(f"`epsilon` ({epsilon}) must be in the interval (0, inf)")
        num_cg_iters = int(num_cg_iters)
        max_backtracks = int(max_backtracks)

        defaults = {
            "max_constraint": max_constraint,
            "num_cg_iters": num_cg_iters,
            "max_backtracks": max_backtracks,
            "backtrack_ratio": backtrack_ratio,
            "hvp_reg_coeff": hvp_reg_coeff,
            "accept_violation": accept_violation,
            "epsilon": epsilon,
        }
        super().__init__(params, defaults)

    # override
    @torch.no_grad()
    def step(
        self,
        loss_fn: "Callable[[], Tensor]",
        constraint_fn: "Callable[[], Tensor]",
    ) -> "Tuple[Tensor, Tensor]":
        """Perform a single optimization step (parameter update).

        Parameters
        ----------
        loss_fn:
            The function that computes the loss. It receives
            no arguments and returns the loss value.
        constraint_fn:
            The function that computes the constraint. It receives
            no arguments and returns the constraint value.

        Returns
        -------
            The reevaluated loss value; the reevaluated constraint value.

        """
        loss, constraint = 0.0, 0.0
        for group in self.param_groups:
            # Collect trainable parameters and gradients
            params = []
            grads = []
            for param in group["params"]:
                if param.grad is not None:
                    params.append(param)
                    grads.append(param.grad.flatten())
            flat_loss_grads = torch.cat(grads)

            # Build Hessian-vector product function
            Ax_fn = self._build_hessian_vector_product(
                constraint_fn, params, group["hvp_reg_coeff"]
            )

            # Compute step direction
            step_dir = self._conjugate_gradient(
                Ax_fn, flat_loss_grads, group["num_cg_iters"]
            )

            # Replace NaN with 0
            step_dir[step_dir.ne(step_dir)] = 0.0

            # Compute step size
            step_size = (
                2.0
                * group["max_constraint"]
                * (1.0 / (step_dir.dot(Ax_fn(step_dir)) + group["epsilon"]))
            ).sqrt()

            if step_size.isnan():
                step_size = 1.0

            descent_step = step_size * step_dir

            # Update parameters using backtracking line search
            loss, constraint = self._backtracking_line_search(
                params,
                descent_step,
                loss_fn,
                constraint_fn,
                group["max_constraint"],
                group["max_backtracks"],
                group["backtrack_ratio"],
                group["accept_violation"],
            )

        return loss, constraint

    def _build_hessian_vector_product(
        self,
        fn: "Callable[[], Tensor]",
        params: "List[Tensor]",
        hvp_reg_coeff: "float" = 1e-5,
    ) -> "Callable[[Tensor], Tensor]":
        """Compute the Hessian-vector product using Pearlmutter's algorithm.

        Parameters
        ----------
        fn:
            The function whose Hessian is to be computed.
        params:
            The function learnable parameters.
        hvp_reg_coeff:
            The regularization coefficient for the Hessian-vector product computation.

        Returns
        -------
            The function that computes the Hessian-vector product.
            It receives as an argument the vector to be multiplied with the
            Hessian and returns the corresponding Hessian-vector product.

        References
        ----------
        .. [1] B. A. Pearlmutter.
               "Fast Exact Multiplication by the Hessian".
               In: Neural Computation. 1994, pp. 147-160.
               URL: https://doi.org/10.1162/neco.1994.6.1.147

        """
        param_shapes = [p.shape or (1,) for p in params]
        with torch.enable_grad():
            f = fn()
        f_grads = torch.autograd.grad(f, params, create_graph=True)

        def compute_hessian_vector_product(vector: "Tensor") -> "Tensor":
            """Compute the product of the Hessian of `fn` and `vector`.

            Parameters
            ----------
            vector:
                The vector to be multiplied with the Hessian.

            Returns
            -------
                The product of the Hessian of `fn` and `vector`.

            """
            unflattened_vectors = [
                x.reshape(shape)
                for x, shape in zip(
                    vector.split(
                        list(map(lambda x: torch.Size(x).numel(), param_shapes))
                    ),
                    param_shapes,
                )
            ]
            assert len(f_grads) == len(unflattened_vectors)

            with torch.enable_grad():
                grad_vector_product = torch.stack(
                    [(g * x).sum() for g, x in zip(f_grads, unflattened_vectors)]
                ).sum()
            hvp = list(
                torch.autograd.grad(grad_vector_product, params, retain_graph=True)
            )
            for i, (hx, p) in enumerate(zip(hvp, params)):
                if hx is None:
                    hvp[i] = torch.zeros_like(p)

            flat_output = torch.cat([h.flatten() for h in hvp])
            return flat_output + hvp_reg_coeff * vector

        return compute_hessian_vector_product

    def _conjugate_gradient(
        self,
        Ax_fn: "Callable[[Tensor], Tensor]",
        b: "Tensor",
        num_cg_iters: "int" = 10,
        residual_tol: "float" = 1e-10,
    ) -> "Tensor":
        """Use conjugate gradient method to solve `Ax = b`.

        Parameters
        ----------
        Ax_fn:
            The function that computes the Hessian-vector product.
            It receives as an argument the vector to be multiplied with the
            Hessian and returns the corresponding Hessian-vector product.
        b:
            Right-hand side of the equation to solve.
        num_cg_iters:
            The number of conjugate gradient iterations for `A^(-1) b` computation.
        residual_tol:
            The tolerance for convergence.

        Returns
        -------
            The solution `x_star` for equation `Ax = b`.

        References
        ----------
        .. [1] M.R. Hestenes, and E. Stiefel.
               "Methods of Conjugate Gradients for Solving Linear Systems".
               In: Journal of Research of the National Bureau of Standards. 1952, pp. 409-435.
               URL: http://dx.doi.org/10.6028/jres.049.044

        """
        p, r = b.clone(), b.clone()
        x = torch.zeros_like(b)
        r_dot_r = r.dot(r)

        for _ in range(num_cg_iters):
            z = Ax_fn(p)
            v = r_dot_r / p.dot(z)
            x += v * p
            r -= v * z
            new_r_dot_r = r.dot(r)
            mu = new_r_dot_r / r_dot_r
            p = r + mu * p

            r_dot_r = new_r_dot_r
            if r_dot_r < residual_tol:
                break
        return x

    def _backtracking_line_search(
        self,
        params: "List[Tensor]",
        descent_step: "Tensor",
        loss_fn: "Callable[[], Tensor]",
        constraint_fn: "Callable[[], Tensor]",
        max_constraint: "float",
        max_backtracks: "int" = 15,
        backtrack_ratio: "float" = 0.8,
        accept_violation: "bool" = False,
    ) -> "Tuple[Tensor, Tensor]":
        """Use backtracking line search to update the parameters.

        Parameters
        ----------
        params:
            The parameters to optimize.
        descent_step:
            The descent step.
        loss_fn:
            The function that computes the loss. It receives
            no arguments and returns the loss value.
        constraint_fn:
            The function that computes the constraint. It receives
            no arguments and returns the constraint value.
        max_constraint:
            The maximum constraint value.
        max_backtracks:
            The maximum number of backtracking line search iterations.
        backtrack_ratio:
            The backtrack ratio.
        accept_violation:
            True to accept the descent step if it violates the line search
            condition after exhausting all backtracking budgets, False otherwise.

        Returns
        -------
            The reevaluated loss value; the reevaluated constraint value.

        References
        ----------
        .. [1] L. Armijo.
               "Minimization of functions having Lipschitz continuous first partial derivatives".
               In: Pacific J. Math. 1966.
               URL: https://msp.org/pjm/1966/16-1/p01.xhtml

        """
        param_shapes = [p.shape or (1,) for p in params]
        descent_steps = [
            x.reshape(shape)
            for x, shape in zip(
                descent_step.split(
                    list(map(lambda x: torch.Size(x).numel(), param_shapes))
                ),
                param_shapes,
            )
        ]
        assert len(descent_steps) == len(params)

        prev_params = [p.clone() for p in params]
        ratios = backtrack_ratio ** torch.arange(max_backtracks)
        loss_before = loss_fn()
        loss, constraint = 0.0, 0.0
        for ratio in ratios:
            for i, step in enumerate(descent_steps):
                params[i] -= ratio * step

            loss = loss_fn()
            constraint = constraint_fn()
            if loss < loss_before and constraint <= max_constraint:
                break

        if not accept_violation:
            warning_msgs = []
            if loss.isnan():
                warning_msgs.append("loss is NaN")
            if constraint.isnan():
                warning_msgs.append("constraint is NaN")
            if loss >= loss_before:
                warning_msgs.append("loss is not improving")
            if constraint >= max_constraint:
                warning_msgs.append("constraint is violated")
            if warning_msgs:
                warning_msg = f"Rejecting the step: line search condition violated because {', '.join(warning_msgs)}"
                warning_msg = " and".join(warning_msg.rsplit(",", 1))
                _LOGGER.warning(warning_msg)
                for i, prev_param in enumerate(prev_params):
                    params[i] = prev_param

        return loss, constraint
