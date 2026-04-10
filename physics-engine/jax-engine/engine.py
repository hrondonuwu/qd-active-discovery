"""
ODE engine
_build_derivative_function - reads the derivatives routing map, returns compute_derivatives(state, params) -> derivative_state
_build_observable_function - reads obs_keys, returns compute_observables(state, params) -> dict of values
_is_state_valid - returns True if state is within bounds
single_rollout - runs one trajectory
batched_rollout - runs B trajectories in parallel via vmap
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.tree_util import tree_map

from jax_engine.types import SimulatorConfiguration, TrajectoryResult
from jax_engine.forces import FORCE_REGISTRY
from jax_engine.integrators import INTEGRATOR_REGISTRY
from jax_engine.observables import OBSERVABLE_REGISTRY


def _build_derivative_function(derivative_map: dict):
    """
    reads the derivatives routing map and return a function

    the routing map has two types of entries:
        q: p            (string)  ->  derivative_state['q'] = state['p']
        p: [gravity]    (list)    ->  derivative_state['p'] = sum of force functions

    derivative_function takes the current state and params and returns the instantaneous rates of change
    """

    force_functions = {}
    for state_key, rhs_config in derivative_map.items():
        if isinstance(rhs_config, list):
            for force_name in rhs_config:
                if force_name not in FORCE_REGISTRY:
                    raise ValueError(
                        f"Unknown force '{force_name}'. "
                        f"Available: {list(FORCE_REGISTRY.keys())}"
                    )
                force_functions[force_name] = FORCE_REGISTRY[force_name]
    
    def compute_derivatives(state: dict, params: dict) -> dict:
        derivative_state = {}
        for k, v in state.items():
            derivative_state[k] = jnp.zeros_like(v)

        for state_key, rhs_config in derivative_map.items():
            if isinstance(rhs_config, str):
                derivative_state[state_key] = state[rhs_config]
            else:
                total = jnp.zeros_like(state[state_key])
                for force_name in rhs_config:
                    total = total + force_functions[force_name](state, params)
                derivative_state[state_key] = total

        return derivative_state
    return compute_derivatives

def _build_observable_function(observable_keys: tuple):
    """
    reads the observable keys returns a function:
        compute_observables(state, params) -> Dict[str, jnp.ndarray]
    that returns the current values of the requested variables
    """

    observable_functions = []
    for key in observable_keys:
        if key not in OBSERVABLE_REGISTRY:
            raise ValueError(
                f"Unknown observable '{key}'."
            )
        observable_functions.append((key, OBSERVABLE_REGISTRY[key]))
    observable_functions = tuple(observable_functions)
    
    def compute_observables(state: dict, params: dict) -> dict:
        if len(observable_functions) == 0:
            return {}
        
        result = {}
        for key, fn in observable_functions:
            result[key] = fn(state, params)
        return result
    
    return compute_observables

def _is_state_valid(state: dict, cfg: SimulatorConfiguration) -> jnp.ndarray:
    """
    True  = state is finite and within bounds
    False = state has exploded or gone NaN
    """
    is_valid = jnp.bool_(True)
    for key in cfg.validation_keys:
        state_values = state[key]

        is_valid = is_valid & jnp.all(jnp.isfinite(state_values))
        is_valid = is_valid & (jnp.max(jnp.abs(state_values)) <= cfg.divergence_threshold)
    return is_valid

def single_rollout(
    initial_state: dict,
    params: dict,
    cfg: SimulatorConfiguration,
    observable_keys: tuple = (),
) -> TrajectoryResult:
    """
    Runs one simulation trajectory.
    Only (n_recorded) steps are stored, intermediate steps are discarded
    """
    compute_derivatives = _build_derivative_function(cfg.derivative_map)
    compute_observables = _build_observable_function(observable_keys)
    integrator = INTEGRATOR_REGISTRY[cfg.integrator]
    n_recorded = cfg.num_steps // cfg.record_interval

    def _inner_step(i, carry):
        """
        Advances physics by one step using the configured integrator
        Runs record_interval times per outer step.
        """

        state, is_valid = carry
        
        candidate = integrator(
            state, compute_derivatives, cfg.dt, params,
            q_key=cfg.q_key, p_key=cfg.p_key,
        )
        still_valid = is_valid & _is_state_valid(candidate, cfg)

        # if still valid: accept candidate. If not: freeze at current state.
        frozen_state = tree_map(
            lambda new, old: jnp.where(still_valid, new, old),
            candidate, state,
        )
        return frozen_state, still_valid
    
    def _outer_step(carry, _):
        """
        runs (record_interval) inner steps, then records a snapshot
        """
        state, is_valid = carry

        new_state, new_is_valid = lax.fori_loop(
            0, cfg.record_interval, _inner_step, (state, is_valid)
        )

        recorded_state = new_state
        recorded_derivatives = compute_derivatives(new_state, params)
        observables = compute_observables(new_state, params)

        return (new_state, new_is_valid), (recorded_state, recorded_derivatives, observables, new_is_valid)
    
    initial_carry = (initial_state, jnp.bool_(True))
    _, raw_trajectory = lax.scan(_outer_step, initial_carry, None, length=n_recorded)
    all_states, all_derivative_states, all_observables, all_valid = raw_trajectory

    return TrajectoryResult(
        states=all_states,
        state_derivatives=all_derivative_states,
        observables=all_observables,
        is_valid=all_valid,
    )

# Defined at module level so JAX only compiles this once. If this were created
# inside batched_rollout, JAX would see a new function object on every call and
# recompile from scratch each time.
_vmapped_rollout = jax.jit(jax.vmap(single_rollout, in_axes=(0, 0, None, None)))
def batched_rollout(
    batch_initial_state: dict,
    batch_params: dict,
    cfg: SimulatorConfiguration,
    observable_keys: tuple = (),
) -> TrajectoryResult:
    """
    runs B simulations in parallel using vmap
    """
    return _vmapped_rollout(batch_initial_state, batch_params, cfg, observable_keys)
