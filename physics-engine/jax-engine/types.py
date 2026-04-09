"""
Core data structures and configurations for the JAX physics simulation.
These are used to configure the compiled graph at trace time and handle trajectory outputs.
"""
from typing import Any, Dict, NamedTuple
import jax.numpy as jnp

# Type Aliases
State = Dict[str, jnp.ndarray]
Parameters = Dict[str, Any]

class SimulatorConfiguration(NamedTuple):
    """
    Configuration parameters for simulator. These are static and not traced by JAX.
    These values are used at trace time only to configure the compiled graph.

    Attributes:
        dt: timestep size
        n_steps: number of steps to simulate
        record_every: how often to record the state
        integrator: key into INTEGRATOR_REGISTRY (e.g. 'velocity_verlet', 'rk4')
        max_val: threshold for when it becomes invalid
        bounds_keys: which state keys to check for validity
        derivatives: routing map defining derivative relationships between state variables
        q_key: position-like state variable key, only for symplectic integrators
        p_key: momentum-like state variable key, only for symplectic integrators
    """
    dt: float
    num_steps: int
    record_interval: int
    integrator: str 
    max_val: float
    bounds_keys: tuple
    derivatives: dict
    q_key: str
    p_key: str

class TrajectoryResult(NamedTuple):
    """
    Output of a single (or batched) rollout.
    n_recorded = num_steps // record_interval
    Under vmap the leading BATCH dimension is prepended automatically.

    Attributes:
        states: the recorded trajectory, each value has (n_recorded, num of particles, dimensions)
        d_states: how fast things are changing at each recorded moment
        observables: extra things to track for debugging
        valid: shape (n_recorded,) bool mask for whether the trajectory is valid at each recorded step
    """
    states: Dict[str, jnp.ndarray]
    d_states: Dict[str, jnp.ndarray]
    observables: Dict[str, jnp.ndarray]
    valid: jnp.ndarray