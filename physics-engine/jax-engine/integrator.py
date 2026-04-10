"""ODE integrators for physics engine

Currently only three integrators:
- semi implicit Euler: fastest, symplectic, 1st order so error of dt^2 per step
- velocity Verlet: more expensive, symplectic, 2nd order so error of dt^3 per step
- RK4: most expensive, not symplectic and energy drifts over long runs, 4th order so error of dt^5 per step
"""
import jax
import jax.numpy as jnp
from jax import tree_util
from jax_engine.types import State, Parameters
from typing import Callable

def semi_implicit_euler_step (
    state: State,
    compute_derivative: Callable,
    dt: float, 
    params: Parameters,
    q_key: str = 'q',
    p_key: str = 'p',
    **kwargs
) -> State:
    """
    Symplectic Euler integrator, updates velocity first than uses that updated velocity
    to update position. Keeps simulations stable
    """
    derivative_state = compute_derivative(state, params)
    acceleration = derivative_state[p_key]

    p_new = state[p_key] + dt * acceleration
    q_new = state[q_key] + dt * p_new

    return {
        **state,
        q_key: q_new,
        p_key: p_new,
    }

def velocity_verlet_step (
    state: State,
    compute_derivative: Callable,
    dt: float, 
    params: Parameters,
    q_key: str = 'q',
    p_key: str = 'p',
    **kwargs
) -> State:
    """
    Velocity Verlet symplectic integrator. Averages forces at the start and end of every
    step for more accuracy.
    """
    q = state[q_key]
    p = state[p_key]

    derivative_state = compute_derivative(state, params)
    acceleration = derivative_state[p_key]

    p_half = p + 0.5 * dt * acceleration
    q_new = q + dt * p_half

    state_half = {**state, q_key: q_new}
    derivative_state_half = compute_derivative(state_half, params)
    acceleration_half = derivative_state_half[p_key]
    
    p_new = p_half + 0.5 * dt * acceleration_half
    return {
        **state,
        q_key: q_new,
        p_key: p_new,
    }

def rk4_step(
    state: State,
    compute_derivative: Callable,
    dt: float, 
    params: Parameters,
    **kwargs
) -> State:
    """
    Runge-Kutta 4th order integrator step, treats the system as a generic first order ODE
    no specific keys are referenced so it works for any state

    not symplectic so energy will drift over long rollouts in Hamiltonian systems
    """

    def advance(s, k, step_size):
        return tree_map(lambda x, dx: x + step_size * dx, s, k)

    k1 = compute_derivative(state, params)
    k2 = compute_derivative(advance(state, k1, 0.5 * dt), params)
    k3 = compute_derivative(advance(state, k2, 0.5 * dt), params)
    k4 = compute_derivative(advance(state, k3, dt), params)

    return tree_map(
        # 1:2:2:1 weighting derived from Simpson's rule for numerical integration
        lambda s, a, b, c, d: s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        state, k1, k2, k3, k4,
    )

INTEGRATOR_REGISTRY = {
    'semi_implicit_euler': semi_implicit_euler_step,
    'velocity_verlet': velocity_verlet_step,
    'rk4': rk4_step,
}