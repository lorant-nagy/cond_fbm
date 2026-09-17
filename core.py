import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


def _check_h(H):
    if not 0 < H < 1:
        raise ValueError("H must be in (0, 1)")


def r(H, t, s):
    _check_h(H)
    return 0.5 * (t ** (2 * H) + s ** (2 * H) - abs(t - s) ** (2 * H))


def d(H):
    _check_h(H)
    return np.sqrt(
        2 * H * gamma(1.5 - H)
        / (gamma(H + 0.5) * gamma(2 - 2 * H))
    )


def k(H, t, s):
    """Volterra kernel k_H(t,s), equation (2.1) in the paper."""
    _check_h(H)
    if not (0 < s < t):
        return 0.0
    if abs(H - 0.5) < 1e-14:
        return 1.0

    term1 = (t / s) ** (H - 0.5) * (t - s) ** (H - 0.5)

    def integrand(z):
        return z ** (H - 1.5) * (z - s) ** (H - 0.5)

    integral, _ = quad(integrand, s, t, limit=200, epsabs=1e-10, epsrel=1e-10)
    term2 = (H - 0.5) * s ** (0.5 - H) * integral
    return d(H) * (term1 - term2)


def psi(H, t, s, u):
    """Prediction kernel Psi_H(t,s|u), Theorem 3.1."""
    _check_h(H)
    if t < u:
        raise ValueError("t must be >= u")
    if not (0 < s < u) or t == u:
        return 0.0
    if abs(H - 0.5) < 1e-14:
        return 0.0

    coeff = -np.sin(np.pi * (H - 0.5)) / np.pi
    factor = s ** (0.5 - H) * (u - s) ** (0.5 - H)

    def integrand(z):
        return z ** (H - 0.5) * (z - u) ** (H - 0.5) / (z - s)

    integral, _ = quad(integrand, u, t, limit=200, epsabs=1e-10, epsrel=1e-10)
    return coeff * factor * integral


def conditional_mean(H, t, past_times, past_values):
    """
    Numerical approximation of Theorem 3.1's conditional mean.

    The Wiener integral is approximated by a step-function sum
        sum Psi(t, s_i* | u) [B(t_{i+1}) - B(t_i)],
    using interval midpoints s_i*.
    """
    _check_h(H)
    times = np.asarray(past_times, dtype=float)
    values = np.asarray(past_values, dtype=float)

    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        raise ValueError("past_times and past_values must be one-dimensional and have equal length")
    if len(times) < 2:
        raise ValueError("at least two past observations are required")
    if not np.isclose(times[0], 0.0):
        raise ValueError("past_times must start at 0")
    if np.any(np.diff(times) <= 0):
        raise ValueError("past_times must be strictly increasing")

    u = times[-1]
    if t < u:
        raise ValueError("prediction time t must be >= the last past time u")
    if np.isclose(t, u):
        return values[-1]
    if abs(H - 0.5) < 1e-14:
        return values[-1]

    midpoints = 0.5 * (times[:-1] + times[1:])
    increments = np.diff(values)
    weights = np.array([psi(H, t, s, u) for s in midpoints])
    return values[-1] - weights @ increments


def conditional_covariance(H, t, s, u):
    """Conditional covariance from Theorem 3.1, evaluated in its stable form."""
    _check_h(H)
    if u < 0 or t < u or s < u:
        raise ValueError("require 0 <= u <= min(t, s)")

    upper = min(t, s)
    if np.isclose(u, upper):
        return 0.0
    if np.isclose(u, 0.0):
        return r(H, t, s)
    if abs(H - 0.5) < 1e-14:
        return upper - u

    value, _ = quad(
        lambda v: k(H, t, v) * k(H, s, v),
        u,
        upper,
        limit=200,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    return value


def conditional_law(H, future_times, past_times, past_values):
    """Return mean vector and covariance matrix of the discretized future law."""
    future = np.asarray(future_times, dtype=float)
    past_times = np.asarray(past_times, dtype=float)
    past_values = np.asarray(past_values, dtype=float)

    if future.ndim != 1 or len(future) == 0:
        raise ValueError("future_times must be a non-empty one-dimensional array")
    u = past_times[-1]
    if np.any(future < u):
        raise ValueError("all future times must be >= u")

    mean = np.array([conditional_mean(H, t, past_times, past_values) for t in future])

    n = len(future)
    cov = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            value = conditional_covariance(H, future[i], future[j], u)
            cov[i, j] = value
            cov[j, i] = value

    return mean, cov
