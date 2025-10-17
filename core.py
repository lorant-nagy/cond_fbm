import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def r(H, t, s):
    return 0.5 * (t**(2*H) + s**(2*H) - np.abs(t - s)**(2*H))


def d(H):
    numerator = 2 * H * gamma(3/2 - H)
    denominator = gamma(H + 1/2) * gamma(2 - 2*H)
    return np.sqrt(numerator / denominator)


def k(H, t, s):
    if s >= t or s <= 0:
        return 0.0
    
    dH = d(H)
    term1 = (t/s)**(H - 0.5) * (t - s)**(H - 0.5)
    
    def integrand(z):
        return z**(H - 1.5) * (z - s)**(H - 0.5)
    
    integral, _ = quad(integrand, s, t, limit=100)
    term2 = (H - 0.5) * s**(0.5 - H) * integral
    
    return dH * (term1 - term2)


def psi(H, t, s, u):
    if s >= u or s <= 0 or t < u:
        return 0.0
    
    coeff = -np.sin(np.pi * (H - 0.5)) / np.pi
    factor = s**(0.5 - H) * (u - s)**(0.5 - H)
    
    def integrand(z):
        numerator = z**(H - 0.5) * (z - u)**(H - 0.5)
        denominator = z - s
        return numerator / denominator
    
    integral, _ = quad(integrand, u, t, limit=100)
    
    return coeff * factor * integral


def r_hat(H, t, s, u):
    if u > min(t, s):
        raise ValueError(f"Conditioning time u={u} must be ≤ min(t, s)")
    
    base_cov = r(H, t, s)
    
    if u <= 1e-10:
        return base_cov
    
    def integrand(v):
        return k(H, t, v) * k(H, s, v)
    
    eps = max(1e-8, u * 1e-5)
    integral, _ = quad(integrand, eps, u, limit=100, epsabs=1e-8, epsrel=1e-8)
    
    return base_cov - integral


def m_hat(H, t, u, past_times, past_values):

    if t < u:
        raise ValueError(f"Prediction time t={t} must be ≥ conditioning time u={u}")
    
    past_times = np.asarray(past_times)
    past_values = np.asarray(past_values)
    
    if len(past_times) != len(past_values):
        raise ValueError(f"Length mismatch: {len(past_times)} times vs {len(past_values)} values")
    
    if not np.isclose(past_times[-1], u, rtol=1e-6):
        raise ValueError(f"Last observation time {past_times[-1]} must equal u={u}")
    
    if np.isclose(t, u, rtol=1e-6):
        return past_values[-1]
    
    BH_u = past_values[-1]

    integral_sum = 0.0
    
    for i in range(len(past_times) - 1):
        t_i = past_times[i]
        t_next = past_times[i + 1]

        if t_i <= 1e-10:
            continue
        
        s_mid = (t_i + t_next) / 2
        
        psi_val = psi(H, t, s_mid, u)

        dBH = past_values[i + 1] - past_values[i]

        integral_sum += psi_val * dBH

    return BH_u - integral_sum


def build_conditional_mean_vector(H, future_times, u, past_times, past_values):

    future_times = np.asarray(future_times)
    past_times = np.asarray(past_times)
    past_values = np.asarray(past_values)
    
    assert np.all(future_times >= u), f"All future times must be ≥ u={u}"
    assert len(past_times) == len(past_values), "Length mismatch between times and values"
    assert np.isclose(past_times[-1], u, rtol=1e-6), f"Last time {past_times[-1]} must equal u={u}"
    
    n_future = len(future_times)
    mean_vector = np.zeros(n_future)
    
    for i, t in enumerate(future_times):
        mean_vector[i] = m_hat(H, t, u, past_times, past_values)
    
    return mean_vector


def build_conditional_covariance_matrix(H, future_times, u):

    future_times = np.asarray(future_times)
    n_future = len(future_times)
    
    assert np.all(future_times >= u), f"All future times must be ≥ u={u}"
    
    cov_matrix = np.zeros((n_future, n_future))
    
    for i in range(n_future):
        for j in range(i, n_future):
            cov_matrix[i, j] = r_hat(H, future_times[i], future_times[j], u)
            
            if i != j:
                cov_matrix[j, i] = cov_matrix[i, j]
    
    return cov_matrix

