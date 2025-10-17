# prediction law for fractional brownian motion

Finite-dimensional implementation of the prediction law for fractional Brownian motion.

Based on: Tommi Sottinen, Lauri Viitasaari, *Prediction law of fractional Brownian motion*, Statistics & Probability Letters, Volume 129, 2017, Pages 155-166, ISSN 0167-7152, https://doi.org/10.1016/j.spl.2017.05.004

## Usage
```python
import numpy as np
from fbm import FBM
from fbm_prediction import build_conditional_mean_vector, build_conditional_covariance_matrix

# Generate FBM past
H = 0.7
f = FBM(n=50, hurst=H, length=1.0, method='daviesharte')
past_times = f.times()
past_values = f.fbm()
u = past_times[-1]

# Predict future
future_times = np.linspace(u, 2.0, 50)
mean_vec = build_conditional_mean_vector(H, future_times, u, past_times, past_values)
cov_mat = build_conditional_covariance_matrix(H, future_times, u)

# Sample conditional paths
samples = np.random.multivariate_normal(mean_vec, cov_mat, size=5)
```

## Files

- `core.py` - Core implementation
- `example.py` - Usage example

## Dependencies

- `numpy`, `scipy`, `fbm`