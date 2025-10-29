# Prediction law for fractional Brownian motion

Finite-dimensional implementation of the prediction law for fractional Brownian motion.

Based on: Tommi Sottinen, Lauri Viitasaari, *Prediction law of fractional Brownian motion*, Statistics & Probability Letters, Volume 129, 2017, Pages 155-166, ISSN 0167-7152, https://doi.org/10.1016/j.spl.2017.05.004

## Usage

```python
import numpy as np
from fbm import FBM
from core import build_conditional_mean_vector, build_conditional_covariance_matrix

# Generate FBM past
H = 0.9
length = 1.0
n = 10
f = FBM(n=n, hurst=H, length=length, method='daviesharte')
past_times = f.times()
past_values = f.fbm()
u = past_times[-1]

# Calculate conditional mean and covariance
T_fut = 2.0
future_times = np.linspace(u, T_fut, n)
mean_vec = build_conditional_mean_vector(H, future_times, u, past_times, past_values)
cov_mat = build_conditional_covariance_matrix(H, future_times, u)

# Sample conditional paths
size = 10
samples = np.random.multivariate_normal(mean_vec, cov_mat, size=size)

# Visualize
import matplotlib.pyplot as plt
delta = length / n
future_times_shifted = future_times + delta
future_times_plot = np.insert(future_times_shifted, 0, past_times[-1])
samples_to_plot = np.insert(samples, 0, past_values[-1], axis=1)

plt.figure(figsize=(10, 6))
plt.plot(past_times, past_values, 'o-', label='Past FBM Path', color='blue')
for i in range(samples_to_plot.shape[0]):
    plt.plot(future_times_plot, samples_to_plot[i], 'o--', label=f'Predicted Path {i+1}', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('FBM Value')
plt.title(f'FBM Prediction (H={H})')
plt.legend()
plt.grid()
plt.show()
```

## Files

- `core.py` - Core implementation

## Dependencies

- `numpy`