# Prediction law for fractional Brownian motion

Numerical implementation of the prediction law in Sottinen and Viitasaari (2017).

For a fractional Brownian motion observed on `[0,u]`, the conditional future is Gaussian with

\[
\hat m_t^H(u)=B_u^H-\int_0^u \Psi_H(t,s\mid u)\,dB_s^H
\]

and

\[
\hat r_H(t,s\mid u)=\int_u^{t\wedge s} k_H(t,v)k_H(s,v)\,dv.
\]

The covariance is evaluated from the formula above. The Wiener integral in the mean is approximated with a discrete past path using stepfunctions.

Based on: T. Sottinen and L. Viitasaari, *Prediction law of fractional Brownian motion*, Statistics & Probability Letters 129 (2017), 155-166.

## Example

```python
import numpy as np
from core import r, conditional_law

H = 0.8
u = 1.0
rng = np.random.default_rng(1)

# Simulate an fBM past on a grid.
past_times = np.linspace(0.0, u, 101)
K = np.array([[r(H, t, s) for s in past_times[1:]]
              for t in past_times[1:]])
past_values = np.r_[0.0, rng.multivariate_normal(np.zeros(100), K)]

# Conditional law of future values.
future_times = np.linspace(u, 2.0, 11)[1:]
mean, cov = conditional_law(H, future_times, past_times, past_values)

samples = rng.multivariate_normal(mean, cov, size=5)
print(mean)
print(samples.shape)
```

Requires `numpy` and `scipy`.
