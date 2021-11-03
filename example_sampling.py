import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Sampling_methods import kernel_herding, cont_herded_gibbs, gm_sample
from scipy.stats import multivariate_normal
import tikzplotlib

matplotlib.use('Qt5Agg')

plt.ion()

d = 2
M = 5
mean = np.array([
    [6, 5],
    [7, 2],
    [1, 3],
    [7, 6],
    [2, 8],
], dtype=np.float64)
cov = np.array([
    [[5, 3], [3, 5]],
    [[1, 0.3], [0.3, 1]],
    [[4, 3], [3, 4]],
    [[2, 0.5], [0.5, 3]],
    [[1, 0], [0, 3]],
], dtype=np.float64)
phi = np.array([0.3, 0.1, 0.2, 0.3, 0.1])

num_samples = 20
sig_kernel = 0.5

samples_herding = kernel_herding(num_samples, d, mean, cov, phi, sig_kernel)
samples_hg = cont_herded_gibbs(num_samples, d, mean, cov, phi, sig_kernel)
samples_rnd = gm_sample(num_samples, d, mean, cov, phi, sig_kernel)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(10,4))
ax1.plot(samples_herding[:, 0], samples_herding[:, 1], '.', markersize=3)
ax2.plot(samples_hg[:, 0], samples_hg[:, 1], '.', markersize=3)
ax3.plot(samples_rnd[:, 0], samples_rnd[:, 1], '.', markersize=3)

samples = np.concatenate((samples_herding, samples_hg, samples_rnd, np.array([[-1, 0], [10, 10]])), axis=0)

for ax in (ax1, ax2, ax3):
    x = np.linspace(samples.min(axis=0) - 1, samples.max(axis=0) + 1, 200)
    grid = np.array(np.meshgrid(*x.T)).T
    z = np.zeros(grid.shape[:-1])
    for m in range(phi.shape[0]):
        z += phi[m] * multivariate_normal(mean=mean[m, :2], cov=cov[m][:2, :2]).pdf(grid)
    ax.contour(x[:, 0], x[:, 1], z.T, 10)

ax1.set_title('kernel herding')
ax2.set_title('continuous herded Gibbs sampling')
ax3.set_title('random sampling')

plt.show()




