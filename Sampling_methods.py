import numpy as np
import numpy.linalg as la
import scipy.optimize
from scipy.stats import rv_discrete, multivariate_normal


def gm_sample(num_samples, d, mean, cov, phi, sig_kernel, random_startingpoints=False, num_startingpoints=None):
    # generate random samples from a Gaussian density via direkt sampling
    M = phi.shape[0]
    samples = np.zeros((num_samples, d))
    phi_sample = rv_discrete(values=(np.arange(M), phi)).rvs(size=num_samples)

    counter = 0
    for i in range(M):
        phi_i = np.sum(phi_sample == i)
        samples[counter:counter + phi_i] = multivariate_normal(mean=mean[i], cov=cov[i]).rvs(size=phi_i)
        counter += phi_i
    return samples


def herding_weight(x, d, mean, cov_inv, phi, T, sig_kernel, samples):
    # herding weight function for kernel herding

    # compute expectation / convolution
    if len(mean.shape) == 1:  # Gaussian
        exp = (2 * np.pi) ** (-d / 2) * la.det(cov_inv) ** 0.5
        exp *= np.exp(-0.5 * (x - mean).T @ cov_inv @ (x - mean))

    else:  # Gaussian Mixture
        exp = phi * (2 * np.pi) ** (-d / 2) * la.det(cov_inv) ** 0.5
        exp *= np.squeeze(np.exp(-0.5 * (x - mean[:, None, :]) @ cov_inv @ (x[:, None] - mean[:, :, None])),
                          axis=(1, 2))
        exp = np.sum(exp)

    # kernel density estimate
    mahalanobis = 1 / sig_kernel ** 2 * np.squeeze((x - samples)[:, None, :] @ (x - samples)[:, :, None], axis=-1)
    kernels = 1 / (T + 1) * (np.sqrt(2 * np.pi) * sig_kernel) ** (-d) * np.sum(np.exp(-0.5 * mahalanobis))

    fun = - (exp - kernels)  # maximization
    return fun


def kernel_herding(num_samples, d, mean, cov, phi, sig_kernel, random_startingpoints=False, num_startingpoints=None):
    """
        Compute deterministic samples from a Gaussian Mixture using kernel herding
        :param num_samples: number of samples
        :param d: dimension
        :param mean: array of means of GM
        :param cov: array of covariances of GM
        :param phi: array of weights of GM
        :param sig_kernel: standard deviation of Gaussian kernel
        :param random_startingpoints: bool, use random starting points for optimization
        :param num_startingpoints: if using random starting points, number of points
        :return: computed samples
    """

    M = 1 if type(phi) == int else phi.shape[0]
    samples_herding = np.zeros((num_samples, d))

    # precomptute covariance inverse
    cov_inv = la.inv(sig_kernel ** 2 * np.eye(d) + cov)

    for i in range(num_samples):
        # sample
        # compute starting points and optimize
        if random_startingpoints:
            fun_val = np.zeros(num_startingpoints)
            x_val = np.zeros((num_startingpoints, d))
            mean_min = mean.min(axis=0) - 2
            mean_max = mean.max(axis=0) + 2
            startingpoints = np.random.random_sample((num_startingpoints, d)) * (mean_max - mean_min) + mean_min

            for j, x0 in enumerate(startingpoints):
                s = scipy.optimize.minimize(herding_weight, x0=x0,
                                            args=(d, mean, cov_inv, phi, i + 1, sig_kernel, samples_herding[:i]),
                                            method='BFGS')
                fun_val[j] = s.fun
                x_val[j] = s.x
        else:
            fun_val = np.zeros(2 * d * M)
            x_val = np.zeros((2 * d * M, d))
            for m in range(M):
                for j in range(d):
                    x0 = cov[j] + mean if M == 1 else cov[m, j] + mean[m]
                    s = scipy.optimize.minimize(herding_weight, x0=x0,
                                                args=(d, mean, cov_inv, phi, i + 1, sig_kernel, samples_herding[:i]),
                                                method='BFGS')
                    fun_val[m * d * 2 + 2 * j] = s.fun
                    x_val[m * d * 2 + 2 * j] = s.x
                    x0 = -cov[j] + mean if M == 1 else -cov[m, j] + mean[m]
                    s = scipy.optimize.minimize(herding_weight, x0=x0,
                                                args=(d, mean, cov_inv, phi, i + 1, sig_kernel, samples_herding[:i]),
                                                method='BFGS')
                    fun_val[m * d * 2 + 2 * j + 1] = s.fun
                    x_val[m * d * 2 + 2 * j + 1] = s.x

        samples_herding[i][:] = x_val[np.argmin(fun_val)]
    return samples_herding


def herded_gibbs_weight(x, d, curr_dim, curr_sample, mean_cond, cov_cond, phi, sig_kernel, samples):
    # Herding weight function for continuous herded Gibbs

    idx = np.arange(d) != curr_dim
    T = samples.shape[0]  # number of samples so wfat

    # compute expectation / convolution
    if type(phi) == int:  # Gaussian
        exp = (2 * np.pi * (cov_cond + sig_kernel ** 2)) ** (-0.5)
        exp *= np.exp(-0.5 * (x - mean_cond) ** 2 / (cov_cond + sig_kernel ** 2))

    else:  # Gaussian Mixture
        exp = (2 * np.pi * (cov_cond + sig_kernel ** 2)) ** (-0.5)
        exp *= np.exp(-0.5 * (x - mean_cond) ** 2 / (cov_cond + sig_kernel ** 2))
        exp = np.sum(phi * exp)

    # kernel densitz estimate
    mahalanobis_weights = 1 / sig_kernel ** 2 * np.squeeze(
        (curr_sample[idx] - samples[:, idx])[:, None, :] @ (curr_sample[idx] - samples[:, idx])[:, :, None],
        axis=(1, 2))
    weights = 1 / ((2 * np.pi) ** 0.5 * sig_kernel) ** (d - 1) * np.exp(- 0.5 * mahalanobis_weights)

    const = np.sum(weights)

    mahalanobis = 1 / sig_kernel ** 2 * (x - samples[:, curr_dim]) ** 2
    kernels = 1 / (np.sqrt(2 * np.pi) * sig_kernel) * np.exp(-0.5 * mahalanobis)

    if const == 0.0:
        fun = (T + 1) / T * exp
    else:
        fun = (T + 1) / T * exp - np.sum(weights * kernels) / const
    return -fun  # maximization


def cont_herded_gibbs(num_samples, d, mean, cov, phi, sig_kernel, random_startingpoints=False, num_startingpoints=None):
    """
    Compute deterministic samples from a Gaussian Mixture using continuous herded gibbs sampling
    :param num_samples: number of samples
    :param d: dimension
    :param mean: array of means of GM
    :param cov: array of covariances of GM
    :param phi: array of weights of GM
    :param sig_kernel: standard deviation of Gaussian kernel
    :param random_startingpoints: bool, use random starting points for optimization
    :param num_startingpoints: if using random starting points, number of points
    :return: computed samples
    """

    samples = np.zeros((num_samples, d))

    d_range = np.arange(d)
    # compute parameters for cond probabilities, naming see Wikipedia
    cond_prob_dtype = np.dtype([
        ("mu1", "f8"),
        ("mu2", "f8", d - 1),
        ("sig11", "f8"),
        ("sig12", "f8", (1, d - 1)),
        ("sig21", "f8", (d - 1, 1)),
        ("sig22", "f8", (d - 1, d - 1)),
        ("sig22i", "f8", (d - 1, d - 1)),
    ])
    if type(phi) == int:  # Gaussian
        cond_prob_param = np.zeros(d, dtype=cond_prob_dtype, )
        for i in range(d):
            idx = d_range != i
            cond_prob_param[i]["mu1"] = mean[i]
            cond_prob_param[i]["mu2"] = mean[idx]
            cond_prob_param[i]["sig11"] = cov[i, i]
            cond_prob_param[i]["sig12"] = cov[i, idx]
            cond_prob_param[i]["sig21"] = cov[idx, i, None]  # keep dims
            cond_prob_param[i]["sig22"] = cov[idx][:, idx]
            cond_prob_param[i]["sig22i"] = la.inv(cov[idx][:, idx])
    else:  # Gaussian Mixture
        M = phi.shape[0]
        cond_prob_param = np.zeros((d, M), dtype=cond_prob_dtype)
        for i in range(d):
            idx = d_range != i
            for m in range(M):
                cond_prob_param[i, m]["mu1"] = mean[m, i]
                cond_prob_param[i, m]["mu2"] = mean[m, idx]
                cond_prob_param[i, m]["sig11"] = cov[m, i, i]
                cond_prob_param[i, m]["sig12"] = cov[m, i, idx]
                cond_prob_param[i, m]["sig21"] = cov[m, idx][:, i, None]
                cond_prob_param[i, m]["sig22"] = cov[m, idx][:, idx]
                cond_prob_param[i, m]["sig22i"] = la.inv(cov[m, idx][:, idx])

    # initialize first sample
    if type(phi) == int:  # Gaussian
        samples[0] = mean[:]  # first sample
        sample_curr = mean.copy()

    else:  # Gaussian mixture - first sample as first kernel herding sample
        s = scipy.optimize.minimize(herding_weight, x0=mean[np.argmax(phi)], args=(
            d, mean, la.inv(sig_kernel ** 2 * np.eye(d) + cov), phi, 1, sig_kernel, samples[:0]), method='BFGS')
        samples[0] = s.x
        sample_curr = samples[0].copy()

    for n in range(1, num_samples):  # first sample already taken
        for i in range(d):
            # compute conditional probabilities
            if type(phi) == int:  # Gaussian
                mean_cond = cond_prob_param[i]["mu1"] + cond_prob_param[i]["sig12"] @ cond_prob_param[i]["sig22i"] @ (
                        sample_curr[d_range != i] -
                        cond_prob_param[i]["mu2"]
                )
                mean_cond = np.squeeze(mean_cond)
                cov_cond = cond_prob_param[i]["sig11"] - cond_prob_param[i]["sig12"] @ cond_prob_param[i]["sig22i"] @ \
                           cond_prob_param[i]["sig21"]
                cov_cond = np.squeeze(cov_cond)
                phi_cond = phi
            else:  # Gaussian Mixture
                mean_cond = cond_prob_param[i]["mu1"] + np.squeeze(
                    cond_prob_param[i]["sig12"] @ cond_prob_param[i]["sig22i"] @ (
                            sample_curr[d_range != i] -
                            cond_prob_param[i]["mu2"]
                    ).reshape((-1, d - 1, 1)), axis=(1, 2))
                cov_cond = cond_prob_param[i]["sig11"] - np.squeeze(
                    cond_prob_param[i]["sig12"] @ cond_prob_param[i]["sig22i"] @ \
                    cond_prob_param[i]["sig21"], axis=(1, 2))
                x2 = sample_curr[np.arange(d) != i]
                phi_cond = phi * (2 * np.pi) ** (-(d - 1) / 2) * la.det(cond_prob_param[i]['sig22']) ** (
                    -0.5) * np.exp(-0.5 * np.squeeze(
                    (x2 - cond_prob_param[i]['mu2']).reshape((-1, 1, d - 1)) @ cond_prob_param[i]['sig22i'] @ (
                            x2 - cond_prob_param[i]['mu2']).reshape((-1, d - 1, 1)), axis=(1, 2)))
                phi_cond /= np.sum(phi_cond)

            if random_startingpoints:
                mean_min = mean_cond.min() - 2
                mean_max = mean_cond.max() + 2
                startpoints = np.random.random_sample(num_startingpoints) * (mean_max - mean_min) + mean_min
            else:  # heuristic starting points
                startpoints = np.array(
                    [mean_cond - cov_cond, mean_cond + cov_cond, mean_cond, mean_cond - 2 * cov_cond,
                     mean_cond + 2 * cov_cond]).flatten()

            # optimize
            x_vals = np.zeros(startpoints.shape)
            fun_vals = np.zeros(startpoints.shape)
            for j, x0 in enumerate(startpoints):
                s = scipy.optimize.minimize(herded_gibbs_weight, x0=x0, args=(
                    d, i, sample_curr, mean_cond, cov_cond, phi_cond, sig_kernel, samples[:n]))
                x_vals[j] = s.x
                fun_vals[j] = s.fun
            sample_curr[i] = x_vals[np.argmin(fun_vals)]

        # save sample after whole sweep
        samples[n][:] = sample_curr[:]
    return samples
