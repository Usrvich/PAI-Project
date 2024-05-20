import math

import numpy
import numpy as np
import scipy
from scipy.stats import laplace, norm, t

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp
from scipy.stats import norm


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    # TODO: enter your code here
    log_p_H1, log_p_H2, log_p_H3 = np.log(0.35), np.log(0.25), np.log(0.40)
    # normal distribution
    log_p_x_H1 = norm.logpdf(x, loc=0, scale=np.sqrt(2))
    # laplace distribution
    log_p_x_H2 = laplace.logpdf(x, loc=0, scale=1)
    # student-t distribution
    log_p_x_H3 = t.logpdf(x, df=4)

    a1 = (- np.sum(x**2) / 4) - len(x) * np.log(2 * np.sqrt(np.pi)) + log_p_H1
    a2 = np.sum(-np.abs(x)) - len(x) * np.log(2) + log_p_H2
    a3 = len(x) * np.log(math.gamma(5/2) / (2 * np.sqrt(np.pi) * math.gamma(2))) - (5/2) * np.sum(np.log(1 + x**2 / 4)) + log_p_H3

    log_p_H1_x = np.sum(log_p_x_H1) + log_p_H1 - logsumexp([a1, a2, a3])
    log_p_H2_x = np.sum(log_p_x_H2) + log_p_H2 - logsumexp([a1, a2, a3])
    log_p_H3_x = np.sum(log_p_x_H3) + log_p_H3 - logsumexp([a1, a2, a3])
    log_p = np.array([log_p_H1_x, log_p_H2_x, log_p_H3_x])
    ### end ###

    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 50 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
