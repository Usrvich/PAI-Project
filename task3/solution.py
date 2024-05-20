"""Solution."""
import time

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm

# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    WhiteKernel,
)

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # pass
        kernel_f = 1.0 * Matern(
            length_scale=1.0, nu=2.5, length_scale_bounds="fixed"
        ) + WhiteKernel(noise_level=0.15**2, noise_level_bounds="fixed")
        kernel_v = 1.0 * ConstantKernel(
            constant_value=4, constant_value_bounds="fixed"
        ) * (
            DotProduct(sigma_0=0, sigma_0_bounds="fixed")
            + Matern(length_scale=1.0, nu=2.5, length_scale_bounds="fixed")
        ) + WhiteKernel(
            noise_level=0.0001**2, noise_level_bounds="fixed"
        )

        self.gp_f = GaussianProcessRegressor(kernel=kernel_f)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v, normalize_y=True)
        self.X_sample = np.empty((0, 1))
        self.Y_sample_f = np.empty((0, 1))
        self.Y_sample_v = np.empty((0, 1))
        self.next_x = None
        self.cnt = 0
        self.beta = 0
        self.lambdas = 100

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        # raise NotImplementedError
        self.next_x = self.optimize_acquisition_function()
        return np.array([[self.next_x]])

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * np.random.rand(
                DOMAIN.shape[0]
            )
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN, approx_grad=True)

            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # remove nans in x
        x = np.nan_to_num(x)
        # TODO: Implement the acquisition function you want to optimize.
        # raise NotImplementedError
        # Predict mean and standard deviation for objective and constraint
        mu_f, std_f = self.gp_f.predict(x.reshape(-1, 1), return_std=True)
        mu_v, std_v = self.gp_v.predict(x.reshape(-1, 1), return_std=True)

        # Compute the EI under the constraint that v(x) is below the threshold
        with np.errstate(divide="warn"):
            improvement = mu_f - self.Y_sample_f.max()
            # penalize if v is likely to be above the threshold
            # print("======")
            # print("mu_v", mu_v)
            # print("std_v", std_v)
            # print("mu_f", mu_f)
            # time.sleep(0.1)
            # self.cnt += 1
            # print(self.cnt)
            # improvement -= max(mu_v - std_v - SAFETY_THRESHOLD, 0) * mu_f * 10
            # improvement -= max(mu_v - SAFETY_THRESHOLD, 0) * mu_f
            # improvement -= max(mu_v + std_v - SAFETY_THRESHOLD, 0) * mu_f
            # improvement -= max(mu_v + 2 * std_v - SAFETY_THRESHOLD, 0) * mu_f * 0.1

            Z = improvement / std_f
            ei = improvement * norm.cdf(Z) + std_f * norm.pdf(Z)
            # ei = norm.cdf(Z)  # use pi

            # avoid sampling from around the initial point
            mask = abs(x.flatten() - self.X_sample.flatten()[0]) <= 0.1
            ei[mask] -= 1e6

            # Apply a mask to only select EI of points that satisfy the constraint
            # We should carefully avoid points that potentially violates the constrant v.

            # print(f"mu_v: {mu_v}")
            # print(f"std_v: {std_v}")
            # print(mu_v, std_v, mu_v + std_v)
            ei -= 0.01 * max(mu_v + std_v, 0)
            ei -= max(mu_v + 1.5 * std_v - SAFETY_THRESHOLD, 0)
            # ei -= 0.01 * max(mu_v + 2 * std_v - SAFETY_THRESHOLD, 0)

        return ei

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        # raise NotImplementedError
        self.X_sample = np.vstack((self.X_sample, x.reshape(-1, 1)))
        self.Y_sample_f = np.vstack((self.Y_sample_f, f))
        self.Y_sample_v = np.vstack((self.Y_sample_v, v))
        # Update Gaussian Process Regressors
        self.gp_f.fit(self.X_sample, self.Y_sample_f)
        self.gp_v.fit(self.X_sample, self.Y_sample_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # raise NotImplementedError
        mask = self.Y_sample_v.flatten() <= SAFETY_THRESHOLD
        safe_f_values = self.Y_sample_f.flatten()[mask]
        if len(safe_f_values) == 0:
            raise ValueError("No safe solution found")

        # avoid penalty of 0.15
        strong_mask = mask & (
            abs(self.X_sample.flatten() - self.X_sample.flatten()[0]) > 0.1
        )
        strong_safe_f_values = self.Y_sample_f.flatten()[strong_mask]
        if len(strong_safe_f_values) > 0:
            safe_f_values = strong_safe_f_values
            mask = strong_mask

        return self.X_sample[mask][np.argmax(safe_f_values)]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        # pass
        import matplotlib.pyplot as plt

        # Generate a sequence of points within the DOMAIN
        x = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 400).reshape(-1, 1)

        # Predict the mean and standard deviation for the objective and constraint functions
        mu_f, std_f = self.gp_f.predict(x, return_std=True)
        mu_v, std_v = self.gp_v.predict(x, return_std=True)

        # Plot the objective function posterior
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.fill_between(
            x.ravel(),
            (mu_f - 1.96 * std_f).ravel(),
            (mu_f + 1.96 * std_f).ravel(),
            alpha=0.1,
            label="Confidence",
        )
        plt.plot(x, mu_f, label="Mean")
        plt.scatter(self.X_sample, self.Y_sample_f, c="r", label="Observations")
        plt.title("GP Posterior of Objective Function")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()

        # Plot the constraint function posterior
        plt.subplot(2, 1, 2)
        plt.fill_between(
            x.ravel(),
            (mu_v - 1.96 * std_v).ravel(),
            (mu_v + 1.96 * std_v).ravel(),
            alpha=0.1,
            label="Confidence",
        )
        plt.plot(x, mu_v, label="Mean")
        plt.scatter(self.X_sample, self.Y_sample_v, c="r", label="Observations")
        plt.axhline(
            y=SAFETY_THRESHOLD, color="k", linestyle="--", label="Safety Threshold"
        )
        plt.title("GP Posterior of Constraint Function")
        plt.xlabel("x")
        plt.ylabel("v(x)")
        plt.legend()

        # Plot the recommendation if required
        if plot_recommendation and self.next_x is not None:
            plt.scatter(
                self.X_sample[-1], self.Y_sample_f[-1], c="g", label="Recommendation"
            )
            plt.legend()

        plt.tight_layout()
        plt.savefig("plot.png")


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---


def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return -np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0
    # return 6.0 if 4.5 <= x <= 5.5 else 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), (
            f"The function next recommendation must return a numpy array of "
            f"shape (1, {DOMAIN.shape[0]})"
        )

        # Obtain objective and constraint observation
        # obj_val = f(x) + np.randn()
        obj_val = f(x) + np.random.randn()
        # cost_val = v(x) + np.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), (
        f"The function get solution must return a point within the"
        f"DOMAIN, {solution} returned instead"
    )

    # Compute regret
    regret = 0 - f(solution)

    unsafe_evals = [v(x) > SAFETY_THRESHOLD for x in agent.X_sample]

    print(
        f"Optimal value: 0\nProposed solution {solution}\nSolution value "
        f"{f(solution)}\nRegret {regret}\nUnsafe-evals {sum(unsafe_evals)}\n"
    )

    agent.plot()


if __name__ == "__main__":
    main()
