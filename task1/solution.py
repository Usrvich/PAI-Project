import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
import torch
import gpytorch


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """


    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        # TODO: Add custom initialization for your model here if necessary
        # Statistics of the training data, used for Standardization
        self.train_x_mean = None
        self.train_x_std = None
        self.train_y_mean = None
        self.train_y_std = None

        # Kmeans for clustering, used for training local GPs
        self.num_clusters = 20
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)

        # Models for each cluster
        self.local_GPs = []
        self.likelihoods = []
        self.training_iter = 100


    def make_predictions(self, test_x_2D: np.ndarray, test_x_AREA: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_x_2D: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_x_AREA: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        # Standardize the test data
        test_x_2D = (test_x_2D - self.train_x_mean) / self.train_x_std
        # Get cluster labels for the test data
        test_labels = self.kmeans.predict(test_x_2D)
     
        # # TODO: Use your GP to estimate the posterior mean and stddev for each city_area here
        gp_mean = np.zeros(test_x_2D.shape[0], dtype=float)
        gp_std = np.zeros(test_x_2D.shape[0], dtype=float)
        # gp_mean = np.zeros((test_x_2D.shape[0]), dtype=float)
        # gp_std = np.zeros((test_x_2D.shape[0]), dtype=float)
        # # gp_mean, gp_std = self.gp.predict(test_x_2D, return_std=True)


        # # TODO: Use the GP posterior to form your predictions here
        # # TODO: find better way for prediction (utilize std)
        # # predictions = gp_mean
        # predictions = np.where(test_x_AREA, gp_mean + gp_std, gp_mean)
        # # print(gp_std)

        # Predict for each cluster
        for i in range(self.num_clusters):
            # Select the data points belonging to the current cluster
            cluster_x_2D = test_x_2D[test_labels == i]
            cluster_x_2D = torch.from_numpy(cluster_x_2D).float()
            # Predict using the local GP
            model = self.local_GPs[i]
            likelihood = self.likelihoods[i]
            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()

            cluster_f_preds = model(cluster_x_2D)
            cluster_gp_mean = cluster_f_preds.mean
            cluster_gp_std = cluster_f_preds.stddev

            gp_mean[test_labels == i] = cluster_gp_mean.detach().numpy()
            gp_std[test_labels == i] = cluster_gp_std.detach().numpy()
        
        # DeStandardization
        predictions = gp_mean * self.train_y_std + self.train_y_mean
        gp_mean = gp_mean * self.train_y_std + self.train_y_mean
        gp_std = gp_std * self.train_y_std
        
        return predictions+1.5*gp_std, gp_mean , gp_std

    def fitting_model(self, train_y: np.ndarray,train_x_2D: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x_2D: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        # TODO: Fit your model here
        # TODO: lots of parameters https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
        # TODO: find better undersampling

        # Get statistics of the training data
        self.train_x_mean = np.mean(train_x_2D, axis=0)
        self.train_x_std = np.std(train_x_2D, axis=0)
        self.train_y_mean = np.mean(train_y)
        self.train_y_std = np.std(train_y)
        # Standardize the training data
        train_x_2D = (train_x_2D - self.train_x_mean) / self.train_x_std
        train_y = (train_y - self.train_y_mean) / self.train_y_std
        # Apply kmeans clustering
        train_labels = self.kmeans.fit_predict(train_x_2D)

        # Fit a GP for each cluster
        for i in range(self.num_clusters):
            # Select the data points belonging to the current cluster
            print(f'Fitting model for cluster {i+1} / {self.num_clusters}')
            cluster_x_2D = train_x_2D[train_labels == i]
            print(f'Cluster size: {cluster_x_2D.shape[0]}')
            cluster_y = train_y[train_labels == i]
            # Convert to torch tensor
            cluster_x_2D = torch.from_numpy(cluster_x_2D).float()
            cluster_y = torch.from_numpy(cluster_y).float()

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(cluster_x_2D, cluster_y, likelihood)

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for i in range(self.training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(cluster_x_2D)
                # Calc loss and backprop gradients
                loss = -mll(output, cluster_y)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, self.training_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
                optimizer.step()
            self.local_GPs.append(model)
            self.likelihoods.append(likelihood)

        # end TODO

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# You don't have to change this function
def cost_function(ground_truth: np.ndarray, predictions: np.ndarray, AREA_idxs: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param AREA_idxs: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(AREA_idx) for AREA_idx in AREA_idxs]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_in_circle(coor, circle_coor):
    """
    Checks if a coordinate is inside a circle.
    :param coor: 2D coordinate
    :param circle_coor: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coor[0] - circle_coor[0])**2 + (coor[1] - circle_coor[1])**2 < circle_coor[2]**2

# You don't have to change this function
def determine_city_area_idx(visualization_xs_2D):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param visualization_xs_2D: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])

    visualization_xs_AREA = np.zeros((visualization_xs_2D.shape[0],))

    for i,coor in enumerate(visualization_xs_2D):
        visualization_xs_AREA[i] = any([is_in_circle(coor, circ) for circ in circles])

    return visualization_xs_AREA

# You don't have to change this function
def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs_2D = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    print("vis")
    # print(visualization_xs_2D)
    visualization_xs_AREA = determine_city_area_idx(visualization_xs_2D)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs_2D, visualization_xs_AREA)
    # predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.scatter(visualization_xs_2D[:, 0], visualization_xs_2D[:, 1], c=predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    # im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    # cbar = fig.colorbar(im, ax = ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def plot_training_data(train_x_2D: np.ndarray, train_x_AREA: np.ndarray, train_y: np.ndarray, output_dir: str = '/results'):
    vmin, vmax = 0.0, 65.0
    fig, ax = plt.subplots()
    ax.set_title('training data of task 1')
    im = ax.scatter(train_x_2D[:,0], train_x_2D[:,1], c=train_y, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    # ax.scatter(train_x_2D[train_x_AREA,0], train_x_2D[train_x_AREA,1], color="red", alpha=0.1)
    figure_path = os.path.join(output_dir, 'train_data.pdf')
    fig.savefig(figure_path)


def extract_city_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_x_2D = np.zeros((train_x.shape[0], 2), dtype=float)
    train_x_AREA = np.zeros((train_x.shape[0],), dtype=bool)
    test_x_2D = np.zeros((test_x.shape[0], 2), dtype=float)
    test_x_AREA = np.zeros((test_x.shape[0],), dtype=bool)

    #TODO: Extract the city_area information from the training and test features
    train_x_2D[:, :] = train_x[:, :2]
    train_x_AREA[:] = train_x[:, 2]
    test_x_2D[:, :] = test_x[:, :2]
    test_x_AREA[:] = test_x[:, 2]
    # end TODO

    assert train_x_2D.shape[0] == train_x_AREA.shape[0] and test_x_2D.shape[0] == test_x_AREA.shape[0]
    assert train_x_2D.shape[1] == 2 and test_x_2D.shape[1] == 2
    assert train_x_AREA.ndim == 1 and test_x_AREA.ndim == 1

    return train_x_2D, train_x_AREA, test_x_2D, test_x_AREA

# you don't have to change this function
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_x_2D, train_x_AREA, test_x_2D, test_x_AREA = extract_city_area_information(train_x, test_x)

    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_y,train_x_2D)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.make_predictions(test_x_2D, test_x_AREA)
    print(predictions)

    if EXTENDED_EVALUATION:
        plot_training_data(train_x_2D, train_x_AREA, train_y, output_dir='.')
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
