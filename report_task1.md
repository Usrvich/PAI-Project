Firstly, we employed standardization to transform the training data (train_x, train_y) to have a mean of 0 and a standard deviation of 1. The mean and variance were calculated based on the training data.

Then we divided the training data (train_x) into 20 clusters using the K-Means. For each cluster, we trained a separate GP regressor. Initially we hoped this method may accelerate fitting process, and distribute responsibility for prediction, thereby enhancing accuracy. Data within each local cluster exhibits stronger correlations, effectively reducing the influence of distant data on local predictions.

For the GP regressor implementation, due to the slow fitting performance of the GaussianProcessRegressor in sklearn, we turned to the GPyTorch. To establish our GP model, we defined an ExactGPModel class, the class is implemented with reference to the GPyTorch tutorial (https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html). We used GaussianLikelihood, assuming consistent observation noise across all inputs. Each regressor was trained for 100 iterations.

In the prediction process, after standardization, we employed previously fitted K-Means classifier to assign each test point to the appropriate cluster. The corresponding local GP model is employed to predict each test point. Finally, we reverse the earlier standardization to obtain the predicted values.

To account for asymmetric costs, we use GP_mean + GP_std as final prediction result.

