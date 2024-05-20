We defined two attributes for SWAG-diagonal: theta_bar and theta2bar. The former one to store the running average of $\bar\theta$, the later one to store the running average of $\bar{\theta^2}$. For full SWAG, we defined the attribute d_hat = {name: collections.deque() for name, param in self.network.named_parameters()} to be the matrix with columns equal to the deviation matrix comprised of columns $D_i$. In update_swag function, the two theta attributes are updated using the running average formula, append new column $\theta-\bar\theta$ to $\hat D$, and remove the first column if the rank of $\hat D$ exceeds deviation_matrix_max_rank.  In the fitting procedure, the two theta attributes are initialized by the parameters of the model before training. In each epoch, we check if (epoch + 1) % self.swag_update_freq == 0 and update SWAG attributes. In calibration procedure, we used the provided prediction threshold 2.0/3.0. In the sample_parameters function, use theta_bar as current_mean, for current_std, instead of just use torch.sqrt(self.theta2bar[name] - current_mean ** 2), we used torch.sqrt(torch.clamp(self.theta2bar[name] - current_mean ** 2,1e-30)), which ensures that each element   in current_std is greater than or equal to 1e-30. We observed that when using the formal code, the final cost is very high, the reason is worthy further investigation. Then calculate sampled parameters using the formula given by the paper. We update batch normalization statistics after sampling parameters in each time. In prediction procedure, we store the result for each sampled model, then use the average of the results as final prediction.



We have defined two attributes for SWAG-diagonal: `theta_bar` and `theta2bar`. The former is used to store the running average of model parameters, denoted as $\bar{\theta}$, while the latter keeps track of the running average of the squared parameters, $\bar{\theta^2}$. For the full SWAG implementation, we introduced the attribute `d_hat`, which is a dictionary initialized with `collections.deque()` for each named parameter in our network. This attribute represents a deviation matrix, with each column corresponding to $D_i$ in the SWAG formulation.

In the `update_swag` function, we update the `theta_bar` and `theta2bar` attributes using the running average formula. Additionally, we append the new deviation vector $\theta - \bar{\theta}$ to $\hat{D}$ and remove the oldest column if the rank of $\hat{D}$ exceeds the `deviation_matrix_max_rank`.

At the beginning of the fitting procedure, we initialize the `theta_bar` and `theta2bar` attributes with the current model parameters. At the end of each epoch, we evaluate if the condition ((epoch + 1) % self.swag_update_freq == 0) holds true to determine whether to update the SWAG statistics.

In the calibration process, we utilize the provided prediction threshold value of $\frac{2.0}{3.0}$. This threshold helps in distinguishing between confident predictions and ambiguous samples.

In the `sample_parameters` function, `theta_bar` is used as `current_mean`. For `current_std`, we use the expression `torch.sqrt(torch.clamp(self.theta2bar[name] - current_mean ** 2, 1e-30))` to ensure numerical stability, as it prevents the standard deviation from being too close to zero, which could cause numerical issues and leads to higher cost.

Afterwards, we calculate the sampled parameters as described in the SWAG paper. Then update the batch normalization statistics after each time of resampling parameters to maintain accurate model statistics.

In the prediction phase, we store the predictions from each sampled model. The final prediction is then determined by taking the average of these stored predictions, providing an estimate that incorporates the uncertainty represented by the SWAG distribution.





The kernel design in `BO_algo` class involves constructing Gaussian Process Regression (GPR) models for both the objective function `f` and the constraint function `v`. The kernel for modeling `f` is a combination of a Matern kernel and a White kernel. The kernel for `v` is more complex, comprising a Constant kernel, a DotProduct kernel, a Matern kernel, and a White kernel. We mainly implemented following functions: 1. The `next_recommendation` function calls the `optimize_acquisition_function` method and returns the point that this method determines as the optimal next sampling location. 2. In `acquisition_function` we first calculate probability of improvement (PI) over the current best value, adjusted by the uncertainty (standard deviation) at each point.  Then subtract `max(mu_v+std_v, 0)` as a penalty term from the PI adjusted by a lambda factor (`self.lambdas`=100). The function also includes additional logic to discourage sampling near the initial point of the domain. A mask is applied to heavily penalize and effectively exclude points that are likely to violate the safety constraint. 3. In `add_data_point`, observed data point is added to the data and the Gaussian process regression model is updated. 4. `get_solution`  identifies the optimal point that maximizes the objective function `f` while satisfying the constraint function `v`. This function also includes an additional check to avoid points that are very close to the initial sampled point.



1. NeuralNetwork: A flexible neural network class, capable of creating models with various numbers of hidden layers and input/output dimensions.

2. Actor: Manages the policy learning process. It uses the `NeuralNetwork` class to map states to actions.  The input dimension is equal to the dimension of state, the output dim is action_dim. It includes methods to clamp log standard deviations and compute actions and their corresponding log probabilities, essential for handling stochastic policies.

3. Critic: Estimates the value function using a separate neural network. The input dimension is equal to state_dim + action_dim, the output dim is 1. This class plays a crucial role in evaluating the actions taken by the actor.

4. Agent: The central piece that orchestrates the training process. It initializes and integrates the actor and critic components, manages the replay buffer for experience replay, and facilitates the interaction with the environment.

5. TrainableParameter: A utility class for defining trainable parameters, such as the entropy temperature in SAC, which balances exploration and exploitation.

The training loop within the `Agent` class is where the SAC algorithm's core logic resides. It involves updating the policy (actor) and the value function (critic) using sampled data from the replay buffer, and soft updates to the target networks. The script also includes a testing phase, running episodes in the environment to evaluate the agent's performance. 



regression task?

mixup: easy to implement. But the precision of the combined feature? can it be used in ehr?







model inversion: client queries the model to find representative training inputs

data extraction: client queires the model to find exact training samples that belong to the training data



$\argmin_{x^*} d(\nabla_\Theta \mathcal{L}(f_\Theta(x^*), y^*),g_k)+\alpha_{reg}\cdot \mathcal{R}(x^*)$

$\argmin_{\tilde x^k} d(\tilde\Theta^k, \Theta^k)+\alpha_{reg}\cdot\frac{1}{E^2}\sum_{e_1,e_2}\mathcal{R}(g(\{\tilde x^k_{e_1,b}\}),g(\{\tilde x ^k_{e_2,b}\}))$

For tabular data: One-hot encode categorical data; use softmax in reconstruction for positivity and unit-sum. Assessing quality: Combine reconstructions with varied initializations; form normalized histograms per cell; compute histogram entropy; higher is better.





