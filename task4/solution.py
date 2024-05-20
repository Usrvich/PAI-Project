import copy
import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal

from utils import ReplayBuffer, get_env, run_episode

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class NeuralNetwork(nn.Module):
    '''
    This class implements a neural network with a variable number of hidden layers and hidden units.
    You may use this function to parametrize your policy and critic networks.
    '''
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int,
                                hidden_layers: int, activation: str = ""):
        super(NeuralNetwork, self).__init__()

        # TODO: Implement this function which should define a neural network
        # with a variable number of hidden layers and hidden units.
        # Here you should define layers which your network will use.
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_size))
        self.layers.append(self.activation)
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(hidden_size, output_dim))
        # NOTE: Not sure if I'm right about how to use this argument
        # tanh will be used for Actor because output should be in range(-1, 1)
        if activation == "":
            pass
        elif activation == "tanh":
            self.layers.append(nn.Tanh())
        else:
            raise NotImplementedError(f"activation '{activation}' is not defined")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass for the neural network you have defined.
        for layer in self.layers:
            s = layer(s)
        return s

class Actor:
    def __init__(self,hidden_size: int, hidden_layers: int, actor_lr: float,
                state_dim: int = 3, action_dim: int = 1, device: torch.device = torch.device('cpu')):
        super(Actor, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.actor_lr = actor_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.setup_actor()

    def setup_actor(self):
        '''
        This function sets up the actor network in the Actor class.
        '''
        # TODO: Implement this function which sets up the actor network.
        # Take a look at the NeuralNetwork class in utils.py.
        # NOTE: actor is a map of state->action.
        self.model = NeuralNetwork(input_dim=self.state_dim, output_dim=self.action_dim, hidden_size=self.hidden_size, hidden_layers=self.hidden_layers, activation="tanh")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.actor_lr)

    def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
        '''
        :param log_std: torch.Tensor, log_std of the policy.
        Returns:
        :param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
        '''
        return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

    def get_action_and_log_prob(self, state: torch.Tensor,
                                deterministic: bool) -> (torch.Tensor, torch.Tensor):
        '''
        :param state: torch.Tensor, state of the agent
        :param deterministic: boolean, if true return a deterministic action
                                otherwise sample from the policy distribution.
        Returns:
        :param action: torch.Tensor, action the policy returns for the state.
        :param log_prob: log_probability of the the action.
        '''
        assert state.shape == (3,) or state.shape[1] == self.state_dim, 'State passed to this method has a wrong shape'
        # action , log_prob = torch.zeros(state.shape[0]), torch.ones(state.shape[0])
        action , log_prob = torch.zeros(self.action_dim), torch.ones(self.action_dim)
        # TODO: Implement this function which returns an action and its log probability.
        # If working with stochastic policies, make sure that its log_std are clamped
        # using the clamp_log_std function.
        # NOTE: I'm confused between log_std and log_prob, and not sure about my code here.
        # Since Actor is a regression model, how do I get prob/std?
        action = self.model(state)
        # log_prob = ???
        if not deterministic:
            std = 1.0
            action += std * np.random.randn()


        assert (action.shape == (self.action_dim, ) and log_prob.shape == (self.action_dim,)) or \
            (action.shape == (state.shape[0], 1) and log_prob.shape == (state.shape[0], 1)), 'Incorrect shape for action or log_prob.'
        return action, log_prob


class Critic:
    def __init__(self, hidden_size: int,
                 hidden_layers: int, critic_lr: int, state_dim: int = 3,
                    action_dim: int = 1,device: torch.device = torch.device('cpu')):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.critic_lr = critic_lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.setup_critic()

    def setup_critic(self):
        # TODO: Implement this function which sets up the critic(s). Take a look at the NeuralNetwork
        # class in utils.py. Note that you can have MULTIPLE critic networks in this class.
        # NOTE; output of critic is value function Q
        # This is a float value (output_dim=1) and is computed based on state-action pair (input_dim=self.state_dim + self.action_dim)
        self.model_base = NeuralNetwork(input_dim=self.state_dim + self.action_dim, output_dim=1, hidden_size=self.hidden_size, hidden_layers=self.hidden_layers)
        self.model_target = NeuralNetwork(input_dim=self.state_dim + self.action_dim, output_dim=1, hidden_size=self.hidden_size, hidden_layers=self.hidden_layers)

        self.model_target.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.model_base.parameters(), lr=self.critic_lr)

class TrainableParameter:
    '''
    This class could be used to define a trainable parameter in your method. You could find it
    useful if you try to implement the entropy temerature parameter for SAC algorithm.
    '''
    def __init__(self, init_param: float, lr_param: float,
                 train_param: bool, device: torch.device = torch.device('cpu')):

        self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
        self.optimizer = optim.Adam([self.log_param], lr=lr_param)

    def get_param(self) -> torch.Tensor:
        return torch.exp(self.log_param)

    def get_log_param(self) -> torch.Tensor:
        return self.log_param


class Agent:
    def __init__(self):
        # Environment variables. You don't need to change this.
        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.min_buffer_size = 1000
        self.max_buffer_size = 100000
        # If your PC possesses a GPU, you should be able to use it for training,
        # as self.device should be 'cuda' in that case.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)

        self.setup_agent()

    def setup_agent(self):
        # TODO: Setup off-policy agent with policy and critic classes.
        # Feel free to instantiate any other parameters you feel you might need.
        # NOTE: Should find optimal parameters (hidden_size, hidden_layers, lr)
        self.actor = Actor(hidden_size=100, hidden_layers=2, actor_lr=1e-3, action_dim=self.action_dim, device=self.device)
        self.critic1 = Critic(hidden_size=100, hidden_layers=2, critic_lr=1e-3, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)
        self.critic2 = Critic(hidden_size=100, hidden_layers=2, critic_lr=1e-3, state_dim=self.state_dim, action_dim=self.action_dim, device=self.device)

    def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
        """
        :param s: np.ndarray, state of the pendulum. shape (3, )
        :param train: boolean to indicate if you are in eval or train mode.
                    You can find it useful if you want to sample from deterministic policy.
        :return: np.ndarray,, action to apply on the environment, shape (1,)
        """
        # TODO: Implement a function that returns an action from the policy for the state s.
        action: np.ndarray = self.actor.model(torch.from_numpy(s)).detach().numpy()
        if train:
            std = 0.5
            action += std * np.random.randn()
            action = action.clip(min=-1, max=1)

        assert action.shape == (1,), 'Incorrect action shape.'
        assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray'
        return action

    @staticmethod
    def run_gradient_update_step(object: Union[Actor, Critic], loss: torch.Tensor):
        '''
        This function takes in a object containing trainable parameters and an optimizer,
        and using a given loss, runs one step of gradient update. If you set up trainable parameters
        and optimizer inside the object, you could find this function useful while training.
        :param object: object containing trainable parameters and an optimizer
        '''
        object.optimizer.zero_grad()
        loss.mean().backward()
        object.optimizer.step()

    def critic_target_update(self, base_net: NeuralNetwork, target_net: NeuralNetwork,
                             tau: float, soft_update: bool):
        '''
        This method updates the target network parameters using the source network parameters.
        If soft_update is True, then perform a soft update, otherwise a hard update (copy).
        :param base_net: source network
        :param target_net: target network
        :param tau: soft update parameter
        :param soft_update: boolean to indicate whether to perform a soft update or not
        '''
        for param_target, param in zip(target_net.parameters(), base_net.parameters()):
            if soft_update:
                param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
            else:
                param_target.data.copy_(param.data)

    def train_agent(self):
        '''
        This function represents one training iteration for the agent. It samples a batch
        from the replay buffer,and then updates the policy and critic networks
        using the sampled batch.
        '''
        # TODO: Implement one step of training for the agent.
        # Hint: You can use the run_gradient_update_step for each policy and critic.
        # Example: self.run_gradient_update_step(self.policy, policy_loss)

        # Batch sampling
        batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = batch
        # print(s_batch.size(), a_batch.size(), r_batch.size(), s_prime_batch.size())  # (200, 3), (200, 1), (200, 3), (200, 1)

        # TODO: Implement Critic(s) update here.
        gamma = 0.99
        alpha = 0.1
        def _calc_entropy(model: Actor, state: torch.Tensor) -> torch.Tensor:
            # NOTE: compute entropy of the model output
            # To do this, we need probability of the model output -> use get_action_and_log_prob??
            return 0
        entropy = _calc_entropy(self.actor.model, s_prime_batch)

        def _calc_loss(critic1: Critic, critic2: Critic) -> torch.Tensor:
            base1 = critic1.model_base
            target1 = critic1.model_target
            base2 = critic2.model_base
            target2 = critic2.model_target
            sa_pair_for_y = torch.concat((s_prime_batch, self.actor.model(s_prime_batch)), dim=-1)
            min_output = torch.min(target1(sa_pair_for_y), target2(sa_pair_for_y))
            y = r_batch + gamma * (min_output + alpha * entropy)
            y = y.detach()  # Detach because we treat it as a label
            sa_pair = torch.concat((s_batch, a_batch), dim=-1)
            base1_output = base1(sa_pair)
            base2_output = base2(sa_pair)
            base1_loss = (base1_output - y) ** 2
            base2_loss = (base2_output - y) ** 2
            return base1_loss, base2_loss

        base1_loss, base2_loss = _calc_loss(self.critic1, self.critic2)
        self.run_gradient_update_step(object=self.critic1, loss=base1_loss)
        self.run_gradient_update_step(object=self.critic2, loss=base2_loss)

        # TODO: Implement Policy update here
        sa_pair_for_actor = torch.cat((s_batch, self.actor.model(s_batch)), dim=-1)
        # NOTE: Sign is flipped because gradient ascent w.r.t value function Q is same as gradient descent w.r.t -Q
        actor_loss = - (self.critic1.model_base(sa_pair_for_actor) + alpha * entropy)
        self.run_gradient_update_step(object=self.actor, loss=actor_loss)

        self.critic_target_update(base_net=self.critic1.model_base, target_net=self.critic1.model_target, tau=0.01, soft_update=True)
        self.critic_target_update(base_net=self.critic2.model_base, target_net=self.critic2.model_target, tau=0.01, soft_update=True)




# This main function is provided here to enable some basic testing.
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    TRAIN_EPISODES = 50
    TEST_EPISODES = 300

    # You may set the save_video param to output the video of one of the evalution episodes, or
    # you can disable console printing during training and testing by setting verbose to False.
    save_video = True
    verbose = True

    agent = Agent()
    env = get_env(g=10.0, train=True)

    for EP in range(TRAIN_EPISODES):
        print(f"epoch {EP:03}", end=" ")
        run_episode(env, agent, None, verbose, train=True)

    if verbose:
        print('\n')

    test_returns = []
    env = get_env(g=10.0, train=False)

    if save_video:
        video_rec = VideoRecorder(env, "pendulum_episode.mp4")

    for EP in range(TEST_EPISODES):
        print(f"epoch {EP:03}", end=" ")
        rec = video_rec if (save_video and EP == TEST_EPISODES - 1) else None
        with torch.no_grad():
            episode_return = run_episode(env, agent, rec, verbose, train=False)
        test_returns.append(episode_return)

    avg_test_return = np.mean(np.array(test_returns))

    print("\n AVG_TEST_RETURN:{:.1f} \n".format(avg_test_return))

    if save_video:
        video_rec.close()
