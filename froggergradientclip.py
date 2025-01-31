import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
import cv2
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
import os
import json 
matplotlib.use('TkAgg')  # Needed for visualization 

# 1. Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_action_dist(action_count, count_per_action, num_actions):
    '''
    Function used for debugging and inspecting model behaviour.
    Prints the percentage an action was selected during an episode
    :param action_count: how many actions were performed during an episode
    :param count_per_action: list with counts for each specific action
    :param num_actions: number of actions
    '''''
    for action in range(num_actions):
        print(f"Model performed ACTION {action} {count_per_action[action] / action_count}%")

def preprocess_state(state):
    '''
    Function that preprocesses the state by first resizing, normalising,
    adding two dimensions (batch, channel) 
    :param state: the current state
    :return: the preprocessed state
    '''''
    resized_state = cv2.resize(state, (128, 128)) / 255.0  # Resize and normalize
    tensor_state = torch.tensor(resized_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Move to device
    if torch.isnan(tensor_state).any() or torch.isinf(tensor_state).any():
        print("Invalid state detected during preprocessing!")
        return torch.zeros_like(tensor_state)  # Replace invalid state with zeros
    return tensor_state

''''
Implementation of Advantage Actor-Critic architecture
'''
class A2CModel(nn.Module):

    def __init__(self, input_shape, num_actions):
        """
        Initialisation function
        :param input_shape: the shape of the input
        :param num_actions: the number of actions the model can take
        """""
        super(A2CModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        ''''
        This part of the code creates a dummy input in 
        order to compute the shape of the flattened feature map
        stored in n_flattend which is used for the actor and critic heads
        '''
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape).to(device)  # Move dummy input to device
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        '''
        Actor head with two linear layers and a softmax activation
        Outputs a probability distribution for the actions
        '''
        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )
        ''''
        Critic head with two linear layers
        Outputs an estimated value which assess how good 
        the current state is
        '''
        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """
        Forward pass through network
        :param x: input (in this ase state
        :return: the actor's output (action probabilites) and critic's output (estimated value)
        """""
        x = x.to(device)  # Ensure input is on the correct device
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        action_probs = torch.clamp(action_probs, min=1e-6, max=1 - 1e-6)  # Add numerical stability
        value = self.critic(features)
        return action_probs, value

def compute_returns(rewards, dones, gamma):
    """
    Computes the discounted returns used to compute the advantage
    :param rewards: the rewards collected during an episode
    :param dones: whether the episode was done or not
    :param gamma: discount factor
    :return: return values
    """""
    returns = []
    G = 0
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1 - done)
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

def train_a2c(model, optimizer, states, actions, rewards, dones, gamma=0.99):
    """
    Function that trains the model after the agent played one game
    It concatenates all states, actions and rewards, computes the advantage and performs a forward pass
    :param model: the model
    :param optimizer: the optimizer
    :param states: list of states
    :param actions: list of actions performed
    :param rewards: list of rewards
    :param dones: list of bools determining whether the game was done
    :param gamma: discount factor set to 0.99
    """""
    states = torch.cat(states).to(device)  # Move states to device
    actions = torch.tensor(actions, dtype=torch.long).to(device)  # Move actions to device
    dones = torch.tensor(dones, dtype=torch.float32).to(device)  # Move dones to device

    # Forward pass
    action_probs, values = model(states)

    # Compute returns and advantages
    returns = compute_returns(rewards, dones, gamma).to(device)
    advantage = returns - values.squeeze()

    # Actor loss
    log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
    actor_loss = -(log_probs * advantage.detach()).mean()

    # Critic loss
    critic_loss = F.mse_loss(values.squeeze(), returns.detach())

    # Total loss
    loss = actor_loss + critic_loss

    # Backpropagation with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
    optimizer.step()

    print(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {loss.item():.4f}")

def save_checkpoint(model, optimizer, filename="a2c_checkpoint.pth"):
    """
    Function used to save the model
    :param model: the model
    :param optimizer: the optimizer
    :param filename: desired filename
    """""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename="a2c_checkpoint.pth"):
    """
    Functions that loads a previously trained model if the file exists
    :param model: the model that we will override with the pre-trained model
    :param optimizer: the optimizer that will be taken from the file as well
    :param filename: the filename of the model we wish to load
    :return: true if the file exists, false otherwise
    """""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {filename}")
        return True
    return False

def save_rewards(reward_history, filename="reward_history.json"):
    """
    Function that saves a reward history to a json file
    :param reward_history: the list of rewards
    :param filename: filename used for saving
    """""
    with open(filename, "w") as f:
        json.dump(reward_history, f)
    print(f"Reward history saved to {filename}")

def load_rewards(filename="reward_history.json"):
    """
    Function that loads the rewards from a json file
    Used if we want to interrupt training and continue later on
    :param filename: the name of the file containing the rewards
    :return: a list of rewards if the file exists, otherwise an empty list
    """""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reward_history = json.load(f)
        print(f"Reward history loaded from {filename}")
        return reward_history
    else:
        print(f"No reward history file found. Starting fresh.")
        return []

def main():
    # This main is used just for visualisation, the model is not trained anymore
    # Choose which environment to initialize with human rendering or without
    #env = gym.make("ALE/Frogger-v5", obs_type='grayscale', frameskip=(1, 4), render_mode="human")
    env = gym.make("ALE/Frogger-v5", obs_type='grayscale', frameskip=(1, 4), render_mode="rgb_array")

    num_actions = env.action_space.n
    input_shape = (1, 128, 128)
    model = A2CModel(input_shape, num_actions).to(device)  # Move model to device
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Load checkpoint from a2c_checkpoint_13500.pth
    checkpoint_file = "a2c_checkpoint_13500.pth"
    if not load_checkpoint(model, optimizer, filename=checkpoint_file):
        print(f"Checkpoint {checkpoint_file} not found! Starting training from scratch.")

    # Load reward history
    reward_history = load_rewards()
    episode_rewards = deque(maxlen=100)
    episode = len(reward_history) * 5

    try:
        while True:
            state, _ = env.reset()
            state = preprocess_state(state)
            done = False
            episode_reward = 0

            while not done:
                env.render()  # Render the environment in human mode
                action_probs, _ = model(state)
                action = torch.multinomial(action_probs, 1).item()
                next_state, reward, done, _, _ = env.step(action)
                state = preprocess_state(next_state)
                episode_reward += reward

            print(f"Episode {episode}, Reward: {episode_reward}")
            episode_rewards.append(episode_reward)
            episode += 1
    except KeyboardInterrupt:
        print("Training interrupted.")
        env.close()

if __name__ == "__main__":
    main()


