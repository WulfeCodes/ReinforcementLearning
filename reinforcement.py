import gymnasium as gym
import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import os
import math 

class ReplayBufferDict:
    def __init__(self, max_games):
        self.buffer = {}  # Store games as dictionary entries
        self.max_games = max_games
        self.current_game = 0  # Track the current game index

    def add_game(self, states, actions, rewards):
        """Add a game experience to the buffer"""
        if len(self.buffer) >= self.max_games:
            oldest_game = min(self.buffer.keys())  # Remove oldest entry
            del self.buffer[oldest_game]

        self.buffer[self.current_game] = {
            "states": states,
            "actions": actions,
            "rewards": rewards
        }
        self.current_game += 1  # Increment game index

    def sample(self, batch_size):
        """Sample a batch of games from the buffer"""
        batch_size = min(batch_size, len(self.buffer))
        sampled_keys = np.random.choice(list(self.buffer.keys()), batch_size, replace=False)
        return {key: self.buffer[key] for key in sampled_keys}

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, fc1_dims, fc2_dims, output_dims, beta, chkpt_dir='tmp/lunar'):
        super(ValueNetwork, self).__init__()

        self.input_dims = input_dims
        self.action_dims = action_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims
        
        # Corrected the architecture
        self.input = nn.Linear(self.input_dims + self.action_dims, fc1_dims)
        self.fc1 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2 = nn.Linear(self.fc2_dims, output_dims)  # Fixed variable name
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '_sac')
        self.device = t.device('cpu')
        self.to(self.device)

    def forward(self, actions, observation):
        # Convert inputs to tensors if they aren't already
        if not isinstance(observation, t.Tensor):
            observation = t.tensor(observation, dtype=t.float32)
        if not isinstance(actions, t.Tensor):
            actions = t.tensor(actions, dtype=t.float32)
            
        # Reshape if needed
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)
            
        # Correctly concatenate tensors
        combined = t.cat([observation, actions], dim=1)
        
        value = self.input(combined)
        value = F.relu(value)
        value = self.fc1(value)
        value = F.relu(value)
        value = self.fc2(value)  # Use correct layer
        
        return value
    
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims, beta, chkpt_dir='tmp/lunar'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.output = nn.Linear(self.fc2_dims, self.output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        # Weight savers!
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, '_sac')
        self.device = t.device('cpu')
        self.to(self.device)

    def forward(self, observation):
        # Convert observation to tensor if it's not already
        if not isinstance(observation, t.Tensor):
            # Handle the case where observation is a tuple from env.reset()
            if isinstance(observation, tuple):
                observation = observation[0]  # Extract the actual observation from the tuple
            observation = t.tensor(observation, dtype=t.float32)
        
        # Ensure the observation has the right shape
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
            
        action_val = self.fc1(observation)
        action_val = F.relu(action_val)
        action_val = self.fc2(action_val)
        action_val = F.relu(action_val)
        logits = self.output(action_val)
        probs = F.softmax(logits, dim=1)
        
        return probs
    
    def get_action(self, observation):
        probs = self.forward(observation)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.item()
    
    def get_log_prob(self, state, action):
        probs = self.forward(state)
        distribution = Categorical(probs)
        return distribution.log_prob(t.tensor(action))
    
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))
        return self.state_dict()


def main():
    env = gym.make("LunarLander-v3", continuous=False)
    # Action space is 4 discrete
    # Observation space is 8 float
    beta = 0.0001
    Actor = ActorNetwork(8, 256, 256, 4, beta)
    Value = ValueNetwork(8, 4, 256, 256, 1, beta)
    n_games = 2000
    ReplayBuffer = ReplayBufferDict(n_games)

    alpha = 0.0001
    lr = 0.001
    gamma = .99
    
    for game in range(n_games):
        # Reset environment and get initial observation
        obs, _ = env.reset()  # Correctly unpack the tuple
        
        currActions = []
        currRewards = []
        currStates = []
        
        done = False
        
        while not done:
            # Convert observation to tensor for network input
            obs_tensor = t.tensor(obs, dtype=t.float32).unsqueeze(0)
            
            # Get action probabilities
            action_probs = Actor.forward(obs_tensor)
            
            # Sample action from distribution
            distribution = Categorical(action_probs)
            action = distribution.sample().item()
            
            # Take step in environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    
            # Store experience
            currStates.append(obs)
            currActions.append(action)
            currRewards.append(reward)
            
            obs = next_obs
            
        # End of episode
        ReplayBuffer.add_game(currStates, currActions, currRewards)
        
        # Update networks based on collected experience
        # Update Value Network
        for i in range(len(currRewards)):
            # Calculate return (sum of future rewards)
            G = 0
            for j in range(i, len(currRewards)):
                G += (gamma**(j-i)) * currRewards[j]
            
            # Convert state and action to tensors
            state_tensor = t.tensor(currStates[i], dtype=t.float32).unsqueeze(0)
            action_onehot = t.zeros(4)
            action_onehot[currActions[i]] = 1
            action_onehot = action_onehot.unsqueeze(0)
            
            # Get current value prediction
            current_value = Value.forward(action_onehot, state_tensor)
            
            # Calculate target value
            target_value = t.tensor([G], dtype=t.float32).unsqueeze(0)
            
            # Compute loss
            value_loss = F.mse_loss(current_value, target_value)
            
            # Update value network
            Value.optimizer.zero_grad()
            value_loss.backward()
            Value.optimizer.step()
        
        # Update Actor Network
        for i in range(len(currActions)):
            # Calculate return
            G = 0
            for j in range(i, len(currRewards)):
                G += currRewards[j]
            
            # Convert state to tensor
            state_tensor = t.tensor(currStates[i], dtype=t.float32).unsqueeze(0)
            
            # Get action probabilities
            action_probs = Actor.forward(state_tensor)
            
            # Create distribution
            distribution = Categorical(action_probs)
            
            # Calculate log probability of taken action
            log_prob = distribution.log_prob(t.tensor([currActions[i]]))
            
            # Calculate actor loss (negative for gradient ascent)
            actor_loss = -log_prob * G
            
            # Update actor network
            Actor.optimizer.zero_grad()
            actor_loss.backward()
            Actor.optimizer.step()
        
        # Save checkpoints
        Actor.save_checkpoint()
        Value.save_checkpoint()
        
        print(f"Game {game+1}/{n_games} completed with total reward: {sum(currRewards)}")

if __name__ == "__main__":
    main()