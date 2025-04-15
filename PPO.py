import gymnasium as gym
import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import os

class ReplayBufferDict:
    def __init__(self, max_games):
        self.buffer = {}
        self.max_games = max_games
        self.current_game = 0

    def add_game(self, states, actions, rewards):
        if len(self.buffer) >= self.max_games:
            oldest_game = min(self.buffer.keys())
            del self.buffer[oldest_game]

        self.buffer[self.current_game] = {
            "states": states,
            "actions": actions,
            "rewards": rewards
        }
        self.current_game += 1

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        sampled_keys = np.random.choice(list(self.buffer.keys()), batch_size, replace=False)
        return {key: self.buffer[key] for key in sampled_keys}

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer[self.current_game-1][key]


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, fc1_dims, fc2_dims, beta, chkpt_dir='/tmp/PPO'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir,'_ppo')
        self.device = t.device('cpu')
        
        self.input = nn.Linear(input_dims, fc1_dims)
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, output_dims)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(self.device)

    def forward(self, state):
        x = self.input(state)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        actions = self.output(x)
        # For continuous action spaces, we can return the raw outputs
        # which represent the policy's parameters directly
        return actions
    
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))
        return self.state_dict()


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, fc1_dims, fc2_dims, beta, chkpt_dir='tmp/PPO'):
        super(ValueNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir,'_ppo')
        self.device = t.device('cpu')
        
        self.input1 = nn.Linear(input_dims, fc1_dims)
        self.fc11 = nn.Linear(fc1_dims, fc2_dims)
        self.output1 = nn.Linear(fc2_dims, output_dims)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(self.device)

    def forward(self, state):
        value = self.input1(state)
        value = F.relu(value)
        value = self.fc11(value)
        value = F.relu(value)
        value = self.output1(value)
        return value
    
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))
        return self.state_dict()


def main():
    env = gym.make("Humanoid-v4")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    beta = 0.0001
    Actor = ActorNetwork(obs_dim, action_dim, 256, 256, beta)
    Value = ValueNetwork(obs_dim, 1, 256, 256, beta)
    
    n_games = 10000
    Replay_Buffer = ReplayBufferDict(n_games)
    gamma = 0.95
    epsilon = 0.002  # PPO clip parameter
    
    for game in range(n_games):
        obs, _ = env.reset()
        currActions = []  # Will store action distributions
        currLogProbs = []  # Will store log probs
        currRewards = []
        currStates = []
        value_to_go = []
        done = False
        counter = 0

        # Collect experience
        while not done:
            obs_tensor = t.tensor(obs, dtype=t.float32).unsqueeze(0)
            
            # Get action distribution from actor
            action_params = Actor.forward(obs_tensor)
            
            # Create distribution - for Humanoid, use Normal distribution
            dist = Normal(action_params, t.ones_like(action_params) * 0.1)  # Fixed std
            
            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            
            # Take step in environment
            action_np = action.detach().numpy().flatten()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Store experience
            currStates.append(obs)
            currActions.append(action_params.detach())  # Store action parameters
            currLogProbs.append(log_prob.detach())  # Store log prob
            currRewards.append(reward)
            
            # Calculate running reward sum
            if counter == 0:
                value_to_go.append(reward)
            else:
                value_to_go.append(reward + value_to_go[counter-1])
                
            obs = next_obs
            counter += 1

        # Add to replay buffer
        Replay_Buffer.add_game(currStates, currLogProbs, currRewards)

        # Calculate advantages
        Alist = []
        A = 0
        lambda_gae = 0.97
        
        # Calculate advantages (handle the out of bounds case)
        for i in range(len(currStates) - 1):
            delta = currRewards[i] + (value_to_go[-1] - currRewards[i]) + gamma * (value_to_go[-1] - currRewards[i+1])
            A += delta * (gamma * lambda_gae) ** i
            Alist.append(A)
        
        # Handle the last state separately
        i = len(currStates) - 1
        delta = currRewards[i] + (value_to_go[-1] - currRewards[i])
        A += delta * (gamma * lambda_gae) ** i
        Alist.append(A)
        
        # Convert to tensor for easier operations
        Alist_tensor = t.tensor(Alist, dtype=t.float32)
        
        # Determine clipping parameter based on advantage sign
        g = 1 + epsilon if A >= 0 else 1 - epsilon
        g_tensor = t.tensor(g, dtype=t.float32)

        # PPO update - actor
        Actor.optimizer.zero_grad()
        actor_loss = 0
        
        for i in range(len(currStates)):
            state_tensor = t.tensor(currStates[i], dtype=t.float32).unsqueeze(0)
            old_log_prob = currLogProbs[i]
            
            # Get new distribution with current policy
            action_params = Actor.forward(state_tensor)
            dist = Normal(action_params, t.ones_like(action_params) * 0.1)
            
            # Sample same action but get new log prob
            action = dist.sample()
            new_log_prob = dist.log_prob(action).sum(-1)
            
            # Calculate ratio
            ratio = t.exp(new_log_prob - old_log_prob)
            
            # Clipped surrogate objective
            surr1 = ratio * Alist_tensor[i]
            surr2 = t.clamp(ratio, 1-epsilon, 1+epsilon) * Alist_tensor[i]
            actor_loss += t.min(surr1, surr2)  # Negative for gradient ascent
        
        actor_loss /= len(currStates)
        actor_loss.backward()
        Actor.optimizer.step()
        
        # PPO update - value network
        Value.optimizer.zero_grad()
        value_loss = 0
        
        for i in range(len(currStates)):
            state_tensor = t.tensor(currStates[i], dtype=t.float32).unsqueeze(0)
            value = Value.forward(state_tensor)
            value_target = Alist_tensor[i]
            value_loss += (value - value_target) ** 2
            
        value_loss /= len(currStates)
        value_loss.backward()
        Value.optimizer.step()
        
        if game % 10 == 0:
            print(f"Game: {game}, Avg Reward: {sum(currRewards)/len(currRewards)}, Actor Loss: {actor_loss.item()}, Value Loss: {value_loss.item()}")


if __name__ == '__main__':
    main()