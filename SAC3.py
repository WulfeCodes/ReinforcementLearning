import torch as t
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import gymnasium as gym
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt  

#TODO reward scaling helped q networks converge, test with constant alpha w/o entropy regularization
#TODO COMPUTATIONAL GRAPH SHIEEEE
#TODO reward scaling or q value clamp, q beta annealing
#TODO test with increased alpha and learning rates

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.ptr = 0  # Current position to write
        self.size = 0  # Current buffer size

        # Pre-allocate memory with float32 for efficiency
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        if(np.isnan(state).any() or np.isnan(action).any() or
        np.isnan(reward).any() or np.isnan(state_).any()):
            print("nan detected, outputting none")

            return 

        index = self.ptr
        self.state_memory[index] = state
        self.new_state_memory[index] = state_  
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.ptr = (self.ptr + 1) % self.mem_size
        self.size = min(self.size + 1, self.mem_size)

    def sample_buffer(self, batch_size):
        # Handle edge case where buffer has fewer samples than batch_size
        max_mem = min(self.size, self.mem_size)
        assert max_mem > 0, "Buffer is empty!"
        batch_size = min(batch_size, max_mem)  # Ensure we don't over-sample
        batch = np.random.choice(max_mem, batch_size, replace=(max_mem < batch_size))
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, states_, actions, rewards, dones
    

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'state_memoryTryNow31.npy'), self.state_memory)
        np.save(os.path.join(save_dir, 'action_memoryTryNow31.npy'), self.action_memory)
        np.save(os.path.join(save_dir, 'reward_memoryTryNow31.npy'), self.reward_memory)
        np.save(os.path.join(save_dir, 'new_state_memoryTryNow31.npy'), self.new_state_memory)
        np.save(os.path.join(save_dir, 'terminal_memoryTryNow31.npy'), self.terminal_memory)
        np.save(os.path.join(save_dir, 'ptrTryNow31.npy'), self.ptr)
        np.save(os.path.join(save_dir, 'sizeTryNow31.npy'), self.size)

    def load(self, load_dir):
        self.state_memory = np.load(os.path.join(load_dir, 'state_memoryTryNow31.npy'))
        self.action_memory = np.load(os.path.join(load_dir, 'action_memoryTryNow31.npy'))
        self.reward_memory = np.load(os.path.join(load_dir, 'reward_memoryTryNow31.npy'))
        self.new_state_memory = np.load(os.path.join(load_dir, 'new_state_memoryTryNow31.npy'))
        self.terminal_memory = np.load(os.path.join(load_dir, 'terminal_memoryTryNow31.npy'))
        self.ptr = np.load(os.path.join(load_dir, 'ptrTryNow31.npy'))
        self.size = np.load(os.path.join(load_dir, 'sizeTryNow31.npy'))
    
class QNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, output_dims, beta,file, chkpt_dir='/tmp1'):
        super(QNetwork,self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims
        self.beta = beta
        self.checkpoint_directory = chkpt_dir

        self.checkpoint_file = os.path.join(chkpt_dir,file)
        self.device = t.device('cpu')

        self.input_layer = nn.Linear(input_dims, fc1_dims)
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)
        self.fc2 = nn.Linear(fc2_dims, output_dims)
        self.optimizer = optim.Adam(self.parameters(), beta)

        #weight initilization using fan in method for leaky relu
        for layer in [self.input_layer, self.fc1]:
            nn.init.kaiming_normal_(
                layer.weight, 
                mode='fan_in', 
                nonlinearity='leaky_relu',
                a=0.01  
            )
        nn.init.uniform_(self.fc2.weight, -1e-3, 1e-3)



    def save_checkpoint(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))

    def forward(self, state, action):
        input = t.cat([state, action], dim=-1)
        currValue = self.input_layer(input)
        currValue = F.leaky_relu(currValue)
        currValue = self.fc1(currValue)
        currValue = F.leaky_relu(currValue)
        currValue = self.fc2(currValue)
        return currValue

class PolicyNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, beta, file_name ,chkpt_dir='/tmp1'):
        super(PolicyNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.beta = beta
        self.checkpoint_directory = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir,file_name)
        self.device = t.device('cpu')
        self.input_layer = nn.Linear(input_dims, fc1_dims)
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)

        #FAN IN WEIGHT INITIALIZATION, HELPS PRVNT exploding/vanishing grads
        for layer in [self.input_layer, self.fc1]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu',a=0.01)
        

        # Output layers for mean and log-std for each action dimension
        self.mean_output = nn.Linear(fc2_dims, n_actions)
        self.log_std_output = nn.Linear(fc2_dims, n_actions)

        #output layers set as uniform for stability
        nn.init.uniform_(self.mean_output.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_output.weight, -1e-3, 1e-3)
        
        self.optimizer = optim.Adam(self.parameters(), beta)

        #entropy annealing!!! here 
        self.log_alpha = t.tensor([np.log(0.3)], requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.log_alpha.data.clamp_(-5, 5) 
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=beta)
        self.target_entropy = t.tensor(-3.0).reshape(1,1)
        # Minimum and maximum log-std value
        self.min_log_std = -5
        self.max_log_std = 2

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))

    def sampleAction(self, state):
        currValue = self.input_layer(state)
        currValue = F.leaky_relu(currValue)
        currValue = self.fc1(currValue)
        currValue = F.leaky_relu(currValue)
        
        # Compute mean and log-std for all actions
        mean = self.mean_output(currValue)
        log_std = self.log_std_output(currValue)
        log_std = t.clamp(log_std, self.min_log_std, self.max_log_std)
        std = t.exp(log_std)

        # Create normal distributions
        normal_dist = t.distributions.Normal(mean, std)
        
        # Reparameterization trick
        z = normal_dist.rsample()

        # Apply tanh to constrain actions
        action = t.tanh(z)
        
        # Compute log probabilities
        #!!BIG CHANGE HERE, TAKING THE LOG PROB of action, instead of sample
        log_prob = normal_dist.log_prob(z)

        # Adjust for tanh squashing
        
        # Sum log probabilities
        log_prob = log_prob.sum().unsqueeze(0).unsqueeze(0)

        # print("log probability: {%f}, action probability: {%f}, std: {%f}, correction {%f}",log_prob,action, std, correction)

        return action, log_prob

# [Previous class implementations remain the same]
# ... (ReplayBuffer, QNetwork, PolicyNetwork classes)

class Visualizer():
    def __init__(self,n_iterations, training_iterations):
    
        self.rewardAccum = []
        self.lossAccumQ2 = []
        self.lossAccumQ1 = []
        self.lossAccumPolicy = []
    
    def plot(self):
        lossAccumPolicy_np = np.array([x.item() if hasattr(x, "item") else x for x in self.lossAccumPolicy])
        lossAccumQ1_np = np.array([x.item() if hasattr(x, "item") else x for x in self.lossAccumQ1])
        lossAccumQ2_np = np.array([x.item() if hasattr(x, "item") else x for x in self.lossAccumQ2])

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Reward Plot
        if len(self.rewardAccum) > 0:
            axes[0].plot(self.rewardAccum, label="Reward", color="blue")
            axes[0].set_title("Episode Rewards")
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Reward")
            axes[0].legend()
        
        # Loss Plot
        if len(self.lossAccumPolicy) > 0:
            axes[1].plot(lossAccumPolicy_np, label="Policy Loss", color="red", alpha=0.7)
            axes[1].plot(lossAccumQ1_np, label="Q1 Loss", color="green", alpha=0.7)
            axes[1].plot(lossAccumQ2_np, label="Q2 Loss", color="purple", alpha=0.7)
            axes[1].set_title("Training Loss")
            axes[1].set_xlabel("Training Step")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def appendReward(self,episode_rewards):

        
        self.rewardAccum.append(episode_rewards)
    
    def appendLoss(self,lossPolicy, lossQ1, lossQ2):
        self.lossAccumPolicy.append(lossPolicy.detach().cpu().numpy())
        self.lossAccumQ1.append(lossQ1.detach().cpu().numpy())
        self.lossAccumQ2.append(lossQ2.detach().cpu().numpy())


def main():
    q1_loss=0
    q2_loss=0
    actor_loss=0
    alpha_loss =0


    # Hyperparameters
    training_iterations = 0
    training_batch_size = 30
    n_iterations = 0
    threshold = 10000

    n_actions = 4  # 2 actions for Lunar Lander
    gamma = 0.99
    tau = 0.05  # Soft update coefficient
    batch_size = 256
    
    resume_training = True
    CHECKPOINT_DIR = '/tmp1'

    # Initialize networks
    visualizer =Visualizer(n_iterations,training_iterations)

    Policy = PolicyNetwork(8, 256, 256, n_actions, 1e-4,'PolicyTryNow31.pt')
    
    QNetwork_base1 = QNetwork(12, 256, 256, 1, 1e-4,'Q1NetTryNow31.pt')
    QNetwork_base2 = QNetwork(12, 256, 256, 1, 1e-4,'Q2NetTryNow31.pt')
    QNetwork_target1 = QNetwork(12, 256, 256, 1, 1e-4,'QT1NetTryNow31.pt')
    QNetwork_target2 = QNetwork(12, 256, 256, 1, 1e-4,'QT2NetTryNow31.pt')
    replay_buff = ReplayBuffer(max_size=1000000, input_shape=(8,), n_actions=n_actions)

    if resume_training:
        try:
            Policy.load_checkpoint()
            QNetwork_base1.load_checkpoint()
            QNetwork_base2.load_checkpoint()
            QNetwork_target2.load_checkpoint()
            QNetwork_target1.load_checkpoint()
            replay_buff.load("/tmp1")

            print("Loaded existing checkpoints and buffer!")
        except FileNotFoundError:
            print("No checkpoints found - starting fresh.")

    else:
        print("normal initialization")
    # Create copies of target networks and freeze their parameters
        QNetwork_target1.load_state_dict(QNetwork_base1.state_dict())
        QNetwork_target2.load_state_dict(QNetwork_base2.state_dict())
    
    for param in QNetwork_target1.parameters():
        param.requires_grad = False
    for param in QNetwork_target2.parameters():
        param.requires_grad = False
    
    # Tracking variables
    total_rewards = []
    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'log_probs': [],
    }
    
    # Environment setup
    env = gym.make("LunarLanderContinuous-v3")

    # Exploration phase
    print("\n--- Starting Exploration Phase ---")
    while n_iterations < threshold:
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_rewards = 0
        while not (terminated or truncated):
            
            observation_tensor = t.tensor(observation, dtype=t.float32)
            # print("observation_tensor shape", observation_tensor.shape)

            # Sample action
            action, action_log_prob = Policy.sampleAction(state=observation_tensor)

            # Convert action to NumPy for environment step
            action_numpy = action.detach().numpy()
            #TODO wtf, why do you do this? 
            # print("action numpy shape", action_numpy.shape)
            # Take step in environmentS
            observation_, reward, terminated, truncated, info = env.step(action_numpy)
                        #trying reward clipping..sigh far too brittle
            # print(observation)
            # Accumulate episode rewards
            episode_rewards += reward
            
            # Store transition
            replay_buff.store_transition(
                observation, 
                action_numpy, 
                reward, 
                observation_, 
                terminated or truncated
            )
            
            observation = observation_ 
            
        visualizer.appendReward(episode_rewards)
        total_rewards.append(episode_rewards)
        print(f"Exploration Iteration {n_iterations + 1}: Episode Reward = {episode_rewards:.2f}")
        n_iterations += 1
    
    # Training phase
    print("\n--- Starting Training Phase ---")
    while training_iterations < training_batch_size:
        print("wtf")
        # Ensure enough samples in replay buffer
        if replay_buff.size < 256:
            print("continuing")
            continue
        batch_count = 0

        while batch_count<batch_size:
        # Sample from replay buffer
            states, states_, actions, rewards, dones = replay_buff.sample_buffer(batch_size=1)

            # Convert to tensors
            states = t.tensor(states, dtype=t.float32)
            states_ = t.tensor(states_, dtype=t.float32)
            actions = t.tensor(actions, dtype=t.float32)
            rewards = t.tensor(rewards, dtype=t.float32)
            dones = t.tensor(dones, dtype=t.float32)

            # Critic (Q-network) update
            #the t.no_grads dont contribute to the gradient computation
            with t.no_grad():
                # Sample next actions
                next_actions, next_log_probs = Policy.sampleAction(states_)
                
                # Compute target Q-values
                target1_q = QNetwork_target1.forward(states_, next_actions)
                target2_q = QNetwork_target2.forward(states_, next_actions)
                target_q = t.min(target1_q, target2_q)
                # next_log_probs = next_log_probs.unsqueeze(-1)
                # rewards = rewards.unsqueeze(-1)
                # dones = dones.unsqueeze(-1)        
        
                # Compute target values
                y = rewards + gamma * (1 - dones) * (target_q - Policy.alpha * next_log_probs.detach())
                #detaching irrelevant calcs in the backprop update!
            # Current Q-values

                #PRINT DEBUG STATEMENTS

                # print(states.shape, states_.shape, rewards.shape, dones.shape,next_actions.shape,next_log_probs.shape, target_q.shape, y.shape)

            current_q1 = QNetwork_base1.forward(states, actions)
            current_q2 = QNetwork_base2.forward(states, actions)
            # print(current_q1,current_q2,target_q)
            # Q-network losses
            q1_loss += pow((y-current_q1),2)
            q2_loss += pow((y-current_q2),2)

            # Actor loss
            sampled_actions, log_probs = Policy.sampleAction(states)
            # print("action dim, log prob dim:", sampled_actions.shape, log_probs.shape)

            with t.no_grad():
                q1_vals = QNetwork_base1.forward(states, sampled_actions)
                q2_vals = QNetwork_base2.forward(states, sampled_actions)
                q_vals = t.min(q1_vals, q2_vals)
                # print("q_vals shape", q_vals.shape)
            
                #detaching irrelevant calcs in the backprop update!
            actor_loss += ((Policy.log_alpha.unsqueeze(0)).detach() * log_probs - q_vals)

            #loss_alpha = -a(log_pi(a|s)+ H)

                #detaching irrelevant calcs in the backprop update!

            alpha_loss += -((Policy.log_alpha.unsqueeze(0)) * (log_probs.detach() + Policy.target_entropy))
            # print("target entropy squozed shaped:", Policy.target_entropy.shape)
            # print("y output shape:",y.shape)
            # print("targ q shape", target_q.shape)
            # print("q base shape:", q_vals.shape)
            # print("actor log prob and q base shape",log_probs.shape,q_vals.shape)
            # print("actor loss shape",actor_loss.shape)
            # print("alpha shapes: ", (Policy.log_alpha.unsqueeze(0)).shape, (Policy.alpha.unsqueeze(0)).shape, alpha_loss.shape)
            batch_count+=1


        # if training_iterations % 10 == 0:
        #     print(f"Training Iteration: {training_iterations}")
        actor_loss/=256.0
        q2_loss/=256.0
        q1_loss/=256.0
        alpha_loss/=256.0

        # print("q1 and q2 loss", q1_loss.shape, q2_loss.shape)

        visualizer.appendLoss(actor_loss,q1_loss,q2_loss)

        training_losses['actor_loss'].append(actor_loss.item())
        training_losses['q1_loss'].append(q1_loss.item())
        training_losses['q2_loss'].append(q2_loss.item())
        training_losses['log_probs'].append(log_probs.item())

        QNetwork_base1.optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        t.nn.utils.clip_grad_norm_(QNetwork_base1.parameters(), max_norm=0.5)
        QNetwork_base1.optimizer.step()


        QNetwork_base2.optimizer.zero_grad()
        q2_loss.backward()
        t.nn.utils.clip_grad_norm_(QNetwork_base2.parameters(), max_norm=0.5)
        QNetwork_base2.optimizer.step()
        # Backward pass
        #clipping the gradients helps prevent exploding or diminished grad updates


        # Then policy
        Policy.optimizer.zero_grad()
        actor_loss.backward()
        t.nn.utils.clip_grad_norm_(Policy.parameters(), max_norm=1.0)
        Policy.optimizer.step()

        # Finally alpha
        Policy.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        t.nn.utils.clip_grad_norm_([Policy.log_alpha], max_norm=0.5)  
        Policy.alpha_optimizer.step()

        if target2_q<target1_q:
            soft_update(QNetwork_base2, QNetwork_target1, tau)
            soft_update(QNetwork_base2, QNetwork_target2, tau)
        
        else:
            soft_update(QNetwork_base1, QNetwork_target1, tau)
            soft_update(QNetwork_base1, QNetwork_target2, tau)
        
        Policy.alpha = Policy.log_alpha.exp()

        actor_loss=0
        q1_loss=0
        q2_loss=0
        alpha_loss=0
        training_iterations += 1

        # if np.mean(training_losses['actor_loss']) < 2 and np.mean(training_losses['q1_loss'])<20 and np.mean(training_losses['actor_loss']) <20:
        #     print("break!")
        #     print(np.mean(training_losses['actor_loss']), np.mean(training_losses['q1_loss']), np.mean(training_losses['q2_loss']) )
        #     break
    # Soft update of target networks

    # Logging

    # Print progress

    # Final training summary
    print("\n--- Training Phase Complete ---")
    print("Average Losses:")
    print(f"  Actor Loss: {np.mean(training_losses['actor_loss']):.4f}")
    print(f"  Q1 Network Loss: {np.mean(training_losses['q1_loss']):.4f}")
    print(f"  Q2 Network Loss: {np.mean(training_losses['q2_loss']):.4f}")
    print(f"Alpha: {Policy.alpha.item():.3f}, Entropy: {-log_probs.mean().item():.3f}")
    visualizer.plot()
    # Save networks
    Policy.save_checkpoint()
    QNetwork_base1.save_checkpoint()
    QNetwork_base2.save_checkpoint()
    QNetwork_target1.save_checkpoint()
    QNetwork_target2.save_checkpoint()
    replay_buff.save('/tmp1')
    print("Checkpoints saved.")

def soft_update(source, target, tau):
    """
    Soft update of the target network parameters.
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

if __name__ == "__main__":
    main()