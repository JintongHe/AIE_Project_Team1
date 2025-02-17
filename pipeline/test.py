from mushroom_rl.core import Core, Agent
import matplotlib.pyplot as plt
import os
import numpy as np
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import random

# Environment setup with custom reward
def ankle_training_reward(state, action, next_state):
    """Composite reward function for ankle coordination training"""
    # State indices based on provided observation variables
    q_pelvis_tx, q_pelvis_tz, q_pelvis_ty = state[0], state[1], state[2]
    q_pelvis_tilt, q_pelvis_list = state[3], state[4]
    dq_pelvis_tx = state[18]  # Forward velocity
    ankle_torque = action[7]  # TD3-generated ankle control

    # -------------------------------------------------
    # 1. Forward Motion Incentive (40% weight)
    # -------------------------------------------------
    target_velocity = 1.5  # m/s
    velocity_ratio = dq_pelvis_tx / target_velocity
    r_forward = 0.4 * np.clip(velocity_ratio, 0, 1)

    # -------------------------------------------------
    # 2. Stability Components (30% weight)
    # -------------------------------------------------
    # Torso orientation penalty
    tilt_penalty = 0.15 * np.abs(q_pelvis_tilt)
    list_penalty = 0.15 * np.abs(q_pelvis_list)
    
    # COM height maintenance
    target_height = 0.9  # Adjust based on your agent's normal standing height
    height_penalty = 0.1 * np.abs(q_pelvis_ty - target_height)
    
    r_stability = 0.3 - (tilt_penalty + list_penalty + height_penalty)

    # -------------------------------------------------
    # 3. Ankle-Specific Rewards (20% weight)
    # -------------------------------------------------
    # Torque efficiency penalty
    torque_penalty = 0.05 * np.square(ankle_torque)
    
    # Foot contact stability (simplified)
    contact_penalty = 0.15 if q_pelvis_tz < 0.05 else 0  # Approximate foot-ground contact
    
    r_ankle = 0.2 - (torque_penalty + contact_penalty)

    # -------------------------------------------------
    # 4. Survival Bonus (10% weight)
    # -------------------------------------------------
    survival_bonus = 0.1 if np.abs(q_pelvis_tilt) < 0.26 and np.abs(q_pelvis_list) < 0.26 else 0  # ~15 degrees

    # -------------------------------------------------
    # Total Reward Calculation
    # -------------------------------------------------
    return r_forward + r_stability + r_ankle + survival_bonus



def get_ankle_substate(state):
    """Extract relevant features for ankle control"""
    return np.concatenate([
        state[10:11],   # q_ankle_angle_r
        state[29:30],   # dq_ankle_angle_r
        state[18:19],   # dq_pelvis_tx (forward velocity)
        state[3:5],     # q_pelvis_tilt/list
        state[34:35]    # dq_ankle_angle_l for symmetry
    ])


# TD3 Network Architecture for Ankle Control
class AnkleActor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)

class AnkleCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim+1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state, action):
        while action.dim() < state.dim():
            action = action.unsqueeze(-1)
        while action.dim() > state.dim():
            action = action.squeeze(-1)
        return self.net(torch.cat([state, action], dim=1))

# TD3 Agent for Ankle Training
class AnkleTD3:
    def __init__(self, input_dim, device='cpu'):
        self.actor = AnkleActor(input_dim).to(device)
        self.actor_target = deepcopy(self.actor)
        self.critic1 = AnkleCritic(input_dim).to(device)
        self.critic2 = AnkleCritic(input_dim).to(device)
        self.critic_target1 = deepcopy(self.critic1)
        self.critic_target2 = deepcopy(self.critic2)
        
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=3e-4
        )
        
        self.replay_buffer = []
        self.batch_size = 256
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.device = device
        self.total_it = 0

    def select_action(self, state, noise_scale=0.1):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)  # Add batch dimension if missing
            action = self.actor(state).cpu().numpy()
            noise = noise_scale * np.random.normal(size=action.shape)
            return np.clip(action + noise, -1, 1)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch and properly stack tensors
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Extract and stack each component
        states = torch.stack([t[0] for t in batch]).to(self.device)
        actions = torch.stack([t[1] for t in batch]).squeeze(-1).to(self.device)  # Remove extra dimension
        rewards = torch.stack([t[2] for t in batch]).to(self.device)
        next_states = torch.stack([t[3] for t in batch]).to(self.device)
        dones = torch.stack([t[4] for t in batch]).to(self.device)

        # TD3 Update Logic
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            target_Q1 = self.critic_target1(next_states, next_actions)
            target_Q2 = self.critic_target2(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * 0.99 * target_Q

        # Critic Update
        current_Q1 = self.critic1(states, actions)
        current_Q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Delayed Policy Update
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            
            # Update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic1.parameters(), self.critic_target1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic_target2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.total_it += 1

def main():

    # Environment setup
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True, reward_type="custom",
                      reward_params=dict(reward_callback=ankle_training_reward))

    # Load expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_423_J_991.255877.msh")
    expert_agent = Agent.load(agent_file_path)
    
    # TD3 initialization for ankle control
      # Ankle-related features
    ankle_agent = AnkleTD3(input_dim=6, device='cuda' if torch.cuda.is_available() else 'mps')
    
    
    for epoch in range(200):
        state = mdp.reset()
        total_reward = 0
        done = False
        
        while not done:
            expert_action = expert_agent.draw_action(state)
            ankle_state = get_ankle_substate(state)
            
            # TD3-generated ankle action with exploration
            ankle_action = ankle_agent.select_action(ankle_state, noise_scale=0.3)
            expert_action[7] = ankle_action.item()  # Right ankle control index

            next_state, reward, done, _ = mdp.step(expert_action)
            
            # Store transition with ankle-specific substate
            ankle_agent.replay_buffer.append((
                torch.FloatTensor(ankle_state),
                torch.FloatTensor([ankle_action]),
                torch.FloatTensor([reward]),
                torch.FloatTensor(get_ankle_substate(next_state)),
                torch.FloatTensor([float(done)])
            ))

            # Prioritized update every 4 steps
            if len(ankle_agent.replay_buffer) >= ankle_agent.batch_size and len(ankle_agent.replay_buffer) % 4 == 0:
                ankle_agent.update()

            total_reward += reward
            state = next_state

        print(f"Epoch {epoch+1}, Total Reward: {total_reward:.2f}")
    # Plotting and model saving logic
    # plt.plot(epoch_rewards)
    # plt.xlabel('Epoch')
    # plt.ylabel('Total Reward')
    # plt.title('Ankle TD3 Training Progress')
    # plt.show()

if __name__ == '__main__':
    main()
