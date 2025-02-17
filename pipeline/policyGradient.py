from mushroom_rl.core import Core, Agent
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from imitation_learning.utils import get_agent
import math

#Reward function
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



#Extract ankle action from action
def get_right_ankle_substate(state):
    """Extract relevant features for ankle control"""
    return np.concatenate([
        state[10:11],   # q_ankle_angle_r
        state[29:30],   # dq_ankle_angle_r
        state[18:19],   # dq_pelvis_tx (forward velocity)
        state[3:4],     # q_pelvis_tilt
        state[4:5],     # q_pelvis_list
        state[34:35]    # dq_ankle_angle_l for symmetry
    ])

#Model for predicting right ankle action
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=3, num_layers=2, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_dim, input_dim)
        self.decoder = nn.Linear(input_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output[-1]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def main():
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True, reward_type="custom", reward_params=dict(reward_callback=ankle_training_reward))

    print("Observation Space:", mdp.info.observation_space.low)
    print("Observation Space Shape:", mdp.info.observation_space.shape)

    print("Observation Variables:")
    for obs in mdp.get_all_observation_keys():
        print(obs)
    print("Action Space:", mdp.info.action_space)
    print("Action Space Shape:", mdp.info.action_space.shape)

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    sw = None  # TensorBoard logging can be added later
    # agent = get_agent(env_id, mdp, use_cuda, sw, conf_path="imitation_learning/confs.yaml")

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_423_J_991.255877.msh")
    agent = Agent.load(agent_file_path)
    # core = Core(agent, mdp)
    # dataset = core.evaluate(n_episodes=1000, render=True)


    # Reset the environment

    # Initialize the model
    input_dim = 6  # Number of features in the substate
    output_dim = 1  # Number of actions
    model = TransformerModel(input_dim, output_dim)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.99
    epoch_rewards = []
    num_epochs = 1000


    # Run evaluation for 1000 episodes
    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        done = False
        step = 0
        epoch_reward = 0.0
        log_probs = []
        rewards = []

        while not done:
            print(f"Epoch {epoch + 1}, Step {step + 1}")
            # Get action from expert
            action = agent.draw_action(state)
            # Get action from ankle model
            right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            model_action = model(right_ankle_substate_tensor).squeeze()

            # Replace expert action with model action
            right_ankle_action = model_action.item()
            action[7] = right_ankle_action

            # Take action in environment
            next_state, reward, done, _ = mdp.step(action)
            epoch_reward += reward

            # Calculate log probability of the action
            log_prob = torch.log(model_action)
            log_probs.append(log_prob)
            rewards.append(reward)

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate policy gradient loss
        policy_gradient_loss = torch.stack(log_probs) * discounted_rewards
        policy_gradient_loss = -policy_gradient_loss.sum()

        # Perform optimization step
        optimizer.zero_grad()
        policy_gradient_loss.backward()
        optimizer.step()

        epoch_rewards.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed with total reward: {epoch_reward}")

    # Plot the rewards after each epoch
    plt.plot(range(1, num_epochs + 1), epoch_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward After Each Epoch')
    plt.show()

    # Save the model weights
    # model_save_path = os.path.join(os.path.dirname(__file__), "right_ankle_model.pth")
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model weights saved to {model_save_path}")

if __name__ == '__main__':
    main()