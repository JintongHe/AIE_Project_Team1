import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
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
    relevant_indices = [
        0,  # q_pelvis_tx
        1,  # q_pelvis_tz
        2,  # q_pelvis_ty
        3,  # q_pelvis_tilt
        4,  # q_pelvis_list
        5,  #q_pelvis_rotation
        6,  #q_hip_flexion_r
        7,  #q_hip_adduction_r
        8,  #q_hip_rotation_r
        9,  #q_knee_angle_r
        10,  #q_ankle_angle_r
        19,  # dq_pelvis_tx
        20,  # dq_pelvis_tz
        21,  # dq_pelvis_ty
        22,  # dq_pelvis_tilt
        23,  # dq_pelvis_list
        24,  #dq_pelvis_rotation
        25,  #dq_hip_flexion_r
        26,  #dq_hip_adduction_r
        27,  #dq_hip_rotation_r
        28,  #dq_knee_angle_r
        29,  #dq_ankle_angle_r
    ]
    right_ankle_substate = state[relevant_indices]
    return right_ankle_substate

def get_action_substate(action):
    # Indices of right ankle related features in the observation space
    right_ankle_indices = [
        7,  # q_ankle_angle_r
    ]
    
    right_ankle_substate = action[right_ankle_indices]
    
    return right_ankle_substate

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state):
        state = state.to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the learning rate for the critic
        self.critic_scheduler.step(critic_loss)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the learning rate for the actor
            self.actor_scheduler.step(actor_loss)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
#Transformer model
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
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, dim_feedforward=128):
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
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x