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
    # return np.concatenate([
    #     state[10:11],   # q_ankle_angle_r
    #     state[29:30],   # dq_ankle_angle_r
    #     state[18:19],   # dq_pelvis_tx (forward velocity)
    #     state[3:4],     # q_pelvis_tilt
    #     state[4:5],     # q_pelvis_list
    #     state[34:35]    # dq_ankle_angle_l for symmetry
    # ])
    return state

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

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
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
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

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
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
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )


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
    device = torch.device("cuda" if use_cuda else "cpu")
    sw = None  # TensorBoard logging can be added later
    # agent = get_agent(env_id, mdp, use_cuda, sw, conf_path="imitation_learning/confs.yaml")

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_423_J_991.255877.msh")
    agent = Agent.load(agent_file_path)
    # core = Core(agent, mdp)
    # dataset = core.evaluate(n_episodes=1000, render=True)


    # Reset the environment

    # Initialize the model
    input_dim = 36  # Number of features in the substate
    output_dim = 1  # Number of actions
    max_action = 100.0
    model = TD3(input_dim, output_dim, max_action)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(input_dim, output_dim)

    # Initialize the optimizer
    epoch_rewards = []
    num_epochs = 1000
    batch_size = 100


    # Run evaluation for 1000 episodes
    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        done = False
        step = 0
        epoch_reward = 0.0

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

            # Store transition in replay buffer
            replay_buffer.add(right_ankle_substate, model_action, get_right_ankle_substate(next_state), reward, done)

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1

            # Train TD3 model
            if replay_buffer.size > batch_size:
                model.train(replay_buffer, batch_size)

        epoch_rewards.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed with total reward: {epoch_reward} and total steps: {step}")

        

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