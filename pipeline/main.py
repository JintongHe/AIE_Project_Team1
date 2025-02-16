from mushroom_rl.core import Core, Agent
# from loco_mujoco import LocoEnv
#
#
# env = LocoEnv.make("HumanoidTorque.walk.perfect")
#
# agent = Agent.load("loco-mujoco/logs/loco_mujoco_evalution_2025-02-09_21-29-47/env_id___HumanoidTorque.walk.perfect/0/agent_epoch_49_J_959.235794.msh")
#
# core = Core(agent, env)
#
# core.evaluate(n_episodes=10, render=True)
#
# env.play_trajectory_from_velocity(n_steps_per_episode=500)
import matplotlib.pyplot as plt
import os
import sys
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from imitation_learning.utils import get_agent

#1)
def get_right_ankle_substate(state):
    # Indices of right ankle related features in the observation space
    right_ankle_indices = [
        10,  # q_ankle_angle_r
        29   # dq_ankle_angle_r
    ]
    
    # Extract the substate with only the right ankle related features
    right_ankle_substate = state[right_ankle_indices]
    
    return right_ankle_substate

def get_action_substate(action):
    # Index of the action related feature
    action_index = 7
    
    # Extract the substate with only the action related feature
    action_substate = action[action_index]
    
    return action_substate

#2)
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the humanoid environment without left ankle movement
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True)

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
agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_495_J_642.787921.msh")
agent = Agent.load(agent_file_path)
# core = Core(agent, mdp)
# dataset = core.evaluate(n_episodes=1000, render=True)
# Reset the environment
state = mdp.reset()
input_dim = 2  # Number of features in the substate
output_dim = 1  # Number of actions
model = SimpleModel(input_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 50
epoch_losses = []
num_epochs = 5000
# #1) Extract ankle features
# state = mdp.reset()
# right_ankle_substate = get_right_ankle_substate(state)
# print("Right Ankle Substate:", right_ankle_substate)

# #2)create model that takes in state and outputs action

# #3) fead ankle features into the model and get action
# right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32)
# action = model(right_ankle_substate_tensor)
# print("Model Action:", action.item())

# #4) calculate loss from difference in actions
# expert_action_state = agent.draw_action(state)
# expert_action = expert_action_state[10]
# expert_action_tensor = torch.tensor(expert_action, dtype=torch.float32)
# loss = F.mse_loss(action, expert_action_tensor)
# print("Loss:", loss.item())
# #4) Replace expert action with model action
# expert_action_state[10] = action.item()
# #5) Step the environment 

# Run evaluation for 1000 episodes
for epoch in range(num_epochs):
    state = mdp.reset()  # Reset the environment for each episode
    right_ankle_substate = get_right_ankle_substate(state)
    done = False
    step = 0
    batch_loss = 0.0

    while not done:
        # Get action from expert
        action = agent.draw_action(state)
        right_ankle_action = get_action_substate(action)
        #Get action from ankle model
        right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32)
        model_action = model(right_ankle_substate_tensor)

        # Calculate loss
        right_ankle_action_tensor = torch.tensor(right_ankle_action, dtype=torch.float32)
        loss = F.mse_loss(model_action, right_ankle_action_tensor)
        batch_loss += loss.item()

        #accumulate gradients
        loss.backward()

        # Perform optimization step every batch_size steps
        if (step + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            #print(f"Batch {step // batch_size + 1} completed with average loss: {batch_loss / batch_size}")
            batch_loss = 0.0

        #replace expert action with model action
        right_ankle_action = model_action.item()
        action[7] = right_ankle_action

        # Print state and action values
        # print(f"Episode {episode + 1}, Step {step + 1}")
        # print(f"State: {state}")
        # print(f"Action: {action}")

        # Take action in environment
        next_state, reward, done, _ = mdp.step(action)

        # 🔹 Render the environment at every step
        #mdp.render()

        # Update state
        state = next_state
        right_ankle_substate = get_right_ankle_substate(state)
        step += 1
    epoch_losses.append(batch_loss)
    print(f"Episode {epoch + 1} completed with loss: {loss.item()}")

# Plot the loss after each epoch
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss After Each Epoch')
plt.show()