import os
import torch
import torch.nn.functional as F
from loco_mujoco import LocoEnv
from mushroom_rl.core import Core, Agent
from ImitationLearning import TransformerModel, get_right_ankle_substate, get_action_substate  # Import necessary functions and classes


# Initialize the humanoid environment
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True)

# Load the expert agent
agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_423_J_991.255877.msh")
agent = Agent.load(agent_file_path)

# Load the model
input_dim = 36  # Number of features in the substate
output_dim = 1  # Number of actions
model = TransformerModel(input_dim, output_dim)
model_load_path = os.path.join(os.path.dirname(__file__), "bestIL.pth")
model.load_state_dict(torch.load(model_load_path))
model.eval()
print(f"Model weights loaded from {model_load_path}")

# Perform rollouts
num_episodes = 10
for episode in range(num_episodes):
    state = mdp.reset()
    done = False
    step = 0
    while not done:
        # Extract right ankle substate
        right_ankle_substate = state
        right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Get action from the model
        model_action = model(right_ankle_substate_tensor).squeeze().item()
        
        # Replace the expert action with the model action
        action = agent.draw_action(state)
        action[7] = model_action
        
        # Take action in the environment
        next_state, reward, done, _ = mdp.step(action)
        
        # Render the environment
        mdp.render()
        
        # Update state
        state = next_state
        step += 1
    print(f"Episode {episode + 1} completed with {step} steps")
