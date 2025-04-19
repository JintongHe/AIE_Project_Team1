import os
import torch
import torch.nn.functional as F
from loco_mujoco import LocoEnv
from mushroom_rl.core import Core, Agent
from ModelsAndUtils import MLP, TransformerModel, get_right_ankle_substate, get_action_substate
import keyboard  # Added for key press detection
import time

# Check if CUDA is available, otherwise fallback to CPU
device = "cuda"
print(f"Using device: {device}")

# Initialize the humanoid environment
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True)

# Load the expert agent
agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_140_prosthesis_inertia.msh")
agent = Agent.load(agent_file_path)

# Load the model
input_dim = 16  # Number of features in the substate
hidden_dim = 128
output_dim = 1  # Number of actions

model = MLP(input_dim, hidden_dim, output_dim)
model_load_path = os.path.join(os.path.dirname(__file__), "mlp_state_16_hidden_128_prosthesis.pth")
model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cuda')))
model.eval()
print(f"Model weights loaded from {model_load_path}")

# Perform rollouts
num_episodes = 10
total_steps = 0
for episode in range(num_episodes):
    state = mdp.reset()
    done = False
    step = 0
    while not done:
        # Check for key press: if 'r' is pressed, break out of the loop to reset the environment.
        if keyboard.is_pressed('r'):
            print("User pressed 'r', resetting environment for next rollout.")
            while keyboard.is_pressed('r'):
                time.sleep(0.1)
            break

        # Create the right ankle substate tensor (here we simply use the full state tensor)
        right_ankle_substate = get_right_ankle_substate(state, input_dim)
        right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Get action from the model
        model_action = model(right_ankle_substate_tensor).squeeze().item()

        # Get the expert action and override the right ankle control (index 7)
        action = agent.draw_action(state)
        vail_agent_action = action[7]
        mlp_agent_action = model_action

        # if step <= 3000:
        action[7] = model_action

        # Take the action in the environment
        next_state, reward, done, _ = mdp.step(action)
        mdp.render()

        # Update the state and step count
        state = next_state
        step += 1
    total_steps += step
    print(f"Episode {episode + 1} completed with {step} steps")

print(f"Average steps per episode: {total_steps / num_episodes}")
