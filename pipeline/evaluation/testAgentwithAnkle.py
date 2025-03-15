import os
import torch
import torch.nn.functional as F
from loco_mujoco import LocoEnv
from mushroom_rl.core import Core, Agent
from ModelsAndUtils import Actor, TransformerModel, get_right_ankle_substate, get_action_substate  # Import necessary functions and classes


# Initialize the humanoid environment
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True)

# Load the expert agent
agent_file_path = os.path.join(os.path.dirname(__file__), "best_real_agent_141.msh")
agent = Agent.load(agent_file_path)

# Load the model
input_dim = 22  # Number of features in the substate
hidden_dim = 64
output_dim = 1  # Number of actions
model = Actor(input_dim, output_dim, 0.4)
model_load_path = os.path.join(os.path.dirname(__file__), "es_actor.pth")
model.load_state_dict(torch.load(model_load_path))
model.eval()
print(f"Model weights loaded from {model_load_path}")

# Perform rollouts
num_episodes = 100
total_steps = 0
for episode in range(num_episodes):
    state = mdp.reset()
    done = False
    step = 0
    while not done:
        # Extract right ankle substate
        right_ankle_substate = get_right_ankle_substate(state)
        right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Get action from the model
        model_action = model(right_ankle_substate_tensor).squeeze().item()

        if model_action < min_action:
            min_action = model_action
        if model_action > max_action:
            max_action = model_action
        
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
    total_steps += step
    print(f"Episode {episode + 1} completed with {step} steps")
    print(f"Min action: {min_action}, Max action: {max_action}")

print(f"Average steps per episode: {total_steps / num_episodes}")
