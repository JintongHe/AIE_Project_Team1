import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import keyboard
import threading
from threading import Lock
from queue import Queue
from loco_mujoco import LocoEnv
from mushroom_rl.core import Agent
from ModelsAndUtils import MLP, get_right_ankle_substate

# Check for available device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# Policy Network for RL model
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)

        # Mean output for continuous actions
        self.mean = torch.nn.Linear(32, action_dim)
        # Log standard deviation network
        self.logstd = torch.nn.Linear(32, action_dim)

        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = F.tanh(self.mean(x))
        std = torch.exp(torch.clamp(self.logstd(x), -9, 0.5))
        return mean, std

    def get_best_action(self, state):
        """Deterministic action selection for testing"""
        mean, _ = self.forward(state)
        return mean


def load_models():
    # Initialize the MLP model (IL)
    input_dim = 36  # Number of features in the substate
    hidden_dim = 128
    output_dim = 1  # Number of actions

    il_model = MLP(input_dim, hidden_dim, output_dim).to(device)
    il_model_path = os.path.join(os.path.dirname(__file__), "mlp_state_36_hidden_128_prosthesis.pth")
    il_model.load_state_dict(torch.load(il_model_path, map_location=torch.device(device)))
    il_model.eval()
    print(f"IL Model loaded from {il_model_path}")

    # Initialize the Policy model (RL)
    rl_model = PolicyNet(input_dim, output_dim).to(device)
    rl_model_path = os.path.join(os.path.dirname(__file__), "new_36_states_survival_pros.pth")
    rl_model.load_state_dict(torch.load(rl_model_path, map_location=torch.device(device)))
    rl_model.eval()
    print(f"RL Model loaded from {rl_model_path}")

    return il_model, rl_model


# Global variables for thread coordination
stop_threads = False
reset_now = False  # Changed from reset_requested to reset_now
print_lock = Lock()


def run_single_rollout(model_type, model, agent, num_episodes=1000, max_steps=10000, thread_id=0):
    """
    Run a single rollout for either IL or RL model

    Args:
        model_type (str): "il" or "rl"
        model: The neural network model
        agent: The expert agent
        num_episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
        thread_id (int): ID of the thread for printing
    """
    global stop_threads, reset_now

    # Initialize environment
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True)

    total_steps = 0

    for episode in range(num_episodes):
        # Check if we need to stop
        if stop_threads:
            break

        # Reset environment
        state = mdp.reset()
        done = False
        step = 0

        with print_lock:
            print(f"[{model_type.upper()}] Episode {episode + 1} starting.")

        while not done and step < max_steps:
            # Check if we need to stop
            if stop_threads:
                break

            # Check if we need to reset the environment
            if reset_now:
                with print_lock:
                    print(f"[{model_type.upper()}] Resetting environment.")
                state = mdp.reset()
                if thread_id == 0:  # Only reset the flag once (from the first thread)
                    reset_now = False
                continue

            # Get right ankle substate
            ankle_substate = get_right_ankle_substate(state, 36)
            ankle_tensor = torch.tensor(ankle_substate, dtype=torch.float32).to(device)

            # Get expert action
            expert_action = agent.draw_action(state)

            # Get model action based on model type
            if model_type == "il":
                # Prepare tensor for IL model
                ankle_tensor = ankle_tensor.unsqueeze(0).unsqueeze(0)
                model_action = model(ankle_tensor).squeeze().item()
            else:  # rl
                model_action = model.get_best_action(ankle_tensor).item()

            # Override the right ankle control
            expert_action[7] = model_action

            # Take action in environment
            next_state, reward, done, _ = mdp.step(expert_action)
            mdp.render()

            # Update state
            state = next_state
            step += 1

        if not stop_threads:
            total_steps += step
            with print_lock:
                print(f"[{model_type.upper()}] Episode {episode + 1} completed with {step} steps")

    if not stop_threads:
        with print_lock:
            print(f"[{model_type.upper()}] Average steps per episode: {total_steps / max(1, episode)}")

    # Close environment
    mdp.close()


def key_monitor():
    """Monitor keyboard inputs for both threads"""
    global stop_threads, reset_now

    while not stop_threads:
        if keyboard.is_pressed('q'):
            with print_lock:
                print("User pressed 'q', quitting all simulations.")
            stop_threads = True
            # Wait to ensure key press is released
            while keyboard.is_pressed('q'):
                time.sleep(0.1)
            break

        if keyboard.is_pressed('r'):
            with print_lock:
                print("User pressed 'r', resetting all environments.")
            reset_now = True
            # Wait to ensure key press is released
            while keyboard.is_pressed('r'):
                time.sleep(0.1)

        time.sleep(0.05)  # Small sleep to reduce CPU usage


def run_rollouts(mode="compare", num_episodes=1000, max_steps=10000):
    """
    Run rollouts with different modes using multi-threading

    Args:
        mode (str):
            - "il" for imitation learning only
            - "rl" for reinforcement learning only
            - "compare" for running both IL and RL in parallel
            - "switch" for toggling between IL and RL with the spacebar
        num_episodes (int): Number of episodes to run
        max_steps (int): Maximum steps per episode
    """
    global stop_threads, reset_now
    stop_threads = False
    reset_now = False

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_140_prosthesis_inertia.msh")
    agent = Agent.load(agent_file_path)

    # Load models
    il_model, rl_model = load_models()

    threads = []

    if mode == "compare":
        # Create threads for both IL and RL
        il_thread = threading.Thread(
            target=run_single_rollout,
            args=("il", il_model, agent, num_episodes, max_steps, 0)
        )

        rl_thread = threading.Thread(
            target=run_single_rollout,
            args=("rl", rl_model, agent, num_episodes, max_steps, 1)
        )

        # Start key monitoring thread
        key_thread = threading.Thread(target=key_monitor)
        key_thread.daemon = True
        key_thread.start()

        # Start rollout threads
        il_thread.start()
        rl_thread.start()

        threads = [il_thread, rl_thread]

    elif mode == "il":
        # Create thread for IL only
        il_thread = threading.Thread(
            target=run_single_rollout,
            args=("il", il_model, agent, num_episodes, max_steps, 0)
        )

        # Start key monitoring thread
        key_thread = threading.Thread(target=key_monitor)
        key_thread.daemon = True
        key_thread.start()

        # Start rollout thread
        il_thread.start()

        threads = [il_thread]

    elif mode == "rl":
        # Create thread for RL only
        rl_thread = threading.Thread(
            target=run_single_rollout,
            args=("rl", rl_model, agent, num_episodes, max_steps, 0)
        )

        # Start key monitoring thread
        key_thread = threading.Thread(target=key_monitor)
        key_thread.daemon = True
        key_thread.start()

        # Start rollout thread
        rl_thread.start()

        threads = [rl_thread]

    elif mode == "switch":
        print("Switch mode is not compatible with multi-threading. Using 'compare' mode instead.")
        return run_rollouts("compare", num_episodes, max_steps)

    # Wait for all threads to complete
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping all threads.")
        stop_threads = True

    print("All rollouts completed.")


if __name__ == "__main__":
    # Parse command line arguments or set defaults
    import argparse

    parser = argparse.ArgumentParser(description='Run combined IL and RL rollouts with multi-threading')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['il', 'rl', 'compare', 'switch'],
                        help='Mode to run the rollouts (il, rl, compare, switch)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=10000,
                        help='Maximum steps per episode')
    args = parser.parse_args()

    print(f"Starting multi-threaded rollouts with {args.episodes} episodes, {args.max_steps} max steps per episode...")
    print("Press 'q' to quit all simulations")
    print("Press 'r' to reset all simulations")

    run_rollouts(mode=args.mode, num_episodes=args.episodes, max_steps=args.max_steps)