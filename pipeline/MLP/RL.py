from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelsAndUtils import MLP, get_right_ankle_substate, get_action_substate


def main():
    #Initialize device
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    else: 
        device = torch.device("cpu")

    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.real"
    mdp = LocoEnv.make(env_id, use_box_feet=True)

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "real_180.msh")
    agent = Agent.load(agent_file_path)

    #Initialize the model
    input_dim = 22  # Number of features in the substate
    output_dim = 1  # Number of actions
    hidden_dim = 64  # Number of hidden units
    model = MLP(input_dim, hidden_dim, output_dim).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 100
    epoch_losses = []
    num_epochs = 500
    patience = 1000  # Number of epochs to wait for improvement
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        step = 0
        batch_loss = 0.0
        epoch_loss = 0.0
        total_reward = 0.0

        # reset gradients
        optimizer.zero_grad()

        for _ in range(batch_size):

            # Get the expert action
            action = agent.draw_action(state)
            right_ankle_action = get_action_substate(action)

            # Get the model prediction
            right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            model_action = model(right_ankle_substate_tensor).squeeze()

            # Take action in environment
            next_state, reward, done, _ = mdp.step(action)
            total_reward += reward

            # Use negative reward as the loss
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            total_loss = -reward_tensor  # Negative reward to maximize it

            # Accumulate gradients
            total_loss.backward()

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1

            # Accumulate losses
            batch_loss += total_loss.item()
            epoch_loss += total_loss.item()

        # Perform optimization step after each batch
        optimizer.step()
        optimizer.zero_grad()  # Reset gradients after each batch

        average_epoch_loss = epoch_loss / batch_size
        epoch_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1} completed with average loss: {average_epoch_loss}")

        # Early stopping check
        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        


#Plot the loss after each epoch
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss After Each Epoch')
    plt.show()

    # Save the model weights
    model_save_path = os.path.join(os.path.dirname(__file__), "Simple.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

if __name__ == '__main__':
    main()