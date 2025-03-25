from mushroom_rl.core import Agent

import os
import matplotlib
matplotlib.use("TkAgg")  # Or try "Qt5Agg" if TkAgg is not available
import matplotlib.pyplot as plt
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
from ModelsAndUtils import MLP, get_right_ankle_substate, get_action_substate


def main():
    # Initialize device with CPU fallback
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # Environment and expert agent setup
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True)

    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_88_original.msh")
    agent = Agent.load(agent_file_path)

    # Initialize the model
    input_dim = 12  # Number of features in the substate
    output_dim = 1  # Number of actions (scalar prediction)
    hidden_dim = 256  # Number of hidden units
    model = MLP(input_dim, hidden_dim, output_dim).to(device)
    model.train()

    # Optimizer and learning rate scheduler setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    total_steps_per_epoch = 8194   # Total training steps per epoch
    mini_batch_size = 64          # Steps to accumulate in one mini-batch
    num_mini_batches = total_steps_per_epoch // mini_batch_size
    epoch_losses = []
    num_epochs = 500
    early_stopping_patience = 50  # Adjust early stopping patience
    best_loss = float('inf')
    epochs_no_improve = 0

    # Define a path to save the best model
    model_save_path = os.path.join(os.path.dirname(__file__), "mlp_state_12_hidden_128_perfect_88.pth")
    state = mdp.reset()
    for epoch in range(num_epochs):
        # Reset environment at the beginning of each epoch
        # state = mdp.reset()
        epoch_loss = 0.0
        done = False

        for mb in range(num_mini_batches):
            inputs = []
            targets = []
            # Accumulate mini-batch data over 128 steps
            for step in range(mini_batch_size):
                if done:
                    print('expert falls')
                    state = mdp.reset()
                # Extract current substate and expert action
                current_substate = get_right_ankle_substate(state)
                expert_action = agent.draw_action(state)
                target_value = get_action_substate(expert_action)

                inputs.append(current_substate)
                targets.append(target_value)

                # Step environment using the expert action
                next_state, reward, done, _ = mdp.step(expert_action)
                state = next_state  # Update state for next step
                # Optionally, handle 'done' if needed:
                # if done:
                #     state = mdp.reset()

            # Convert lists to tensors with proper batch dimensions:
            # inputs: [mini_batch_size, 22]
            input_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
            # targets: [mini_batch_size]; model outputs shape [mini_batch_size, 1] then squeezed
            target_tensor = torch.tensor(targets, dtype=torch.float32).to(device).squeeze(1)

            # Forward pass on the mini-batch
            model_output = model(input_tensor)  # Expected shape: [mini_batch_size, 1]
            model_output = model_output.squeeze(1)  # Now shape: [mini_batch_size]

            # Compute MSE loss over the mini-batch
            loss = F.mse_loss(model_output, target_tensor)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate loss (weighted by mini-batch size)
            epoch_loss += loss.item() * mini_batch_size

        # Compute average loss over the entire epoch
        average_epoch_loss = epoch_loss / total_steps_per_epoch
        epoch_losses.append(average_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}: Average Loss: {average_epoch_loss:.6f}, LR: {current_lr}")

        # Step the scheduler based on the average epoch loss
        scheduler.step(average_epoch_loss)

        # Checkpointing: save model if current epoch loss outperforms previous best
        if average_epoch_loss < best_loss:
            best_loss = average_epoch_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Improved performance: saving model at epoch {epoch + 1} with loss {best_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Plot the training loss over epochs
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss After Each Epoch')
    plt.show()

    print(f"Best model saved to {model_save_path}")


if __name__ == '__main__':
    main()
