from mushroom_rl.core import Core, Agent
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import os
import sys
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import sys
from imitation_learning.utils import get_agent
import math


#Extract ankle features from state
def get_right_ankle_substate(state):
    # Indices of right ankle related features in the observation space
    #right_ankle_indices = [
    #    10,  # q_ankle_angle_r
    #    29   # dq_ankle_angle_r
    #]
    
    #right_ankle_substate = state[right_ankle_indices]
    
    #return right_ankle_substate
    return state

#Extract ankle action from action
def get_action_substate(action):
    # Index of the action related feature
    action_index = 7
    
    # Extract the substate with only the action related feature
    action_substate = action[action_index]
    
    return action_substate

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



def main():
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True)

    print("Observation Space:", mdp.info.observation_space.low)
    print("Observation Space Shape:", mdp.info.observation_space.shape)
    print("Observation Space:", mdp.info.observation_space.low)
    print("Observation Space Shape:", mdp.info.observation_space.shape)

    print("Observation Variables:")
    for obs in mdp.get_all_observation_keys():
        print(obs)
    print("Action Space:", mdp.info.action_space)
    print("Action Space Shape:", mdp.info.action_space.shape)
    print("Observation Variables:")
    for obs in mdp.get_all_observation_keys():
        print(obs)
    print("Action Space:", mdp.info.action_space)
    print("Action Space Shape:", mdp.info.action_space.shape)


    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_423_J_991.255877.msh")
    agent = Agent.load(agent_file_path)


    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    else: 
        device = torch.device("cpu")

    #Initialize the model
    state = mdp.reset()
    input_dim = 36  # Number of features in the substate
    output_dim = 1  # Number of actions
    model = TransformerModel(input_dim, output_dim).to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    batch_size = 1000
    epoch_losses = []
    num_epochs = 200

    # Run evaluation for 1000 episodes
    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        done = False
        step = 0
        batch_loss = 0.0
        epoch_loss = 0.0
        total_reward = 0.0

        while not done:
            print(f"Epoch {epoch + 1}, Step {step + 1}")
            # Get action from expert
            action = agent.draw_action(state)
            right_ankle_action = get_action_substate(action)
            #Get action from ankle model
            right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            model_action = model(right_ankle_substate_tensor).squeeze()

            # Calculate loss imitating expert action
            right_ankle_action_tensor = torch.tensor(right_ankle_action, dtype=torch.float32).to(device)
            loss = F.mse_loss(model_action, right_ankle_action_tensor)
            batch_loss += loss.item()
            epoch_loss += loss.item()

            #replace expert action with model action
            # right_ankle_action = model_action.item()
            # action[7] = right_ankle_action

            # Take action in environment
            next_state, reward, done, _ = mdp.step(action)
            total_reward += reward

            #accumulate gradients
            loss.backward()

            # Perform optimization step every batch_size steps
            if (step + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"Batch {step // batch_size + 1} completed with average loss: {batch_loss / batch_size}")
                batch_loss = 0.0

            
            # 🔹 Render the environment at every step
            #mdp.render()

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1
        average_epoch_loss = epoch_loss/(step+1)
        epoch_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1} completed with average loss: {average_epoch_loss}")

    # Plot the loss after each epoch
    plt.plot(range(1, num_epochs + 1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss After Each Epoch')
    plt.show()

    # Save the model weights
    model_save_path = os.path.join(os.path.dirname(__file__), "right_ankle_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

if __name__ == '__main__':
    main()
            