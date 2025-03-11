from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelsAndUtils import PPO, get_right_ankle_substate, get_action_substate

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

    # Initialize PPO agent
    state_dim = 22  # Number of features in the substate
    action_dim = 1  # Number of actions
    ppo_agent = PPO(state_dim, action_dim)

    num_episodes = 10000
    max_steps = 20
    epoch_losses = []
    for episode in range(num_episodes):
        state = mdp.reset()
        episode_reward = 0
        states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
        steps = 0
        for step in range(max_steps):

            #get action from PPO agent
            right_ankle_substate = get_right_ankle_substate(state)
            action, log_prob = ppo_agent.get_action(right_ankle_substate)
            action_torch = torch.tensor(action).to(device)
            

            # Get the expert action
            full_action = agent.draw_action(state)
            expert_action = get_action_substate(full_action)
            expert_action = torch.tensor(expert_action).to(device)
            full_action[7] = action

            #calculate imitation loss
            imitation_loss = F.mse_loss(action_torch, expert_action)

            next_state, reward, done, _ = mdp.step(full_action)

            total_reward = -1 * imitation_loss.item()

            #print(f"Imitation Loss: {imitation_loss.item()}, Total Reward: {total_reward}")
            
            states.append(right_ankle_substate)
            actions.append(action.item())
            rewards.append(total_reward)
            next_states.append(get_right_ankle_substate(next_state))
            dones.append(done)
            log_probs.append(log_prob.item())

            episode_reward += total_reward
            steps += 1

            if done:
                state = mdp.reset()
            else:
                state = next_state

        # Update PPO agent
        #print(log_probs)
        loss = ppo_agent.update(states, actions, rewards, next_states, dones, log_probs)

        print(f"Episode {episode + 1}, Reward: {episode_reward/steps}, Loss: {loss} , Steps: {steps}")

        epoch_losses.append(episode_reward/steps)
        


    # Save the trained PPO model
    ppo_agent.save("ppo_model.pth")

    #Plot the loss after each epoch
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss After Each Epoch')
    plt.show()
    

if __name__ == '__main__':
    main()

