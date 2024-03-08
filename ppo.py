# import torch
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset

# class PPO:
#     def __init__(self, env, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, gae_lambda=0.95, 
#                  ppo_epochs=10, mini_batch_size=64, ppo_clip=0.2):
#         self.env = env
#         self.actor_lr = actor_lr
#         self.critic_lr = critic_lr
#         self.gamma = gamma
#         self.gae_lambda = gae_lambda
#         self.ppo_epochs = ppo_epochs
#         self.mini_batch_size = mini_batch_size
#         self.ppo_clip = ppo_clip

#         # Initialize actor and critic networks for each agent
#         self.rl_agents = [agent for agent in env.rl_powerplants + env.rl_storages if agent.learning]
#         for agent in self.rl_agents:
#             agent.actor = # initialize actor network
#             agent.critic = # initialize critic network
#             agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=self.actor_lr)
#             agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=self.critic_lr)


#     def collect_trajectories(self):
#         data = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        
#         # Example loop to collect data - Adjust according to your environment
#         for _ in range(self.num_steps):
#             state = self.env.get_current_state()
#             action = self.policy(state)  # Assuming policy method exists and returns an action for given state
#             next_state, reward, done, _ = self.env.step(action)  # Interact with the environment

#             data['states'].append(state)
#             data['actions'].append(action)
#             data['rewards'].append(reward)
#             data['next_states'].append(next_state)
#             data['dones'].append(done)

#             if done:
#                 self.env.reset()  # Reset the environment if done

#         return data


#     def _compute_gae(self, rewards, values, masks, next_value):
#         # Computes Generalized Advantage Estimation
#         values = values + [next_value]
#         gae = 0
#         returns = []
#         for step in reversed(range(len(rewards))):
#             delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
#             gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
#             returns.insert(0, gae + values[step])
#         return returns

#     def create_mini_batches(self, data, mini_batch_size):
#         # Prepares mini-batches
#         tensor_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
#         dataset = TensorDataset(tensor_data['states'], tensor_data['actions'],
#                                 tensor_data['log_probs'], tensor_data['returns'],
#                                 tensor_data['advantages'])
#         data_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
#         return data_loader

#     def update_mini_batch(self, mini_batch, agent):
#         # Updates the policy using a mini-batch
#         states, actions, old_log_probs, returns, advantages = mini_batch

#         # Get current policy outputs
#         log_probs, entropy = agent.actor.get_log_probs(states, actions)
#         values = agent.critic(states).squeeze()

#         # Calculate ratios
#         ratios = torch.exp(log_probs - old_log_probs.detach())

#         # Calculate surrogate losses
#         surr1 = ratios * advantages
#         surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
#         policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

#         # Update actor
#         agent.actor_optimizer.zero_grad()
#         policy_loss.backward()
#         agent.actor_optimizer.step()

#         # Update critic
#         critic_loss = F.mse_loss(values, returns)
#         agent.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         agent.critic_optimizer.step()

#     def _get_mini_batch_indices(self, dataset_size, mini_batch_size):
#         indices = np.arange(dataset_size)
#         np.random.shuffle(indices)
#         mini_batches = [indices[i:i + mini_batch_size] for i in range(0, dataset_size, mini_batch_size)]
#         return mini_batches

#     def update_policy(self, data):
#         states = torch.tensor(data['states'], dtype=torch.float32)
#         actions = torch.tensor(data['actions'], dtype=torch.float32)
#         rewards = torch.tensor(data['rewards'], dtype=torch.float32)
#         next_states = torch.tensor(data['next_states'], dtype=torch.float32)
#         dones = torch.tensor(data['dones'], dtype=torch.float32)

#         # Calculate advantages and returns
#         values = self.critic(states)
#         next_values = self.critic(next_states)
#         returns, advantages = self._compute_gae(rewards, dones, values, next_values)

#         for _ in range(self.ppo_epochs):
#             mini_batch_indices = self._get_mini_batch_indices(len(states), self.mini_batch_size)

#             for indices in mini_batch_indices:
#                 mb_states = states[indices]
#                 mb_actions = actions[indices]
#                 mb_returns = returns[indices]
#                 mb_advantages = advantages[indices]

#                 # Update critic
#                 values = self.critic(mb_states)
#                 critic_loss = F.mse_loss(values.squeeze(-1), mb_returns)

#                 self.critic_optimizer.zero_grad()
#                 critic_loss.backward()
#                 self.critic_optimizer.step()

#                 # Update actor
#                 old_log_probs = self.actor.get_log_probs(mb_states, mb_actions)
#                 new_log_probs = self.actor.get_log_probs(mb_states, mb_actions)
#                 ratio = torch.exp(new_log_probs - old_log_probs)

#                 surr1 = ratio * mb_advantages
#                 surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * mb_advantages
#                 actor_loss = -torch.min(surr1, surr2).mean()

#                 self.actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 self.actor_optimizer.step()
                    
                    
    