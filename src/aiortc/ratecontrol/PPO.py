import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
# if(torch.cuda.is_available()): 
# 	device = torch.device('cuda:2')
# 	torch.cuda.empty_cache()
# 	print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
# 	print("Device set to : cpu")
print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.is_terminals = []
	
	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.is_terminals[:]

	@property
	def length(self):
		return len(self.rewards)

	@property
	def size(self):
		targets = [self.actions, self.states, self.logprobs]
		sz = sum([len(t) for t in targets])
		return sz
		
class ActorCritic(nn.Module):
	def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, evaluate=False):
		super(ActorCritic, self).__init__()

		self.has_continuous_action_space = has_continuous_action_space
		self.is_evaluate = evaluate
		
		if has_continuous_action_space:
			self.action_dim = action_dim
			self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
		# actor
		# network reference: OnRL https://dl.acm.org/doi/10.1145/3372224.3419186
		if has_continuous_action_space :
			self.actor = nn.Sequential(
							nn.Linear(state_dim, 64),
							# nn.Tanh(),
							nn.LeakyReLU(),
							nn.Linear(64, 32),
							# nn.Tanh(),
							nn.LeakyReLU(),
							nn.Linear(32, action_dim),
						)
		else:
			self.actor = nn.Sequential(
							nn.Linear(state_dim, 64),
							# nn.Tanh(),
							nn.LeakyReLU(),
							nn.Linear(64, 32),
							# nn.Tanh(),
							nn.LeakyReLU(),
							nn.Linear(32, action_dim),
							nn.Softmax(dim=-1)
						)
		# critic
		self.critic = nn.Sequential(
						nn.Linear(state_dim, 64),
						# nn.Tanh(),
						nn.LeakyReLU(),
						nn.Linear(64, 32),
						# nn.Tanh(),
						nn.LeakyReLU(),
						nn.Linear(32, 1)
					)
		
	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def forward(self):
		raise NotImplementedError
	
	def act(self, state):
		if self.has_continuous_action_space:
			action_mean = self.actor(state)
			cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
			dist = MultivariateNormal(action_mean, cov_mat)
		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)
		
		if not self.is_evaluate:
			action = dist.sample()
		else:
			# choose the action with hightest probability under evaluation (choose the first action when there are multiple actions)
			action = torch.where(dist.probs == torch.max(dist.probs))
			action = action[0]
		action_logprob = dist.log_prob(action)
		
		return action.detach(), action_logprob.detach()

	def evaluate(self, state, action):

		if self.has_continuous_action_space:
			action_mean = self.actor(state)
			
			action_var = self.action_var.expand_as(action_mean)
			cov_mat = torch.diag_embed(action_var).to(device)
			dist = MultivariateNormal(action_mean, cov_mat)
			
			# For Single Action Environments.
			if self.action_dim == 1:
				action = action.reshape(-1, self.action_dim)
		else:
			action_probs = self.actor(state)
			dist = Categorical(action_probs)
		action_logprobs = dist.log_prob(action)
		dist_entropy = dist.entropy()
		state_values = self.critic(state)
		
		return action_logprobs, state_values, dist_entropy

	def get_value(self, state):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			state_value = self.critic(state).cpu().item()
		pass
		return state_value


class PPO:
	def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, entropy_factor, action_std_init=0.6, evaluate=False):

		self.has_continuous_action_space = has_continuous_action_space
		self.is_evaluate = evaluate

		if has_continuous_action_space:
			self.action_std = action_std_init

		self.gamma = gamma
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs


		self.update_cnt = 0
		"""dynamic parameters"""
		self.entropy_factor = entropy_factor

		self.buffer = RolloutBuffer()

		self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, evaluate=self.is_evaluate).to(device)
		self.optimizer = torch.optim.Adam([
						{'params': self.policy.actor.parameters(), 'lr': lr_actor},
						{'params': self.policy.critic.parameters(), 'lr': lr_critic}
					])

		self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, evaluate=self.is_evaluate).to(device)
		self.policy_old.load_state_dict(self.policy.state_dict())

		self.MseLoss = nn.MSELoss()

	def set_action_std(self, new_action_std):
		if self.has_continuous_action_space:
			self.action_std = new_action_std
			self.policy.set_action_std(new_action_std)
			self.policy_old.set_action_std(new_action_std)
		else:
			print("--------------------------------------------------------------------------------------------")
			print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
			print("--------------------------------------------------------------------------------------------")

	def decay_action_std(self, action_std_decay_rate, min_action_std):
		print("--------------------------------------------------------------------------------------------")
		if self.has_continuous_action_space:
			self.action_std = self.action_std - action_std_decay_rate
			self.action_std = round(self.action_std, 4)
			if (self.action_std <= min_action_std):
				self.action_std = min_action_std
				print("setting actor output action_std to min_action_std : ", self.action_std)
			else:
				print("setting actor output action_std to : ", self.action_std)
			self.set_action_std(self.action_std)

		else:
			print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
		print("--------------------------------------------------------------------------------------------")

	def select_action(self, state, force_action=None):
		"""
		action_: pre-determine the action to take. Used to update the solution in subprocess of PPOParallelSolution.
		"""
		with torch.no_grad():
			state = torch.FloatTensor(state).to(device)
			if force_action is None:
				action, action_logprob = self.policy_old.act(state)
			else:
				action, action_logprob = force_action

		if not self.is_evaluate:
			self.buffer.states.append(state)
			self.buffer.actions.append(action)
			self.buffer.logprobs.append(action_logprob)

		if self.has_continuous_action_space:
			return action.detach().cpu().numpy().flatten()
		else:
			return action.item(), action, action_logprob

	def cumulated_reward(self, rewards, is_terminals=None):
		if not is_terminals:
			is_terminals = [False] * len(rewards)
		cum_rewards = []
		discounted_reward = 0
		for reward, is_terminals in zip(reversed(rewards), reversed(is_terminals)):
			if is_terminals:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			cum_rewards.insert(0, discounted_reward)
		return cum_rewards

	def update(self):
		# Monte Carlo estimate of returns
		# rewards = []
		# discounted_reward = 0
		# for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
		# 	if is_terminal:
		# 		discounted_reward = 0
		# 	discounted_reward = reward + (self.gamma * discounted_reward)
		# 	rewards.insert(0, discounted_reward)
		rewards = self.cumulated_reward(self.buffer.rewards, self.buffer.is_terminals)


		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		# Normalizing the rewards : NOTE: is it a good idea? https://ai.stackexchange.com/a/10204/51116
		# rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
		old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
		old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

		# history
		history = {
			"actor_loss" : [],
			"entropy_loss" : [],
			"critic_loss" : [],
			"loss" : [],
			"dist_entropy" : [],
			"ev_critic" : []
		}

		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = rewards - state_values.detach()
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			# final loss of clipped objective PPO
			# loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.03*dist_entropy
			loss_actor = -torch.min(surr1, surr2)
			loss_entropy = -self.entropy_factor(self.update_cnt) * dist_entropy
			loss_critic = 0.5 * self.MseLoss(state_values, rewards)
			loss = loss_actor + loss_critic + loss_entropy

			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

			# record
			with torch.no_grad():
				# explained variance of critic network
				ev_critic = 1 - (rewards.cpu().numpy() - state_values.cpu().numpy()).var() / rewards.cpu().numpy().var()
				# history
				history['actor_loss'].append(loss_actor.mean().item())
				history['entropy_loss'].append(loss_entropy.mean().item())
				history['critic_loss'].append(loss_critic.mean().item())
				history['loss'].append(loss.mean().item())
				history['dist_entropy'].append(dist_entropy.mean().item())
				history['ev_critic'].append(ev_critic)


		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()

		# counter
		self.update_cnt += 1

		# average history
		for key, value in history.items():
			history[key] = np.mean(value)

		return history

	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

	def load_state_dict(self, state_dict):
		self.policy_old.load_state_dict(state_dict)
		self.policy.load_state_dict(state_dict)
		
		
################################## Play-Back Policy ##################################
class PlayBackPolicy():
	def __init__(self, playback: list) -> None:
		self.playback = playback
		self.action = None
		pass

	def select_action(self, *args):
		if self.playback:
			self.action = self.playback.pop(0)
		else:
			print('WARNING: play back runs out, reuse the last action')
		return self.action

################################## Optimal Policy ##################################
# class OptimalPolicy():
# 	def __init__(self, trace_file, **kwargs) -> None:
# 		self.load(trace_file)

# 	def load(self, trace_file):
# 		trace = SimpleEmulator.get_trace(trace_file, ("time", "bandwith", "loss_rate", "delay"))
# 		trace = np.array(trace)
# 		# bandwidth
# 		rate_map = np.linspace(0.1, 2.5, 25, endpoint=True) * 1.0e6
# 		bandwidth_bps = trace[:,1] * 8
# 		playback = []
# 		for bandwidth in bandwidth_bps.tolist():
# 			tmp = np.where(rate_map < bandwidth)[0]
# 			action = tmp[-1] if tmp.size else 0
# 			playback.append(action)
# 		# timestamps
# 		timestamps = trace[:, 0]
# 		self.playback = (timestamps, playback)

# 	def select_action(self, cur_time):
# 		idx = np.where(self.playback[0] < cur_time)[0]
# 		action = self.playback[1][idx[-1] if idx.size else 0]
# 		return action
