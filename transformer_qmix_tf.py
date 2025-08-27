import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerMixingNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, obs_dim, d_model=64, nhead=4, num_layers=2, hidden_dim=64):
        super(TransformerMixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Embeddings
        self.q_embed = nn.Linear(1, d_model)  # Embed scalar Q-values
        self.state_embed = nn.Linear(num_agents * obs_dim, d_model)  # Embed state entities
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers for MLP weights
        self.w1_out = nn.Linear(d_model, hidden_dim)
        self.b1_out = nn.Linear(d_model, hidden_dim)
        self.w2_out = nn.Linear(d_model, hidden_dim)
        self.b2_out = nn.Linear(d_model, 1)

        # Recurrent vectors
        self.register_buffer('recurrent_vectors', torch.zeros(3, d_model))

    def forward(self, agent_qs, state):
        batch_size = agent_qs.size(0)
        device = agent_qs.device

        # Embed Q-values: [batch_size, num_agents, 1] -> [batch_size, num_agents, d_model]
        q_embed = self.q_embed(agent_qs.unsqueeze(-1))

        # Split state into entities: [batch_size, num_agents * num_agents * obs_dim] -> [batch_size, num_agents, num_agents * obs_dim]
        state_entities = state.view(batch_size, self.num_agents, self.num_agents * self.obs_dim)
        # Embed state entities: [batch_size, num_agents, d_model]
        state_embed = self.state_embed(state_entities)

        # Expand recurrent vectors: [3, d_model] -> [batch_size, 3, d_model]
        recurrent_embed = self.recurrent_vectors.unsqueeze(0).expand(batch_size, -1, -1)

        # Form graph: [batch_size, num_agents + 3 + num_agents, d_model]
        graph = torch.cat([q_embed, recurrent_embed, state_embed], dim=1)

        # Positional encoding
        graph = self.pos_encoder(graph)

        # Transformer processing
        graph_out = self.transformer_encoder(graph)  # [batch_size, num_agents + 3 + num_agents, d_model]

        # Split output
        q_out = graph_out[:, :self.num_agents, :]  # [batch_size, num_agents, d_model]
        recurrent_out = graph_out[:, self.num_agents:self.num_agents + 3, :]  # [batch_size, 3, d_model]
        # State outputs are ignored

        # Generate MLP weights
        w1 = torch.abs(self.w1_out(q_out))  # [batch_size, num_agents, hidden_dim]
        b1 = self.b1_out(recurrent_out[:, 0, :])  # [batch_size, hidden_dim]
        w2 = torch.abs(self.w2_out(recurrent_out[:, 1, :]))  # [batch_size, hidden_dim]
        b2 = F.relu(self.b2_out(recurrent_out[:, 2, :]))  # [batch_size, 1]

        # Compute Q_total
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)  # [batch_size, 1, num_agents]
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = b2.view(batch_size, 1, 1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [batch_size, 1, hidden_dim]
        q_total = torch.bmm(hidden, w2) + b2  # [batch_size, 1, 1]

        # Update recurrent vectors (mean across batch for next step)
        self.recurrent_vectors = recurrent_out.mean(dim=0).detach()

        return q_total.squeeze(-1)  # [batch_size, 1]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerQNetwork(nn.Module):
    def __init__(self, obs_action_dim, action_dim, num_agents, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.obs_action_dim = obs_action_dim  # num_agents * obs_dim + 1
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.d_model = d_model
        self.input_embed = nn.Linear(obs_action_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.encoder_layers = encoder_layers
        self.output_layer = nn.Linear(d_model, action_dim)

    def forward(self, history, history_mask=None, agent_id=None):
        if len(history.shape) == 2:
            history = history.unsqueeze(0)
        batch_size, seq_len, obs_action_dim = history.shape
        x = self.input_embed(history)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        if history_mask is not None:
            src_key_padding_mask = (history_mask == 0).all(dim=-1)
        else:
            src_key_padding_mask = None
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)
        x = x[:, -1, :]
        q_values = self.output_layer(x)  # [batch_size, action_dim]
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        history, actions, rewards, next_history, dones = experience
        history = np.array(history, dtype=np.float32)
        next_history = np.array(next_history, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)
        self.buffer.append((history, actions, rewards, next_history, dones))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        histories, actions, rewards, next_histories, dones = zip(*batch)
        try:
            histories = np.stack(histories)  # [batch_size, num_agents, seq_len, obs_action_dim]
            actions = np.stack(actions)  # [batch_size, num_agents]
            rewards = np.stack(rewards)  # [batch_size, num_agents]
            next_histories = np.stack(next_histories)
            dones = np.stack(dones)
        except ValueError as e:
            print(f"Error in sample: {e}")
            print(f"History shapes: {[h.shape for h in histories]}")
            raise
        return histories, actions, rewards, next_histories, dones

    def __len__(self):
        return len(self.buffer)


class TransformerAgent:
    def __init__(self, params, device, num_agents):
        self.device = device
        self.obs_dim = params['obs_dim']
        self.action_dim = params['action_dim']
        self.num_agents = num_agents
        self.q_net = TransformerQNetwork(
            obs_action_dim=params['obs_dim'] * num_agents + 1,
            action_dim=params['action_dim'],
            num_agents=num_agents,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)
        self.target_q_net = TransformerQNetwork(
            obs_action_dim=params['obs_dim'] * num_agents + 1,
            action_dim=params['action_dim'],
            num_agents=num_agents,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # 添加混合网络
        self.state_dim = num_agents * num_agents * self.obs_dim  # 假设全局状态为 global_obs 展平
        self.mixing_net = TransformerMixingNetwork(num_agents, self.state_dim, obs_dim=params['obs_dim'], d_model=64, nhead=4, num_layers=2, hidden_dim=64).to(device)
        self.target_mixing_net = TransformerMixingNetwork(num_agents, self.state_dim, obs_dim=params['obs_dim'], d_model=64, nhead=4, num_layers=2, hidden_dim=64).to(device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
        # 优化器包括 Q 网络和混合网络
        self.optimizer = optim.Adam(
            list(self.q_net.parameters()) + list(self.mixing_net.parameters()),
            lr=params['lr']
        )
        self.memory = ReplayBuffer(params['buffer_size'])
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.tau = params['tau']
        self.current_episode = 0
        self.sequence_length = params['sequence_length']
        self.alpha = params.get('hysteretic_alpha', 0.001)
        self.beta = params.get('hysteretic_beta', 0.0001)
        self.save_dir = "saved_models"
        os.makedirs(self.save_dir, exist_ok=True)
        self.history_buffer = {}
        self.mask_buffer = {}
        self.best_reward = -float('inf')

    def save_model(self, episode=None, reward=None):
        if reward is not None and reward > self.best_reward:
            self.best_reward = reward
            save_path = os.path.join(self.save_dir, "best_tf_qix_tf_model.pth")
            torch.save({
                'q_net_state': self.q_net.state_dict(),
                'target_net_state': self.target_q_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'best_reward': self.best_reward,
                'episode': episode
            }, save_path)
            print(f"最佳模型已保存至 {save_path}，奖励: {reward:.2f}")

    def get_action(self, obs, agent_id, eval_mode, obs_mask=None):
        # Ensure obs is a 2D array [num_agents, obs_dim]
        if isinstance(obs, list):
            obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]  # Add agent dimension if single agent
        if obs.ndim != 2 or obs.shape[0] != self.num_agents or obs.shape[1] != self.obs_dim:
            raise ValueError(f"Expected obs to be 2D [num_agents, obs_dim], got shape {obs.shape}")

        # Ensure obs_mask is consistent
        if obs_mask is not None:
            if isinstance(obs_mask, list):
                obs_mask = np.array(obs_mask, dtype=np.float32)
            if obs_mask.ndim == 1:
                obs_mask = obs_mask[np.newaxis, :]
            if obs_mask.shape != obs.shape:
                raise ValueError(f"obs_mask shape {obs_mask.shape} does not match obs shape {obs.shape}")

        # Initialize history and mask buffers
        if agent_id not in self.history_buffer:
            self.history_buffer[agent_id] = []
            self.mask_buffer[agent_id] = []

        history = self.history_buffer[agent_id]
        mask_history = self.mask_buffer[agent_id] if obs_mask is not None else []

        # Pad history if needed
        expected_obs_action_dim = self.num_agents * self.obs_dim + 1
        if len(history) < self.sequence_length:
            padding = np.zeros(expected_obs_action_dim, dtype=np.float32)
            history = [padding] * (self.sequence_length - len(history)) + history
            if obs_mask is not None:
                mask_padding = np.zeros(expected_obs_action_dim, dtype=np.float32)
                mask_history = [mask_padding] * (self.sequence_length - len(mask_history)) + mask_history
        else:
            history = history[-self.sequence_length:]
            if obs_mask is not None:
                mask_history = mask_history[-self.sequence_length:]

        # Flatten obs and concatenate with a placeholder action
        obs_flat = obs.flatten()
        if obs_flat.shape[0] != self.num_agents * self.obs_dim:
            raise ValueError(f"Flattened obs has shape {obs_flat.shape}, expected ({self.num_agents * self.obs_dim},)")
        obs_action = np.concatenate([obs_flat, np.array([0], dtype=np.float32)])

        # Convert history to tensor
        try:
            history_array = np.array(history, dtype=np.float32)
            if history_array.shape != (self.sequence_length, expected_obs_action_dim):
                raise ValueError(
                    f"History shape {history_array.shape} does not match expected {(self.sequence_length, expected_obs_action_dim)}")
            history_tensor = torch.FloatTensor(history_array).to(self.device)
            if len(history_tensor.shape) == 2:
                history_tensor = history_tensor.unsqueeze(0)  # [1, seq_len, num_agents * obs_dim + 1]
        except ValueError as e:
            print(f"Error creating history_tensor: {e}")
            print(f"History shapes: {[np.array(h).shape for h in history]}")
            raise

        # Handle mask tensor
        mask_tensor = None
        if obs_mask is not None:
            mask_flat = obs_mask.flatten()
            mask_action = np.concatenate([mask_flat, np.ones(1, dtype=np.float32)])
            # Create mask_array before appending mask_action
            try:
                mask_array = np.array(mask_history, dtype=np.float32)
                if mask_array.shape != (self.sequence_length, expected_obs_action_dim):
                    raise ValueError(
                        f"Mask history shape {mask_array.shape} does not match expected {(self.sequence_length, expected_obs_action_dim)}")
                mask_tensor = torch.FloatTensor(mask_array).to(self.device)
                if len(mask_tensor.shape) == 2:
                    mask_tensor = mask_tensor.unsqueeze(0)
            except ValueError as e:
                print(f"Error creating mask_tensor: {e}")
                print(f"Mask history shapes: {[np.array(m).shape for m in mask_history]}")
                raise
            # Append mask_action after creating mask_array
            mask_history.append(mask_action)
            self.mask_buffer[agent_id] = mask_history[-self.sequence_length:]  # Ensure length is capped

        # Get action
        if eval_mode or np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_net(history_tensor, mask_tensor)
            action = torch.argmax(q_values).item()
        else:
            action = np.random.randint(self.action_dim)

        # Update history with the actual action
        obs_action[-1] = action
        self.history_buffer[agent_id].append(obs_action)
        self.history_buffer[agent_id] = self.history_buffer[agent_id][-self.sequence_length:]  # Cap history length

        return action

    def decay_epsilon(self):
        if self.current_episode <= 50:
            decay_step = (1.0 - self.epsilon_min) / 50
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_step)
        else:
            self.epsilon = self.epsilon_min

    def update_target_network(self):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_mixing_net.parameters(), self.mixing_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def store_experience(self, experiences):
        # experiences: [(obs, action, reward, next_obs, done)]
        obs = np.array([exp[0] for exp in experiences], dtype=np.float32)  # [num_agents, num_agents, obs_dim]
        actions = np.array([exp[1] for exp in experiences], dtype=np.int64)  # [num_agents]
        rewards = np.array([exp[2] for exp in experiences], dtype=np.float32)  # [num_agents]
        next_obs = np.array([exp[3] for exp in experiences], dtype=np.float32)  # [num_agents, num_agents, obs_dim]
        dones = np.array([exp[4] for exp in experiences], dtype=np.bool_)  # [num_agents]

        if len(experiences) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} experiences, got {len(experiences)}")

        history_arrays = []
        next_history_arrays = []
        for agent_id in range(self.num_agents):
            history = self.history_buffer.get(agent_id, [])
            mask_history = self.mask_buffer.get(agent_id, []) if agent_id in self.mask_buffer else []
            expected_obs_action_dim = self.num_agents * self.obs_dim + 1
            if len(history) < self.sequence_length:
                padding = np.zeros(expected_obs_action_dim, dtype=np.float32)
                history = [padding] * (self.sequence_length - len(history)) + history
                if mask_history:
                    mask_padding = np.zeros(expected_obs_action_dim, dtype=np.float32)
                    mask_history = [mask_padding] * (self.sequence_length - len(mask_history)) + mask_history
            else:
                history = history[-self.sequence_length:]
                if mask_history:
                    mask_history = mask_history[-self.sequence_length:]

            # 只为有效动作更新历史
            if actions[agent_id] != -1:  # 有效动作
                obs_action = np.concatenate([obs[agent_id].flatten(), np.array([actions[agent_id]], dtype=np.float32)])
                next_obs_action = np.concatenate(
                    [next_obs[agent_id].flatten(), np.array([actions[agent_id]], dtype=np.float32)])
                next_history = history[1:] + [next_obs_action] if len(history) == self.sequence_length else history + [
                    next_obs_action]
                if len(next_history) < self.sequence_length:
                    padding = np.zeros(expected_obs_action_dim, dtype=np.float32)
                    next_history = [padding] * (self.sequence_length - len(next_history)) + next_history
                history_array = np.array(history, dtype=np.float32)
                next_history_array = np.array(next_history, dtype=np.float32)
                expected_shape = (self.sequence_length, expected_obs_action_dim)
                if history_array.shape != expected_shape:
                    print(f"Invalid history shape for agent {agent_id}: {history_array.shape}")
                    raise ValueError(f"History shape mismatch, expected {expected_shape}")
                if next_history_array.shape != expected_shape:
                    print(f"Invalid next_history shape for agent {agent_id}: {next_history_array.shape}")
                    raise ValueError(f"Next history shape mismatch, expected {expected_shape}")
                self.history_buffer[agent_id] = next_history[-self.sequence_length:]
                if mask_history:
                    mask_action = np.concatenate(
                        [np.zeros(self.num_agents * self.obs_dim, dtype=np.float32), np.ones(1, dtype=np.float32)])
                    mask_history.append(mask_action)
                    self.mask_buffer[agent_id] = mask_history[-self.sequence_length:]
            else:
                # 无效动作，保持历史不变
                history_array = np.array(history, dtype=np.float32)
                next_history_array = np.array(history, dtype=np.float32)  # 复制当前历史

            history_arrays.append(history_array)
            next_history_arrays.append(next_history_array)

        # 存储所有智能体的经验
        self.memory.push((history_arrays, actions, rewards, next_history_arrays, dones))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        experiences = self.memory.sample(self.batch_size)
        histories, actions, rewards, next_histories, dones = experiences
        histories = torch.FloatTensor(histories).to(self.device)  # [batch_size, num_agents, seq_len, obs_action_dim]
        actions = torch.LongTensor(actions).to(self.device)  # [batch_size, num_agents]
        rewards = torch.FloatTensor(rewards).to(self.device)  # [batch_size, num_agents]
        next_histories = torch.FloatTensor(next_histories).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 构造全局状态
        states = histories[:, :, -1, :-1].reshape(self.batch_size,
                                                  -1)  # [batch_size, num_agents * num_agents * obs_dim], e.g., [512, 108]
        next_states = next_histories[:, :, -1, :-1].reshape(self.batch_size,
                                                            -1)  # [batch_size, num_agents * num_agents * obs_dim]

        # 调试打印
        print(
            f"train_step: histories.shape={histories.shape}, states.shape={states.shape}, actions.shape={actions.shape}")

        all_q_values = []
        valid_mask = (actions != -1)  # [batch_size, num_agents]
        for agent_id in range(self.num_agents):
            agent_history = histories[:, agent_id]
            q_values = self.q_net(agent_history)  # [batch_size, action_dim]
            q_selected = torch.zeros(self.batch_size, device=self.device)
            valid_indices = valid_mask[:, agent_id]
            if valid_indices.any():
                valid_actions = actions[:, agent_id][valid_indices]
                valid_q_selected = q_values[valid_indices].gather(1, valid_actions.unsqueeze(1)).squeeze(1)
                q_selected[valid_indices] = valid_q_selected
            all_q_values.append(q_selected)
        agent_qs = torch.stack(all_q_values, dim=1)  # [batch_size, num_agents], e.g., [512, 3]

        q_total = self.mixing_net(agent_qs, states)  # [batch_size, 1]

        with torch.no_grad():
            all_target_q_values = []
            for agent_id in range(self.num_agents):
                next_q_values = self.q_net(next_histories[:, agent_id])
                target_next_q_values = self.target_q_net(next_histories[:, agent_id])
                max_actions = torch.argmax(next_q_values, dim=1)
                target_q_selected = target_next_q_values.gather(1, max_actions.unsqueeze(1)).squeeze(1)
                all_target_q_values.append(target_q_selected)
            target_agent_qs = torch.stack(all_target_q_values, dim=1)
            target_q_total = self.target_mixing_net(target_agent_qs, next_states)
            target_q = rewards.mean(dim=1, keepdim=True) + self.gamma * target_q_total * (~dones[:, 0].unsqueeze(1))

        loss = F.mse_loss(q_total, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target_network()

        return loss.item()


class MultiAgentTransformer:
    def __init__(self, num_agents, params, device):
        self.num_agents = num_agents
        self.params = params
        self.device = device
        self.agent = TransformerAgent(self.params, self.device, num_agents)  # Pass num_agents

    def get_actions(self, global_obs, flag_decision, eval_mode, obs_masks=None):
        # actions = []
        # for agent_id, obs in enumerate(observations):
        #     if flag_decision[agent_id] == 1:
        #         actions.append(self.agent.get_action(obs, agent_id, eval_mode))
        # return actions
        actions = []
        for agent_id in range(self.num_agents):
            if flag_decision[agent_id] == 1:
                # Pass the entire global_obs and obs_masks
                action = self.agent.get_action(global_obs, agent_id, eval_mode, obs_masks)
                actions.append(action)
        return actions
            # return [self.agent.get_action(obs, agent_id, eval_mode, obs_mask)
            #         for agent_id, (obs, obs_mask) in enumerate(zip(observations, obs_masks))]

    def update_training_progress(self, episode):
        self.agent.current_episode = episode
        self.agent.decay_epsilon()

    def store_experiences(self, all_experiences):
        self.agent.store_experience(all_experiences)

    def train(self):
        loss = self.agent.train_step()
        self.agent.epsilon = max(
            self.params['epsilon_min'],
            self.agent.epsilon * self.params['epsilon_decay']
        )
        return loss