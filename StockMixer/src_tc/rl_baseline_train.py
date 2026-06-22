import os
import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

from src_tc.config import load_config
from src_tc.data import StockDatasetGPU, make_offsets
from src_tc.utils import set_seed, configure_torch
from src_tc.metrics import evaluate_prediction_metrics
from src_tc.backtest import backtest_topk

class PPOActorCritic(nn.Module):
    def __init__(self, lookback, num_features, hidden_size=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_size * lookback, 1)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_size * lookback, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        original_shape = state.shape
        if len(original_shape) == 3:
            state = state.unsqueeze(0)
            
        T_ep, N, L, F_dim = state.shape
        x = self.feature_extractor(state) 
        x = x.view(T_ep, N, -1) 
        
        logits = self.actor_head(x).squeeze(-1) 
        alpha = F.softplus(logits) + 1.0 
        
        v = self.critic_head(x).squeeze(-1) 
        value = v.mean(dim=1) 
        
        if len(original_shape) == 3:
            alpha = alpha.squeeze(0)
            value = value.squeeze(0)
            
        return alpha, value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

class PPOAgent:
    def __init__(self, lookback, num_features, device, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = PPOActorCritic(lookback, num_features).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = PPOActorCritic(lookback, num_features).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        
    def select_action(self, state, memory):
        with torch.no_grad():
            alpha, value = self.policy_old(state)
            dist = Dirichlet(alpha)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.values.append(value)
        return action
        
    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        old_states = torch.stack(memory.states, dim=0).detach() 
        old_actions = torch.stack(memory.actions, dim=0).detach() 
        old_logprobs = torch.stack(memory.logprobs, dim=0).detach() 
        old_values = torch.stack(memory.values, dim=0).detach() 
        
        advantages = rewards - old_values
        
        for _ in range(self.K_epochs):
            alpha, state_values = self.policy(old_states)
            dist = Dirichlet(alpha)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class PortfolioEnv:
    def __init__(self, dataset, offsets, cost_bps=10.0, reward_type='default'):
        self.dataset = dataset
        self.offsets = offsets
        self.cost_bps = cost_bps / 10000.0
        self.reward_type = reward_type
        self.current_step = 0
        self.prev_w = None
        self.cumulative_return = 1.0
        self.peak_return = 1.0
        
    def reset(self):
        self.current_step = 0
        self.prev_w = torch.zeros(self.dataset.eod.shape[0], device=self.dataset.device)
        self.cumulative_return = 1.0
        self.peak_return = 1.0
        return self._get_state()
        
    def _get_state(self):
        if self.current_step >= len(self.offsets):
            return None
        offset = self.offsets[self.current_step]
        data_batch, mask_batch, price_batch, gt_batch = self.dataset.get_batch(offset)
        return data_batch, mask_batch, price_batch, gt_batch
        
    def step(self, action_w):
        state = self._get_state()
        if state is None:
            return None, 0.0, True
            
        data_batch, mask_batch, price_batch, gt_batch = state
        mask_float = mask_batch.squeeze()
        
        valid_w = action_w * mask_float
        sum_w = valid_w.sum() + 1e-12
        w = valid_w / sum_w
        
        gt = gt_batch.squeeze()
        gross_return = torch.sum(w * gt)
        
        if self.current_step == 0:
            turnover = torch.sum(torch.abs(w))
        else:
            prev_w_evolved = self.prev_w * (1 + gt)
            sum_prev_w = prev_w_evolved.sum() + 1e-12
            prev_w_evolved = prev_w_evolved / sum_prev_w
            turnover = torch.sum(torch.abs(w - prev_w_evolved))
            
        tc = turnover * self.cost_bps
        base_reward = gross_return - tc
        reward_val = base_reward.item()
        
        self.cumulative_return *= (1.0 + reward_val)
        self.peak_return = max(self.peak_return, self.cumulative_return)
        
        if self.reward_type == 'sortino':
            drawdown = (self.cumulative_return / self.peak_return) - 1.0
            if drawdown < -0.10:
                reward_val -= 1.0  # Heavy downside risk penalty
            elif drawdown < 0:
                reward_val += drawdown * 0.5  # Linear downside penalty
        
        self.prev_w = w.detach()
        self.current_step += 1
        
        done = self.current_step >= len(self.offsets)
        next_state = self._get_state() if not done else None
        return next_state, reward_val, done

def evaluate_agent(agent, env):
    state = env.reset()
    all_actions, all_gt, all_mask = [], [], []
    while state is not None:
        data_batch, mask_batch, price_batch, gt_batch = state
        with torch.no_grad():
            alpha, _ = agent.policy(data_batch)
            action = alpha / alpha.sum()
            
        all_actions.append(action.cpu().numpy())
        all_gt.append(gt_batch.squeeze().cpu().numpy())
        all_mask.append(mask_batch.squeeze().cpu().numpy())
        
        state, _, done = env.step(action)
        
    preds = np.stack(all_actions, axis=1)
    gts = np.stack(all_gt, axis=1)
    masks = np.stack(all_mask, axis=1)
    
    pm = evaluate_prediction_metrics(preds, gts, masks)
    bt = backtest_topk(preds, gts, masks, topk=5, cost_bps=10.0)
    return preds, gts, masks, pm, bt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--reward_type', type=str, default='default')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.seed = args.seed
    cfg.output_dir = f"results/{cfg.market_name.lower()}_ppo_seed{cfg.seed}"
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    set_seed(cfg.seed)
    configure_torch(cfg.speed_mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = StockDatasetGPU(cfg, device)
    train_offsets, valid_offsets, test_offsets = make_offsets(
        cfg.valid_index, cfg.test_index, dataset.trade_dates,
        cfg.lookback_length, cfg.steps
    )
    
    env = PortfolioEnv(dataset, train_offsets, cost_bps=10.0, reward_type=args.reward_type)
    valid_env = PortfolioEnv(dataset, valid_offsets, cost_bps=10.0, reward_type=args.reward_type)
    test_env = PortfolioEnv(dataset, test_offsets, cost_bps=10.0, reward_type=args.reward_type)
    
    agent = PPOAgent(cfg.lookback_length, cfg.fea_num, device)
    memory = PPOMemory()
    
    best_score = -1e18
    patience = 0
    
    print(f"Starting PPO Training on {cfg.market_name} (Seed {cfg.seed})")
    
    for epoch in range(1, cfg.epochs + 1):
        state = env.reset()
        ep_reward = 0
        while state is not None:
            data_batch, _, _, _ = state
            action = agent.select_action(data_batch, memory)
            state, reward, done = env.step(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            ep_reward += reward
            
        agent.update(memory)
        memory.clear()
        
        _, _, _, valid_pm, valid_bt = evaluate_agent(agent, valid_env)
        valid_score = valid_bt['net_sharpe']
        
        print(f"Epoch {epoch:03d} | Reward: {ep_reward:.4f} | Valid Net Sharpe: {valid_score:.4f} | Valid Turnover: {valid_bt['avg_turnover']:.4f}")
        
        if valid_score > best_score:
            best_score = valid_score
            torch.save(agent.policy.state_dict(), os.path.join(cfg.output_dir, 'best_ppo.pt'))
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping triggered.")
                break
                
    agent.policy.load_state_dict(torch.load(os.path.join(cfg.output_dir, 'best_ppo.pt')))
    
    test_preds, test_gts, test_masks, test_pm, test_bt = evaluate_agent(agent, test_env)
    print(f"\nFinal Test Results:")
    print(f"Net Sharpe: {test_bt['net_sharpe']:.4f}")
    print(f"Turnover: {test_bt['avg_turnover']:.4f}")
    print(f"Max Drawdown: {test_bt['max_drawdown']:.4f}")
    
    def get_scalars(bt_dict):
        return {k: v for k, v in bt_dict.items() if not isinstance(v, np.ndarray)}
        
    metrics = {'test_prediction_metrics': test_pm, 'test_backtest': get_scalars(test_bt)}
    with open(os.path.join(cfg.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    np.savez(os.path.join(cfg.output_dir, 'test_predictions.npz'), pred=test_preds, gt=test_gts, mask=test_masks)
        
if __name__ == '__main__':
    main()
