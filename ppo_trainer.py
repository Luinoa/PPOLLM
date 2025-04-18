# ppo_trainer.py
import torch
import numpy as np
import torch.nn as nn

class PPOTrainer:
    def __init__(self, agent, args, device, writer):
        self.agent = agent
        self.args = args
        self.device = device
        self.writer = writer

        self.policy_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.agent.actor.parameters()),
            lr=args.policy_learning_rate, eps=1e-5, weight_decay=0
        )
        self.value_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.critic.parameters()),
            lr=args.value_learning_rate, eps=1e-5
        )

    def update(self, experience_list, global_step, is_warmup):
        """
        experience_list: list of dictionaries, each from a separate task.
                         Each dict contains obs, actions, logprobs, rewards, dones, values
        """
        args = self.args
        all_obs, all_actions, all_logprobs, all_rewards = [], [], [], []
        all_dones, all_values, all_returns, all_advantages = [], [], [], []

        for exp in experience_list:
            obs = exp['obs']  # [T, ...]
            actions = exp['actions']
            logprobs = exp['logprobs']
            rewards = exp['rewards']
            dones = exp['dones']
            values = exp['values']
            next_obs = exp['next_obs']
            next_done = exp['next_done']

            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1)
                T = rewards.shape[0]
                advantages = torch.zeros_like(rewards, device=self.device)
                lastgaelam = 0
                for t in reversed(range(T)):
                    if t == T - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values

            all_obs.append(obs)
            all_actions.append(actions)
            all_logprobs.append(logprobs)
            all_rewards.append(rewards)
            all_dones.append(dones)
            all_values.append(values)
            all_returns.append(returns)
            all_advantages.append(advantages)

        # Flatten across all experience segments (i.e., list of variable-length tensors)
        b_obs = torch.cat(all_obs, dim=0)
        b_actions = torch.cat(all_actions, dim=0)
        b_logprobs = torch.cat(all_logprobs, dim=0)
        b_values = torch.cat(all_values, dim=0)
        b_returns = torch.cat(all_returns, dim=0)
        b_advantages = torch.cat(all_advantages, dim=0)

        # Shuffle for batching
        batch_size = b_obs.shape[0]
        b_inds = np.arange(batch_size)
        np.random.shuffle(b_inds)

        # Value Update
        kl_explode = False
        for start in range(0, batch_size, args.value_minibatch_size):
            if kl_explode:
                break
            end = start + args.value_minibatch_size
            mb_inds = b_inds[start:end]
            newvalue = self.agent.get_value(b_obs[mb_inds]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            loss = v_loss * args.vf_coef
            self.value_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
            self.value_optimizer.step()

        if is_warmup:
            return {"value_loss": v_loss.item(), "policy_loss": 0.0, "approx_kl": 0.0}

        # Policy Update
        self.policy_optimizer.zero_grad()
        total_approx_kl = torch.tensor(0.0, device=self.device)
        pg_loss = entropy_loss = old_approx_kl = torch.tensor(0.0, device=self.device)
        clipfracs = []
        policy_update_steps = 0

        for start in range(0, batch_size, args.policy_minibatch_size):
            end = start + args.policy_minibatch_size
            mb_inds = b_inds[start:end]

            # Gradient accumulation
            if policy_update_steps % args.gradient_checkpointing_steps == 0:
                total_approx_kl = torch.tensor(0.0, device=self.device)

            # Get action values
            _, newlogprob, entropy, _ = self.agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds], is_warmup, return_value=False
            )

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                total_approx_kl += approx_kl / args.gradient_checkpointing_steps
                clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss
            loss /= args.gradient_checkpointing_steps  # normalize for accumulation

            loss.backward()

            policy_update_steps += 1
            if policy_update_steps % args.gradient_checkpointing_steps == 0:
                if args.target_kl is not None and total_approx_kl > args.target_kl:
                    self.policy_optimizer.zero_grad()
                    kl_explode = True
                    policy_update_steps -= args.gradient_checkpointing_steps
                    break  # early stopping

                nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()

        # Logging
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/policy_update_times", policy_update_steps // args.gradient_checkpointing_steps,
                               global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)

        return {"value_loss": v_loss.item(), "policy_loss": pg_loss.item(), "approx_kl": approx_kl.item()}

    def reset_optimizers(self, args, update):
        """ warm up the learning rate of the optimizers. """
        num_updates = args.total_timesteps // args.batch_size
        num_critic_warm_up_updates = args.critic_warm_up_steps // args.batch_size
        frac = 1.0 - (update - 1.0 - num_critic_warm_up_updates) / num_updates
        self.policy_optimizer.param_groups[0]["lr"] = frac * args.policy_learning_rate
        self.value_optimizer.param_groups[0]["lr"] = frac * args.value_learning_rate

    def get_optimizer_info(self):
        """Return optimizer learning rates for logging purposes."""
        info = {
            "policy_lr": self.policy_optimizer.param_groups[0]["lr"],
            "value_lr": self.value_optimizer.param_groups[0]["lr"],
        }
        return info