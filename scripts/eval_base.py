"""Evaluate a base-policy checkpoint on the fixed eval pool.

Uses env.eval_mode = True, which switches the sampler to a deterministic
mapping env i -> eval_pool[(eval_offset + i) % n_eval]. Every eval run
sees the same rows, so S_mean / S_min / S_max are comparable across
checkpoints.

Reports, per scale bucket and overall:
  - S_mean / S_min / S_max  (max_consecutive_success per episode)
  - mean orientation error (deg) at the end of each episode
  - fraction of episodes that reached success at least once

Usage:
    python DexReMoERebuild/scripts/eval_base.py \
        --checkpoint logs/rsl_rl/base_policy/<ts>/model_final.pt \
        --num_envs 64 --episodes 64
"""

import builtins
builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False

import argparse
import os


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--episodes", type=int, default=64,
                   help="Total episodes to run. Split evenly across the eval pool.")
    p.add_argument("--cache_root", type=str,
                   default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache_isaac_validated")
    p.add_argument("--eval_n_per_file", type=int, default=10,
                   help="Rows sampled from each train file in the eval pool.")
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=None,
                   help="Rows per reset chunk. Defaults to num_envs.")
    return p.parse_args()


ARGS = _parse_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True})

import carb  # noqa: E402
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/cloud",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
SimulationContext._apply_render_settings_from_cfg = lambda self: None

from dexremoe_tasks.envs import DexReMoEReorientEnvCfg, DexReMoEReorientEnv  # noqa: E402
from dexremoe_agents.networks import DexReMoEActorCritic, OBS_LAYOUT  # noqa: E402

import rsl_rl.runners.on_policy_runner as _opr  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def _flatten_obs(obs_dict):
    policy = obs_dict["policy"]
    privileged = obs_dict["privileged"]
    point_cloud = obs_dict["point_cloud"].reshape(obs_dict["point_cloud"].shape[0], -1)
    size_encoding = obs_dict["size_encoding"]
    return torch.cat([policy, privileged, point_cloud, size_encoding], dim=-1)


class _Wrapper:
    def __init__(self, env):
        self.env = env
        self.unwrapped = env
        self.cfg = env.cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = env.action_manager.total_action_dim
        self.num_obs = OBS_LAYOUT.total
        self.num_privileged_obs = OBS_LAYOUT.total
        self.max_episode_length = env.max_episode_length

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, v):
        self.env.episode_length_buf = v

    def get_observations(self):
        obs_dict = self.env.observation_manager.compute(update_history=False)
        return _flatten_obs(obs_dict), {"observations": obs_dict}

    def reset(self):
        obs_dict, extras = self.env.reset()
        extras = dict(extras) if extras else {}
        extras["observations"] = obs_dict
        return _flatten_obs(obs_dict), extras

    def step(self, actions):
        obs_dict, rew, term, trunc, extras = self.env.step(actions)
        extras = dict(extras) if extras else {}
        extras["observations"] = obs_dict
        extras["time_outs"] = trunc
        return _flatten_obs(obs_dict), rew, term | trunc, extras


def main():
    # ---- env ----
    cfg = DexReMoEReorientEnvCfg()
    cfg.cache_root = ARGS.cache_root
    cfg.scene.num_envs = ARGS.num_envs
    cfg.data_bucket = None
    cfg.seed = 0
    cfg.eval_n_per_file = ARGS.eval_n_per_file
    cfg.eval_seed = ARGS.eval_seed

    env = DexReMoEReorientEnv(cfg)
    env.eval_mode = True   # setter triggers reset; sampling becomes deterministic

    wrapped = _Wrapper(env)

    # ---- policy ----
    _opr.DexReMoEActorCritic = DexReMoEActorCritic
    ckpt = torch.load(ARGS.checkpoint, map_location=wrapped.device, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)

    train_cfg = {
        "policy": {
            "class_name": "DexReMoEActorCritic",
            "init_noise_std": 0.5,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.002,
            "num_learning_epochs": 5,
            "num_mini_batches": 2,
            "learning_rate": 5e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.02,
            "max_grad_norm": 1.0,
        },
        "num_steps_per_env": 8,
        "save_interval": 100,
        "empirical_normalization": False,
        "logger": "tensorboard",
    }
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=wrapped.device)
    runner.alg.policy.load_state_dict(model_state)
    policy = runner.get_inference_policy(device=wrapped.device)

    # ---- eval loop ----
    cmd = env.command_manager.get_term("object_pose")

    n_eps_target = ARGS.episodes
    per_env_eps: list[list[dict]] = [[] for _ in range(env.num_envs)]
    active = np.zeros(env.num_envs, dtype=bool)
    ep_start_scale = np.zeros(env.num_envs, dtype=np.float32)
    ep_count = 0

    obs, extras = wrapped.reset()
    obs = obs.to(wrapped.device)
    active[:] = True
    ep_start_scale[:] = env.active_scale().cpu().numpy()

    # We need dtheta at termination; grab it from cmd metrics each step
    steps = 0
    max_steps = 600 + 10  # slightly over max episode length

    while ep_count < n_eps_target and steps < max_steps * (n_eps_target // env.num_envs + 2):
        with torch.no_grad():
            actions = policy(obs)

        # Read cur_max BEFORE step: env.step() resets cmd._max_consec for
        # terminated envs, so post-step reads give 0 for the very episodes
        # we need to score.
        cur_max = cmd.get_max_consecutive_success().cpu().numpy()

        obs, _rew, dones, extras = wrapped.step(actions)
        obs = obs.to(wrapped.device)
        steps += 1

        done_ids = dones.cpu().numpy() if torch.is_tensor(dones) else np.asarray(dones)
        for i in np.where(done_ids)[0]:
            if not active[i]:
                continue
            per_env_eps[i].append({
                "S": float(cur_max[i]),
                "scale": float(ep_start_scale[i]),
            })
            ep_count += 1
            active[i] = False

        if not active.any():
            obs, extras = wrapped.reset()
            obs = obs.to(wrapped.device)
            active[:] = True
            ep_start_scale[:] = env.active_scale().cpu().numpy()

    # ---- aggregate ----
    all_eps = [e for env_eps in per_env_eps for e in env_eps]
    if not all_eps:
        print("[eval] no episodes completed")
        return

    S = np.array([e["S"] for e in all_eps], dtype=np.float32)
    scales = np.array([e["scale"] for e in all_eps], dtype=np.float32)

    def bucket_mask(s, lo, hi):
        return (s > lo) & (s <= hi)

    print("=" * 70, flush=True)
    print(f"[eval] episodes     = {len(all_eps)}", flush=True)
    print(f"[eval] S_mean       = {S.mean():.3f}", flush=True)
    print(f"[eval] S_min        = {S.min():.3f}", flush=True)
    print(f"[eval] S_max        = {S.max():.3f}", flush=True)
    print(f"[eval] success_rate = {(S > 0).mean() * 100:.1f}%", flush=True)

    for name, lo, hi in (("b0", -1.0, 0.74), ("b1", 0.74, 0.78), ("b2", 0.78, 0.82), ("b3", 0.82, 10.0)):
        m = bucket_mask(scales, lo, hi)
        if m.sum() > 0:
            print(f"[eval] {name}: n={m.sum():3d}  S_mean={S[m].mean():.3f}  "
                  f"success={(S[m] > 0).mean() * 100:.1f}%", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
