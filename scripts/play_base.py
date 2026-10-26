"""Visualize a base-policy checkpoint in the DexReMoE env.

Opens a render window and runs the checkpoint's deterministic policy
(`act_inference`, no sampling noise). Watch:
  - does the hand keep the object in the fingers (no drop)?
  - does the object rotate towards the goal marker?
  - any jitter / slipping / hand opening?

Usage:
    python DexReMoERebuild/scripts/play_base.py \
        --checkpoint logs/rsl_rl/base_policy/<ts>/model_final.pt
"""

import builtins
builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False

import argparse
import os


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--cache_root", type=str,
                   default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache_isaac_validated")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


ARGS = _parse_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["PRIVACY_CONSENT"] = "Y"

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": False})

import carb  # noqa: E402
carb.settings.get_settings().set(
    "/persistent/isaac/asset_root/cloud",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
)

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


def _build_policy(checkpoint, device):
    _opr.DexReMoEActorCritic = DexReMoEActorCritic
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state_dict", ckpt)

    dummy_env = type("Dummy", (), {})()
    # OnPolicyRunner requires a proper VecEnv to build policy dims; we only need the class.
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
    return ckpt, model_state, train_cfg


def main():
    cfg = DexReMoEReorientEnvCfg()
    cfg.cache_root = ARGS.cache_root
    cfg.scene.num_envs = ARGS.num_envs
    cfg.data_bucket = None
    cfg.seed = ARGS.seed
    cfg.viewer.eye = (1.5, 1.5, 1.5)

    env = DexReMoEReorientEnv(cfg)
    env.sim.play()

    wrapped = _Wrapper(env)

    ckpt, model_state, train_cfg = _build_policy(ARGS.checkpoint, wrapped.device)

    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=wrapped.device)
    runner.alg.policy.load_state_dict(model_state)
    policy = runner.get_inference_policy(device=wrapped.device)

    cmd = env.command_manager.get_term("object_pose")

    print("=" * 70, flush=True)
    print(f"[play] checkpoint     = {ARGS.checkpoint}", flush=True)
    print(f"[play] iter in ckpt   = {ckpt.get('iter', 'unknown')}", flush=True)
    print(f"[play] num_envs       = {ARGS.num_envs}", flush=True)
    print("=" * 70, flush=True)

    obs, _ = wrapped.reset()
    obs = obs.to(wrapped.device)

    ep = 0
    step_in_ep = 0
    while True:
        with torch.no_grad():
            actions = policy(obs)
        obs, _rew, dones, _ = wrapped.step(actions)
        obs = obs.to(wrapped.device)
        step_in_ep += 1
        env.sim.render()

        if step_in_ep % 30 == 0:
            dtheta = cmd.metrics["hold_counter"][0].item()
            consec = cmd.metrics["consecutive_success"][0].item()
            print(f"[play] ep{ep} step{step_in_ep:3d}  "
                  f"consec={consec:.0f}  hold={dtheta:.0f}  "
                  f"max_consec={cmd.get_max_consecutive_success()[0].item():.0f}", flush=True)

        if dones.any():
            print(f"[play] ep{ep} terminated after {step_in_ep} steps  "
                  f"max_consec={cmd.get_max_consecutive_success()[0].item():.0f}", flush=True)
            ep += 1
            step_in_ep = 0


if __name__ == "__main__":
    main()
