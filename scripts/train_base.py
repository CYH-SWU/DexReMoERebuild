"""DexReMoE base-policy training entry point.

Flow:
    SimulationApp -> isaaclab -> DexReMoEReorientEnv -> VecEnv wrapper
    -> DexReMoEActorCritic (injected into rsl_rl runners namespace)
    -> OnPolicyRunner.learn -> checkpoint(s).

Key details:
  - RSL-RL only sees a flat observation tensor. The wrapper concatenates the
    four env obs groups into a single OBS_LAYOUT.total-dim tensor
    (policy | privileged | point_cloud | size_encoding).
  - The `class_name` in rsl_rl's train_cfg is resolved with `eval()` inside
    rsl_rl.runners.on_policy_runner, so DexReMoEActorCritic must be injected
    into that module's namespace before the runner is constructed.
  - RSL-RL's own `empirical_normalization` is disabled; all normalization
    lives inside DexReMoEActorCritic.

Usage:
    python DexReMoERebuild/scripts/train_base.py \
        --num_envs 128 --max_iter 3000 --seed 0
"""

import builtins
builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False

import argparse
import datetime as _dt
import os

# --- parse our args BEFORE starting Kit (Kit eats unknown argv) ---
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--max_iter", type=int, default=3000)
    p.add_argument("--save_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # --- PPO hyper-parameters, aligned with paper Table II ---
    # learning_rate: paper 5e-3
    # num_steps_per_env (horizon length): paper 8, ours 24.
    #   Paper ran 32768 envs. At 64 envs, T=8 gives only 512 samples/iter and
    #   a rollout that is far shorter than the GAE effective horizon (~20
    #   with lam=0.95), so value bootstrap is inaccurate and every episode is
    #   split across 75 iterations. T=24 matches the GAE horizon, keeps the
    #   on-policy assumption healthy (ratio stays near 1), and gives 1536
    #   samples/iter. T > ~40 would violate on-policy assumptions.
    # desired_kl (kl threshold): paper 0.02
    # clip_param: paper 0.2 (fixed below)
    # gamma: paper 0.99 (fixed below)
    # lam (PPO tau): paper 0.95 (fixed below)
    #
    # Paper minibatch = 16384 over 262144 total samples = 16 mini-batches.
    # At 64 env * 24 horizon = 1536 samples, 16 splits gives 96 samples/batch,
    # which is noisy. `--num_mini_batches` can be overridden; a rule of thumb
    # is `num_envs * num_steps_per_env / 256` (target ~256 per batch).
    # 5e-3 is the paper value, but it assumes 262144 samples/iter. At 64 env
    # * 24 horizon = 1536 samples/iter the gradient is ~170x noisier and KL
    # routinely overshoots desired_kl. Starting at 3e-3 with the adaptive
    # scheduler (which is already in RSL-RL) is a gentler warm start.
    p.add_argument("--learning_rate", type=float, default=3e-3)
    p.add_argument("--num_steps_per_env", type=int, default=24)
    p.add_argument("--num_mini_batches", type=int, default=None,
                   help="Override mini-batch count. Default: max(1, min(16, num_envs * horizon // 256)), "
                        "i.e. target ~256 samples/batch while capping at the paper ratio 1/16.")
    p.add_argument("--init_noise_std", type=float, default=0.5,
                   help="Std of the action distribution at init. Action scale is 0.175; "
                        "0.5 -> joint jitter of ~0.09 rad per step.")
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None,
                   help="If set, checkpoint and tensorboard logs go to "
                        "logs/rsl_rl/base_policy/<run_name>/ instead of a timestamp.")
    p.add_argument("--cache_root", type=str,
                   default="/home/chen/my_project/DexReMoERebuild/assets/grasp_cache_isaac_validated")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--headless", action="store_true", default=True)
    # placeholder so a caller can pass the usual `--headless` and not fail
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

import torch  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

SimulationContext._apply_render_settings_from_cfg = lambda self: None

from dexremoe_tasks.envs import DexReMoEReorientEnvCfg, DexReMoEReorientEnv  # noqa: E402
from dexremoe_agents.networks import DexReMoEActorCritic, OBS_LAYOUT  # noqa: E402

import rsl_rl.runners.on_policy_runner as _opr  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


# --------------------------------------------------------------------- wrapper


def _flatten_obs(obs_dict):
    """Dict[str, Tensor] -> (N, OBS_LAYOUT.total) in the canonical order."""
    policy = obs_dict["policy"]
    privileged = obs_dict["privileged"]
    point_cloud = obs_dict["point_cloud"].reshape(obs_dict["point_cloud"].shape[0], -1)
    size_encoding = obs_dict["size_encoding"]
    return torch.cat([policy, privileged, point_cloud, size_encoding], dim=-1)


class DexReMoEVecEnvWrapper:
    """Flatten the 4 env obs groups into one tensor for RSL-RL.

    Duck-types the subset of IsaacLab's `RslRlVecEnvWrapper` that RSL-RL's
    `OnPolicyRunner` actually touches.
    """

    def __init__(self, env):
        self.env = env
        self.unwrapped = env
        self.cfg = env.cfg
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_actions = env.action_manager.total_action_dim
        self.num_obs = OBS_LAYOUT.total
        # critic uses the same 1660-dim tensor; RSL-RL will pass it through
        # as `privileged_obs` so `evaluate()` can slice it the same way.
        self.num_privileged_obs = OBS_LAYOUT.total
        self.max_episode_length = env.max_episode_length
        self.episode_length_buf = env.episode_length_buf

    # --- RSL-RL API ---

    def get_observations(self):
        # update_history=False: this call is only used to inspect obs shapes in
        # OnPolicyRunner.__init__ and at the start of learn(). Updating the
        # history buffer here would corrupt the 3-step window for the first
        # real step.
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
        # RSL-RL uses time_outs to bootstrap value for truncated episodes;
        # without it, every timed-out episode is treated as terminal.
        extras["time_outs"] = trunc
        dones = term | trunc
        return _flatten_obs(obs_dict), rew, dones, extras

    # --- episode_length_buf is reassigned by RSL-RL when
    #     init_at_random_ep_len=True; forward reads and writes to the env.

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    # --- misc passthrough ---

    def close(self):
        self.env.close()


# ----------------------------------------------------------------------- main


def main():
    # ---- env ----
    cfg = DexReMoEReorientEnvCfg()
    cfg.cache_root = ARGS.cache_root
    cfg.scene.num_envs = ARGS.num_envs
    cfg.data_bucket = None   # base stage: stratified 1/3 across size buckets
    cfg.seed = ARGS.seed
    env = DexReMoEReorientEnv(cfg)

    # env construction does NOT reset it: object_pos_init is all zeros until
    # reset runs, so the very first step would see object_far fire for every
    # env (||obj - 0|| >> 3*size). Reset once to populate all per-env state.
    env.reset()

    wrapped = DexReMoEVecEnvWrapper(env)

    print("=" * 78, flush=True)
    print(f"[train] num_envs           = {ARGS.num_envs}", flush=True)
    print(f"[train] obs total          = {OBS_LAYOUT.total}", flush=True)
    print(f"[train] action dim         = {wrapped.num_actions}", flush=True)
    print(f"[train] max_episode_length = {wrapped.max_episode_length}", flush=True)
    print(f"[train] data_bucket        = {cfg.data_bucket}", flush=True)
    print("=" * 78, flush=True)

    # ---- inject policy class so rsl_rl's eval() can find it ----
    _opr.DexReMoEActorCritic = DexReMoEActorCritic

    # ---- log dir ----
    if ARGS.log_dir is not None:
        log_dir = ARGS.log_dir
    elif ARGS.run_name is not None:
        log_dir = os.path.join(
            "/home/chen/my_project/DexReMoERebuild/logs/rsl_rl/base_policy",
            ARGS.run_name,
        )
    else:
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(
            "/home/chen/my_project/DexReMoERebuild/logs/rsl_rl/base_policy", ts
        )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] log_dir            = {log_dir}", flush=True)

    # ---- resolve num_mini_batches ----
    if ARGS.num_mini_batches is None:
        total = ARGS.num_envs * ARGS.num_steps_per_env
        num_mb = max(1, min(16, total // 256))
    else:
        num_mb = ARGS.num_mini_batches
    print(f"[train] num_mini_batches   = {num_mb} "
          f"({ARGS.num_envs * ARGS.num_steps_per_env // num_mb} samples/batch)", flush=True)

    # ---- agent cfg (plain dict, rsl_rl reads it directly) ----
    train_cfg = {
        "policy": {
            "class_name": "DexReMoEActorCritic",
            "init_noise_std": ARGS.init_noise_std,
            "noise_std_type": "scalar",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,             # paper
            "entropy_coef": 0.002,         # not given in paper
            "num_learning_epochs": 5,      # not given in paper (rsl_rl default)
            "num_mini_batches": num_mb,
            "learning_rate": ARGS.learning_rate,   # paper 5e-3
            "schedule": "adaptive",
            "gamma": 0.99,                 # paper
            "lam": 0.95,                   # paper (PPO tau)
            "desired_kl": 0.02,            # paper (kl threshold)
            "max_grad_norm": 1.0,
        },
        "num_steps_per_env": ARGS.num_steps_per_env,
        "save_interval": ARGS.save_interval,
        "empirical_normalization": False,   # normalization lives inside the policy
        "logger": "tensorboard",
    }

    runner = OnPolicyRunner(
        wrapped, train_cfg, log_dir=log_dir, device=wrapped.device
    )

    if ARGS.resume is not None:
        print(f"[train] resuming from {ARGS.resume}", flush=True)
        runner.load(ARGS.resume)

    # ---- learn ----
    try:
        runner.learn(
            num_learning_iterations=ARGS.max_iter,
            init_at_random_ep_len=True,
        )
    finally:
        # save final checkpoint regardless of how we exit
        final_path = os.path.join(log_dir, "model_final.pt")
        try:
            runner.save(final_path)
            print(f"[train] saved final checkpoint: {final_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train] failed to save final checkpoint: {e}", flush=True)


if __name__ == "__main__":
    main()
