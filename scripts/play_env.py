"""Interactive visualizer for the DexReMoE env.

Opens a window and runs the env so you can visually confirm:
  - hand at z=1.0, palm up
  - object at fingertip (z~1.12)
  - action=0 keeps the hand at the nominal light-close grasp (q_nominal)
  - zero action for many steps leaves the grasp stable

Close the window or Ctrl+C to exit.
"""

import builtins
builtins.ISAAC_LAUNCHED_FROM_TERMINAL = False
import os
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
os.environ['PRIVACY_CONSENT'] = 'Y'

from isaacsim import SimulationApp
app = SimulationApp({'headless': False})
import carb
carb.settings.get_settings().set(
    '/persistent/isaac/asset_root/cloud',
    'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1',
)

import torch
from isaaclab.sim import SimulationContext
SimulationContext._apply_render_settings_from_cfg = lambda self: None

from dexremoe_tasks.envs import DexReMoEReorientEnvCfg, DexReMoEReorientEnv

REPO = '/home/chen/my_project'
CACHE_ROOT = os.path.join(REPO, 'DexReMoERebuild', 'assets', 'grasp_cache_isaac_validated')

cfg = DexReMoEReorientEnvCfg()
cfg.cache_root = CACHE_ROOT
cfg.scene.num_envs = 1
cfg.sim.render_interval = 1
cfg.viewer.eye = (1.5, 1.5, 1.5)

env = DexReMoEReorientEnv(cfg)
env.sim.play()

rob = env.scene['robot']
act_term = env.action_manager.get_term('joint_pos')

def print_state(tag):
    st = env.active_root_state_w()
    print(f'[{tag}]', flush=True)
    print(f'  joint_pos[0:4] = {[round(x,3) for x in rob.data.joint_pos[0,:4].tolist()]}', flush=True)
    print(f'  obj pos        = {[round(x,3) for x in st["pos"][0].tolist()]}', flush=True)
    print(f'  hand root      = {[round(x,3) for x in rob.data.root_pos_w[0].tolist()]}', flush=True)

obs, _ = env.reset()
print_state('after reset')

print('\n[PLAY] action=0 for 5 steps -> hand should settle to q_nominal (light close).', flush=True)
zeros = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for _ in range(5):
    env.step(zeros)
    env.sim.render()
print_state('after 5 steps of action=0')

print('\n[PLAY] Running zero-action episodes. Watch: object stays in hand, fingers hold grasp.', flush=True)
print('[PLAY] Close the window or Ctrl+C to exit.', flush=True)
while True:
    env.step(zeros)
    env.sim.render()
