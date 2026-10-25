"""DexReMoE base-policy env package.

Env-only. Networks (PointNet++, mu_e, policy, gate) live in the training scripts.
The env exposes raw observation groups and physical state; all learning-side
compression happens in the trainer.
"""

__version__ = "0.1.0"
