"""Pure-math verification of HORA -> IsaacLab base frame transform.

No Isaac Sim needed. Compares:
  - hand base rotation in HORA (Isaac Gym) vs IsaacLab (USD)
  - object position relative to hand base, before and after R_delta
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def wxyz_to_rot(q_wxyz):
    return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


# Hand base poses
hand_pos_hora = np.array([0.0, 0.0, 0.5])
q_hora_wxyz = np.array([0.5, 0.5, -0.5, 0.5])

hand_pos_isaac = np.array([0.0, 0.0, 0.5])
q_isaac_wxyz = np.array([0.25755104422569275, 0.28304505348205566, 0.6833301782608032, -0.621782124042511])

R_hora = wxyz_to_rot(q_hora_wxyz)
R_isaac = wxyz_to_rot(q_isaac_wxyz)

print("=== Hand base rotation matrices ===")
print("R_hora (columns = HORA base axes in world):")
print(np.round(R_hora.as_matrix(), 3))
print("R_isaac (columns = IsaacLab base axes in world):")
print(np.round(R_isaac.as_matrix(), 3))

# Object pose from dataset row 0
obj_pos_hora = np.array([7.995844e-03, 5.407836e-04, 6.274506e-01])
obj_quat_hora_xyzw = np.array([-0.43293193, -0.12931427, 0.10492182, 0.88591146])  # xyzw

R_obj_hora = Rot.from_quat(obj_quat_hora_xyzw)

print("\n=== Object relative to hand base (HORA frame) ===")
rel_hora = obj_pos_hora - hand_pos_hora
print(f"rel position (world): {np.round(rel_hora, 4)}")
rel_in_base_hora = R_hora.inv().apply(rel_hora)
print(f"rel position (in HORA base frame): {np.round(rel_in_base_hora, 4)}")
print("  -> this is where the object sits w.r.t. the palm in HORA's own coordinates")

# Transform: R_delta = R_isaac * R_hora^{-1}
R_delta = R_isaac * R_hora.inv()
print("\n=== R_delta = R_isaac * R_hora^{-1} ===")
print(np.round(R_delta.as_matrix(), 3))

# Transform object position
rel_delta = R_delta.apply(rel_hora)
obj_pos_isaac = hand_pos_isaac + rel_delta
print("\n=== Object position after transform ===")
print(f"rel position (world, transformed): {np.round(rel_delta, 4)}")
print(f"abs position (IsaacLab world): {np.round(obj_pos_isaac, 4)}")

rel_in_base_isaac = R_isaac.inv().apply(rel_delta)
print(f"rel position (in IsaacLab base frame): {np.round(rel_in_base_isaac, 4)}")

# The key check: rel_in_base_isaac should equal rel_in_base_hora
print("\n=== Sanity check ===")
print(f"rel in HORA base: {np.round(rel_in_base_hora, 4)}")
print(f"rel in Isaac base: {np.round(rel_in_base_isaac, 4)}")
print(f"difference: {np.round(rel_in_base_isaac - rel_in_base_hora, 6)}")
print("  -> should be ~0 if the transform is correct")

# Transform object orientation
R_obj_isaac = R_delta * R_obj_hora
q_obj_isaac_xyzw = R_obj_isaac.as_quat()
q_obj_isaac_wxyz = np.array([q_obj_isaac_xyzw[3], q_obj_isaac_xyzw[0], q_obj_isaac_xyzw[1], q_obj_isaac_xyzw[2]])
print("\n=== Object orientation after transform ===")
print(f"q_hora (wxyz): {np.round(q_hora_wxyz, 4)}")
print(f"q_obj_hora (wxyz): {np.round([obj_quat_hora_xyzw[3], obj_quat_hora_xyzw[0], obj_quat_hora_xyzw[1], obj_quat_hora_xyzw[2]], 4)}")
print(f"q_obj_isaac (wxyz): {np.round(q_obj_isaac_wxyz, 4)}")

# Alternative: reverse direction, in case R_delta should be R_hora * R_isaac^{-1}
R_delta_rev = R_hora * R_isaac.inv()
rel_rev = R_delta_rev.apply(rel_hora)
obj_pos_rev = hand_pos_isaac + rel_rev
print("\n=== Reverse direction (for comparison) ===")
print(f"obj position if R_delta reversed: {np.round(obj_pos_rev, 4)}")
rel_in_base_rev = R_isaac.inv().apply(rel_rev)
print(f"rel in Isaac base (reversed): {np.round(rel_in_base_rev, 4)}")
print(f"difference vs HORA base: {np.round(rel_in_base_rev - rel_in_base_hora, 6)}")
