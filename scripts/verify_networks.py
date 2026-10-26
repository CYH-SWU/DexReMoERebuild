"""Static correctness checks for dexremoe_agents networks.

No env, no IsaacLab, no GPU required. Pure tensor operations on CPU.

Covers:
  - module import
  - PointNet / MuE / RunningMeanStd shapes and behaviour
  - DexReMoEActorCritic construction, forward, act, evaluate, log_prob
  - gradient flow through the full network (including the in-place masked
    normalizer, which is the riskiest spot)
  - normalization sanity (RunningMeanStd updates, LayerNorm output stats)
  - init scaling (actor last layer -> initial action near zero)
  - export_encoders / load_encoders / freeze_encoders
"""

import math

import torch

from dexremoe_agents.networks import (
    OBS_LAYOUT,
    PRIV_CONTINUOUS_INDICES,
    DexReMoEActorCritic,
    MuE,
    PointNet,
    RunningMeanStd,
)


RESULTS = []


def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}" + (f" | {detail}" if detail else "")
    print(line, flush=True)
    RESULTS.append((name, bool(ok), detail))


def section(title):
    print(f"\n===== {title} =====", flush=True)


# ------------------------------------------------------------------ 1. layout

def check_layout():
    section("1. OBS_LAYOUT")
    L = OBS_LAYOUT
    report("policy slice", L.policy_start == 0 and L.policy_dim == 96, f"{L.policy_start}:{L.policy_start + L.policy_dim}")
    report("privileged slice", L.priv_start == 96 and L.priv_dim == 23, f"{L.priv_start}:{L.priv_start + L.priv_dim}")
    report("point_cloud slice", L.point_start == 119 and L.point_flat_dim == 1536, f"{L.point_start}:{L.point_start + L.point_flat_dim}")
    report("size_encoding slice", L.size_start == 1655 and L.size_dim == 5, f"{L.size_start}:{L.size_start + L.size_dim}")
    report("total = 1660", L.total == 1660, f"total={L.total}")
    report("policy + priv + point + size == total",
           L.policy_dim + L.priv_dim + L.point_flat_dim + L.size_dim == L.total)

    n_cont = len(PRIV_CONTINUOUS_INDICES)
    report("PRIV_CONTINUOUS_INDICES = 11 dims", n_cont == 11, f"n={n_cont}, idx={PRIV_CONTINUOUS_INDICES}")
    report("continuous idx within [0, 23)", all(0 <= i < 23 for i in PRIV_CONTINUOUS_INDICES))
    # 常量/离散必须是补集
    excluded = set(range(23)) - set(PRIV_CONTINUOUS_INDICES)
    expected_excluded = {1, 3, 4, 5, 9, 10, 11, 12, 19, 20, 21, 22}  # friction, com, quat, one-hot
    report("excluded set matches design", excluded == expected_excluded, f"excluded={sorted(excluded)}")


# ------------------------------------------------------------------ 2. pointnet

def check_pointnet():
    section("2. PointNet")
    net = PointNet(out_dim=32)
    n, p = 4, 512
    pts = torch.rand(n, p, 3) * 2 - 1  # in [-1, 1]
    out = net(pts)
    report("output shape (N, 32)", out.shape == (n, 32), f"got {tuple(out.shape)}")

    # LayerNorm -> each sample has ~zero mean and ~unit variance over 32 dims
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    report("LayerNorm mean ~ 0", mean.abs().max() < 1e-3, f"max|mean|={mean.abs().max().item():.2e}")
    report("LayerNorm std ~ 1", (std - 1).abs().max() < 0.1, f"max|std-1|={(std - 1).abs().max().item():.3f}")

    # permutation sensitivity: PointNet should be (nearly) permutation invariant
    perm = torch.randperm(p)
    out2 = net(pts[:, perm])
    diff = (out - out2).abs().max().item()
    report("permutation invariant", diff < 1e-4, f"max diff={diff:.2e}")

    n_params = sum(p_.numel() for p_ in net.parameters())
    report("parameter count reasonable", 300_000 < n_params < 2_000_000, f"{n_params:,}")


# ------------------------------------------------------------------ 3. mu_e

def check_mu_e():
    section("3. MuE")
    net = MuE(in_dim=55, hidden=128, out_dim=66)
    n = 4
    e_t = torch.randn(n, 55)
    out = net(e_t)
    report("output shape (N, 66)", out.shape == (n, 66), f"got {tuple(out.shape)}")
    mean = out.mean(dim=-1)
    std = out.std(dim=-1)
    report("LayerNorm mean ~ 0", mean.abs().max() < 1e-3, f"max|mean|={mean.abs().max().item():.2e}")
    report("LayerNorm std ~ 1", (std - 1).abs().max() < 0.15, f"max|std-1|={(std - 1).abs().max().item():.3f}")


# ------------------------------------------------------------------ 4. rms

def check_rms():
    section("4. RunningMeanStd")
    rms = RunningMeanStd(8)
    rms.train()

    # feed three batches from N(5, 4)
    x_batches = [torch.randn(1024, 8) * 4 + 5 for _ in range(3)]
    for x in x_batches:
        rms(x)

    report("mean approaches 5", (rms.mean - 5).abs().max() < 0.6,
           f"mean~{[round(v, 2) for v in rms.mean.tolist()[:3]]}")
    report("var approaches 16", (rms.var.sqrt() - 4).abs().max() < 0.6,
           f"std~{[round(v, 2) for v in rms.var.sqrt().tolist()[:3]]}")

    # normalized output should be roughly zero mean unit std
    y = rms(torch.randn(2048, 8) * 4 + 5)
    report("normalized output mean ~ 0", y.mean().abs() < 0.2, f"mean={y.mean().item():.3f}")
    report("normalized output std ~ 1", (y.std() - 1).abs() < 0.2, f"std={y.std().item():.3f}")

    # eval mode must not update stats
    rms.eval()
    mean_before = rms.mean.clone()
    rms(torch.randn(64, 8) * 100 + 50)
    report("eval mode does not update", torch.equal(rms.mean, mean_before))


# ------------------------------------------------------------------ 5. actor_critic forward

def make_actor_critic(noise_std=0.01):
    return DexReMoEActorCritic(
        num_actor_obs=OBS_LAYOUT.total,
        num_critic_obs=OBS_LAYOUT.total,
        num_actions=16,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=noise_std,
    )


def check_actor_critic_shapes():
    section("5. DexReMoEActorCritic shapes")
    ac = make_actor_critic(noise_std=0.01)
    ac.train()
    n = 8
    obs = torch.randn(n, OBS_LAYOUT.total)

    # fill the point-cloud slice with values in [-1, 1] to mimic unit-scale clouds
    L = OBS_LAYOUT
    obs[:, L.point_start:L.point_start + L.point_flat_dim] = torch.randn(n, L.point_flat_dim) * 0.3

    a = ac.act(obs)
    report("act() -> (N, 16)", a.shape == (n, 16), f"got {tuple(a.shape)}")

    lp = ac.get_actions_log_prob(a)
    report("log_prob -> (N,)", lp.shape == (n,), f"got {tuple(lp.shape)}")

    ai = ac.act_inference(obs)
    report("act_inference() -> (N, 16)", ai.shape == (n, 16), f"got {tuple(ai.shape)}")

    v = ac.evaluate(obs)
    report("evaluate() -> (N, 1)", v.shape == (n, 1), f"got {tuple(v.shape)}")

    report("action_std finite", torch.isfinite(ac.action_std).all().item(),
           f"std~{ac.action_std.mean().item():.3f}")
    report("entropy finite", torch.isfinite(ac.entropy).all().item(),
           f"entropy~{ac.entropy.mean().item():.3f}")


# ------------------------------------------------------------------ 6. gradient flow

def check_backward():
    section("6. Backward pass (in-place normalizer risk)")
    ac = make_actor_critic(noise_std=0.01)
    ac.train()
    n = 8
    L = OBS_LAYOUT
    obs = torch.randn(n, L.total)
    obs[:, L.point_start:L.point_start + L.point_flat_dim] = torch.randn(n, L.point_flat_dim) * 0.3

    a = ac.act(obs)
    v = ac.evaluate(obs)
    # PPO's actual actor loss path: -log_prob(actions) - entropy_coef * entropy.
    # `act()` samples without reparameterization, so gradient must flow through
    # log_prob / entropy, not through the sampled action tensor.
    log_prob = ac.get_actions_log_prob(a)
    entropy = ac.entropy
    loss = -log_prob.mean() - 0.01 * entropy.mean() + v.pow(2).mean()

    try:
        loss.backward()
        ok = True
        err = ""
    except Exception as e:  # noqa: BLE001
        ok = False
        err = f"{type(e).__name__}: {e}"
    report("loss.backward() succeeds", ok, err)
    if not ok:
        return

    # encoder grads must reach the actor's pointnet + mu_e
    actor_pn_grad = sum(p.grad.abs().sum().item() for p in ac.actor.encoder.pointnet.parameters() if p.grad is not None)
    actor_mu_grad = sum(p.grad.abs().sum().item() for p in ac.actor.encoder.mu_e.parameters() if p.grad is not None)
    report("grad reaches actor PointNet", actor_pn_grad > 0, f"sum|grad|={actor_pn_grad:.3e}")
    report("grad reaches actor MuE", actor_mu_grad > 0, f"sum|grad|={actor_mu_grad:.3e}")

    critic_pn_grad = sum(p.grad.abs().sum().item() for p in ac.critic.encoder.pointnet.parameters() if p.grad is not None)
    report("grad reaches critic PointNet", critic_pn_grad > 0, f"sum|grad|={critic_pn_grad:.3e}")

    # actor and critic encoders must be DIFFERENT objects
    same_pn = ac.actor.encoder.pointnet is ac.critic.encoder.pointnet
    same_mu = ac.actor.encoder.mu_e is ac.critic.encoder.mu_e
    report("actor / critic PointNet independent", not same_pn)
    report("actor / critic MuE independent", not same_mu)


# ------------------------------------------------------------------ 7. init scale

def check_init_scale():
    section("7. Init scale (actor output near zero)")
    torch.manual_seed(0)
    ac = make_actor_critic(noise_std=0.01)
    ac.eval()
    n = 64
    L = OBS_LAYOUT
    obs = torch.randn(n, L.total) * 0.5
    obs[:, L.point_start:L.point_start + L.point_flat_dim] = torch.randn(n, L.point_flat_dim) * 0.3

    with torch.no_grad():
        a = ac.act_inference(obs)
    abs_mean = a.abs().mean().item()
    abs_max = a.abs().max().item()
    report("initial action |mean| < 0.1", abs_mean < 0.1, f"|a| mean={abs_mean:.4f}")
    report("initial action |max| < 0.5", abs_max < 0.5, f"|a| max={abs_max:.4f}")


# ------------------------------------------------------------------ 8. encoder export

def check_export_freeze():
    section("8. export / load / freeze encoders")
    ac = make_actor_critic(noise_std=0.01)
    enc = ac.export_encoders()
    keys = sorted(enc.keys())
    expected = ["actor_mu_e", "actor_pointnet", "critic_mu_e", "critic_pointnet"]
    report("export keys", keys == expected, f"{keys}")

    # tweak one weight, then reload to confirm it is restored
    with torch.no_grad():
        first_key = next(iter(ac.actor.encoder.pointnet.head[0].weight.view(-1).unbind()))
        original = first_key.item()
        ac.actor.encoder.pointnet.head[0].weight.view(-1)[0] = original + 1.0
    ac.load_encoders(enc)
    restored = ac.actor.encoder.pointnet.head[0].weight.view(-1)[0].item()
    report("load_encoders restores weights", abs(restored - original) < 1e-6,
           f"orig={original:.4f} restored={restored:.4f}")

    ac.freeze_encoders()
    frozen = []
    for name, mod in (
        ("actor_pointnet", ac.actor.encoder.pointnet),
        ("actor_mu_e", ac.actor.encoder.mu_e),
        ("critic_pointnet", ac.critic.encoder.pointnet),
        ("critic_mu_e", ac.critic.encoder.mu_e),
    ):
        if not all(not p.requires_grad for p in mod.parameters()):
            frozen.append(name)
    report("freeze_encoders works", not frozen, f"still trainable: {frozen}")


# ------------------------------------------------------------------ 9. param count

def check_param_count():
    section("9. Parameter count")
    ac = make_actor_critic(noise_std=0.01)
    total = sum(p.numel() for p in ac.parameters())
    actor_enc = sum(p.numel() for p in ac.actor.encoder.parameters())
    critic_enc = sum(p.numel() for p in ac.critic.encoder.parameters())
    actor_mlp = sum(p.numel() for p in ac.actor.mlp.parameters())
    critic_mlp = sum(p.numel() for p in ac.critic.mlp.parameters())
    print(f"  total        {total:>12,}", flush=True)
    print(f"  actor_enc    {actor_enc:>12,}", flush=True)
    print(f"  critic_enc   {critic_enc:>12,}", flush=True)
    print(f"  actor_mlp    {actor_mlp:>12,}", flush=True)
    print(f"  critic_mlp   {critic_mlp:>12,}", flush=True)
    report("total params < 5M", total < 5_000_000, f"{total:,}")
    report("actor_enc approx critic_enc", abs(actor_enc - critic_enc) < 100, f"{actor_enc:,} vs {critic_enc:,}")


# ------------------------------------------------------------------ 10. save/load

def check_state_dict():
    section("10. state_dict round-trip")
    ac1 = make_actor_critic(noise_std=0.01)
    ac1.train()
    n = 8
    L = OBS_LAYOUT
    obs = torch.randn(n, L.total)
    obs[:, L.point_start:L.point_start + L.point_flat_dim] = torch.randn(n, L.point_flat_dim) * 0.3
    for _ in range(2):
        ac1.act(obs)

    sd = ac1.state_dict()
    has_norm = any("policy_norm.mean" in k for k in sd.keys())
    report("state_dict contains normalizer stats", has_norm,
           f"policy_norm.mean present={has_norm}")

    ac2 = make_actor_critic(noise_std=0.01)
    ac2.load_state_dict(sd)
    ac1.eval(); ac2.eval()
    with torch.no_grad():
        o1 = ac1.act_inference(obs)
        o2 = ac2.act_inference(obs)
    diff = (o1 - o2).abs().max().item()
    report("state_dict round-trip output matches", diff < 1e-6, f"max diff={diff:.2e}")
    report("normalizer stats restored", torch.equal(ac1.actor.encoder.policy_norm.mean, ac2.actor.encoder.policy_norm.mean))


# ------------------------------------------------------------------ main

def main():
    torch.manual_seed(0)
    check_layout()
    check_pointnet()
    check_mu_e()
    check_rms()
    check_actor_critic_shapes()
    check_backward()
    check_init_scale()
    check_export_freeze()
    check_param_count()
    check_state_dict()

    section("SUMMARY")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{passed}/{len(RESULTS)} checks passed", flush=True)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL: {name}  {detail}", flush=True)


if __name__ == "__main__":
    main()
