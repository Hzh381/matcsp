# Stage 2：同元素原子置换 + 平移的蒙特卡洛群平均。

import math

import torch

from matformer_csp.gmm.padding import pad_flat, wrap


def sample_permutation(atom_types: torch.Tensor, node2graph: torch.Tensor) -> torch.Tensor:
    """
    采样只交换同种元素的置换（对每个晶体分别进行）。

    做法：按 (graph, type) 分组，组内随机打乱。
    返回 perm，使得 F[perm] 是置换后的坐标。
    """
    # 原子序数 <= 100，用 graph*128+type 作为分组 id
    gid = node2graph * 128 + atom_types
    target_sorted = torch.argsort(gid)
    source_sorted = torch.argsort(gid.to(torch.float32) * 2.0 + torch.rand_like(gid, dtype=torch.float32))
    perm = torch.empty_like(target_sorted)
    perm[target_sorted] = source_sorted
    return perm


def apply_symmetry(F: torch.Tensor, atom_types: torch.Tensor, node2graph: torch.Tensor,
                   batch_size: int, generator=None):
    """对分数坐标施加随机的同元素置换 + 平移，并 wrap 到 [0,1)。"""
    perm = sample_permutation(atom_types, node2graph)
    tau = torch.rand(batch_size, 3, device=F.device, generator=generator)
    return wrap(F[perm] + tau[node2graph])


def transform_pair(F0, F1, atom_types, node2graph, batch_size, generator=None):
    """对源和目标坐标施加同一个置换 + 平移（Stage 3 step 21）。"""
    perm = sample_permutation(atom_types, node2graph)
    tau = torch.rand(batch_size, 3, device=F0.device, generator=generator)
    shift = tau[node2graph]
    return wrap(F0[perm] + shift), wrap(F1[perm] + shift)


@torch.no_grad()
def sample_transformed_padded(F, num_atoms, atom_types, node2graph, max_n, J,
                              generator=None):
    """
    生成 J 个经过随机对称变换的 padded 坐标，用于群平均。

    返回:
        xs:   list，长度 J，每个 (B, 3*max_n)
        mask: (B, 3*max_n) bool
    """
    batch_size = num_atoms.numel()
    _, mask = pad_flat(F, num_atoms, max_n)
    xs = []
    for _ in range(J):
        Ft = apply_symmetry(F, atom_types, node2graph, batch_size, generator=generator)
        x, _ = pad_flat(Ft, num_atoms, max_n)
        xs.append(x)
    return xs, mask


def group_average_log_likelihood(F, num_atoms, atom_types, node2graph, max_n,
                                 gmm, J, generator=None):
    """
    计算 log gbar_c^F(F|A) = log (1/J) Σ_j N_w(w(P_j F + 1 τ_j^T))。

    返回 (B, K)。
    """
    xs, mask = sample_transformed_padded(
        F, num_atoms, atom_types, node2graph, max_n, J, generator=generator
    )
    lps = torch.stack([gmm.log_likelihood(x, mask) for x in xs], dim=0)
    return torch.logsumexp(lps, dim=0) - math.log(J)
