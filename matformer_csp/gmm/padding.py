# Stage 0：把晶体表示 (A, F, k) 整理成 GMM 可用的张量。
# 分数坐标按数据集实际最大原子数 max_n 在末尾补零，mask 标记有效维度。

import torch


def wrap(x: torch.Tensor) -> torch.Tensor:
    """把分数坐标或坐标向量取模到 [0, 1)。"""
    return x % 1.0


def pad_flat(F: torch.Tensor, num_atoms: torch.Tensor, max_n: int):
    """
    把拼接后的分数坐标补零到 (B, 3*max_n)。

    F: (N_total, 3)
    num_atoms: (B,)
    返回:
        x:    (B, 3*max_n)
        mask: (B, 3*max_n) bool，True 表示有效维度
    """
    B = num_atoms.numel()
    D = 3 * max_n
    x = F.new_zeros(B, D)
    mask = torch.zeros(B, D, dtype=torch.bool, device=F.device)
    offset = 0
    for b, n in enumerate(num_atoms.tolist()):
        L = 3 * n
        x[b, :L] = F[offset:offset + n].reshape(-1)
        mask[b, :L] = True
        offset += n
    return x, mask


def unpad_flat(x: torch.Tensor, num_atoms: torch.Tensor) -> torch.Tensor:
    """把 (B, 3*max_n) 还原成拼接的 (N_total, 3)。"""
    out = []
    for b, n in enumerate(num_atoms.tolist()):
        out.append(x[b, :3 * n].reshape(n, 3))
    return torch.cat(out, dim=0)


def build_from_cached(cached_data, max_n: int | None = None, device="cpu"):
    """
    从 CrystDataset.cached_data 构建 GMM 训练张量。

    cached_data 的元素包含 graph_arrays:
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms, lattice_polar)
    """
    fracs, types, ks, nums = [], [], [], []
    for d in cached_data:
        frac, atom_types, _, _, _, _, n, k = d["graph_arrays"]
        fracs.append(torch.as_tensor(frac, dtype=torch.float32, device=device))
        types.append(torch.as_tensor(atom_types, dtype=torch.long, device=device))
        ks.append(torch.as_tensor(k, dtype=torch.float32, device=device))
        nums.append(int(n))

    num_atoms = torch.tensor(nums, dtype=torch.long, device=device)
    if max_n is None:
        max_n = int(num_atoms.max().item())

    F = torch.cat(fracs, dim=0)
    atom_types = torch.cat(types, dim=0)
    node2graph = torch.repeat_interleave(
        torch.arange(len(nums), device=device), num_atoms
    )
    k = torch.stack(ks, dim=0)
    F_pad, mask = pad_flat(F, num_atoms, max_n)

    return {
        "F": F,
        "atom_types": atom_types,
        "node2graph": node2graph,
        "num_atoms": num_atoms,
        "k": k,
        "F_pad": F_pad,
        "mask": mask,
        "max_n": max_n,
    }
