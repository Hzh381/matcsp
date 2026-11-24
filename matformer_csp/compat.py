# 纯 PyTorch/PyG 的 torch_scatter 兼容层。
# 这样项目不再依赖 torch_scatter/torch_sparse 等编译扩展，
# 避免 torch 与扩展 wheel 的 ABI 不匹配（undefined symbol 等）。

import torch
from torch_geometric.utils import softmax as _pyg_softmax


def _prod(values):
    out = 1
    for v in values:
        out *= v
    return out


def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """
    等价于 torch_scatter.scatter。

    src:   (N, ...) 或 (..., D, ...)
    index: src 去掉最后一维后的形状；沿 dim 指定每个元素的目标下标
    """
    if dim < 0:
        dim += src.dim()
    index = index.to(device=src.device, dtype=torch.long)

    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    dim_size = int(dim_size)

    # 把要 scatter 的维度挪到最前，其余维度视为 batch / 特征
    src_p = src.movedim(dim, 0)          # (D, *other, *trailing)
    idx_p = index.movedim(dim, 0)        # (D, *other)
    D = src_p.shape[0]
    other = src_p.shape[1:idx_p.dim()]
    trailing = src_p.shape[idx_p.dim():]
    n_other = _prod(other)

    src_flat = src_p.reshape(D, n_other, *trailing).reshape(-1, *trailing)
    idx_flat = idx_p.reshape(D, n_other)
    # 输出按 (target, *other, *trailing) 排列，因此 flat index = target * n_other + other_idx
    offsets = torch.arange(n_other, device=src.device).view(1, n_other)
    idx_flat = (idx_flat * n_other + offsets).reshape(-1)

    out = src.new_zeros((dim_size * n_other,) + tuple(trailing))
    if reduce == "sum":
        out.index_add_(0, idx_flat, src_flat)
    elif reduce == "mean":
        out.index_add_(0, idx_flat, src_flat)
        cnt = torch.bincount(idx_flat, minlength=dim_size * n_other).to(src.dtype)
        view = (dim_size * n_other,) + (1,) * len(trailing)
        out = out / cnt.clamp_min(1).view(view)
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")

    out = out.reshape((dim_size,) + tuple(other) + tuple(trailing))
    return out.movedim(0, dim)


def scatter_add(src, index, dim=0, out=None):
    res = scatter(src, index, dim=dim, reduce="sum")
    if out is not None:
        out.copy_(res)
        return out
    return res


def segment_coo(src, index, dim_size, reduce="sum"):
    return scatter(src, index, dim=0, dim_size=dim_size, reduce=reduce)


def segment_csr(src, indptr, reduce="sum"):
    """indptr 是 CSR 的累计偏移，长度 B+1；返回每个 segment 的聚合结果。"""
    indptr = indptr.to(src.device)
    lengths = (indptr[1:] - indptr[:-1]).clamp_min(0)
    n_seg = lengths.numel()
    seg = torch.repeat_interleave(torch.arange(n_seg, device=src.device), lengths)
    out = scatter(src, seg, dim=0, dim_size=n_seg, reduce=reduce)
    return out.to(src.dtype)


def scatter_softmax(src, index, dim=0, dim_size=None):
    if dim != 0:
        raise NotImplementedError("compat.scatter_softmax 只支持 dim=0")
    return _pyg_softmax(src, index, num_nodes=dim_size)
