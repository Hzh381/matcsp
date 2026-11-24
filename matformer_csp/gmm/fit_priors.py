# Stage 0–2：拟合 GMM 先验并保存每个 split 的模式责任 r_ic。
#
# 用法:
#   python -m matformer_csp.gmm.fit_priors data=mp_20 gmm.out=cache/gmm/mp_20.pt

import math
import sys
from pathlib import Path

# 直接以文件方式运行时，把仓库根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment

from matformer_csp.common.utils import PROJECT_ROOT
from matformer_csp.gmm.lattice_gmm import LatticeGMM, effective_k
from matformer_csp.gmm.padding import build_from_cached
from matformer_csp.gmm.symmetry import sample_transformed_padded
from matformer_csp.gmm.wrapped_gmm import WrappedDiagonalGMM


def _resp_path(save_path: str) -> Path:
    p = Path(save_path)
    return p.with_name(p.stem + "_gmm_resp.pt")


def _get_dataset(cfg_split):
    return hydra.utils.instantiate(cfg_split, _recursive_=False)


def _build_split_data(split_cfg, max_n, device):
    dataset = _get_dataset(split_cfg)
    return build_from_cached(dataset.cached_data, max_n=max_n, device=device)


def _joint_resp(data, lat_gmm, wrap_gmm, pi, J, generator, max_n):
    """用联合混合计算每个样本的模式责任 r_ic。"""
    lp_lat = lat_gmm.log_likelihood(data["k"])
    xs, mask = sample_transformed_padded(
        data["F"], data["num_atoms"], data["atom_types"], data["node2graph"],
        max_n, J, generator=generator,
    )
    lps = torch.stack([wrap_gmm.log_likelihood(x, mask) for x in xs], dim=0)
    lp_f = torch.logsumexp(lps, dim=0) - math.log(J)
    logp = torch.log(pi.clamp_min(1e-12))[None, :] + lp_lat + lp_f
    return F.softmax(logp, dim=-1), xs, mask


@hydra.main(config_path=str(PROJECT_ROOT / "matformer_csp" / "conf"),
            config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    g = cfg.gmm
    device = g.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[GMM] CUDA 不可用，回退到 CPU")
        device = "cpu"

    generator = torch.Generator(device=device)
    generator.manual_seed(int(g.seed))

    train_cfg = cfg.data.datamodule.datasets.train
    val_cfg = cfg.data.datamodule.datasets.val[0]
    test_cfg = cfg.data.datamodule.datasets.test[0]

    # 先确定数据集实际最大原子数
    train_ds = _get_dataset(train_cfg)
    val_ds = _get_dataset(val_cfg)
    test_ds = _get_dataset(test_cfg)
    observed_max = max(
        int(d["graph_arrays"][6])
        for ds in (train_ds, val_ds, test_ds)
        for d in ds.cached_data
    )
    max_n = observed_max if g.max_atoms is None else max(observed_max, int(g.max_atoms))
    print(f"[GMM] max_n = {max_n}")

    # 拟合用训练数据（可下采样）
    fit_items = train_ds.cached_data
    if g.max_fit_samples is not None:
        n_fit = min(int(g.max_fit_samples), len(fit_items))
        idx = torch.randperm(len(fit_items), generator=torch.Generator().manual_seed(int(g.seed)))[:n_fit].tolist()
        fit_items = [fit_items[i] for i in idx]
        print(f"[GMM] 使用 {n_fit} 条训练样本拟合")

    fit_data = build_from_cached(fit_items, max_n=max_n, device=device)

    # Stage 1.4–6：晶格 GMM
    lat_gmm = LatticeGMM.fit(
        fit_data["k"], K_max=int(g.K_max), seed=int(g.seed)
    )
    print(f"[GMM] 晶格 K_eff = {effective_k(lat_gmm.weights, g.eps)}")

    # Stage 1.7–9：坐标 wrapped GMM
    wrap_gmm = WrappedDiagonalGMM.fit(
        fit_data["F_pad"], fit_data["mask"],
        K_max=int(g.K_max), alpha=float(g.alpha),
        sigma_prior=float(g.sigma_prior),
        min_sigma=float(g.min_sigma), max_sigma=float(g.max_sigma),
        max_iter=int(g.wrapped_iters), seed=int(g.seed),
    )
    print(f"[GMM] 坐标 K_eff = {effective_k(wrap_gmm.weights, g.eps)}")

    # 用硬责任共现矩阵对齐两套分量编号
    r_lat = lat_gmm.responsibilities(fit_data["k"])
    r_f = wrap_gmm.responsibilities(fit_data["F_pad"], fit_data["mask"])
    co = (r_lat.T @ r_f).detach().cpu().numpy()
    row, col = linear_sum_assignment(-co)
    # perm[晶格分量] = 坐标分量，使两套编号对齐
    perm = torch.empty(lat_gmm.K, dtype=torch.long)
    perm[torch.as_tensor(row, dtype=torch.long)] = torch.as_tensor(col, dtype=torch.long)
    wrap_gmm.reorder(perm.to(device))

    # Stage 2.12–14：联合 EM，共享混合权重 pi
    pi = 0.5 * (lat_gmm.weights + wrap_gmm.weights)
    pi = pi / pi.sum().clamp_min(1e-8)

    J = int(g.group_samples)
    for it in range(int(g.joint_iters)):
        r, xs, mask = _joint_resp(fit_data, lat_gmm, wrap_gmm, pi, J, generator, max_n)
        M = fit_data["k"].shape[0]

        # 更新共享权重与两个空间的分量参数
        pi = ((r.sum(dim=0) + g.alpha) / (M + lat_gmm.K * g.alpha)).clamp_min(0)
        pi = pi / pi.sum().clamp_min(1e-8)
        lat_gmm.update(fit_data["k"], r)

        x_cat = torch.cat(xs, dim=0)
        mask_cat = mask.repeat(J, 1)
        r_cat = r.repeat(J, 1) / J
        wrap_gmm.update_from_data(
            x_cat, mask_cat, r_cat,
            alpha=float(g.alpha), sigma_prior=float(g.sigma_prior),
        )
        print(f"[GMM] joint iter {it + 1}/{g.joint_iters}  K_eff={effective_k(pi, g.eps)}")

    # 剪掉无效分量
    keep = pi > g.eps
    if keep.sum() == 0:
        keep[pi.argmax()] = True
    idx = keep.nonzero(as_tuple=True)[0]
    pi = pi[idx] / pi[idx].sum()
    lat_gmm.reorder(idx)
    wrap_gmm.reorder(idx)
    k_eff = int(idx.numel())
    print(f"[GMM] 最终 K_eff = {k_eff}, weights = {pi.detach().cpu().tolist()}")

    # 让保存的 GMM 对象也带上联合权重，便于单独使用 log_prob
    lat_gmm.weights = pi.clone()
    wrap_gmm.weights = pi.clone()

    # 保存先验
    out_path = Path(g.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "lattice": lat_gmm.state_dict(),
        "coords": wrap_gmm.state_dict(),
        "weights": pi.detach().cpu(),
        "max_n": max_n,
        "K_eff": k_eff,
        "eps": float(g.eps),
        "alpha": float(g.alpha),
        "group_samples": J,
        "joint_iters": int(g.joint_iters),
        "dataset": str(train_cfg.name),
    }, out_path)
    print(f"[GMM] 先验已保存到 {out_path}")

    # 为 train/val/test 计算并保存模式责任
    for split_name, split_cfg in (("train", train_cfg), ("val", val_cfg), ("test", test_cfg)):
        data = build_from_cached(_get_dataset(split_cfg).cached_data, max_n=max_n, device=device)
        r, _, _ = _joint_resp(data, lat_gmm, wrap_gmm, pi, J, generator, max_n)
        resp_path = _resp_path(split_cfg.save_path)
        torch.save(r.detach().cpu(), resp_path)
        print(f"[GMM] {split_name} resp -> {resp_path}  shape={tuple(r.shape)}")


if __name__ == "__main__":
    main()
