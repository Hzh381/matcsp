# Stage 1.4–6：6D 晶格参数的截断 Dirichlet Process 贝叶斯 GMM。
# 用 sklearn 的 BayesianGaussianMixture 做变分贝叶斯拟合，再转成 torch 参数供流模型使用。

import math

import torch
import torch.nn.functional as F


class LatticeGMM:
    """6 维晶格表示 k 的贝叶斯高斯混合。"""

    def __init__(self, weights: torch.Tensor, means: torch.Tensor, covs: torch.Tensor):
        self.weights = weights          # (K,)
        self.means = means              # (K, 6)
        self.covs = covs                # (K, 6, 6)

    @property
    def K(self) -> int:
        return self.weights.numel()

    def to(self, device):
        self.weights = self.weights.to(device)
        self.means = self.means.to(device)
        self.covs = self.covs.to(device)
        return self

    def log_likelihood(self, k: torch.Tensor) -> torch.Tensor:
        """返回每个分量的对数似然（不含混合权重），形状 (M, K)。"""
        diff = k[:, None, :] - self.means[None, :, :]                  # (M,K,6)
        cov = self.covs + 1e-6 * torch.eye(6, device=k.device)
        inv = torch.linalg.inv(cov)                                    # (K,6,6)
        logdet = torch.logdet(cov)                                     # (K,)
        maha = torch.einsum("mki,kij,mkj->mk", diff, inv, diff)
        return -0.5 * (6 * math.log(2 * math.pi) + logdet[None, :] + maha)

    def log_prob(self, k: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(k) + torch.log(self.weights.clamp_min(1e-12))[None, :]

    def responsibilities(self, k: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.log_prob(k), dim=-1)

    def sample(self, comp_idx: torch.Tensor) -> torch.Tensor:
        """按分量索引采样晶格 k，形状 (B, 6)。"""
        mu = self.means[comp_idx]
        cov = self.covs[comp_idx] + 1e-6 * torch.eye(6, device=mu.device)
        L = torch.linalg.cholesky(cov)
        eps = torch.randn_like(mu)
        return mu + torch.einsum("bij,bj->bi", L, eps)

    def update(self, k: torch.Tensor, r: torch.Tensor, reg: float = 1e-6):
        """按联合责任 r 更新均值/协方差（权重由调用方统一更新）。"""
        Nk = r.sum(dim=0).clamp_min(1e-8)                              # (K,)
        self.means = (r.T @ k) / Nk[:, None]
        diff = k[:, None, :] - self.means[None, :, :]                  # (M,K,6)
        cov = torch.einsum("mk,mki,mkj->kij", r, diff, diff) / Nk[:, None, None]
        eye = torch.eye(6, device=k.device)
        self.covs = cov + reg * eye

    def reorder(self, perm: torch.Tensor):
        """按 perm 重排分量，用于和坐标 GMM 对齐编号。"""
        self.weights = self.weights[perm]
        self.means = self.means[perm]
        self.covs = self.covs[perm]

    def state_dict(self):
        return {
            "weights": self.weights.detach().cpu(),
            "means": self.means.detach().cpu(),
            "covs": self.covs.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, sd):
        return cls(sd["weights"].float(), sd["means"].float(), sd["covs"].float())

    @classmethod
    def fit(cls, k: torch.Tensor, K_max: int, seed: int = 0, reg_covar: float = 1e-6):
        """优先用 sklearn 的 DP-BGMM；失败时回退到简单 EM。"""
        try:
            from sklearn.mixture import BayesianGaussianMixture

            gmm = BayesianGaussianMixture(
                n_components=K_max,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1.0 / K_max,
                reg_covar=reg_covar,
                max_iter=500,
                tol=1e-3,
                random_state=seed,
            )
            gmm.fit(k.detach().cpu().numpy())
            weights = torch.as_tensor(gmm.weights_, dtype=k.dtype, device=k.device)
            means = torch.as_tensor(gmm.means_, dtype=k.dtype, device=k.device)
            covs = torch.as_tensor(gmm.covariances_, dtype=k.dtype, device=k.device)
            return cls(weights, means, covs)
        except Exception as e:
            print(f"[LatticeGMM] sklearn BGMM 不可用，改用 EM 兜底: {e}")
            return cls._fit_em(k, K_max, seed, reg_covar)

    @classmethod
    def _fit_em(cls, k: torch.Tensor, K_max: int, seed: int, reg_covar: float,
                iters: int = 50, tol: float = 1e-4):
        """不依赖 sklearn 的加权 EM 初始化/迭代。"""
        torch.manual_seed(seed)
        M = k.shape[0]
        idx = torch.randperm(M, device=k.device)[:min(K_max, M)]
        means = k[idx].clone()
        if means.shape[0] < K_max:
            pad = means[-1:].repeat(K_max - means.shape[0], 1)
            means = torch.cat([means, pad], dim=0)
        var = k.var(dim=0, unbiased=False).clamp_min(1e-3)
        covs = torch.diag(var).repeat(K_max, 1, 1) + reg_covar * torch.eye(6, device=k.device)
        weights = torch.full((K_max,), 1.0 / K_max, device=k.device)
        inst = cls(weights, means, covs)

        for _ in range(iters):
            r = inst.responsibilities(k)
            old = inst.weights.clone()
            inst.update(k, r, reg=reg_covar)
            inst.weights = (r.sum(dim=0) + 1.0 / K_max) / (M + 1.0)
            inst.weights = inst.weights / inst.weights.sum().clamp_min(1e-8)
            if (inst.weights - old).abs().max().item() < tol:
                break
        return inst


def effective_k(weights: torch.Tensor, eps: float) -> int:
    """Stage 1.6 / 1.9：有效模式数。"""
    return int((weights > eps).sum().item())
