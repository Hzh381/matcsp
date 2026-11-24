# Stage 1.7–9：分数坐标的 wrapped 对角 GMM。
# 用有限周期镜像 (-1,0,1) 近似 wrapped normal，并用变分/EM 近似更新参数。
# 注意：这是“对角协方差 + 有限镜像”的实用近似，不是精确的 3N 维 wrapped VB。

import math

import torch
import torch.nn.functional as F

from matformer_csp.gmm.padding import wrap

LOG2PI = math.log(2 * math.pi)


class WrappedDiagonalGMM:
    def __init__(self, weights, means, log_sigmas, min_sigma=1e-3, max_sigma=0.5):
        self.weights = weights          # (K,)
        self.means = means              # (K, D)
        self.log_sigmas = log_sigmas    # (K, D)
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    @property
    def K(self):
        return self.weights.numel()

    @property
    def sigmas(self):
        return torch.exp(self.log_sigmas).clamp(self.min_sigma, self.max_sigma)

    def to(self, device):
        self.weights = self.weights.to(device)
        self.means = self.means.to(device)
        self.log_sigmas = self.log_sigmas.to(device)
        return self

    def log_likelihood(self, x, mask, chunk=2048):
        """
        每维有限镜像 wrapped 对数似然，padding 维不参与。
        x: (M,D)，mask: (M,D) bool -> 返回 (M,K)。
        """
        out = []
        mu = self.means[None, :, :]
        sig = self.sigmas[None, :, :]
        log_sig = torch.log(sig)
        for start in range(0, x.shape[0], chunk):
            xc = x[start:start + chunk]
            mc = mask[start:start + chunk].float()
            x3 = xc[:, None, :]                                     # (m,1,D)
            terms = []
            for n in (-1.0, 0.0, 1.0):
                z = x3 + n
                terms.append(-0.5 * (LOG2PI + 2 * log_sig + ((z - mu) / sig) ** 2))
            lp = torch.logsumexp(torch.stack(terms, dim=0), dim=0)  # (m,K,D)
            out.append((lp * mc[:, None, :]).sum(dim=-1))
        return torch.cat(out, dim=0)

    def log_prob(self, x, mask):
        return self.log_likelihood(x, mask) + torch.log(self.weights.clamp_min(1e-12))[None, :]

    def responsibilities(self, x, mask):
        return F.softmax(self.log_prob(x, mask), dim=-1)

    @torch.no_grad()
    def update_from_data(self, x, mask, r, alpha=0.1, sigma_prior=1e-3, chunk=2048):
        """按责任 r 做一步变分/EM 更新（有限镜像后验 + Dirichlet 先验）。"""
        K = self.K
        D = self.means.shape[1]
        device = x.device
        Nk = torch.zeros(K, device=device)
        Nkd = torch.zeros(K, D, device=device)
        sum_z = torch.zeros(K, D, device=device)
        sum_z2 = torch.zeros(K, D, device=device)

        mu = self.means[None, :, :]
        sig = self.sigmas[None, :, :]
        log_sig = torch.log(sig)

        for start in range(0, x.shape[0], chunk):
            xc = x[start:start + chunk]
            mc = mask[start:start + chunk].float()
            rc = r[start:start + chunk]
            x3 = xc[:, None, :]                                     # (m,1,D)

            # 有限镜像后验 q_i(n) ∝ N(x+n; mu, sigma)
            logp = []
            for n in (-1.0, 0.0, 1.0):
                z = x3 + n
                logp.append(-0.5 * (LOG2PI + 2 * log_sig + ((z - mu) / sig) ** 2))
            logp = torch.stack(logp, dim=0)                         # (3,m,K,D)
            q = torch.softmax(logp, dim=0)

            z = sum(q[i] * (x3 + n) for i, n in enumerate((-1.0, 0.0, 1.0)))
            z2 = sum(q[i] * (x3 + n) ** 2 for i, n in enumerate((-1.0, 0.0, 1.0)))
            w = rc[:, :, None] * mc[:, None, :]                     # (m,K,D)

            Nk += rc.sum(dim=0)
            Nkd += w.sum(dim=0)
            sum_z += (w * z).sum(dim=0)
            sum_z2 += (w * z2).sum(dim=0)

        # 混合权重：对称 Dirichlet 先验（用有效权重和，兼容 J 个变换样本拼接）
        total = r.sum().clamp_min(1e-8)
        self.weights = (Nk + alpha) / (total + K * alpha)

        has = Nkd > 0
        mu_new = sum_z / Nkd.clamp_min(1e-8)
        var_new = sum_z2 / Nkd.clamp_min(1e-8) - mu_new ** 2 + sigma_prior ** 2
        var_new = var_new.clamp(self.min_sigma ** 2, self.max_sigma ** 2)

        self.means = torch.where(has, mu_new, self.means)
        self.log_sigmas = torch.where(has, 0.5 * torch.log(var_new), self.log_sigmas)

    def sample(self, comp_idx, num_atoms):
        """按分量索引采样，返回拼接后的 (N_total, 3)。"""
        mu = self.means[comp_idx]
        sig = self.sigmas[comp_idx]
        y = mu + sig * torch.randn_like(mu)
        out = []
        for b, n in enumerate(num_atoms.tolist()):
            out.append(wrap(y[b, :3 * n].reshape(n, 3)))
        return torch.cat(out, dim=0)

    def reorder(self, perm):
        self.weights = self.weights[perm]
        self.means = self.means[perm]
        self.log_sigmas = self.log_sigmas[perm]

    def state_dict(self):
        return {
            "weights": self.weights.detach().cpu(),
            "means": self.means.detach().cpu(),
            "log_sigmas": self.log_sigmas.detach().cpu(),
            "min_sigma": self.min_sigma,
            "max_sigma": self.max_sigma,
        }

    @classmethod
    def from_state_dict(cls, sd):
        inst = cls(sd["weights"].float(), sd["means"].float(), sd["log_sigmas"].float(),
                   sd.get("min_sigma", 1e-3), sd.get("max_sigma", 0.5))
        return inst

    @classmethod
    def fit(cls, x, mask, K_max, alpha=0.1, sigma_prior=1e-3,
            min_sigma=1e-3, max_sigma=0.5, max_iter=30, seed=0, tol=1e-4):
        """先用 sklearn 对角 BGMM 初始化，再跑有限镜像 EM/VB。"""
        means, log_sigmas, weights = cls._init_from_bgmm(
            x, K_max, seed, min_sigma, max_sigma
        )
        inst = cls(weights, means, log_sigmas, min_sigma, max_sigma)

        for it in range(max_iter):
            r = inst.responsibilities(x, mask)
            old_w = inst.weights.clone()
            inst.update_from_data(x, mask, r, alpha=alpha, sigma_prior=sigma_prior)
            if (inst.weights - old_w).abs().max().item() < tol:
                break
        return inst

    @staticmethod
    def _init_from_bgmm(x, K_max, seed, min_sigma, max_sigma):
        M, D = x.shape
        device = x.device
        try:
            from sklearn.mixture import BayesianGaussianMixture

            bgmm = BayesianGaussianMixture(
                n_components=K_max,
                covariance_type="diag",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1.0 / K_max,
                max_iter=200,
                tol=1e-3,
                random_state=seed,
            )
            bgmm.fit(x.detach().cpu().numpy())
            means = torch.as_tensor(bgmm.means_, dtype=x.dtype, device=device)
            var = torch.as_tensor(bgmm.covariances_, dtype=x.dtype, device=device)
            weights = torch.as_tensor(bgmm.weights_, dtype=x.dtype, device=device)
        except Exception:
            # 兜底：均值取 0，方差取中间值，权重均匀
            means = torch.zeros(K_max, D, dtype=x.dtype, device=device)
            var = torch.full((K_max, D), 0.1 ** 2, dtype=x.dtype, device=device)
            weights = torch.full((K_max,), 1.0 / K_max, dtype=x.dtype, device=device)

        var = var.clamp(min_sigma ** 2, max_sigma ** 2)
        log_sigmas = 0.5 * torch.log(var)
        return means, log_sigmas, weights


def effective_k(weights, eps):
    return int((weights > eps).sum().item())
