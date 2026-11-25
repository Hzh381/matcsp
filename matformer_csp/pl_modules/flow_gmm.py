# Stage 3–4：GMM 先验 + 模式耦合流匹配（CSP 版本）。
# 通过覆写 CSPFlow 的两个钩子实现，不重写原 flow 的训练/采样逻辑。

import os
from pathlib import Path

import torch

from matformer_csp.common.utils import PROJECT_ROOT, torch_load
from matformer_csp.gmm.lattice_gmm import LatticeGMM
from matformer_csp.gmm.padding import wrap
from matformer_csp.gmm.symmetry import sample_permutation
from matformer_csp.gmm.wrapped_gmm import WrappedDiagonalGMM
from matformer_csp.pl_modules.flow import CSPFlow


def _load_prior(path):
    return torch_load(path, map_location="cpu")


class GMMPriorMixin:
    """给 flow 类提供 GMM 先验采样与置换/平移变换。"""

    def _init_gmm(self, gmm_prior_path):
        # 兼容从 checkpoint / hparams.yaml 恢复时的两种结构：
        # 1) 训练时 hydra 直接把 model 配置展开为 kwargs，gmm_prior_path 为顶层键；
        # 2) hparams.yaml 把模型配置嵌在 model 子节点下。
        if gmm_prior_path is None and hasattr(self, "hparams"):
            if "gmm_prior_path" in self.hparams:
                gmm_prior_path = self.hparams["gmm_prior_path"]
            elif hasattr(self.hparams, "model"):
                model_cfg = self.hparams.model
                if hasattr(model_cfg, "get"):
                    gmm_prior_path = model_cfg.get("gmm_prior_path", None)
        if gmm_prior_path is None:
            raise ValueError("GMM 流需要 model.gmm_prior_path 指向 fit_priors.py 的输出")
        if not self.lattice_polar:
            raise ValueError("GMM 流要求 lattice_polar=true（晶格 GMM 是 6D 极分解）")

        # 相对路径优先按当前工作目录解析，失败则回退到仓库根目录
        if not os.path.isabs(gmm_prior_path) and not Path(gmm_prior_path).exists():
            gmm_prior_path = str(Path(PROJECT_ROOT) / gmm_prior_path)

        prior = _load_prior(gmm_prior_path)
        self.gmm_weights = prior["weights"].float()
        self.lat_gmm = LatticeGMM.from_state_dict(prior["lattice"])
        self.wrap_gmm = WrappedDiagonalGMM.from_state_dict(prior["coords"])
        self.gmm_max_n = int(prior["max_n"])
        self.gmm_K = int(prior["K_eff"])
        # 训练/验证的 forward 使用每个样本的模式后验；采样的 sample 使用全局权重
        self._use_gmm_resp = False

    def forward(self, batch, *args, **kwargs):
        self._use_gmm_resp = True
        return super().forward(batch, *args, **kwargs)

    def sample(self, batch, *args, **kwargs):
        self._use_gmm_resp = False
        return super().sample(batch, *args, **kwargs)

    def _ensure_gmm_device(self):
        device = self.device
        if self.lat_gmm.weights.device != device:
            self.lat_gmm.to(device)
            self.wrap_gmm.to(device)
            self.gmm_weights = self.gmm_weights.to(device)

    def sample_prior(self, batch):
        """Stage 3.18–20 / Stage 4.29–30：按模式责任采样 (k0, F0_base)。"""
        self._ensure_gmm_device()
        B = batch.num_graphs

        # 训练/验证时使用预计算的联合模式责任；采样时只用全局混合权重
        r = getattr(batch, "gmm_resp", None) if self._use_gmm_resp else None
        if r is not None and r.numel() > 0:
            r = r.to(self.device).float().view(B, -1)
            if r.shape[1] != self.gmm_weights.numel():
                r = None
        if r is None:
            c = torch.multinomial(self.gmm_weights.clamp_min(1e-12), B, replacement=True)
        else:
            c = torch.multinomial(r.clamp_min(1e-12), 1).squeeze(1)

        k0 = self.lat_gmm.sample(c)                    # (B, 6)
        f0 = self.wrap_gmm.sample(c, batch.num_atoms)  # (N_total, 3)
        return k0, f0

    def transform_for_flow(self, batch, f0, f1=None):
        """Stage 3.21：对源和目标坐标施加同一个同元素置换 + 平移。"""
        B = batch.num_graphs
        tau = torch.rand(B, 3, device=f0.device)
        shift = tau[batch.batch]

        # DNG 生成时可能没有已知原子类型，此时只做平移
        atom_types = getattr(batch, "atom_types", None)
        if atom_types is None:
            f0 = wrap(f0 + shift)
            if f1 is not None:
                f1 = wrap(f1 + shift)
            return f0, f1

        perm = sample_permutation(atom_types, batch.batch)
        f0 = wrap(f0[perm] + shift)
        if f1 is not None:
            f1 = wrap(f1[perm] + shift)
        return f0, f1


class CSPFlowGMM(GMMPriorMixin, CSPFlow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gmm(kwargs.get("gmm_prior_path"))
