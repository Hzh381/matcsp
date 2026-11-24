# Stage 3–4：GMM 先验 + 模式耦合流匹配（DNG 版本，同时生成原子类型）。
# 类型先验/速度沿用原 flow_w_type，仅替换晶格/坐标先验。

from matformer_csp.pl_modules.flow_gmm import GMMPriorMixin
from matformer_csp.pl_modules.flow_w_type import CSPFlow as CSPFlowType


class CSPFlowGMMType(GMMPriorMixin, CSPFlowType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_gmm(kwargs.get("gmm_prior_path"))
