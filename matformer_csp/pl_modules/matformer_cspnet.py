# Matformer 解码器：接口与 CSPNet 对齐，内部用 MatformerConv 做消息传递。
# 这样所有 CrystalFlow 的 flow / diffusion / energy 实验只需把 decoder 换成 matformer 即可。

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from matformer_csp.pl_modules.edge_utils import gen_edges
from matformer_csp.pl_modules.matformer_conv import MatformerConv
from matformer_csp.pl_modules.rbf import RBFExpansion

MAX_ATOMIC_NUM = 100


class MatformerCSPNet(nn.Module):
    """
    用 Matformer 周期图 Transformer 实现 CSPNet 的解码器接口。

    输入:
        t: (B, latent_dim) 时间/噪声嵌入
        atom_types: (N,) 原子序数，或 pred_type 时的连续概率 (N, 100)
        frac_coords: (N, 3) 分数坐标
        lattices_rep: (B, lattice_dim) 极分解表示，或 (B, 3, 3) 晶格矩阵
        num_atoms: (B,) 每个晶体的原子数
        node2graph: (N,) 每个原子所属晶体索引
        lattices_mat: (B, 3, 3) 建图用晶格矩阵（默认等于 lattices_rep）
        cemb / guide_indicator: 条件引导
    输出:
        (lattice_out, coord_out) 或 (lattice_out, coord_out, type_out)，或标量
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 256,
        lattice_dim: int = 9,
        cemb_dim: int = 1,
        num_layers: int = 5,
        heads: int = 4,
        edge_rbf_bins: int = 128,
        max_atoms: int = 100,
        cutoff: float = 6.0,
        max_neighbors: int = 20,
        edge_style: str = "knn",
        ln: bool = False,
        ip: bool = True,
        smooth: bool = False,
        pred_type: bool = False,
        pred_scalar: bool = False,
        type_encoding=None,
        dropout: float = 0.0,
        **kwargs,  # 接收 CSPNet 专属参数（num_freqs/rec_emb/...），对 Matformer 无效
    ):
        super().__init__()
        self.lattice_dim = lattice_dim
        self.ip = ip
        self.smooth = smooth
        self.pred_type = pred_type
        self.pred_scalar = pred_scalar
        self.type_encoding = type_encoding

        # 建图参数
        self.edge_style = edge_style
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

        # 节点编码：离散原子序数用 Embedding，连续类型概率用 Linear
        if type_encoding is None:
            if smooth:
                self.node_embedding = nn.Linear(MAX_ATOMIC_NUM, hidden_dim)
            else:
                self.node_embedding = nn.Embedding(MAX_ATOMIC_NUM, hidden_dim)
        else:
            self.node_embedding = nn.Linear(type_encoding.out_dim, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        # 边编码：笛卡尔键长 RBF 展开后映射到 hidden_dim
        self.rbf = RBFExpansion(vmin=0.0, vmax=8.0, bins=edge_rbf_bins)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_rbf_bins, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Matformer 消息传递层
        self.layers = nn.ModuleList(
            [
                MatformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers

        # 条件引导：每层一个条件偏置
        for i in range(num_layers):
            self.add_module(f"cemb_mixin_{i}", nn.Linear(hidden_dim, hidden_dim, bias=False))
            self.add_module(
                f"cemb_adapter_{i}",
                nn.Sequential(
                    nn.Linear(cemb_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                ),
            )

        self.final_layer_norm = nn.LayerNorm(hidden_dim) if ln else nn.Identity()

        # 输出头
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, lattice_dim),
        )
        if pred_type:
            type_out_dim = MAX_ATOMIC_NUM if type_encoding is None else type_encoding.out_dim
            self.type_out = nn.Linear(hidden_dim, type_out_dim)
        if pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        t,
        atom_types,
        frac_coords,
        lattices_rep,
        num_atoms,
        node2graph,
        lattices_mat=None,
        cemb=None,
        guide_indicator=None,
    ):
        if lattices_mat is None:
            lattices_mat = lattices_rep

        # 1. 按当前结构现场建周期图
        edges, frac_diff = gen_edges(
            self.edge_style,
            self.cutoff,
            self.max_neighbors,
            num_atoms,
            frac_coords,
            lattices_mat,
            node2graph,
        )
        edge2graph = node2graph[edges[0]]

        # 2. 分数坐标差 -> 笛卡尔位移 -> 键长 RBF
        cart_diff = torch.einsum("ei,eij->ej", frac_diff, lattices_mat[edge2graph])
        edge_attr = self.edge_mlp(self.rbf(torch.norm(cart_diff, dim=-1)))

        # 3. 节点初始特征 = 原子类型 embedding + 时间嵌入
        if self.type_encoding is None and not self.smooth:
            node_features = self.node_embedding(atom_types - 1)
        else:
            node_features = self.node_embedding(atom_types)
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = self.atom_latent_emb(torch.cat([node_features, t_per_atom], dim=1))

        # 4. Matformer 消息传递
        for i, layer in enumerate(self.layers):
            if cemb is not None:
                mixin = self._modules[f"cemb_mixin_{i}"]
                adapter = self._modules[f"cemb_adapter_{i}"]
                cemb_bias = (
                    mixin(adapter(cemb)) * guide_indicator[:, None]
                ).repeat_interleave(num_atoms, dim=0)
                node_features = node_features + cemb_bias
            node_features = layer(node_features, edges, edge_attr)

        node_features = self.final_layer_norm(node_features)
        coord_out = self.coord_out(node_features)

        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        # 5. 晶格速度头：图特征 + 时间嵌入
        lattice_out = self.lattice_out(torch.cat([graph_features, t], dim=1))
        lattice_out = lattice_out.view(lattices_rep.shape)
        if self.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices_rep)

        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out
        return lattice_out, coord_out
