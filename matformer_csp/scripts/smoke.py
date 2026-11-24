# 轻量 smoke test：不依赖 pymatgen/lightning，可在只装了 torch + torch_geometric 的环境运行。
#
#   python matformer_csp/scripts/smoke.py

import sys
from pathlib import Path

# 直接以文件方式运行时，把仓库根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

import torch
from torch_geometric.data import Batch, Data

from matformer_csp.gmm.lattice_gmm import LatticeGMM
from matformer_csp.gmm.padding import pad_flat, wrap
from matformer_csp.gmm.symmetry import (
    group_average_log_likelihood,
    sample_permutation,
    transform_pair,
)
from matformer_csp.gmm.wrapped_gmm import WrappedDiagonalGMM
from matformer_csp.pl_modules.matformer_cspnet import MatformerCSPNet


def make_batch():
    data_list = []
    for n in (4, 5, 6):
        data_list.append(
            Data(
                frac_coords=torch.rand(n, 3),
                atom_types=torch.randint(1, 21, (n,)),
                lattice_polar=torch.randn(1, 6),
                num_atoms=n,
                num_nodes=n,
            )
        )
    batch = Batch.from_data_list(data_list)
    lattices_mat = torch.eye(3).repeat(batch.num_graphs, 1, 1)
    return batch, lattices_mat


def test_matformer_decoder():
    batch, lattices_mat = make_batch()
    t = torch.randn(batch.num_graphs, 16)
    common = dict(
        hidden_dim=32, latent_dim=16, num_layers=2, heads=2,
        edge_rbf_bins=16, edge_style="fc", ip=False,
    )

    # CSP： (B,6) 晶格速度 + (N,3) 坐标速度
    model = MatformerCSPNet(lattice_dim=6, **common)
    pred_l, pred_f = model(
        t, batch.atom_types, batch.frac_coords, batch.lattice_polar,
        batch.num_atoms, batch.batch, lattices_mat=lattices_mat,
    )
    assert pred_l.shape == (batch.num_graphs, 6), pred_l.shape
    assert pred_f.shape == (batch.num_nodes, 3), pred_f.shape
    (pred_l.sum() + pred_f.sum()).backward()
    assert any(p.grad is not None for p in model.parameters())

    # DNG：连续类型概率输入 + 类型速度输出
    model_t = MatformerCSPNet(lattice_dim=6, pred_type=True, smooth=True, **common)
    type_probs = torch.randn(batch.num_nodes, 100)
    pred_l, pred_f, pred_t = model_t(
        t, type_probs, batch.frac_coords, batch.lattice_polar,
        batch.num_atoms, batch.batch, lattices_mat=lattices_mat,
    )
    assert pred_t.shape == (batch.num_nodes, 100), pred_t.shape

    # 条件引导
    model_g = MatformerCSPNet(lattice_dim=6, cemb_dim=8, **common)
    cemb = torch.randn(batch.num_graphs, 8)
    guide = torch.ones(batch.num_graphs)
    pred_l, pred_f = model_g(
        t, batch.atom_types, batch.frac_coords, batch.lattice_polar,
        batch.num_atoms, batch.batch, lattices_mat=lattices_mat,
        cemb=cemb, guide_indicator=guide,
    )
    assert pred_l.shape == (batch.num_graphs, 6)

    # Energy：标量输出
    model_s = MatformerCSPNet(lattice_dim=6, pred_scalar=True, **common)
    pred_e = model_s(
        t, batch.atom_types, batch.frac_coords, batch.lattice_polar,
        batch.num_atoms, batch.batch, lattices_mat=lattices_mat,
    )
    assert pred_e.shape == (batch.num_graphs, 1), pred_e.shape

    print("[smoke] Matformer decoder OK")


def test_gmm():
    torch.manual_seed(0)
    M, max_n = 24, 5
    num_atoms = torch.randint(3, max_n + 1, (M,))
    F = torch.rand(num_atoms.sum(), 3)
    atom_types = torch.randint(1, 10, (num_atoms.sum(),))
    node2graph = torch.repeat_interleave(torch.arange(M), num_atoms)
    x, mask = pad_flat(F, num_atoms, max_n)
    k = torch.randn(M, 6)

    # 晶格 GMM
    lat = LatticeGMM.fit(k, K_max=4, seed=0)
    lp = lat.log_prob(k)
    assert lp.shape == (M, 4), lp.shape
    r = lat.responsibilities(k)
    assert torch.allclose(r.sum(-1), torch.ones(M), atol=1e-5)
    assert lat.sample(torch.randint(0, 4, (7,))).shape == (7, 6)

    # 坐标 wrapped GMM
    gmm = WrappedDiagonalGMM.fit(x, mask, K_max=4, max_iter=3, seed=0)
    ll = gmm.log_likelihood(x, mask)
    assert ll.shape == (M, 4), ll.shape
    assert torch.isfinite(ll).all()
    f0 = gmm.sample(torch.randint(0, 4, (M,)), num_atoms)
    assert f0.shape == (num_atoms.sum(), 3), f0.shape
    assert ((f0 >= 0) & (f0 < 1)).all()

    # 置换只交换同种元素；群平均有限
    perm = sample_permutation(atom_types, node2graph)
    assert torch.equal(atom_types[perm], atom_types)
    f0p, f1p = transform_pair(F, wrap(F + 0.1), atom_types, node2graph, M)
    assert f0p.shape == F.shape and f1p.shape == F.shape
    gbar = group_average_log_likelihood(
        F, num_atoms, atom_types, node2graph, max_n, gmm, J=3
    )
    assert gbar.shape == (M, 4), gbar.shape
    assert torch.isfinite(gbar).all()

    # 联合责任
    pi = torch.full((4,), 0.25)
    logp = torch.log(pi)[None, :] + lat.log_likelihood(k) + gbar
    r = torch.softmax(logp, -1)
    assert torch.allclose(r.sum(-1), torch.ones(M), atol=1e-5)

    print("[smoke] GMM Stage 0–2 math OK")


def test_flow_optional():
    try:
        import hydra  # noqa: F401
        import lightning  # noqa: F401
        from matformer_csp.pl_modules.flow import CSPFlow  # noqa: F401
    except ImportError as e:
        print(f"[smoke] 跳过 flow 测试（缺少依赖: {e}）")
        return
    print("[smoke] flow 依赖已安装；请用 run.py 的 fast_dev_run 做完整验证")


if __name__ == "__main__":
    test_matformer_decoder()
    test_gmm()
    test_flow_optional()
    print("[smoke] 全部通过")
