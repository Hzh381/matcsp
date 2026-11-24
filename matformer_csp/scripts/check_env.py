# 环境自检：打印 Python/torch/CUDA/关键依赖版本，并验证 PyG 的 scatter 路径。
#
#   python matformer_csp/scripts/check_env.py

import importlib
import sys


def main():
    print(f"python: {sys.version.split()[0]}")

    try:
        import torch
        print(
            f"torch: {torch.__version__} | "
            f"cuda build: {torch.version.cuda} | "
            f"cuda available: {torch.cuda.is_available()}"
        )
    except Exception as e:
        print(f"torch: MISSING ({type(e).__name__}: {e})")
        return

    modules = [
        "torch_geometric",
        "lightning",
        "hydra",
        "omegaconf",
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "pymatgen",
        "pyxtal",
        "p_tqdm",
        "smact",
        "matminer",
    ]
    for name in modules:
        try:
            mod = importlib.import_module(name)
            print(f"{name}: {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"{name}: MISSING ({type(e).__name__}: {e})")

    # 验证 PyG 的纯 PyTorch scatter 路径（项目不再依赖 torch_scatter）
    try:
        from torch_geometric.utils import scatter
        x = scatter(torch.ones(3), torch.tensor([0, 0, 1]), dim=0, reduce="sum")
        print(f"PyG scatter 调用: OK ({x.tolist()})")
    except Exception as e:
        print(f"PyG scatter 调用: FAILED ({type(e).__name__}: {e})")

    print("提示: 主 CSP 实验不依赖 torch_scatter/torch_sparse；gemnet/gnn 可选模型除外。")


if __name__ == "__main__":
    main()
