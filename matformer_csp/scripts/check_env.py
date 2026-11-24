# 环境自检：打印 Python/torch/CUDA/PyG 扩展及关键依赖版本。
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
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
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
        "torchdyn",
    ]
    for name in modules:
        try:
            mod = importlib.import_module(name)
            print(f"{name}: {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"{name}: MISSING ({type(e).__name__}: {e})")

    # 真正调用一次 torch_scatter，确认 ABI 匹配（只 import 不够）
    try:
        from torch_scatter import scatter
        x = scatter(torch.ones(3), torch.tensor([0, 0, 1]), dim=0, reduce="sum")
        print(f"torch_scatter 调用: OK ({x.tolist()})")
    except Exception as e:
        print(f"torch_scatter 调用: FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
