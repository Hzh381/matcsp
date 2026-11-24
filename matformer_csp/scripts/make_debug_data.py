# 从 mp_20 的 CSV 截取少量样本，生成 data/mp_20_debug，用于 fast_dev_run 验证。
#
#   python matformer_csp/scripts/make_debug_data.py --source $MATFORMER_CSP_DATA_ROOT/mp_20

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="mp_20 数据目录（含 train/val/test.csv）")
    parser.add_argument("--out", default="data/mp_20_debug")
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        df = pd.read_csv(src / f"{split}.csv", nrows=n)
        df.to_csv(out / f"{split}.csv", index=False)
        print(f"{split}: {df.shape} -> {out / (split + '.csv')}")


if __name__ == "__main__":
    main()
