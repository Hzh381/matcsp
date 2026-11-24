import sys
from pathlib import Path

# 直接以文件方式运行时，把仓库根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import omegaconf

from matformer_csp.common.utils import PROJECT_ROOT
from matformer_csp.run import finetune


@hydra.main(config_path=str(PROJECT_ROOT / "matformer_csp" / "conf"), config_name="default", version_base="1.3")
def main(cfg: omegaconf.DictConfig):
    finetune(cfg)


if __name__ == "__main__":
    main()
