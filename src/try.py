
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

from omegaconf import DictConfig
import torch
import hydra

import random
import numpy as np

from datasets import uspto50k_dataset

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'config')
def main(cfg: DictConfig):
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    datamodule = uspto50k_dataset.USPTO50KDataModule(cfg)
    datamodule.prepare_data()
    dataset_infos = uspto50k_dataset.USPTO50Kinfos(datamodule=datamodule)
    
if __name__ == '__main__':
    main()
