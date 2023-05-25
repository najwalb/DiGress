import sys
sys.path.append('..')

# These imports are tricky because they use c++, do not move them
from rdkit import Chem
try:
    import graph_tool
except ModuleNotFoundError:
    pass

import os
import warnings
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import numpy as np
import wandb


import numpy as np
import random
import pathlib
from sklearn.model_selection import train_test_split

project_dir = pathlib.Path(os.path.realpath(__file__)).parents[1]
 
warnings.filterwarnings("ignore", category=PossibleUserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@hydra.main(version_base='1.1', config_path='../configs', config_name=f'config')
def main(cfg: DictConfig):
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    dataset = cfg.dataset.name if cfg.dataset.number=='' else cfg.dataset.name + '-' + str(cfg.dataset.number)
    print(f'Processing dataset: {dataset}\n')

    splits = ['train', 'test', 'val']
    molecules = []
    data_path = os.path.join(project_dir, 'data', dataset, 'raw')
    
    for split in splits:
        path = os.path.join(data_path, f'{split}.csv')
        lines = open(path, 'r').readlines()
        for l in lines:
            rxn = l.strip()
            print(f'rxn {rxn}\n')
            molecules.extend(rxn.split('>>')[0].split('.'))
            molecules.extend(rxn.split('>>')[1].split('.'))
 
    #print([mol for mol, cnt in collections.Counter(molecules).items() if cnt > 1])
    print(f'total nb of molecules: {len(molecules)}\n')
    print(f'total nb of duplicate molecules: {len(molecules) - len(set(molecules))}\n')
    molecules = list(set(molecules))
    train, test = train_test_split(molecules, test_size=0.2, shuffle=True)
    test, val = train_test_split(test, test_size=0.5, shuffle=True)

    assert len(train)+len(test)+len(val)==len(molecules)

    open(os.path.join(data_path, 'train_mols.csv'), 'w').writelines(f'{m}\n' for m in set(train))
    open(os.path.join(data_path, 'test_mols.csv'), 'w').writelines(f'{m}\n' for m in set(test))
    open(os.path.join(data_path, 'val_mols.csv'), 'w').writelines(f'{m}\n' for m in set(val))

if __name__ == '__main__':
    main()
