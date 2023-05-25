import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics

atom_types = ['O', 'C', 'N', 'I', 'H', 'Cl', 'Si', 'F', 'Br', 'N+1', 'O-1', 'S', 'B', 'N-1', 'Zn+1', 
              'Cu', 'Sn', 'P+1', 'Mg+1', 'C-1', 'P', 'S+1', 'S-1', 'Se', 'Zn', 'Mg']
atom_type_offset = 1 # where to start atom type indexing. 1 if no node type, 0 otherwise

# starting from 1 because considering 0 an edge type (= no edge)
bond_types = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE]
bond_type_offset = 1

def get_mol_nodes(mol):
    atoms = mol.GetAtoms()

    for i, atom in enumerate(atoms):
        at = atom.GetSymbol() if atom.GetFormalCharge()==0 else atom.GetSymbol()+f'{atom.GetFormalCharge():+}'
        atom_type = torch.tensor([atom_types.index(at)+atom_type_offset], 
                                  dtype=torch.long) # needs to be int for one hot
        atom_types_ = torch.cat((atom_types_, atom_type), dim=0) if i > 0 else atom_type
    
    atom_feats = F.one_hot(atom_types_, num_classes=len(atom_types)+atom_type_offset).float()

    return atom_feats

def get_mol_edges(mol, offset=1):
    '''
        Input:
            offset (optional): default: 1. To account for 'no bond' type.
    '''
    for i, b in enumerate(mol.GetBonds()):
        beg_atom_idx = b.GetBeginAtom().GetIdx()
        end_atom_idx = b.GetEndAtom().GetIdx()

        e_beg = torch.tensor([beg_atom_idx+offset, end_atom_idx+offset], dtype=torch.long).unsqueeze(-1)
        e_end = torch.tensor([end_atom_idx+offset, beg_atom_idx+offset], dtype=torch.long).unsqueeze(-1)
        e_type = torch.tensor([bond_types.index(b.GetBondType())+bond_type_offset, 
                               bond_types.index(b.GetBondType())+bond_type_offset], dtype=torch.long) # needs to be int for one hot

        begs = torch.cat((begs, e_beg), dim=0) if i > 0 else e_beg
        ends = torch.cat((ends, e_end), dim=0) if i > 0 else e_end
        edge_type = torch.cat((edge_type, e_type), dim=0) if i > 0 else e_type

    if len(mol.GetBonds())>0:
        edge_index = torch.cat((begs, ends), dim=1).mT.contiguous()
        edge_attr = F.one_hot(edge_type, num_classes=len(bond_types)+bond_type_offset).float() # add 1 to len of bonds to account for no edge
    else:
        edge_index = torch.tensor([]).long().reshape(2,0)
        edge_attr = torch.tensor([]).float().reshape(0, len(bond_types)+bond_type_offset)

    return edge_index, edge_attr

def mol_to_graph(mol, offset=0):
    if type(mol)==str: mol = Chem.MolFromSmiles(mol)
    Chem.RemoveStereochemistry(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = Chem.AddHs(mol, explicitOnly=True)
    
    m_nodes = get_mol_nodes(mol=mol)
    m_edge_index, m_edge_attr = get_mol_edges(mol=mol, offset=offset)

    return m_nodes, m_edge_index, m_edge_attr

def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]

class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data

class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data

class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data

class USPTO50KDataset(InMemoryDataset):

    def __init__(self, stage, root, remove_h: bool, target_prop=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.target_prop = target_prop
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['train_mols.csv', 'val_mols.csv', 'test_mols.csv']

    @property
    def processed_file_names(self):
        return ['train_mols.pt', 'val_mols.pt', 'test_mols.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        graphs = []
        for i, smiles in enumerate(open(self.raw_paths[self.file_idx], 'r')):
            mol = Chem.MolFromSmiles(smiles.strip())
            nodes, edge_index, edge_attr = mol_to_graph(mol, offset=0)
            y = torch.zeros((1,0), dtype=torch.float)
            graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)
            graphs.append(graph)

        torch.save(self.collate(graphs), self.processed_paths[self.file_idx])

class USPTO50KDataModule(MolecularDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        self.datadist_dir = cfg.dataset.datadist_dir
        if cfg.dataset.number!='':
            self.datadir += '-'+str(cfg.dataset.number)
            self.datadist_dir += '-'+str(cfg.dataset.number)
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h

    def prepare_data(self) -> None:
        transform = None
        target = None

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': USPTO50KDataset(stage='train', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                        target_prop=target, transform=transform),
                    'val': USPTO50KDataset(stage='val', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                      target_prop=target, transform=transform),
                    'test': USPTO50KDataset(stage='test', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                       target_prop=target, transform=transform)}
        super().prepare_data(datasets)

    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond
        multiplier = torch.tensor([0, 1, 2, 3])

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]
                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0]==atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()

        return valencies
    
class USPTO50Kinfos(AbstractDatasetInfos):
    def __init__(self, cfg, datamodule, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.max_n_nodes = 50
        self.max_weight = 500
        self.atom_weights = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 
                             11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 
                             19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1}
        self.valencies = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                          1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.datamodule = datamodule
        self.name = 'uspto50k-mols'
        self.atom_decoder = ['none']+atom_types
        self.bond_decoder = ['none']+bond_types

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, datamodule.datadist_dir, 'processed')
        node_count_path = os.path.join(root_path, 'n_counts_mol.txt')
        atom_type_path = os.path.join(root_path, 'atom_types_mol.txt')
        edge_type_path = os.path.join(root_path, 'edge_types_mol.txt')
        valency_dist_path = os.path.join(root_path, 'valency_dist.txt')
        paths_exist = os.path.exists(node_count_path) and os.path.exists(atom_type_path)\
                      and os.path.exists(edge_type_path) and os.path.exists(valency_dist_path)
        
        if not recompute_statistics and paths_exist:
            # use the same distributions for all subsets of the dataset
            self.n_nodes = torch.from_numpy(np.loadtxt(node_count_path))
            self.node_types = torch.from_numpy(np.loadtxt(atom_type_path))
            self.edge_types = torch.from_numpy(np.loadtxt(edge_type_path))
            self.valency_dist = torch.from_numpy(np.loadtxt(valency_dist_path))
        else:
            print('Recomputing\n')
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(node_count_path, self.n_nodes.cpu().numpy())
            self.node_types = datamodule.node_types()                                   
            print("Distribution of node types", self.node_types)
            np.savetxt(atom_type_path, self.node_types.cpu().numpy())
            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(edge_type_path, self.edge_types.cpu().numpy())
            self.valency_dist = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", self.valency_dist)
            np.savetxt(valency_dist_path, self.valency_dist.numpy())
            self.valency_distribution = self.valency_dist

        super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)

