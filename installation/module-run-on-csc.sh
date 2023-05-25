## install torch==1.10 
# module purge
# module use /appl/soft/ai/rhel7/modulefiles/
# module load pytorch/1.10

# export PYTHONUSERBASE=/projappl/project_2006950/digress

# pip uninstall torch
# pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
# pip install --user torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip install --user torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Regular install
module purge
module load python-data

export PYTHONUSERBASE=/projappl/project_2006950/digress
pip install --user seaborn
pip install --user pyyaml
pip install --user imageio
pip install --user numpy>=1.11.1
pip install --user matplotlib==3.6.1
pip install --user scipy
pip install --user tqdm
pip install --user wandb
pip install --user hydra-core
pip install --user pytorch_lightning==1.7.2
pip install --user torchmetrics
pip install --user torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install --user torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

pip install --user networkx==2.8.7
pip install --user overrides
pip install --user pygsp
pip install --user pyemd

pip install --user rdkit==2023.3.1
pip install --user git+https://github.com/igor-krawczuk/mini-moses@master

## use conda env locally
# module load python-data
# export PATH="/projappl/project_2006950/digress/bin:$PATH"

## use with scripts
# module purge
# module load gcc/11.3.0
# module load cuda/11.7.0
# module load python-data

# export PYTHONPATH=/projappl/project_2006950/digress/lib/python3.10/site-packages/

## for pytorch 1.10 
# module purge

# module use /appl/soft/ai/rhel7/modulefiles/
# module load python-data
# module load pytorch/1.10
# export PATH="/projappl/project_2006950/digress/lib/python3.8/site-packages/:$PATH"
# export PYTHONPATH=/projappl/project_2006950/digress/lib/python3.8/site-packages/

