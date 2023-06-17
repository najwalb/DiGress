## FIGURE OUT HOW TO HAVE MORE THAN ONE CONDA ENV ON TRITON

# DELETE ALL CONDA ENVS
# conda remove -n myenv --all
# rm -rf $WRKDIR/.conda_pkgs
# rm -rf $WRKDIR/.conda_envs
# rm -rf $WRKDIR/.local

#### only run the code below once
module load miniconda

# mkdir $WRKDIR/.conda_pkgs
# mkdir $WRKDIR/.conda_envs

# conda config --append pkgs_dirs ~/.conda/pkgs
# conda config --append envs_dirs ~/.conda/envs
# conda config --prepend pkgs_dirs $WRKDIR/.conda_pkgs
# conda config --prepend envs_dirs $WRKDIR/.conda_envs
conda create -n digress
conda activate digress 
# conda install pip

#### run following code to install packages
pip install pandas==1.5.3
pip install seaborn
pip install pyyaml
pip install imageio
pip install numpy>=1.11.1
pip install matplotlib==3.6.1
pip install scipy
pip install tqdm
pip install wandb
pip install hydra-core
pip install torch -i https://download.pytorch.org/whl/cu116
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install pytorch_lightning==1.7.2
pip install torchmetrics

pip install networkx==2.8.7
pip install overrides
pip install pygsp
pip install pyemd

pip install rdkit==2023.3.1
pip install git+https://github.com/igor-krawczuk/mini-moses@master
pip install -e ../

