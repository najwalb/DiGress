#### run following code to install packages
conda create --name digress
conda activate digress
conda install pip

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
pip install pytorch_lightning==1.7.2
pip install torchmetrics
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

pip install ipdb
pip install pdbpp
pip install networkx==2.8.7
pip install overrides
pip install pygsp
pip install pyemd

pip install rdkit==2023.3.1
pip install git+https://github.com/igor-krawczuk/mini-moses@master
pip install -e .
