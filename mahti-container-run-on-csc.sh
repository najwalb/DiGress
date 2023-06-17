module purge
module load gcc
module load git
module load tykky
export PATH=/appl/opt/python/3.10.9-gnu112/bin:$PATH

mkdir /projappl/project_2006950/digress

pip-containerize new --prefix /projappl/project_2006950/digress torch.txt
pip-containerize update /projappl/project_2006950/digress --post-install req.txt
pip-containerize update /projappl/project_2006950/digress --post-install local-install.txt
export PATH="/projappl/project_2006950/digress/bin:$PATH" 
