import sys
sys.path.append('..')

import os
import subprocess

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.makedirs(dir)
        
conda_env = 'digress'
script_name = "main.py"
job_directory = os.path.join('../outputs/', script_name.split('.py')[0])

output_dir = os.path.join(job_directory, 'out')
jobids_file = os.path.join(job_directory, 'jobids.txt')

# Make top level directories
mkdir_p(job_directory)
mkdir_p(output_dir)

args_ = ['qm9_no_h'] 

for arg in args_:
    print(f"Creating job {arg}... ")
    job_file = os.path.join(job_directory, f"{arg}.job")
    # data_file = os.path.join(data_dir, f'{file}_rxn.pickle')
    # output_file = os.path.join(data_dir, f'{file}_graph.pickle')

    # check data file exists:
    # assert os.path.exists(data_file), f'Data file {data_file} not found! Aborting submission.'

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={arg}_%a.job\n") # add time stamp?
        fh.writelines(f"#SBATCH --output={output_dir}/{arg}_%a.out\n")
        fh.writelines(f"#SBATCH --error={output_dir}/{arg}_%a.err\n")
        fh.writelines("#SBATCH --partition=gpu\n")
        fh.writelines("#SBATCH --constraint=ampere\n")
        fh.writelines("#SBATCH --gres=gpu:a100:1\n")
        fh.writelines("#SBATCH --cpus-per-task=4\n")
        fh.writelines("#SBATCH --mem-per-cpu=10G\n")
        fh.writelines("#SBATCH --array=1-1\n")
        fh.writelines("#SBATCH --mail-type=ALL\n")
        fh.writelines("#SBATCH --mail-user=najwa.laabid@aalto.fi\n")
        fh.writelines("#SBATCH --time=15:00:00\n")
        fh.writelines("module load miniconda\n")
        fh.writelines(f"source activate {conda_env}\n\n")
        fh.writelines(f"python3 {script_name} +experiment={arg}.yaml dataset=qm9"+
                      f" hydra.run.dir=../experiments/mol/trained_models/{arg}\n")

    result = subprocess.run(args="sbatch", stdin=open(job_file, 'r'), capture_output=True)
    job_id = result.stdout.decode("utf-8").strip().split('job ')[1]

    with open(jobids_file, 'w') as f:
        f.write(f"{arg}.job: {job_id}")
    
    print(f"=== Submitted to Slurm with ID {job_id}.")



