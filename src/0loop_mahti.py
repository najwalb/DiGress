# import sys
# sys.path.append('..')
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
    # data_file = os.path.join(data_dir, f"{arg}.csv")
    # output_file = os.path.join(data_dir, f'rxn_{arg}.csv')

    # check data file exists:
    # assert os.path.exists(data_file), f'Data file {data_file} not found! Aborting submission.'
    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name={arg}_%a.job\n") # add time stamp?
        fh.writelines(f"#SBATCH --output={output_dir}/{arg}_%a.out\n")
        fh.writelines(f"#SBATCH --error={output_dir}/{arg}_%a.err\n")
        fh.writelines("#SBATCH --account=project_2007775\n")
        # fh.writelines("#SBATCH --partition=gpusmall\n")
        # fh.writelines("#SBATCH --gres=gpu:a100:1\n")
        fh.writelines("#SBATCH --cpus-per-task=4\n")
        fh.writelines("#SBATCH --time=01:00:00\n")
        fh.writelines("#SBATCH --array=1-1\n")
        fh.writelines("module purge\n")
        fh.writelines("module load git\n\n")
        fh.writelines("module load gcc\n\n")
        # fh.writelines("module load tykky\n\n")
        # fh.writelines("export PATH=/appl/opt/python/3.10.9-gnu112/bin:$PATH\n")
        # #fh.writelines(f"export PYTHONPATH=/projappl/project_2006950/{conda_env}/lib/python3.10/site-packages/\n")
        fh.writelines(f"export PATH='/projappl/project_2006950/{conda_env}/bin:$PATH'\n")
        # fh.writelines(f"python3 try_module.py")
        fh.writelines(f"python3 {script_name} +experiment={arg}.yaml dataset=qm9"+
                      f" hydra.run.dir=../experiments/mol/trained_models/{arg}\n")

    result = subprocess.run(args="sbatch", stdin=open(job_file, 'r'), capture_output=True)
    if 'job' not in result.stdout.decode("utf-8"):
        print(result)
    else:
        job_id = result.stdout.decode("utf-8").strip().split('job ')[1]

        with open(jobids_file, 'a') as f:
            f.write(f"{arg}.job: {job_id}\n")
        print(f"=== Submitted to Slurm with ID {job_id}.")


