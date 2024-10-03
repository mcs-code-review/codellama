#!/bin/bash
# Partition for the job:
#SBATCH --partition=deeplearn
##SBATCH --partition=gpu-h100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="34b-instruct-few-shot"

# The project ID which this job should run under:
#SBATCH --account="punim2247"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:4
# Slurm QoS:
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg5

# Requested memory per node:
#SBATCH --mem=100G

# Use this email address:
#SBATCH --mail-user=mukhammad.karimov@student.unimelb.edu.au

# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-72:0:00

# Standard output and error log
#SBATCH -o logs/34b-instruct-few-shot-%N.%j.out # STDOUT
#SBATCH -e logs/34b-instruct-few-shot-%N.%j.err # STDERR

# Run the job from the directory where it was launched (default)

# The modules to load:
echo "Current modules:"
echo "$(module list)"
echo "Loading modules..."
module load foss/2022a
module load CUDA/12.2.0
module load NCCL/2.19.4-CUDA-12.2.0
module load UCX-CUDA/1.14.1-CUDA-12.2.0
module load cuDNN/8.9.3.28-CUDA-12.2.0
module load GCCcore/11.3.0
module load Python/3.10.4
echo "Loaded modules:"
echo "$(module list)"

# The job command(s):
source ~/venvs/codellama/bin/activate

### CodeReviewer ###

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-cr-bm25-1.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-cr-bm25-2.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-cr-bm25-3.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-cr-bm25-4.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

### CodeReviewer with Ownership ###

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot-with-ownership/codellama-34b-instruct-cr-pkg_aco_bm25-3.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 8192 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot-with-ownership/codellama-34b-instruct-cr-pkg_rso_bm25-3.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 8192 \
    --max_batch_size 10 \
    --debug True

### CodeReviewerNew ###

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-crn-bm25-1.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-crn-bm25-2.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-crn-bm25-3.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot/codellama-34b-instruct-crn-bm25-4.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 4096 \
    --max_batch_size 10 \
    --debug True

### CodeReviewerNew with Ownership ###

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot-with-ownership/codellama-34b-instruct-crn-pkg_aco_bm25-3.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 8192 \
    --max_batch_size 10 \
    --debug True

torchrun --nproc_per_node 4 code_review_instructions.py \
    --ckpt_dir ./ckpt/CodeLlama-34b-Instruct/ \
    --tokenizer_path ./ckpt/CodeLlama-34b-Instruct/tokenizer.model \
    --conf_path ../config/few-shot-with-ownership/codellama-34b-instruct-crn-pkg_rso_bm25-3.json \
    --temperature 0.0 \
    --top_p 0.95 \
    --max_seq_len 8192 \
    --max_batch_size 10 \
    --debug True

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -c -n -s
my-job-stats -a -n -s
nvidia-smi
