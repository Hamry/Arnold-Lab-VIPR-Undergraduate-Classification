#!/bin/bash

#=========================================================================================
# SLURM SUBMISSION SCRIPT FOR A PYTHON JOB ON GACRC (SAPELO2)
#
# This script is refined based on the official GACRC Sapelo2 documentation.
# It requests computational resources and runs a Python script.
#
# To use:
#   1. Save this file as 'run_python.slurm'.
#   2. Place your Python script (e.g., 'my_script.py') in the same directory.
#   3. Edit the #SBATCH directives below to request the resources you need.
#   4. Change 'my_script.py' to the name of your Python script.
#   5. Submit the job from your terminal using: sbatch run_python.slurm
#=========================================================================================

#-----------------------------------------------------------------------------------------
# SECTION 1: SLURM DIRECTIVES
#
# These lines, starting with #SBATCH, are instructions for the Slurm scheduler.
# They are not comments, so do not remove the #.
#-----------------------------------------------------------------------------------------

# Job Name: Give your job a descriptive name.
#SBATCH --job-name=my_python_job

# Partition: Specify the queue you want to run on. 'batch' is a general-purpose queue.
# Other common partitions include 'gpu_a100' for GPUs or 'highmem' for memory-intensive jobs.
#SBATCH --partition=batch

# Account: If you are part of a specific research group with an allocation, specify it here.
# #SBATCH --account=my_research_group

# Number of Tasks: For a simple Python script, this is almost always 1.
#SBATCH --ntasks=1

# CPUs per Task: Request the number of CPU cores for your job.
# This is important for multi-threaded applications. For a standard script, 1 or 2 is fine.
#SBATCH --cpus-per-task=2

# Memory: Request the amount of RAM your job needs.
# Specify units like K, M, G, T (e.g., 16G for 16 gigabytes).
#SBATCH --mem=8G

# Wall Time: The maximum time your job will run (format: D-HH:MM:SS).
# The scheduler will kill your job if it exceeds this time. Be generous but realistic.
#SBATCH --time=1-00:00:00 # 1 day

# Output and Error Files: Define where the standard output and error streams go.
# %x will be replaced by the job name, %j by the job ID. This is a GACRC recommended format.
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# Email Notifications: Get emails when your job starts, ends, or fails.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_myid@uga.edu # <-- IMPORTANT: REPLACE WITH YOUR EMAIL

#-----------------------------------------------------------------------------------------
# SECTION 2: JOB EXECUTION
#
# This is where you list the commands you want your job to run.
#-----------------------------------------------------------------------------------------

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submission Directory: $SLURM_SUBMIT_DIR"

# Navigate to the directory where the job was submitted from.
# This is a critical step recommended by GACRC to ensure the script
# can find your Python file and any other necessary input/output files.
cd $SLURM_SUBMIT_DIR

# Load necessary modules.
# It's crucial to load the same software environment you used for development/testing.
# Use 'ml spider Python' to see all available Python versions.
ml Python/3.10.4-GCCcore-11.3.0

# Run your Python script.
# Replace 'my_script.py' with the actual name of your script.
echo "Running Python script..."
python my_script.py

echo "Job finished at $(date)"

