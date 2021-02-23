import os

print(os.environ.get('CONDA_PREFIX'))
os.system("srun  -c2 --gres=gpu:1 --pty bash")