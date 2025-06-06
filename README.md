# fms-flwr

## Send to randi:

```bash
rsync -avht \
  --delete \
  --exclude ".venv/" \
  --exclude ".idea/" \
  --exclude "logs/" \
  ~/Documents/chicago/fms-flwr \
  randi:/gpfs/data/bbj-lab/users/burkh4rt
```

## Install dependencies

```bash
cd /gpfs/data/bbj-lab/users/burkh4rt/fms-flwr
mkdir logs
python3 -m venv .venv
source .venv/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Interactive run:

```bash
systemd-run --scope --user tmux new -s gpuq
srun -p gpuq \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --time=1:00:00 \
  --job-name=adhoc \
  --pty bash -i

source .venv/bin/activate
now() { TZ=America/Chicago date +%Y-%m-%dT%H%M%S%z ; }
flwr run . | tee logs/run-$(now).log
```

## Slurm:

```bash
sbatch slurm.sh
```

<!--

Format:
```
isort src/
black src/
shfmt -w *.sh
prettier --write --print-width 81 --prose-wrap always *.md
```

-->
