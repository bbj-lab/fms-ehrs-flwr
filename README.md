# fed-flwr

## Send to randi:

```bash
rsync -avht \
  --delete \
  --exclude ".venv/" \
  --exclude ".idea/" \
  --exclude "logs/" \
  ~/Documents/chicago/fed-flwr \
  randi:/gpfs/data/bbj-lab/users/burkh4rt

systemd-run --scope --user tmux new -s gpuq3
srun -p gpuq \
  --gres=gpu:3 \
  --cpus-per-task=3 \
  --time=8:00:00 \
  --job-name=adhoc \
  --pty bash -i

cd /gpfs/data/bbj-lab/users/burkh4rt/fed-flwr
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Run the simulation engine

```bash
mkdir logs
now() { TZ=America/Chicago date +%Y-%m-%dT%H%M%S%z ; }
flwr run . | tee logs/run-$(now).log
```
