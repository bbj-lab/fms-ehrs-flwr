# fms-ehrs-flwr

This [flower](https://flower.ai) app performs federated training of a FM on
tokenized EHR data.

## Install

```bash
git clone https://github.com/bbj-lab/fms-ehrs-flwr
cd cd fms-ehrs-flwr
mkdir -p logs
# pip install uv
uv venv --python=$(which python3) venv
source venv/bin/activate
uv pip install --torch-backend=cu128 --link-mode=copy -e .
```

## Interactive run

You can develop code on a single gpu and use the `gpudev` partition which
generally has good availability. (This is not efficient, so I would use this
primarily for troubleshooting/debugging.)

```bash
systemd-run --scope --user tmux new -s gpuq
srun -p gpudev \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --time=1:00:00 \
  --job-name=adhoc \
  --pty bash -i

source venv/bin/activate
now() { TZ=America/Chicago date +%Y-%m-%dT%H%M%S%z ; }
flwr run . | tee logs/run-$(now).log
```

## Slurm

There's a second configuration that runs 3 gpu's on the `gpuq` partition:

```bash
jid=$(sbatch --parsable slurm.sh)
```

## Monitoring

-   Running [nvtop](https://github.com/Syllo/nvtop) on the node running the job
    (`srun --jobid=$jid --pty nvtop`) should give you something like this:

    ![](img/nvtop.png)

-   Other statistics and real-time output is available on
    [weights and biases](wandb.ai).

<!--

Format:
```
ruff format .
shfmt -w .
prettier --write *.md
```

-->
