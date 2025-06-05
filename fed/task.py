#!/usr/bin/env python3

"""
utilities
"""

import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from flwr.common import Context
from fms_ehrs.framework.dataset import Datasets
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


training_args = SFTConfig(
    output_dir="/gpfs/data/bbj-lab/users/burkh4rt/test-fed",
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    save_strategy="best",
    ddp_find_unused_parameters=False,
    max_steps=1_000,
)


def load_data(partition_id: int, num_partitions: int, n_epochs: int, context: Context):
    dataset = Datasets(
        data_version=context.run_config["data-version"],
        data_dir=context.run_config["data-dir"],
        i_part=partition_id,
        n_parts=num_partitions,
    )
    return (
        dataset.get_train_dataset(n_epochs=n_epochs),
        dataset.get_val_dataset(),
        dataset.vocab,
    )


def train(net, trainloader, testloader):
    net.to("cuda")
    trainer = SFTTrainer(
        model=net,
        train_dataset=trainloader,
        eval_dataset=testloader,
        args=training_args,
    )
    trainer.train()


def test(net, testloader):
    net.to("cuda")
    trainer = SFTTrainer(
        model=net,
        eval_dataset=testloader,
        args=training_args,
    )
    return trainer.evaluate()["eval_loss"]


def get_weights(net):
    return [val.cpu().to(torch.float32).numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
