#!/usr/bin/env python3

"""
utilities
"""

import collections
import warnings

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from flwr.common import Context
from fms_ehrs.framework.dataset import Datasets
from transformers import AutoConfig, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
disable_progress_bar()
transformers.logging.set_verbosity_error()


def get_training_args(context: Context, **kwargs):

    max_steps = (
        (
            context.run_config["total-training-tokens"]
            * context.run_config["local-epochs"]
        )
        // context.run_config["max-seq-length"]
        // context.run_config["per-device-train-batch-size"]
        // context.run_config["gradient-accumulation-steps"]
    )

    return SFTConfig(
        report_to="wandb",
        output_dir="/gpfs/data/bbj-lab/users/burkh4rt/test-fed",
        max_seq_length=context.run_config["max-seq-length"],
        per_device_train_batch_size=context.run_config["per-device-train-batch-size"],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=context.run_config["gradient-accumulation-steps"],
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        eval_strategy="steps",
        save_strategy="best",
        ddp_find_unused_parameters=False,
        max_steps=max_steps,
        **kwargs,
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


def train(net, trainloader, testloader, context):
    net.to("cuda")
    trainer = SFTTrainer(
        model=net,
        train_dataset=trainloader,
        eval_dataset=testloader,
        args=get_training_args(context),
    )
    trainer.train()


def test(net, testloader, context):
    net.to("cuda")
    trainer = SFTTrainer(
        model=net,
        eval_dataset=testloader,
        args=get_training_args(context),
    )
    return trainer.evaluate()["eval_loss"]


def get_net(vocab):
    hf_config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        vocab_size=len(vocab),
        bos_token_id=vocab("TL_START"),
        eos_token_id=vocab("TL_END"),
        pad_token_id=vocab("PAD"),
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
    )
    return AutoModelForCausalLM.from_config(hf_config)


def get_weights(net):
    return [val.cpu().to(torch.float32).numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
