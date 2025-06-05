#!/usr/bin/env python3

"""
things the server does
"""

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fms_ehrs.framework.dataset import Datasets
from transformers import AutoConfig, AutoModelForCausalLM

from .task import get_weights


def server_fn(context: Context):

    dataset = Datasets(
        data_version=context.run_config["data-version"],
        data_dir=context.run_config["data-dir"],
    )

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initialize global model
    hf_config = AutoConfig.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        vocab_size=len(dataset.vocab),
        bos_token_id=dataset.vocab("TL_START"),
        eos_token_id=dataset.vocab("TL_END"),
        pad_token_id=dataset.vocab("PAD"),
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
    )
    net = AutoModelForCausalLM.from_config(hf_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
