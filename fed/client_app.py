#!/usr/bin/env python3

"""
things clients do
"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoConfig, AutoModelForCausalLM

from .task import get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, self.testloader)
        return get_weights(self.net), 1, {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader)
        return float(loss), 1, {}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, vocab = load_data(partition_id, num_partitions, 1, context)

    # Load model
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
    net = AutoModelForCausalLM.from_config(hf_config)
    net.to("cuda")

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
