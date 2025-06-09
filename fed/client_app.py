#!/usr/bin/env python3

"""
things clients do
"""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from .task import get_net, get_weights, load_data, set_weights, test, train


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

    trainloader, valloader, vocab = load_data(
        context.node_config["partition-id"],
        context.node_config["num-partitions"],
        context.run_config["local-epochs"],
        context,
    )

    return FlowerClient(
        get_net(vocab).to("cuda"),
        trainloader,
        valloader,
        context.run_config["local-epochs"],
    ).to_client()


app = ClientApp(client_fn)
