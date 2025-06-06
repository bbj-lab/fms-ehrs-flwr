#!/usr/bin/env python3

"""
things the server does
"""

import pathlib

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from fms_ehrs.framework.dataset import Datasets

from .save_fed_avg import SaveFedAvg
from .task import get_net, get_weights


def server_fn(context: Context):

    dataset = Datasets(
        data_version=context.run_config["data-version"],
        data_dir=pathlib.Path(context.run_config["data-dir"]).expanduser().resolve(),
    )

    net = get_net(dataset.vocab).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    strategy = SaveFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        net=net,
        context=context,
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
