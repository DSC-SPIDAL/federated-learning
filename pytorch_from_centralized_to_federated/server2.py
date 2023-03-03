

import flwr as fl
import cifar
import utils
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import log_loss

from typing import Dict, List, Tuple

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_FEDBN: bool = True

trainloader, testloader, num_examples = cifar.load_data()

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: cifar.Net):
    """Return an evaluation function for server-side evaluation."""


    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Set model parameters, evaluate model on local test dataset, return result

        model.train()
        if USE_FEDBN:
            keys = [k for k in model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

        loss, accuracy = cifar.test(model, testloader, device=DEVICE)
        return float(loss),  {"accuracy": float(accuracy)}

    return evaluate

if __name__ == "__main__":
    model = cifar.Net().to(DEVICE).train()


    _ = model(next(iter(trainloader))[0].to(DEVICE))
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
