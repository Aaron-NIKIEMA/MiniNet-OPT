"""Protocoles d’expériences et comparaisons d’hyperparamètres."""

import numpy as np

from model import init_params
from model import backward, forward
from losses import bce
from optimizers import adam_update, initialize_state, sgd_update
from regularization import parameter_norm
from train import train_model


def compare_learning_rates(
    X_train,
    y_train,
    X_val,
    y_val,
    optimizer,
    learning_rates,
    epochs=100,
    batch_size=32,
    seed=42,
    optimizer_kwargs=None,
):
    """Compare plusieurs learning rates en sélectionnant sur la validation."""
    results = {}

    for learning_rate in learning_rates:
        params = init_params(seed=seed)
        _, history = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            params,
            optimizer=optimizer,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            optimizer_kwargs=optimizer_kwargs,
        )
        results[learning_rate] = {
            "train_loss": history["train_loss"][-1],
            "val_loss": history["val_loss"][-1],
            "val_accuracy": history["val_accuracy"][-1],
        }

    return results


def compare_l2_with_constraint(
    X_train,
    y_train,
    X_val,
    y_val,
    coefficient,
    learning_rate=0.05,
    epochs=100,
    batch_size=32,
    seed=42,
):
    """Compare L2 et contrainte en prenant R égal à la norme obtenue avec L2."""
    l2_params, l2_history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        init_params(seed=seed),
        optimizer="sgd",
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        l2_coefficient=coefficient,
    )
    radius = parameter_norm(l2_params)
    constrained_params, constrained_history = train_model(
        X_train,
        y_train,
        X_val,
        y_val,
        init_params(seed=seed),
        optimizer="projected_gradient",
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        constraint_radius=radius,
    )
    return {
        "radius": radius,
        "l2_params": l2_params,
        "l2_history": l2_history,
        "constrained_params": constrained_params,
        "constrained_history": constrained_history,
    }


def verify_pytorch_reference(X, y, params, sgd_rate=0.05, adam_rate=0.01):
    """Compare forward, loss, gradients et pas SGD/Adam avec PyTorch."""
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional

    class TorchNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 16)
            self.layer2 = nn.Linear(16, 8)
            self.layer3 = nn.Linear(8, 1)

        def forward(self, inputs):
            hidden1 = torch.relu(self.layer1(inputs))
            hidden2 = torch.relu(self.layer2(hidden1))
            return torch.sigmoid(self.layer3(hidden2))

    def create_torch_model():
        model = TorchNetwork().double()
        with torch.no_grad():
            for layer, weight_key, bias_key in (
                (model.layer1, "W1", "b1"),
                (model.layer2, "W2", "b2"),
                (model.layer3, "W3", "b3"),
            ):
                layer.weight.copy_(torch.as_tensor(params[weight_key]))
                layer.bias.copy_(torch.as_tensor(params[bias_key].reshape(-1)))
        return model

    X_torch = torch.as_tensor(X, dtype=torch.float64)
    y_torch = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float64)

    predictions, cache = forward(X, params)
    numpy_loss = bce(predictions, y.reshape(-1, 1))
    numpy_grads = backward(y.reshape(-1, 1), params, cache)

    reference = create_torch_model()
    torch_predictions = reference(X_torch)
    torch_loss = torch_functional.binary_cross_entropy(torch_predictions, y_torch)
    torch_loss.backward()

    torch_layers = (reference.layer1, reference.layer2, reference.layer3)
    weight_keys = ("W1", "W2", "W3")
    bias_keys = ("b1", "b2", "b3")
    gradient_errors = []
    for layer, weight_key, bias_key in zip(torch_layers, weight_keys, bias_keys):
        gradient_errors.append(
            np.max(np.abs(numpy_grads[weight_key] - layer.weight.grad.detach().numpy()))
        )
        gradient_errors.append(
            np.max(np.abs(numpy_grads[bias_key] - layer.bias.grad.detach().numpy().reshape(1, -1)))
        )

    numpy_sgd = sgd_update(params, numpy_grads, sgd_rate)
    torch_sgd = create_torch_model()
    torch_optimizer = torch.optim.SGD(torch_sgd.parameters(), lr=sgd_rate)
    torch_optimizer.zero_grad()
    torch_functional.binary_cross_entropy(torch_sgd(X_torch), y_torch).backward()
    torch_optimizer.step()

    numpy_adam, _ = adam_update(
        params, numpy_grads, initialize_state("adam", params), adam_rate
    )
    torch_adam = create_torch_model()
    torch_optimizer = torch.optim.Adam(
        torch_adam.parameters(), lr=adam_rate, betas=(0.9, 0.999), eps=1e-8
    )
    torch_optimizer.zero_grad()
    torch_functional.binary_cross_entropy(torch_adam(X_torch), y_torch).backward()
    torch_optimizer.step()

    def parameter_error(model, numpy_params):
        layers = (model.layer1, model.layer2, model.layer3)
        errors = []
        for layer, weight_key, bias_key in zip(layers, weight_keys, bias_keys):
            errors.append(np.max(np.abs(numpy_params[weight_key] - layer.weight.detach().numpy())))
            errors.append(np.max(np.abs(numpy_params[bias_key] - layer.bias.detach().numpy().reshape(1, -1))))
        return max(errors)

    return {
        "forward_max_abs_error": np.max(
            np.abs(predictions - torch_predictions.detach().numpy())
        ),
        "loss_abs_error": abs(numpy_loss - torch_loss.item()),
        "gradient_max_abs_error": max(gradient_errors),
        "sgd_parameter_max_abs_error": parameter_error(torch_sgd, numpy_sgd),
        "adam_parameter_max_abs_error": parameter_error(torch_adam, numpy_adam),
    }
