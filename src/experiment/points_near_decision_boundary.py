import torch
from torch.utils.data import DataLoader, TensorDataset

from omegaconf import DictConfig
from hydra.utils import instantiate
import hydra
import logging
import utils

log = logging.getLogger(__name__)


def get_fabric(config: DictConfig):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_components(config: DictConfig, fabric):
    model = fabric.setup(instantiate(config.network))
    return model


def get_dataloader(config: DictConfig, fabric):
    dataloader = instantiate(config.dataset)
    return fabric.setup_dataloaders(dataloader)


def select_points_from_each_class(
        dataloader: DataLoader, no_classes: int = 10, return_dl: bool = False
):
    class_points = {}

    for X, y in dataloader:
        for i in range(len(y)):
            label = y[i].item()
            if label not in class_points:
                class_points[label] = X[i]
            if len(class_points) == no_classes:
                break
        if len(class_points) == no_classes:
            break

    if return_dl:
        tensors = torch.stack(list(class_points.values()))
        labels = torch.tensor(list(class_points.keys()))
        dataset = TensorDataset(tensors, labels)
        return DataLoader(dataset, shuffle=False)

    return class_points


def find_the_most_uncertain_points(
        model: torch.nn.Module,
        dataset: dict,
        perturbation_eps: float,
) -> DataLoader:
    alphas = torch.linspace(0, 1.0, 100)
    results = {}
    first_label = list(dataset.keys())[0]
    first_tensor = dataset[first_label]

    with torch.no_grad():
        for class_label, current_tensor in dataset.items():
            if class_label == first_label:
                continue

            prev_connected = current_tensor.unsqueeze(0) if current_tensor.ndim == 3 else current_tensor

            for alpha in alphas:
                current_connected = (1 - alpha) * first_tensor + alpha * current_tensor
                current_connected = current_connected.unsqueeze(0) if current_connected.ndim == 3 else current_connected
                eps = perturbation_eps * torch.ones_like(current_connected)

                _, _, y_pred, _ = model(current_connected, eps, use_softmax=False)
                y_pred = torch.argmax(y_pred, dim=-1)

                if y_pred.item() == class_label:
                    results[class_label] = prev_connected
                    break

                prev_connected = current_connected

    tensors = torch.stack(list(results.values())).squeeze(1)
    labels = torch.tensor(list(results.keys()))
    dataset = TensorDataset(tensors, labels)
    return DataLoader(dataset, shuffle=False)


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    log.info("Launching Fabric")
    fabric = get_fabric(config)

    log.info("Building network")
    network = get_components(config, fabric)

    log.info("Setting up dataloader")
    dataloader = get_dataloader(config, fabric)

    with fabric.init_tensor():
        for batch_idx, batch in enumerate(dataloader):
            # Placeholder for actual training/inference logic
            if hasattr(network, "action"):
                network.action()
            else:
                out = network(batch[0])
                log.info(f"Output shape: {out.shape}")
            if batch_idx >= config.exp.get("max_batches", 1):
                break


