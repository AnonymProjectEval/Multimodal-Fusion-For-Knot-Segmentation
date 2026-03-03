from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast


TensorOrCollection = Union[torch.Tensor, Mapping[str, Any], Sequence[Any]]


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(data: TensorOrCollection, device: torch.device) -> TensorOrCollection:
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, MutableMapping):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data


def _resolve_model_inputs(
    batch: Mapping[str, Any],
    input_keys: Union[str, Sequence[str]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    if isinstance(input_keys, str):
        return batch[input_keys]
    if len(input_keys) == 1:
        return batch[input_keys[0]]
    return tuple(batch[k] for k in input_keys)


def default_forward_train(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    input_keys: Union[str, Sequence[str]] = "vol",
    target_key: str = "seg",
    pass_batch_to_model: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = batch[target_key]

    if pass_batch_to_model:
        logits = model(batch)
        return logits, target

    model_inputs = _resolve_model_inputs(batch, input_keys)
    if torch.is_tensor(model_inputs):
        logits = model(model_inputs)
    else:
        logits = model(*model_inputs)
    return logits, target


def default_forward_val(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    input_keys: Union[str, Sequence[str]] = "vol",
    target_key: str = "seg",
    roi_size: Tuple[int, int, int] = (256, 256, 256),
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    pass_batch_to_model: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = batch[target_key]

    if pass_batch_to_model:
        logits = model(batch)
        return logits, target

    if isinstance(input_keys, str):
        x = batch[input_keys]
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        return logits, target

    if len(input_keys) == 1:
        x = batch[input_keys[0]]
        logits = sliding_window_inference(
            inputs=x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        return logits, target

    raise ValueError(
        "default_forward_val supports one tensor input key only for sliding window inference. "
        "For intermediate fusion or late fusion with multiple inputs, pass a custom val_forward_fn."
    )


def _metric_to_float(metric_obj: Any, metric_value: Any) -> float:
    if metric_value is not None:
        if torch.is_tensor(metric_value):
            return float(metric_value.detach().cpu().item())
        return float(metric_value)

    if hasattr(metric_obj, "aggregate"):
        agg = metric_obj.aggregate()
        if isinstance(agg, (list, tuple)):
            agg = agg[0]
        if torch.is_tensor(agg):
            return float(agg.detach().cpu().item())
        return float(agg)

    raise ValueError("Could not convert validation metric to float.")


def _safe_state_dict(obj: Any) -> Optional[dict]:
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return None


def train_model(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metric_fn: Any,
    device: Union[str, torch.device],
    exp_dir: Union[str, os.PathLike],
    num_epochs: int = 300,
    scheduler: Any = None,
    accum_steps: int = 1,
    use_amp: bool = True,
    input_keys: Union[str, Sequence[str]] = "vol",
    target_key: str = "seg",
    train_forward_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
    val_forward_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
    train_forward_kwargs: Optional[Dict[str, Any]] = None,
    val_forward_kwargs: Optional[Dict[str, Any]] = None,
    roi_size: Tuple[int, int, int] = (256, 256, 256),
    sw_batch_size: int = 1,
    overlap: float = 0.25,
    post_pred: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    post_label: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    validate_every: int = 1,
    early_stopping_patience: int = 50,
    min_delta: float = 0.0,
    max_grad_norm: Optional[float] = None,
    save_every_epoch: bool = False,
    seed: Optional[int] = None,
    save_numpy_history: bool = True,
    save_json_history: bool = True,
) -> Dict[str, Any]:
    """
    Shared training loop for fusion experiments.

    Early fusion can use defaults with input_keys set to "vol".

    Intermediate fusion and late fusion can reuse this by passing custom
    train_forward_fn and val_forward_fn, especially for sliding window inference.
    """

    if seed is not None:
        seed_everything(seed)

    device = torch.device(device)
    model = model.to(device)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "train_log.txt"

    train_forward_fn = train_forward_fn or default_forward_train
    val_forward_fn = val_forward_fn or default_forward_val
    train_forward_kwargs = train_forward_kwargs or {}
    val_forward_kwargs = val_forward_kwargs or {}

    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_metric": [],
        "lr": [],
        "epoch_seconds": [],
    }

    best_metric = -float("inf")
    best_epoch = -1
    no_improve_count = 0

    def _log(msg: str) -> None:
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _log("=" * 80)
    _log(f"Experiment directory: {exp_dir}")
    _log(f"Device: {device}")
    _log(f"Epochs: {num_epochs}, AMP: {use_amp}, Accum steps: {accum_steps}")
    _log(f"Validate every: {validate_every}, Patience: {early_stopping_patience}")
    _log("=" * 80)

    config_dump = {
        "num_epochs": num_epochs,
        "accum_steps": accum_steps,
        "use_amp": bool(use_amp),
        "input_keys": [input_keys] if isinstance(input_keys, str) else list(input_keys),
        "target_key": target_key,
        "roi_size": list(roi_size),
        "sw_batch_size": sw_batch_size,
        "overlap": overlap,
        "validate_every": validate_every,
        "early_stopping_patience": early_stopping_patience,
        "min_delta": min_delta,
        "max_grad_norm": max_grad_norm,
        "seed": seed,
    }
    with open(exp_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_train_loss = 0.0
        n_train_batches = 0
        accum_counter = 0

        for batch in train_loader:
            batch = move_to_device(batch, device)

            with autocast(enabled=scaler.is_enabled()):
                logits, target = train_forward_fn(
                    model,
                    batch,
                    input_keys=input_keys,
                    target_key=target_key,
                    pass_batch_to_model=False,
                    **train_forward_kwargs,
                )
                loss = loss_fn(logits, target)
                loss_to_backprop = loss / accum_steps

            scaler.scale(loss_to_backprop).backward()

            accum_counter += 1
            if accum_counter == accum_steps:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0

            running_train_loss += float(loss.detach().cpu().item())
            n_train_batches += 1

        if accum_counter > 0:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss_epoch = running_train_loss / max(1, n_train_batches)

        should_validate = (epoch % validate_every == 0) or (epoch == num_epochs)
        val_loss_epoch = np.nan
        val_metric_epoch = np.nan

        if should_validate:
            model.eval()
            if hasattr(metric_fn, "reset"):
                metric_fn.reset()

            running_val_loss = 0.0
            n_val_batches = 0
            metric_out = None

            with torch.no_grad():
                for batch in val_loader:
                    batch = move_to_device(batch, device)

                    with autocast(enabled=scaler.is_enabled()):
                        logits, target = val_forward_fn(
                            model,
                            batch,
                            input_keys=input_keys,
                            target_key=target_key,
                            roi_size=roi_size,
                            sw_batch_size=sw_batch_size,
                            overlap=overlap,
                            pass_batch_to_model=False,
                            **val_forward_kwargs,
                        )
                        val_loss = loss_fn(logits, target)

                    running_val_loss += float(val_loss.detach().cpu().item())
                    n_val_batches += 1

                    if post_pred is not None and post_label is not None:
                        pred_list = [post_pred(x) for x in decollate_batch(logits)]
                        label_list = [post_label(y) for y in decollate_batch(target)]
                        metric_out = metric_fn(y_pred=pred_list, y=label_list)
                    else:
                        metric_out = metric_fn(y_pred=logits, y=target)

            val_loss_epoch = running_val_loss / max(1, n_val_batches)
            val_metric_epoch = _metric_to_float(metric_fn, metric_out)

            if hasattr(metric_fn, "reset"):
                metric_fn.reset()

            improved = val_metric_epoch > (best_metric + min_delta)
            if improved:
                best_metric = val_metric_epoch
                best_epoch = epoch
                no_improve_count = 0

                best_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": _safe_state_dict(scheduler),
                    "best_metric": best_metric,
                    "history": history,
                }
                torch.save(best_ckpt, ckpt_dir / "best_model.pt")
            else:
                no_improve_count += 1

            if scheduler is not None:
                try:
                    scheduler.step(val_loss_epoch)
                except TypeError:
                    scheduler.step()
        else:
            if scheduler is not None:
                try:
                    scheduler.step()
                except TypeError:
                    pass

        epoch_seconds = time.time() - t0
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(float(val_loss_epoch) if not np.isnan(val_loss_epoch) else np.nan)
        history["val_metric"].append(float(val_metric_epoch) if not np.isnan(val_metric_epoch) else np.nan)
        history["lr"].append(current_lr)
        history["epoch_seconds"].append(epoch_seconds)

        msg = (
            f"Epoch {epoch:03d}/{num_epochs:03d} | "
            f"train loss {train_loss_epoch:.4f} | "
            f"val loss {val_loss_epoch:.4f} | "
            f"val metric {val_metric_epoch:.4f} | "
            f"lr {current_lr:.6g} | "
            f"time {epoch_seconds:.1f}s"
        )
        if should_validate:
            msg += f" | best {best_metric:.4f} at epoch {best_epoch:03d}"
        _log(msg)

        last_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": _safe_state_dict(scheduler),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "history": history,
        }
        torch.save(last_ckpt, ckpt_dir / "last_model.pt")

        if save_every_epoch:
            torch.save(last_ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")

        if save_numpy_history:
            np.save(exp_dir / "epoch.npy", np.asarray(history["epoch"], dtype=np.int32))
            np.save(exp_dir / "train_loss.npy", np.asarray(history["train_loss"], dtype=np.float32))
            np.save(exp_dir / "val_loss.npy", np.asarray(history["val_loss"], dtype=np.float32))
            np.save(exp_dir / "val_metric.npy", np.asarray(history["val_metric"], dtype=np.float32))
            np.save(exp_dir / "lr.npy", np.asarray(history["lr"], dtype=np.float32))
            np.save(exp_dir / "epoch_seconds.npy", np.asarray(history["epoch_seconds"], dtype=np.float32))

        if save_json_history:
            with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        if should_validate and no_improve_count >= early_stopping_patience:
            _log(
                f"Early stopping at epoch {epoch:03d}. "
                f"No improvement for {no_improve_count} validation checks."
            )
            break

    _log("=" * 80)
    _log(f"Training finished. Best validation metric: {best_metric:.4f} at epoch {best_epoch:03d}")
    _log(f"Best checkpoint: {ckpt_dir / 'best_model.pt'}")
    _log("=" * 80)

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "best_model_path": str(ckpt_dir / "best_model.pt"),
        "last_model_path": str(ckpt_dir / "last_model.pt"),
        "log_path": str(log_path),
    }