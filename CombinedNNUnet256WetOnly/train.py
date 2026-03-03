import os
import sys
import logging
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer

def _init_logger(model_dir: str) -> logging.Logger:
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(model_dir, "train.log")
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        to_stdout = logging.StreamHandler(sys.stdout)
        to_stdout.setFormatter(fmt)
        to_file = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        to_file.setFormatter(fmt)
        logger.addHandler(to_stdout)
        logger.addHandler(to_file)
    return logger

def _make_inferer(roi_size=(256, 256, 256), overlap=0.25, sw_batch_size=1, mode="gaussian"):
    return SlidingWindowInferer(
        roi_size=tuple(int(x) for x in roi_size),
        sw_batch_size=int(sw_batch_size),
        overlap=float(overlap),
        mode=str(mode),
    )

def train(
    model,
    data_in,
    max_epochs: int,
    device: torch.device,
    model_dir: str,
    *,
    val_infer_roi=(256, 256, 256),
    val_infer_overlap=0.25,
    val_sw_batch_size=1,
    lr=1e-4,
    weight_decay=1e-5,
    step_size=10,
    gamma=0.8,
    accum_steps=2,
    early_stop_patience=30,
    use_amp=True,
):
    train_loader, _, val_loader = data_in
    logger = _init_logger(model_dir)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and torch.cuda.is_available()))
    autocast = torch.amp.autocast

    inferer = _make_inferer(roi_size=val_infer_roi, overlap=val_infer_overlap, sw_batch_size=val_sw_batch_size)

    best_metric = -1.0
    best_metric_epoch = -1
    early_stop_counter = 0

    loss_train_hist, dice_train_hist = [], []
    loss_val_hist,   dice_val_hist  = [], []

    logger.info("=== Training start ===")
    logger.info("Max epochs: %d | LR: %.1e | Accum steps: %d | AMP: %s", max_epochs, lr, accum_steps, str(use_amp))
    logger.info("VAL SlidingWindow roi=%s overlap=%.2f sw_batch=%d", tuple(val_infer_roi), val_infer_overlap, val_sw_batch_size)

    for epoch in range(max_epochs):
        logger.info("-" * 40)
        logger.info("Epoch %d / %d", epoch + 1, max_epochs)

        model.train()
        epoch_loss, steps = 0.0, 0
        dice_metric.reset()
        optimizer.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            vol = batch["vol"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)

            with autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
                out  = model(vol)
                loss = loss_function(out, seg) / accum_steps

            scaler.scale(loss).backward()

            if (it + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accum_steps
            steps += 1

            with torch.no_grad():
                preds = torch.argmax(torch.softmax(out, dim=1), dim=1, keepdim=True)
                trues = seg.long()
                dice_metric(preds, trues)

        epoch_loss /= max(1, steps)
        epoch_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        logger.info("Train  | loss %.4f | dice %.4f", epoch_loss, epoch_dice)
        loss_train_hist.append(epoch_loss)
        dice_train_hist.append(epoch_dice)
        np.save(os.path.join(model_dir, "loss_train.npy"),   loss_train_hist)
        np.save(os.path.join(model_dir, "metric_train.npy"), dice_train_hist)

        model.eval()
        val_loss, val_steps = 0.0, 0
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                vvol = batch["vol"].to(device, non_blocking=True)
                vseg = batch["seg"].to(device, non_blocking=True)

                with autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
                    vout  = inferer(vvol, model)
                    vloss = loss_function(vout, vseg)

                val_loss += vloss.item()
                val_steps += 1

                vpred = torch.argmax(torch.softmax(vout, dim=1), dim=1, keepdim=True)
                vtrue = vseg.long()
                dice_metric(vpred, vtrue)

        val_loss /= max(1, val_steps)
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        logger.info("Val    | loss %.4f | dice %.4f", val_loss, val_dice)
        loss_val_hist.append(val_loss)
        dice_val_hist.append(val_dice)
        np.save(os.path.join(model_dir, "loss_val.npy"),   loss_val_hist)
        np.save(os.path.join(model_dir, "metric_val.npy"), dice_val_hist)

        if val_dice > best_metric:
            best_metric, best_metric_epoch = val_dice, epoch + 1
            torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
            early_stop_counter = 0
            logger.info("Saved new best model (epoch %d)", epoch + 1)
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            logger.info("Early stopping triggered (patience %d).", early_stop_patience)
            break

        scheduler.step()
        logger.info("Current LR %.6f", scheduler.get_last_lr()[0])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Training done. Best val dice %.4f at epoch %d", best_metric, best_metric_epoch)
