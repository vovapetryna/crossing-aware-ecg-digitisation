"""Time-budgeted training of the ordinate/existence U-Net on real series crops."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from .dataset import CacheIndex, WindowDataset
from .decoding import decode_ordinate_logits
from .model import TimmTraceUNet
from .predict import evaluate_cache
from .zip_io import CompetitionSource


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trace_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], coordinate_weight: float = 0.5, derivative_weight: float = 0.1, existence_weight: float = 0.25, l2_weight: float = 0.0, crossing_weight: float = 0.5) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["ordinate_logits"][:, 0]
    mask = batch["ordinate_mask"].bool()
    columns = mask.sum().clamp_min(1)
    ordinate_ce = -(batch["ordinate_target"] * logits.log_softmax(dim=1)).sum(dim=1)[mask].sum() / columns
    existence = functional.binary_cross_entropy_with_logits(outputs["existence_logits"][:, 0], batch["path_exists"])
    decoded = decode_ordinate_logits(outputs["ordinate_logits"], outputs["existence_logits"], top_k=5)
    scale = 1.0 / max(logits.shape[1] - 1, 1)
    prediction, target = decoded.y * scale, batch["ordinate_y"] * scale
    coordinate = functional.smooth_l1_loss(prediction[mask], target[mask], reduction="sum", beta=0.01) / columns
    adjacent = mask[:, 1:] & mask[:, :-1]
    derivative = functional.smooth_l1_loss(
        (prediction[:, 1:] - prediction[:, :-1])[adjacent], (target[:, 1:] - target[:, :-1])[adjacent],
        reduction="sum", beta=0.01,
    ) / adjacent.sum().clamp_min(1)
    total = ordinate_ce + existence_weight * existence + coordinate_weight * coordinate * 100 + derivative_weight * derivative * 100
    # Squared pixel error ~ squared mV error, the quantity the metric sums; dominated by QRS columns.
    squared = ((decoded.y - batch["ordinate_y"])[mask].abs().clamp(max=30.0) ** 2).sum() / columns
    total = total + l2_weight * squared
    crossing_parts = {}
    if "crossing_logits" in outputs and "crossing" in batch:
        band = batch["crossing_mask"]
        positive = batch["crossing"] > 0.5
        bce = functional.binary_cross_entropy_with_logits(
            outputs["crossing_logits"][band], batch["crossing"][band],
            pos_weight=torch.tensor(10.0, device=band.device)) if band.any() else outputs["crossing_logits"].sum() * 0
        offset = (outputs["crossing_offset"][positive] - batch["crossing_offset"][positive]).abs().mean() if positive.any() \
            else outputs["crossing_offset"].sum() * 0
        total = total + crossing_weight * (bce + 4.0 * offset)
        crossing_parts = {"crossing_bce": bce.item(), "crossing_offset_err": offset.item()}
    return total, {
        **crossing_parts,
        "l2_px2": squared.item(),
        "ordinate_ce": ordinate_ce.item(), "existence_bce": existence.item(),
        "coordinate_px": coordinate.item() / scale, "derivative_px": derivative.item() / scale,
    }


def build_model(config: dict, device: torch.device, pretrained: bool = True) -> torch.nn.Module:
    model = TimmTraceUNet(config["encoder"], pretrained=pretrained, decoder_width=int(config.get("decoder_width", 64)),
                          crossing_head=bool(config.get("crossing_head", False)), in_chans=int(config.get("in_chans", 4)))
    return model.to(device, memory_format=torch.channels_last)


def train(config: dict) -> dict:
    torch.manual_seed(int(config["seed"]))
    device = select_device(config.get("device", "auto"))
    output_dir = Path(config["output_dir"])
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    stride = int(config.get("label_stride", 1))
    train_index = CacheIndex(config["cache_dir"], "train", min_contrast=float(config.get("min_contrast", 0.0)), label_stride=stride)
    validation_index = CacheIndex(config["cache_dir"], "validation", label_stride=stride)
    source = CompetitionSource(config["source"])
    signal_cache: dict[str, dict] = {}

    def signals(record_id: str) -> dict:
        if record_id not in signal_cache:
            signal_cache[record_id] = source.signals(record_id)
        return signal_cache[record_id]

    dataset = WindowDataset(train_index, int(config["window"]), 10**8, seed=int(config["seed"]),
                            mask_key=config.get("mask_key", "ordinate_mask"), crossing=bool(config.get("crossing_head", False)))
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]), persistent_workers=int(config["num_workers"]) > 0,
        prefetch_factor=4 if int(config["num_workers"]) > 0 else None,
    )
    model = build_model(config, device)
    ema = None
    if config.get("ema_decay"):
        target_decay = float(config["ema_decay"])

        @torch.no_grad()
        def ema_update(ema_params, current_params, num_averaged):
            # Warm-up: decay ramps from 0 towards the target so the random initial weights are forgotten early.
            decay = min(target_decay, (1.0 + float(num_averaged)) / (10.0 + float(num_averaged)))
            for averaged, current in zip(ema_params, current_params):
                if averaged.is_floating_point():
                    averaged.lerp_(current, 1.0 - decay)
                else:
                    averaged.copy_(current)

        ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_update, use_buffers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    budget = float(config["time_budget_minutes"]) * 60.0
    warmup = int(config.get("warmup_steps", 200))
    accumulation = int(config["gradient_accumulation"])
    validate_every = float(config["validate_every_minutes"]) * 60.0
    info = {"device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "train_images": len(train_index),
            "validation_images": len(validation_index), "config": config}
    (output_dir / "run_config.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "config"}))

    started = time.perf_counter()
    last_validation = started
    best, history, step, running = -math.inf, [], 0, {}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for micro_step, batch in enumerate(loader):
        elapsed = time.perf_counter() - started
        progress = min(elapsed / budget, 1.0)
        lr = float(config["learning_rate"]) * min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(batch["image"].contiguous(memory_format=torch.channels_last))
        outputs = {key: value.float() for key, value in outputs.items()}
        loss, parts = trace_loss(outputs, batch, l2_weight=float(config.get("l2_weight", 0.0)))
        (loss / accumulation).backward()
        for key, value in {"loss": loss.item(), **parts}.items():
            running[key] = running.get(key, 0.0) + value
        if (micro_step + 1) % accumulation:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update_parameters(model)
        step += 1
        if step % 50 == 0:
            summary = {key: round(value / (50 * accumulation), 4) for key, value in running.items()}
            print(json.dumps({"step": step, "minutes": round(elapsed / 60, 1), "lr": lr, **summary}), flush=True)
            running = {}
        finished = elapsed >= budget
        if finished or time.perf_counter() - last_validation >= validate_every:
            scored = ema.module if ema is not None else model
            metrics = evaluate_cache(scored, validation_index, signals, device, limit=config.get("validation_limit"))
            model.train()
            last_validation = time.perf_counter()
            record = {"step": step, "minutes": round((last_validation - started) / 60, 1), **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            state = {"checkpoint_schema_version": 2, "model_type": "physionet", "step": step,
                     "model_state": scored.state_dict(), "config": config, "validation": metrics}
            torch.save(state, output_dir / "checkpoints" / "last.pt")
            if metrics["snr_db"] > best:
                best = metrics["snr_db"]
                torch.save(state, output_dir / "checkpoints" / "best.pt")
            (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        if finished:
            break
    report = {"best_snr_db": best, "steps": step, "elapsed_minutes": round((time.perf_counter() - started) / 60, 1), "history": history}
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key=json overrides")
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text())
    for item in arguments.set:
        key, value = item.split("=", 1)
        config[key] = json.loads(value)
    print(json.dumps(train(config)["best_snr_db"]))


if __name__ == "__main__":
    main()
