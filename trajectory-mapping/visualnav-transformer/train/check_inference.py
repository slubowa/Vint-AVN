import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

# --- If your script requires these:
# from warmup_scheduler import GradualWarmupScheduler
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.optimization import get_scheduler
# (If you don't actually need them for simple inference, you can remove.)

# --- Import your model + data + train_eval utilities
from vint_train.models.vint.vint import ViNT
from vint_train.training.train_eval_loop import load_model
from vint_train.data.vint_dataset_2 import ViNT_Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViNT model on a test set.")
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the YAML config file (same as used for training)."
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU device ID to use. Default=0."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override the batch size if desired. Otherwise uses config['batch_size']."
    )
    parser.add_argument(
        "--no_shuffle", action="store_true",
        help="If set, do NOT shuffle the test DataLoader."
    )
    # You can add any other commandline overrides you need here.
    return parser.parse_args()

def main():
    args = parse_args()

    # ---------------------------------------------------------
    #  1) LOAD CONFIG
    # ---------------------------------------------------------
    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    # Overwrite defaults with users config file:
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Optionally override batch size
    if args.batch_size is not None:
        config["eval_batch_size"] = args.batch_size

    # If for some reason your code expects `train` in config:
    config["train"] = False  # ensure were only doing inference

    # ---------------------------------------------------------
    #  2) DEVICE + SEED
    # ---------------------------------------------------------
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    # ---------------------------------------------------------
    #  3) BUILD MODEL (same hyperparams as training)
    # ---------------------------------------------------------
    # For model_type == "vint", e.g.:
    if config["model_type"] != "vint":
        raise ValueError("This script is only set up for ViNT. Adjust if using GNM/NoMaD/etc.")

    model = ViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],          # e.g. "efficientnet-b0"
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------
    #  4) LOAD CHECKPOINT (if any)
    # ---------------------------------------------------------
    # Usually your train script sets config["load_run"] or config["project_folder"]
    # to find the checkpoint. Or you store it in config["checkpoint_path"].
    # Lets assume you used "load_run" as in your train script:
    if "load_run" in config:
        ckpt_dir = os.path.join("logs", config["load_run"])
        ckpt_path = os.path.join(ckpt_dir, "latest.pth")
        print(f"[INFO] Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        load_model(model, config["model_type"], checkpoint)
    else:
        # If you prefer direct path, do:
        # ckpt_path = config["checkpoint_path"]
        # ...
        print("[WARNING] No checkpoint specified in config['load_run']. Using random weights.")

    # ---------------------------------------------------------
    #  5) BUILD TEST DATASET + DATALOADER
    # ---------------------------------------------------------
    # This mirrors your training scripts data pipeline.
    # Lets assume you only want to evaluate on one dataset. For example:
    test_dataset = None
    for dataset_name, data_config in config["datasets"].items():
        if "test" in data_config:
            # We found a dataset that has a "test" split
            test_dataset = ViNT_Dataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=data_config.get("waypoint_spacing", 1),
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=data_config.get("negative_mining", True),
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config.get("context_type", "temporal"),
                end_slack=data_config.get("end_slack", 0),
                goals_per_obs=data_config.get("goals_per_obs", 1),
                normalize=config["normalize"],
                goal_type=config["goal_type"],
            )
            break  # Just pick the first dataset with a test split

    if test_dataset is None:
        raise ValueError(
            "No test dataset found in config['datasets']. "
            "Make sure at least one dataset has a 'test' key."
        )

    # Dataloader
    eval_batch_size = config.get("eval_batch_size", config["batch_size"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=(False if args.no_shuffle else True),
        num_workers=config["num_workers"],
        drop_last=False
    )

    print(f"[INFO] Test dataset: {test_dataset.dataset_name}, size = {len(test_dataset)}")

    # ---------------------------------------------------------
    #  6) SINGLE-BATCH INFERENCE
    # ---------------------------------------------------------
    if len(test_loader) == 0:
        print("[ERROR] Test loader is empty! No samples to evaluate.")
        return

    # Grab one batch from the test set
    batch = next(iter(test_loader))

    # If your dataset returns (obs_img, goal_img, label_actions, dist, etc.),
    # adapt as needed. Lets assume the first 2 are images, next is label_actions, etc.
    obs_img, goal_img = batch[0], batch[1]
    label_actions = batch[2] if len(batch) > 2 else None

    print("obs_img.shape =", obs_img.shape)
    print("goal_img.shape =", goal_img.shape)
    if label_actions is not None:
        print("label_actions.shape =", label_actions.shape)

    # Move images to GPU
    obs_img = obs_img.to(device)
    goal_img = goal_img.to(device)

    with torch.no_grad():
        dist_pred, action_pred = model(obs_img, goal_img)

    print("\n[Inference results]")
    print("dist_pred.shape =", dist_pred.shape)           # Should be [B, 1] if your model outputs a single distance
    print("action_pred.shape =", action_pred.shape)       # Should be [B, len_traj_pred, X] if your model is set up that way

    # Print first samples predicted waypoints
    if action_pred.shape[0] > 0:
        print("First sample predicted waypoints:\n", action_pred[0])
    if action_pred.shape[0] > 1:
        print("Second sample predicted waypoints:\n", action_pred[1])

    print("[INFO] Inference complete.")

if __name__ == "__main__":
    main()