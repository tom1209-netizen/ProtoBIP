import argparse
import os
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.trainutils import get_cls_dataset
from utils.validate import generate_cam, validate
from model.model import ClsNetwork
from utils.pyutils import set_seed
import torch.nn as nn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the saved best_cam.pth model checkpoint.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # --- Setup ---
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)

    # --- Create Output Directory for Visualizations ---
    vis_dir = os.path.join(os.path.dirname(args.checkpoint), "test_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"CAM visualizations will be saved to: {vis_dir}")
    # Update config to point to the new output directory
    cfg.work_dir.pred_dir = vis_dir

    # --- Initialize Model ---
    print("\nInitializing model...")
    model = ClsNetwork(backbone=cfg.model.backbone.config,
                    stride=cfg.model.backbone.stride,
                    cls_num_classes=cfg.dataset.cls_num_classes,
                    num_prototypes_per_class=cfg.model.num_prototypes_per_class,
                    prototype_feature_dim=cfg.model.prototype_feature_dim,
                    n_ratio=cfg.model.n_ratio,
                    pretrained=False) # No need to load default pretrained weights, we're loading our own

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print("Model weights loaded successfully.")

    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Prepare Test Dataset ---
    print("\nPreparing test dataset...")
    # Use the 'test' split
    _, test_dataset = get_cls_dataset(cfg, split="test", enable_rotation=False, p=0.0)
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

    test_loader = DataLoader(test_dataset,
                            batch_size=1, # Process one image at a time
                            shuffle=False,
                            num_workers=min(10, os.cpu_count()),
                            pin_memory=True)

    # --- Optional: Re-run Evaluation to Confirm Score ---
    print("\n1. Re-running evaluation on test dataset...")
    print("-" * 50)
    loss_function = nn.BCEWithLogitsLoss().to(device) # Needed for validate function
    test_all_acc4, test_avg_acc4, fuse234_score, test_cls_loss = validate(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
    )
    print("Evaluation results:")
    print(f"Test avg acc4: {test_avg_acc4:.4f}")
    print(f"Test mIOU: {fuse234_score[:-1].mean():.4f}")

    # --- Generate CAMs for Visualization ---
    print("\n2. Generating CAMs for visualization...")
    print("-" * 50)
    generate_cam(model=model, data_loader=test_loader, cfg=cfg)

    print(f"\n✓ Done. Visualizations saved in {vis_dir}")

if __name__ == "__main__":
    main()