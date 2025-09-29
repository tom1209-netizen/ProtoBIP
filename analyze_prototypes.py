import argparse
import os
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to the config file to get class info.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the saved best_cam.pth model checkpoint.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    num_classes = cfg.dataset.cls_num_classes
    num_prototypes_per_class = cfg.model.num_prototypes_per_class
    class_names = ['TUM', 'STR', 'LYM', 'NEC'] # Or load from config if available

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at {args.checkpoint}")
        return

    # Load to CPU, as we don't need a GPU for this analysis
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # --- Extract Prototypes ---
    if 'model' in checkpoint and 'prototypes' in checkpoint['model']:
        prototypes = checkpoint['model']['prototypes']
        print(f"Successfully extracted prototypes tensor with shape: {prototypes.shape}")
    else:
        print("ERROR: Could not find 'prototypes' tensor in the checkpoint's model state_dict.")
        return

    # --- Calculate Cosine Similarity ---
    # Normalize each prototype vector to unit length
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
    # Calculate the similarity matrix using a dot product
    similarity_matrix = torch.matmul(prototypes_norm, prototypes_norm.T)
    
    # Convert to NumPy for plotting
    similarity_matrix_np = similarity_matrix.cpu().numpy()

    # --- Visualize the Heatmap ---
    print("Generating prototype similarity heatmap...")
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(similarity_matrix_np, cmap='viridis', vmin=-1, vmax=1)
    
    ax.set_title('Cosine Similarity Between Learned Prototypes', fontsize=16)
    ax.set_xlabel('Prototype Index', fontsize=12)
    ax.set_ylabel('Prototype Index', fontsize=12)

    # --- Add Class Labels and Gridlines for Readability ---
    ticks = []
    tick_labels = []
    for i in range(num_classes):
        # Position the tick in the middle of the class block
        tick_pos = i * num_prototypes_per_class + num_prototypes_per_class / 2
        ticks.append(tick_pos)
        tick_labels.append(f'Class {i}\n({class_names[i]})')
        
        # Draw a grid line to separate the classes
        if i > 0:
            line_pos = i * num_prototypes_per_class
            ax.axhline(line_pos, color='white', linewidth=2.5)
            ax.axvline(line_pos, color='white', linewidth=2.5)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, rotation=90, va='center')

    # --- Save the Plot ---
    output_path = os.path.join(os.path.dirname(args.checkpoint), "prototype_similarity_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Done. Heatmap saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()