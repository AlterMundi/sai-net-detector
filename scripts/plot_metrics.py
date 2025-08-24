#!/usr/bin/env python3
"""
Real-time Training Metrics Plotter
Generates metric evolution plots every 5 epochs during Stage 2 training
Focuses on the 6 most important metrics for performance comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import os

def plot_training_metrics(csv_path, save_dir, current_epoch=None):
    """
    Plot the 6 most important training metrics
    """
    # Read CSV data
    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if len(df) == 0:
        print("No data in CSV file")
        return
    
    # Define the 4 most important metrics for performance comparison (no losses for scale)
    metrics = {
        'metrics/mAP50(B)': {'color': '#FF6B35', 'label': 'mAP@0.5', 'style': '-', 'linewidth': 3},
        'metrics/mAP50-95(B)': {'color': '#F7931E', 'label': 'mAP@0.5:0.95', 'style': '-', 'linewidth': 2.5},
        'metrics/precision(B)': {'color': '#4CAF50', 'label': 'Precision', 'style': '--', 'linewidth': 2},
        'metrics/recall(B)': {'color': '#2196F3', 'label': 'Recall', 'style': '--', 'linewidth': 2}
    }
    
    # Create figure with high DPI for clarity
    plt.figure(figsize=(14, 10), dpi=150)
    plt.style.use('dark_background')
    
    # Plot each metric
    for metric, config in metrics.items():
        if metric in df.columns:
            epochs = df['epoch'].values
            values = df[metric].values
            
            plt.plot(epochs, values, 
                    color=config['color'], 
                    label=config['label'],
                    linestyle=config['style'],
                    linewidth=config['linewidth'],
                    marker='o' if len(epochs) < 20 else None,
                    markersize=4,
                    alpha=0.9)
    
    # Customize plot
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=14, fontweight='bold')
    plt.title(f'SAI-Net Stage 2 Key Performance Metrics\nEpoch {len(df)} - H200 SAGRADO Configuration', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend with better positioning
    plt.legend(loc='center right', bbox_to_anchor=(1.15, 0.5), 
              fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add current epoch indicator if provided
    if current_epoch is not None and current_epoch <= len(df):
        plt.axvline(x=current_epoch, color='yellow', linestyle='--', 
                   alpha=0.7, linewidth=2, label=f'Current: Epoch {current_epoch}')
    
    # Add performance annotations for latest epoch
    if len(df) > 0:
        latest = df.iloc[-1]
        if 'metrics/mAP50(B)' in latest:
            map50 = latest['metrics/mAP50(B)'] * 100 if latest['metrics/mAP50(B)'] < 1 else latest['metrics/mAP50(B)']
            plt.text(0.02, 0.98, f'Latest mAP@0.5: {map50:.1f}%', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
                    verticalalignment='top')
    
    # Tight layout and save
    plt.tight_layout()
    
    # Generate filename
    epoch_num = len(df)
    filename = f'metrics_evolution_epoch_{epoch_num:03d}.png'
    save_path = Path(save_dir) / filename
    
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor='#1a1a1a', edgecolor='none')
        print(f"✅ Saved metrics plot: {save_path}")
        
        # Also save as latest for easy access
        latest_path = Path(save_dir) / 'metrics_latest.png'
        plt.savefig(latest_path, dpi=150, bbox_inches='tight',
                   facecolor='#1a1a1a', edgecolor='none')
        
    except Exception as e:
        print(f"❌ Error saving plot: {e}")
    
    plt.close()

def monitor_training(csv_path, interval=30):
    """
    Monitor training and generate plots every 5 epochs
    """
    save_dir = Path(csv_path).parent
    last_epoch_plotted = 0
    
    print(f"🔍 Monitoring training metrics: {csv_path}")
    print(f"📊 Saving plots to: {save_dir}")
    print(f"⏱️  Plot generation: Every 5 epochs")
    
    while True:
        try:
            if Path(csv_path).exists():
                df = pd.read_csv(csv_path)
                current_epoch = len(df)
                
                # Generate plot every 5 epochs or if it's the first epoch
                if (current_epoch % 5 == 0 and current_epoch > last_epoch_plotted) or \
                   (current_epoch == 1 and last_epoch_plotted == 0):
                    
                    print(f"📈 Generating plot for epoch {current_epoch}")
                    plot_training_metrics(csv_path, save_dir, current_epoch)
                    last_epoch_plotted = current_epoch
                
                # Also generate final plot if training seems complete (no updates for a while)
                elif current_epoch > last_epoch_plotted:
                    # Wait a bit more to see if training continues
                    time.sleep(60)
                    df_check = pd.read_csv(csv_path)
                    if len(df_check) == current_epoch:  # No new epochs added
                        print(f"📈 Generating final plot for epoch {current_epoch}")
                        plot_training_metrics(csv_path, save_dir, current_epoch)
                        last_epoch_plotted = current_epoch
                        
        except Exception as e:
            print(f"⚠️  Error monitoring: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAI-Net Training Metrics Plotter")
    parser.add_argument("--csv", type=str, 
                       default="/workspace/sai-net-detector/runs/h200_stage2_pyrosdis3/results.csv",
                       help="Path to results.csv file")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor training and auto-generate plots")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_training(args.csv, args.interval)
    else:
        # Generate single plot
        save_dir = Path(args.csv).parent
        plot_training_metrics(args.csv, save_dir)