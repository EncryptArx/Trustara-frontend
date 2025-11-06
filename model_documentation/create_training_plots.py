#!/usr/bin/env python3
"""
Create Training Progress Visualizations
Generate comprehensive plots from training history
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_history():
    """Load training history from JSON file"""
    with open('training_history.json', 'r') as f:
        return json.load(f)

def create_training_plots():
    """Create comprehensive training visualizations"""
    
    # Load data
    history = load_training_history()
    epochs = [h['epoch'] for h in history]
    
    # Extract metrics
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    train_balanced_acc = [h['train_balanced_acc'] for h in history]
    val_balanced_acc = [h['val_balanced_acc'] for h in history]
    lr = [h['lr'] for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced DeepFake Model Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Loss Curves
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 2. Accuracy Curves
    axes[0, 1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy Curves', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0.8, 1.0)
    
    # 3. Balanced Accuracy
    axes[0, 2].plot(epochs, train_balanced_acc, 'b-', label='Training Balanced Acc', linewidth=2)
    axes[0, 2].plot(epochs, val_balanced_acc, 'r-', label='Validation Balanced Acc', linewidth=2)
    axes[0, 2].set_title('Balanced Accuracy', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Balanced Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0.8, 1.0)
    
    # 4. Learning Rate Schedule
    axes[1, 0].plot(epochs, lr, 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 5. Real vs Fake Detection
    train_real_recall = [h['train_real_recall'] for h in history]
    train_fake_recall = [h['train_fake_recall'] for h in history]
    val_real_recall = [h['val_real_recall'] for h in history]
    val_fake_recall = [h['val_fake_recall'] for h in history]
    
    axes[1, 1].plot(epochs, train_real_recall, 'b-', label='Train Real', linewidth=2)
    axes[1, 1].plot(epochs, train_fake_recall, 'g-', label='Train Fake', linewidth=2)
    axes[1, 1].plot(epochs, val_real_recall, 'r--', label='Val Real', linewidth=2)
    axes[1, 1].plot(epochs, val_fake_recall, 'm--', label='Val Fake', linewidth=2)
    axes[1, 1].set_title('Real vs Fake Detection', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0.8, 1.0)
    
    # 6. Training Time per Epoch
    epoch_times = [h['epoch_time'] for h in history]
    axes[1, 2].plot(epochs, epoch_times, 'purple', linewidth=2, marker='o')
    axes[1, 2].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Time (seconds)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add best epoch annotation
    best_epoch = max(history, key=lambda x: x['val_balanced_acc'])['epoch']
    best_acc = max(history, key=lambda x: x['val_balanced_acc'])['val_balanced_acc']
    
    for ax in axes.flat:
        ax.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        if ax == axes[0, 1]:  # Accuracy plot
            ax.annotate(f'Best: Epoch {best_epoch}\nAcc: {best_acc:.3f}', 
                       xy=(best_epoch, best_acc), xytext=(best_epoch+2, best_acc-0.02),
                       arrowprops=dict(arrowstyle='->', color='orange'),
                       fontsize=10, color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved as 'training_progress.png'")
    print(f"Best epoch: {best_epoch} with {best_acc:.3f} validation balanced accuracy")

def create_performance_summary():
    """Create performance summary visualization"""
    
    history = load_training_history()
    best_epoch_data = max(history, key=lambda x: x['val_balanced_acc'])
    
    # Create performance summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Summary', fontsize=16, fontweight='bold')
    
    # 1. Final Metrics Bar Chart
    metrics = ['Accuracy', 'Balanced Acc', 'Real Recall', 'Fake Recall']
    values = [
        best_epoch_data['val_acc'],
        best_epoch_data['val_balanced_acc'],
        best_epoch_data['val_real_recall'],
        best_epoch_data['val_fake_recall']
    ]
    
    bars = ax1.bar(metrics, values, color=['#2E8B57', '#FF6B6B', '#4CAF50', '#F44336'])
    ax1.set_title('Final Validation Metrics', fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0.9, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training vs Validation Comparison
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    
    ax2.plot(epochs, train_acc, 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
    ax2.set_title('Training vs Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 1.0)
    
    # 3. Loss Reduction
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    ax3.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax3.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax3.set_title('Loss Reduction Over Time', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Learning Rate Schedule
    lr = [h['lr'] for h in history]
    ax4.plot(epochs, lr, 'g-', linewidth=2)
    ax4.set_title('Learning Rate Schedule', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance summary saved as 'performance_summary.png'")

if __name__ == "__main__":
    print("Creating training visualizations...")
    create_training_plots()
    create_performance_summary()
    print("All visualizations created successfully!")
