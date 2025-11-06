#!/usr/bin/env python3
"""
Advanced Model Testing Script
Test the new advanced deepfake detection model and compare with previous models
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
# import seaborn as sns  # Optional dependency

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Import our models
from advanced_anti_overfitting_training import AdvancedDeepFakeModel
from detectors.cnn_lstm_detector import SimpleDeepFakeModel, DeepFakeModel

class ModelTester:
    """Comprehensive model testing framework"""
    
    def __init__(self, project_root=PROJECT_ROOT):
        self.project_root = project_root
        self.models_dir = project_root / "models"
        self.data_dir = project_root / "data" / "images"
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'batch_size': 32,
            'num_test_samples': 1000,  # Test on 1000 samples
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }
        
        print(f"Using device: {self.test_config['device']}")
        print(f"Models directory: {self.models_dir}")
        print(f"Data directory: {self.data_dir}")
    
    def load_models(self):
        """Load all available models for comparison"""
        models = {}
        
        # Load Advanced Model (new)
        try:
            advanced_model = AdvancedDeepFakeModel(dropout_rate=0.5, attention_dim=256)
            advanced_path = self.models_dir / "advanced_deepfake_detector_best.pt"
            if advanced_path.exists():
                advanced_model.load_state_dict(torch.load(advanced_path, map_location=self.test_config['device']))
                advanced_model.to(self.test_config['device'])
                advanced_model.eval()
                models['Advanced Model (New)'] = advanced_model
                print("✅ Advanced Model loaded successfully")
            else:
                print("❌ Advanced model not found")
        except Exception as e:
            print(f" Failed to load Advanced Model: {e}")
        
        # Load Simple Model (old)
        try:
            simple_model = SimpleDeepFakeModel(dropout_rate=0.3)
            simple_path = self.models_dir / "simple_deepfake_detector.pt.best"
            if simple_path.exists():
                simple_model.load_state_dict(torch.load(simple_path, map_location=self.test_config['device']))
                simple_model.to(self.test_config['device'])
                simple_model.eval()
                models['Simple Model (Old)'] = simple_model
                print(" Simple Model loaded successfully")
            else:
                print(" Simple model not found")
        except Exception as e:
            print(f" Failed to load Simple Model: {e}")
        
        return models
    
    def prepare_test_data(self):
        """Prepare test dataset"""
        print("Preparing test data...")
        
        # Get sample images
        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "deepfake"
        
        real_images = list(real_dir.glob("*.jpg"))[:self.test_config['num_test_samples']//2]
        fake_images = list(fake_dir.glob("*.jpg"))[:self.test_config['num_test_samples']//2]
        
        test_samples = []
        for img_path in real_images:
            test_samples.append((str(img_path), 0))  # 0 for real
        for img_path in fake_images:
            test_samples.append((str(img_path), 1))  # 1 for fake
        
        print(f"Test samples: {len(test_samples)}")
        print(f"  Real: {len(real_images)}")
        print(f"  Fake: {len(fake_images)}")
        
        return test_samples
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            return transform(image).unsqueeze(0)
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def test_model(self, model, model_name, test_samples):
        """Test a single model"""
        print(f"\n Testing {model_name}...")
        
        predictions = []
        targets = []
        confidences = []
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for i, (image_path, true_label) in enumerate(test_samples):
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(test_samples)}")
                
                # Preprocess image
                input_tensor = self.preprocess_image(image_path)
                if input_tensor is None:
                    continue
                
                input_tensor = input_tensor.to(self.test_config['device'])
                
                # Measure inference time
                start_time = time.time()
                output = model(input_tensor)
                inference_time = time.time() - start_time
                
                # Get prediction
                if hasattr(model, 'classifier') and len(output.shape) > 1:
                    # Handle different output shapes
                    if output.shape[1] == 1:
                        prob = torch.sigmoid(output.squeeze()).item()
                    else:
                        prob = torch.softmax(output, dim=1)[:, 1].item()
                else:
                    prob = torch.sigmoid(output.squeeze()).item()
                
                pred = 1 if prob > 0.5 else 0
                
                predictions.append(pred)
                targets.append(true_label)
                confidences.append(prob)
                inference_times.append(inference_time)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        balanced_acc = balanced_accuracy_score(targets, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Real and fake detection rates
        real_samples = [i for i, label in enumerate(targets) if label == 0]
        fake_samples = [i for i, label in enumerate(targets) if label == 1]
        
        real_accuracy = sum(1 for i in real_samples if predictions[i] == 0) / len(real_samples) if real_samples else 0
        fake_accuracy = sum(1 for i in fake_samples if predictions[i] == 1) / len(fake_samples) if fake_samples else 0
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'confusion_matrix': cm.tolist(),
            'avg_inference_time': np.mean(inference_times),
            'total_samples': len(predictions),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Balanced Accuracy: {balanced_acc:.3f}")
        print(f"   Real Detection: {real_accuracy:.3f}")
        print(f"   Fake Detection: {fake_accuracy:.3f}")
        print(f"   Avg Inference Time: {np.mean(inference_times):.4f}s")
        
        return results
    
    def create_comparison_plots(self, results_list):
        """Create comparison visualizations"""
        print("\n Creating comparison plots...")
        
        # Extract metrics for plotting
        model_names = [r['model_name'] for r in results_list]
        accuracies = [r['accuracy'] for r in results_list]
        balanced_accs = [r['balanced_accuracy'] for r in results_list]
        real_accs = [r['real_accuracy'] for r in results_list]
        fake_accs = [r['fake_accuracy'] for r in results_list]
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Accuracy comparison
        axes[0, 0].bar(model_names, accuracies, color=['#2E8B57', '#FF6B6B'])
        axes[0, 0].set_title('Overall Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0.8, 1.0)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Balanced accuracy comparison
        axes[0, 1].bar(model_names, balanced_accs, color=['#2E8B57', '#FF6B6B'])
        axes[0, 1].set_title('Balanced Accuracy')
        axes[0, 1].set_ylabel('Balanced Accuracy')
        axes[0, 1].set_ylim(0.8, 1.0)
        for i, v in enumerate(balanced_accs):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Real vs Fake detection
        x = np.arange(len(model_names))
        width = 0.35
        axes[1, 0].bar(x - width/2, real_accs, width, label='Real Detection', color='#4CAF50')
        axes[1, 0].bar(x + width/2, fake_accs, width, label='Fake Detection', color='#F44336')
        axes[1, 0].set_title('Real vs Fake Detection')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0.8, 1.0)
        
        # Confusion matrices
        for i, result in enumerate(results_list):
            cm = np.array(result['confusion_matrix'])
            im = axes[1, 1].imshow(cm, interpolation='nearest', cmap='Blues')
            axes[1, 1].set_title(f'{result["model_name"]} Confusion Matrix')
            axes[1, 1].set_xlabel('Predicted')
            axes[1, 1].set_ylabel('Actual')
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center')
            break  # Only show first model's confusion matrix
        
        plt.tight_layout()
        plot_path = self.results_dir / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f" Comparison plot saved to: {plot_path}")
        return plot_path
    
    def run_comprehensive_test(self):
        """Run comprehensive testing of all models"""
        print("🚀 Starting Comprehensive Model Testing")
        print("=" * 60)
        
        # Load models
        models = self.load_models()
        if not models:
            print(" No models loaded. Exiting.")
            return
        
        # Prepare test data
        test_samples = self.prepare_test_data()
        if not test_samples:
            print(" No test data available. Exiting.")
            return
        
        # Test each model
        results_list = []
        for model_name, model in models.items():
            results = self.test_model(model, model_name, test_samples)
            results_list.append(results)
        
        # Create comparison plots
        self.create_comparison_plots(results_list)
        
        # Save results
        results_file = self.results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"\n Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 TESTING SUMMARY")
        print("=" * 60)
        
        for result in results_list:
            print(f"\n {result['model_name']}:")
            print(f"  Overall Accuracy: {result['accuracy']:.3f}")
            print(f"  Balanced Accuracy: {result['balanced_accuracy']:.3f}")
            print(f"  Real Detection: {result['real_accuracy']:.3f}")
            print(f"  Fake Detection: {result['fake_accuracy']:.3f}")
            print(f"  Avg Inference Time: {result['avg_inference_time']:.4f}s")
            print(f"  Total Samples: {result['total_samples']}")
        
        # Find best model
        best_model = max(results_list, key=lambda x: x['balanced_accuracy'])
        print(f"\n BEST MODEL: {best_model['model_name']}")
        print(f"   Balanced Accuracy: {best_model['balanced_accuracy']:.3f}")
        
        return results_list

def main():
    """Main testing function"""
    print(" DeepSecure Model Testing Suite")
    print("=" * 50)
    
    tester = ModelTester()
    results = tester.run_comprehensive_test()

    print("\n Testing completed successfully!")
    return results

if __name__ == "__main__":
    main()
