# cloud_train_enhanced.py - Training script with overfitting prevention

import torch
import torch.nn as nn
import os
import json
import time
from datetime import datetime
import logging
from FaceDataset import main, FelixClassifier, InceptionResnetV1
import pytorch_trainer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import numpy as np

# Set up logging for remote monitoring
def setup_logging(log_dir='training_logs'):
    """Set up comprehensive logging for remote monitoring"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def save_metrics_json(history, filepath):
    """Save training metrics as JSON for easy parsing"""
    json_history = {}
    for key, value in history.items():
        if isinstance(value, list):
            json_history[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        else:
            json_history[key] = float(value) if hasattr(value, 'item') else value
    
    with open(filepath, 'w') as f:
        json.dump(json_history, f, indent=2)

def create_progress_callback(log_dir='training_logs'):
    """Create callback to save progress metrics"""
    def progress_callback(trainer, epoch):
        metrics_file = os.path.join(log_dir, 'current_metrics.json')
        save_metrics_json(trainer.history, metrics_file)
        
        if trainer.history['val_acc']:
            logging.info(f"Epoch {epoch+1}: Train Acc: {trainer.history['train_acc'][-1]:.4f}, "
                        f"Val Acc: {trainer.history['val_acc'][-1]:.4f}, "
                        f"Val Loss: {trainer.history['val_loss'][-1]:.4f}")
        
        progress_file = os.path.join(log_dir, 'progress.txt')
        with open(progress_file, 'w') as f:
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"Train Accuracy: {trainer.history['train_acc'][-1]:.4f}\n")
            if trainer.history['val_acc']:
                f.write(f"Validation Accuracy: {trainer.history['val_acc'][-1]:.4f}\n")
                f.write(f"Validation Loss: {trainer.history['val_loss'][-1]:.4f}\n")
            f.write(f"Best Epoch: {trainer.history['best_epoch']+1}\n")
            f.write(f"Best Val Acc: {trainer.history['best_val_acc']:.4f}\n")
            f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return progress_callback

class EnhancedFelixClassifier(nn.Module):
    """Enhanced classifier with dropout and regularization"""
    def __init__(self, base_model, dropout_rate=0.5, hidden_size=256):
        super(EnhancedFelixClassifier, self).__init__()
        self.base_model = base_model
        
        # Get the output size of the base model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 160, 160)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.base_model = self.base_model.cuda()
            features = self.base_model(dummy_input)
            feature_size = features.shape[1]
        
        # Enhanced classifier with more dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # Slightly less dropout in second layer
            nn.Linear(128, 2)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.base_model(x)
        x = self.classifier(features)
        return x

class AugmentedFaceDataset(Dataset):
    """Custom dataset with strong augmentation for face images"""
    def __init__(self, root_dir, transform=None, is_training=True):
        self.root_dir = root_dir
        self.is_training = is_training
        self.classes = ['notfelix', 'felix']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        # Strong augmentation for training
        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(160, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))  # Cutout augmentation
            ])
        else:
            # Minimal augmentation for validation
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loaders_with_augmentation(data_path, batch_size=32, val_split=0.2):
    """Create data loaders with strong augmentation"""
    # Get all samples
    all_samples = []
    for class_name in ['felix', 'notfelix']:
        class_dir = os.path.join(data_path, class_name)
        if os.path.exists(class_dir):
            class_idx = 1 if class_name == 'felix' else 0
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    all_samples.append((img_path, class_idx))
    
    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_split))
    
    # Create temporary directories for train/val split
    temp_train_dir = os.path.join(data_path, 'temp_train')
    temp_val_dir = os.path.join(data_path, 'temp_val')
    
    # Use the original data loader with augmentation
    train_dataset = AugmentedFaceDataset(data_path, is_training=True)
    val_dataset = AugmentedFaceDataset(data_path, is_training=False)
    
    # Override samples for proper split
    train_dataset.samples = all_samples[:split_idx]
    val_dataset.samples = all_samples[split_idx:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_felix_classifier_cloud(data_path, epochs=70, log_dir='training_logs'):
    """Main training function with overfitting prevention"""
    
    # Setup logging
    log_file = setup_logging(log_dir)
    logging.info("Starting Felix classifier training with overfitting prevention")
    logging.info(f"Data path: {data_path}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # Load data with augmentation
        logging.info("Loading data with augmentation...")
        train_loader, val_loader = create_data_loaders_with_augmentation(data_path, batch_size=16)
        logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create model
        logging.info("Creating enhanced model with dropout...")
        base_model = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            base_model = base_model.to('cuda')
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Create enhanced classifier with dropout
        felix_classifier = EnhancedFelixClassifier(base_model, dropout_rate=0.5, hidden_size=256)
        if torch.cuda.is_available():
            felix_classifier = felix_classifier.to('cuda')
        
        # Setup training components with L2 regularization
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            felix_classifier.classifier.parameters(), 
            lr=0.0005,  # Lower learning rate
            weight_decay=0.001  # L2 regularization
        )
        
        # More aggressive scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Create trainer
        trainer = pytorch_trainer.PyTorchTrainer(
            model=felix_classifier, 
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler=scheduler
        )
        
        # Create progress callback
        progress_cb = create_progress_callback(log_dir)
        
        # Add early stopping callback
        def early_stop_callback(trainer, epoch):
            if len(trainer.history['val_acc']) > 10:
                recent_val = trainer.history['val_acc'][-10:]
                recent_train = trainer.history['train_acc'][-10:]
                
                # Calculate average gap
                avg_gap = np.mean([t - v for t, v in zip(recent_train, recent_val)])
                
                if avg_gap > 0.3:  # If gap > 30%
                    logging.warning(f"Large train-val gap detected: {avg_gap:.2f}")
                    
                    # Reduce learning rate more aggressively
                    for param_group in trainer.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    logging.info(f"Reduced learning rate to {trainer.optimizer.param_groups[0]['lr']}")
        
        # Start training
        logging.info("Starting training...")
        start_time = time.time()
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            early_stopping=True,
            patience=15,  # Reduced patience
            save_path=os.path.join(log_dir, 'felix_classifier_enhanced.pth'),
            save_best_only=True,
            verbose=1,
            callbacks=[progress_cb, early_stop_callback]
        )
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time/60:.2f} minutes")
        
        # Save final metrics
        final_metrics_file = os.path.join(log_dir, 'final_metrics.json')
        save_metrics_json(history, final_metrics_file)
        
        # Evaluate on validation set
        logging.info("Evaluating model...")
        val_loss, val_acc = trainer.evaluate(val_loader)
        logging.info(f"Final validation accuracy: {val_acc:.4f}")
        
        # Get detailed classification metrics
        try:
            metrics = trainer.get_classification_metrics(
                data_loader=val_loader,
                class_names=["Not Felix", "Felix"]
            )
            
            if metrics:
                report_file = os.path.join(log_dir, 'classification_report.json')
                with open(report_file, 'w') as f:
                    json.dump(metrics['classification_report'], f, indent=2)
                
                logging.info("Classification Report:")
                logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            logging.warning(f"Could not generate classification metrics: {e}")
        
        # Create summary
        summary = {
            'training_completed': True,
            'final_val_accuracy': val_acc,
            'final_val_loss': val_loss,
            'best_validation_accuracy': history['best_val_acc'],
            'best_epoch': history['best_epoch'] + 1,
            'total_epochs_trained': len(history['train_loss']),
            'training_time_minutes': training_time / 60,
            'model_path': os.path.join(log_dir, 'felix_classifier_enhanced.pth'),
            'timestamp': datetime.now().isoformat(),
            'overfitting_prevention': {
                'dropout_rate': 0.5,
                'weight_decay': 0.001,
                'augmentation': 'strong',
                'early_stopping_patience': 15
            }
        }
        
        summary_file = os.path.join(log_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("Training completed successfully!")
        logging.info(f"Model saved to: {os.path.join(log_dir, 'felix_classifier_enhanced.pth')}")
        
        return history, trainer
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        error_file = os.path.join(log_dir, 'error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Training failed at: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n")
        raise

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/root/gpu_train/data/processed" # Update this path
    EPOCHS = 70
    LOG_DIR = "training_logs_enhanced"
    
    # Print configuration
    print("=== Enhanced Training Configuration ===")
    print(f"Data Path: {DATA_PATH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
    print("=== Overfitting Prevention Measures ===")
    print("- Strong data augmentation")
    print("- Dropout layers (0.5)")
    print("- L2 regularization (0.001)")
    print("- Reduced learning rate (0.0005)")
    print("- Early stopping (patience=15)")
    print("- Batch normalization")
    print("=" * 40)
    
    # Verify data path exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path does not exist: {DATA_PATH}")
        print("Please update DATA_PATH in this script to point to your processed data directory")
        exit(1)
    
    felix_dir = os.path.join(DATA_PATH, 'felix')
    notfelix_dir = os.path.join(DATA_PATH, 'notfelix')
    
    if not os.path.exists(felix_dir) or not os.path.exists(notfelix_dir):
        print("ERROR: Data directories not found")
        exit(1)
    
    felix_count = len([f for f in os.listdir(felix_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    notfelix_count = len([f for f in os.listdir(notfelix_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {felix_count} Felix images")
    print(f"Found {notfelix_count} Not-Felix images")
    print(f"Total dataset size: {felix_count + notfelix_count} images")
    
    if felix_count < 50 or notfelix_count < 50:
        print("WARNING: Very small dataset detected. Consider collecting more data.")
    
    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Start training
    history, trainer = train_felix_classifier_cloud(
        data_path=DATA_PATH,
        epochs=EPOCHS,
        log_dir=LOG_DIR
    )