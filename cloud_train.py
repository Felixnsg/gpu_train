# cloud_train.py - Modified training script for cloud GPU

import torch
import os
import json
import time
from datetime import datetime
import logging
from FaceDataset import main, FelixClassifier, InceptionResnetV1
import pytorch_trainer

# Set up logging for remote monitoring
def setup_logging(log_dir='training_logs'):
    """Set up comprehensive logging for remote monitoring"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_file

def save_metrics_json(history, filepath):
    """Save training metrics as JSON for easy parsing"""
    # Convert numpy arrays to lists for JSON serialization
    json_history = {}
    for key, value in history.items():
        if isinstance(value, list):
            # Convert any numpy values to native Python types
            json_history[key] = [float(v) if hasattr(v, 'item') else v for v in value]
        else:
            json_history[key] = float(value) if hasattr(value, 'item') else value
    
    with open(filepath, 'w') as f:
        json.dump(json_history, f, indent=2)

def create_progress_callback(log_dir='training_logs'):
    """Create callback to save progress metrics"""
    def progress_callback(trainer, epoch):
        # Save current metrics to JSON
        metrics_file = os.path.join(log_dir, 'current_metrics.json')
        save_metrics_json(trainer.history, metrics_file)
        
        # Log current progress
        if trainer.history['val_acc']:
            logging.info(f"Epoch {epoch+1}: Train Acc: {trainer.history['train_acc'][-1]:.4f}, "
                        f"Val Acc: {trainer.history['val_acc'][-1]:.4f}, "
                        f"Val Loss: {trainer.history['val_loss'][-1]:.4f}")
        
        # Save a simple progress file for easy monitoring
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

def train_felix_classifier_cloud(data_path, epochs=70, log_dir='training_logs'):
    """Main training function optimized for cloud GPU"""
    
    # Setup logging
    log_file = setup_logging(log_dir)
    logging.info("Starting Felix classifier training on cloud GPU")
    logging.info(f"Data path: {data_path}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # Load data
        logging.info("Loading data...")
        train_loader, test_loader = main(data_path)
        logging.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
        
        # Create model
        logging.info("Creating model...")
        base_model = InceptionResnetV1(pretrained='vggface2').eval()
        if torch.cuda.is_available():
            base_model = base_model.to('cuda')
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Create classifier
        felix_classifier = FelixClassifier(base_model)
        if torch.cuda.is_available():
            felix_classifier = felix_classifier.to('cuda')
        
        # Setup training components
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(felix_classifier.classifier.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
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
        
        # Start training
        logging.info("Starting training...")
        start_time = time.time()
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=epochs,
            early_stopping=True,
            patience=35,
            save_path=os.path.join(log_dir, 'felix_classifier.pth'),
            save_best_only=True,
            verbose=1,
            callbacks=[progress_cb]
        )
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time/60:.2f} minutes")
        
        # Save final metrics
        final_metrics_file = os.path.join(log_dir, 'final_metrics.json')
        save_metrics_json(history, final_metrics_file)
        
        # Evaluate on test set
        logging.info("Evaluating model...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        logging.info(f"Final test accuracy: {test_acc:.4f}")
        
        # Get detailed classification metrics
        try:
            metrics = trainer.get_classification_metrics(
                data_loader=test_loader,
                class_names=["Not Felix", "Felix"]
            )
            
            if metrics:
                # Save classification report
                report_file = os.path.join(log_dir, 'classification_report.json')
                with open(report_file, 'w') as f:
                    json.dump(metrics['classification_report'], f, indent=2)
                
                logging.info("Classification Report:")
                logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            logging.warning(f"Could not generate classification metrics: {e}")
        
        # Create summary file
        summary = {
            'training_completed': True,
            'final_test_accuracy': test_acc,
            'final_test_loss': test_loss,
            'best_validation_accuracy': history['best_val_acc'],
            'best_epoch': history['best_epoch'] + 1,
            'total_epochs_trained': len(history['train_loss']),
            'training_time_minutes': training_time / 60,
            'model_path': os.path.join(log_dir, 'felix_classifier.pth'),
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(log_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info("Training completed successfully!")
        logging.info(f"Model saved to: {os.path.join(log_dir, 'felix_classifier.pth')}")
        logging.info(f"Logs and metrics saved to: {log_dir}")
        
        return history, trainer
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        # Save error info
        error_file = os.path.join(log_dir, 'error.txt')
        with open(error_file, 'w') as f:
            f.write(f"Training failed at: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n")
        raise

if __name__ == "__main__":
    # Configuration - UPDATE THIS PATH TO YOUR DATA LOCATION
    DATA_PATH = "/root/gpu_train/data/processed"

  # Update this path
    EPOCHS = 70
    LOG_DIR = "training_logs"
    
    # Print configuration
    print("=== Training Configuration ===")
    print(f"Data Path: {DATA_PATH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
    print("=" * 35)
    
    # Verify data path exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data path does not exist: {DATA_PATH}")
        print("Please update DATA_PATH in this script to point to your processed data directory")
        exit(1)
    
    # Check for felix and notfelix directories
    felix_dir = os.path.join(DATA_PATH, 'felix')
    notfelix_dir = os.path.join(DATA_PATH, 'notfelix')
    
    if not os.path.exists(felix_dir):
        print(f"ERROR: Felix data directory not found: {felix_dir}")
        exit(1)
    
    if not os.path.exists(notfelix_dir):
        print(f"ERROR: NotFelix data directory not found: {notfelix_dir}")
        exit(1)
    
    felix_count = len([f for f in os.listdir(felix_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    notfelix_count = len([f for f in os.listdir(notfelix_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Found {felix_count} Felix images")
    print(f"Found {notfelix_count} Not-Felix images")
    
    if felix_count == 0 or notfelix_count == 0:
        print("ERROR: No images found in one or both directories")
        exit(1)
    
    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Start training
    history, trainer = train_felix_classifier_cloud(
        data_path=DATA_PATH,
        epochs=EPOCHS,
        log_dir=LOG_DIR
    )