"""
PyTorch Training Utilities

A modular toolkit for training, evaluating, and deploying PyTorch models.
"""

import torch
import time
import copy
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
try:
    from tqdm import tqdm
except ImportError:
    # Create a simple tqdm fallback if not installed
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    plt_available = True
except ImportError:
    plt_available = False

try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    sklearn_available = True
except ImportError:
    sklearn_available = False


class PyTorchTrainer:
    """
    A reusable trainer for PyTorch models that handles training loops, evaluation,
    and various utilities to make training and evaluation more convenient.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer with a model, loss function, optimizer, and optional scheduler.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to run on ('cuda' or 'cpu'), if None, will be automatically determined
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'best_epoch': 0,
            'best_val_acc': 0.0,
            'best_val_loss': float('inf')
        }
        
        # Best model weights
        self.best_model_weights = copy.deepcopy(self.model.state_dict())
        
        # Optional callback functions
        self.callbacks = []
    
    def train(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: Optional[torch.utils.data.DataLoader] = None, 
        epochs: int = 10, 
        early_stopping: bool = False, 
        patience: int = 5, 
        save_path: Optional[str] = None,
        save_best_only: bool = True,
        monitor: str = 'val_acc',  # 'val_acc' or 'val_loss'
        mode: str = 'max',  # 'max' for accuracy, 'min' for loss
        verbose: int = 1,
        callbacks: List[Callable] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set (optional)
            epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            save_path: Path to save the best model
            save_best_only: Whether to save only the best model
            monitor: Metric to monitor for early stopping ('val_acc' or 'val_loss')
            mode: Mode for monitoring ('max' for accuracy, 'min' for loss)
            verbose: Verbosity level (0: silent, 1: progress bar per epoch, 2: progress bar per batch)
            callbacks: List of callback functions to call at the end of each epoch
            
        Returns:
            history: Dictionary containing training history
        """
        since = time.time()
        
        # Initialize variables for early stopping
        no_improve_epochs = 0
        best_monitored_metric = -float('inf') if mode == 'max' else float('inf')
        
        # Set up callbacks
        if callbacks:
            self.callbacks = callbacks
        
        # Create directory for save_path if it doesn't exist
        if save_path and not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        for epoch in range(epochs):
            if verbose > 0:
                print(f'Epoch {epoch+1}/{epochs}')
                print('-' * 10)
            
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, verbose)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Store learning rate
            current_lr = self._get_current_lr()
            self.history['learning_rates'].append(current_lr)
            
            if verbose > 0:
                print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} LR: {current_lr:.6f}')
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, verbose=(verbose > 1))
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                if verbose > 0:
                    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
                
                # Determine if this is the best model
                current_metric = val_acc if monitor == 'val_acc' else -val_loss
                is_best = (mode == 'max' and current_metric > best_monitored_metric) or \
                          (mode == 'min' and current_metric < best_monitored_metric)
                
                if is_best:
                    best_monitored_metric = current_metric
                    self.history['best_epoch'] = epoch
                    
                    if monitor == 'val_acc':
                        self.history['best_val_acc'] = val_acc
                    else:
                        self.history['best_val_loss'] = val_loss
                    
                    # Save best model weights
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                    
                    # Save model to disk if requested
                    if save_path is not None:
                        if save_best_only:
                            self._save_model(save_path, {
                                'epoch': epoch,
                                'acc': val_acc,
                                'loss': val_loss,
                                'lr': current_lr
                            })
                        else:
                            # Save with epoch number in filename
                            filename, ext = os.path.splitext(save_path)
                            epoch_save_path = f"{filename}_epoch{epoch+1}{ext}"
                            self._save_model(epoch_save_path, {
                                'epoch': epoch,
                                'acc': val_acc,
                                'loss': val_loss,
                                'lr': current_lr
                            })
                    
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                
                # Update ReduceLROnPlateau scheduler here with the validation metric
                if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Pass appropriate metric based on monitor setting
                    metric_value = val_acc if self.scheduler.mode == 'max' else val_loss
                    self.scheduler.step(metric_value)
                
                # Early stopping
                if early_stopping and no_improve_epochs >= patience:
                    if verbose > 0:
                        print(f'Early stopping triggered after epoch {epoch+1}')
                    break
            
            # Call callbacks
            for callback in self.callbacks:
                callback(self, epoch)
            
            if verbose > 0:
                print()
        
        # Calculate total time
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        
        # Print best results
        if 'best_val_acc' in self.history and self.history['best_val_acc'] > 0:
            print(f'Best val Acc: {self.history["best_val_acc"]:.4f} at epoch {self.history["best_epoch"]+1}')
        
        # Load best model weights
        self.model.load_state_dict(self.best_model_weights)
        
        return self.history
    
    def _train_epoch(self, train_loader, verbose=1):
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training set
            verbose: Verbosity level
            
        Returns:
            epoch_loss: Average loss for the epoch
            epoch_acc: Accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Set up progress bar if requested
        if verbose > 1:
            train_loader = tqdm(train_loader, desc="Training")
        
        # Iterate over data
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                
                # Handle different output types (logits, tuple, etc.)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take the first element if tuple
                
                # Handle binary classification vs multi-class
                if outputs.shape[1] == 1:  # Binary
                    preds = (outputs > 0).float()
                else:  # Multi-class
                    _, preds = torch.max(outputs, 1)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            if preds.dtype == torch.float32 and labels.dtype == torch.long:
                # Handle case where predictions are float but labels are long
                running_corrects += torch.sum(preds.long() == labels.data)
            else:
                running_corrects += torch.sum(preds == labels.data)
        
        # We don't update ReduceLROnPlateau here - we'll do it after validation
        # Only update epoch-based schedulers here
        if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def evaluate(self, data_loader, verbose=False):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            verbose: Whether to show progress bar
            
        Returns:
            loss: Average loss
            accuracy: Accuracy score
        """
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        
        if verbose:
            data_loader = tqdm(data_loader, desc="Evaluating")
        
        # Disable gradients
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Handle binary classification vs multi-class
                if outputs.shape[1] == 1:  # Binary
                    preds = (outputs > 0).float()
                else:  # Multi-class
                    _, preds = torch.max(outputs, 1)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                if preds.dtype == torch.float32 and labels.dtype == torch.long:
                    running_corrects += torch.sum(preds.long() == labels.data)
                else:
                    running_corrects += torch.sum(preds == labels.data)
        
        # Calculate metrics
        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)
        
        return epoch_loss, epoch_acc.item()
    
    def predict(self, data_loader, return_probs=False):
        """
        Make predictions on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            return_probs: Whether to return probabilities instead of class predictions
            
        Returns:
            predictions: Numpy array of predictions or probabilities
            true_labels: Numpy array of true labels if available
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        has_labels = True
        
        # Use tqdm if available
        loader = tqdm(data_loader, desc="Predicting")
        
        # Disable gradients
        with torch.no_grad():
            for batch in loader:
                # Check if we have labels (support both tuple and list)
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    inputs, labels = batch
                    
                    # Convert labels to tensor if needed
                    if not isinstance(labels, torch.Tensor):
                        if isinstance(labels, list):
                            labels = torch.tensor(labels)
                        else:
                            labels = torch.tensor([labels])
                    
                    all_labels.append(labels.cpu().numpy())
                else:
                    inputs = batch
                    has_labels = False
                
                # Handle the case where inputs is a list
                if isinstance(inputs, list):
                    try:
                        # Try to stack if it's a list of tensors
                        inputs = torch.stack(inputs)
                    except:
                        # If that fails, try the first element (which might be a tensor)
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            inputs = inputs[0]
                        else:
                            raise TypeError(f"Cannot convert inputs to tensor. Got: {type(inputs)}")
                
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if return_probs:
                    # Get probabilities
                    if outputs.shape[1] == 1:  # Binary
                        probs = torch.sigmoid(outputs)
                        # Convert to two-column format [1-p, p]
                        probs = torch.cat([1 - probs, probs], dim=1)
                    else:  # Multi-class
                        probs = torch.softmax(outputs, dim=1)
                    
                    all_probs.append(probs.cpu().numpy())
                else:
                    # Get class predictions
                    if outputs.shape[1] == 1:  # Binary
                        preds = (outputs > 0).float()
                    else:  # Multi-class
                        _, preds = torch.max(outputs, 1)
                    
                    all_preds.append(preds.cpu().numpy())
        
        # Concatenate results
        if return_probs:
            predictions = np.vstack(all_probs)
        else:
            if all_preds and all_preds[0].ndim == 1:
                predictions = np.concatenate(all_preds)
            else:
                predictions = np.vstack(all_preds)
        
        if has_labels:
            true_labels = np.concatenate(all_labels)
            return predictions, true_labels
        else:
            return predictions
                
    def visualize_predictions(self, data_loader, class_names=None, num_images=16, figsize=(20, 10)):
        """
        Visualize model predictions on a batch of images.
        
        Args:
            data_loader: DataLoader containing images to visualize
            class_names: List of class names for labels
            num_images: Number of images to visualize
            figsize: Figure size (width, height) in inches
            
        Returns:
            None: Displays the images with predictions
        """
        if not plt_available:
            print("Warning: matplotlib is not available. Cannot visualize predictions.")
            return
            
        if class_names is None:
            # Default to generic class names if not provided
            num_classes = 2  # Assuming binary classification by default
            try:
                # Try to infer number of classes from model's last layer
                if hasattr(self.model, 'classifier'):
                    if hasattr(self.model.classifier, 'out_features'):
                        num_classes = self.model.classifier.out_features
                
                # Also try checking if the model has an fc layer (common in some architectures)
                if hasattr(self.model, 'fc'):
                    if hasattr(self.model.fc, 'out_features'):
                        num_classes = self.model.fc.out_features
            except:
                pass
                
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        self.model.eval()
        
        # Get a batch of images
        images_so_far = 0
        fig = plt.figure(figsize=figsize)
        
        denorm = None
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions
                outputs = self.model(inputs)
                
                # Handle different output types
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get probabilities and predicted classes
                if outputs.shape[1] == 1:  # Binary
                    probs = torch.sigmoid(outputs)
                    # Convert to two-column format for binary case
                    probs_display = torch.cat([1 - probs, probs], dim=1)
                    preds = (probs > 0.5).long().squeeze()
                else:  # Multi-class
                    probs_display = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                
                # Plot images with predictions
                for i in range(inputs.size(0)):
                    images_so_far += 1
                    
                    # Create subplot
                    ax = plt.subplot(num_images // 4 + 1, 4, images_so_far)
                    ax.axis('off')
                    
                    # Get the image
                    image = inputs[i].cpu().clone()
                    
                    # Denormalize if this is the first time
                    if denorm is None:
                        # Try to determine if normalization was applied and reverse it
                        if image.min() < 0 or image.max() > 1.0:
                            # Assume standard normalization with ImageNet values
                            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                            image = image * std + mean
                    
                    # Convert to numpy for display
                    image = image.permute(1, 2, 0).numpy()
                    
                    # Clip values to valid range for display
                    image = np.clip(image, 0, 1)
                    
                    # Get prediction info
                    pred_idx = preds[i].item()
                    true_idx = labels[i].item()
                    prob = probs_display[i, pred_idx].item()
                    
                    # Set title color based on correctness
                    title_color = 'green' if pred_idx == true_idx else 'red'
                    
                    # Create title with prediction and ground truth
                    title = f"Pred: {class_names[pred_idx]} ({prob:.2f})\nTrue: {class_names[true_idx]}"
                    
                    # Show image with title
                    ax.imshow(image)
                    ax.set_title(title, color=title_color)
                    
                    if images_so_far == num_images:
                        plt.tight_layout()
                        plt.show()
                        return
            
            # If we didn't fill all subplots
            if images_so_far < num_images:
                plt.tight_layout()
                plt.show()
                
    def predict_single_image(self, image_tensor, class_names=None, show_image=True):
        """
        Make prediction on a single image.
        
        Args:
            image_tensor: Preprocessed image tensor [C,H,W]
            class_names: List of class names
            show_image: Whether to display the image with prediction
            
        Returns:
            prediction: Class prediction (int)
            probability: Confidence score
            class_name: Name of predicted class (if class_names provided)
        """
        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError("Input must be a torch tensor")
            
        # Add batch dimension if not present
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Default class names if not provided
        if class_names is None:
            # Try to infer number of classes from model
            num_classes = 2  # Assuming binary by default
            try:
                if hasattr(self.model, 'classifier'):
                    if hasattr(self.model.classifier, 'out_features'):
                        num_classes = self.model.classifier.out_features
                elif hasattr(self.model, 'fc'):
                    if hasattr(self.model.fc, 'out_features'):
                        num_classes = self.model.fc.out_features
            except:
                pass
                
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            
            # Get probabilities and prediction
            if output.shape[1] == 1:  # Binary
                prob = torch.sigmoid(output).item()
                pred_class = 1 if prob > 0.5 else 0
                probability = prob if pred_class == 1 else 1 - prob
            else:  # Multi-class
                probs = torch.softmax(output, dim=1)
                prob, pred_class = torch.max(probs, 1)
                pred_class = pred_class.item()
                probability = prob.item()
        
        # Show image if requested
        if show_image and plt_available:
            # Get the image
            img = image_tensor[0].cpu().clone()
            
            # Try to denormalize
            if img.min() < 0 or img.max() > 1.0:
                # Assume standard normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
            
            # Convert to numpy for display
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            # Display
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Prediction: {class_names[pred_class]}\nProbability: {probability:.4f}")
            plt.axis('off')
            plt.show()
        
        return pred_class, probability, class_names[pred_class]
    
    def get_classification_metrics(self, data_loader, class_names=None):
        """
        Calculate and return classification metrics.
        
        Args:
            data_loader: DataLoader for the dataset
            class_names: List of class names
            
        Returns:
            metrics: Dictionary containing classification metrics
        """
        if not sklearn_available:
            print("Warning: scikit-learn is not available. Cannot compute classification metrics.")
            return None
        
        y_pred, y_true = self.predict(data_loader)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot training and validation loss and accuracy.
        """
        if not plt_available:
            print("Warning: matplotlib is not available. Cannot plot training history.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        if len(self.history['val_loss']) > 0:
            ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        if len(self.history['val_acc']) > 0:
            ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot learning rate
        if len(self.history['learning_rates']) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(self.history['learning_rates'])
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.grid(True)
            plt.show()
    
    def load_best_model(self):
        """
        Load the best model weights found during training.
        """
        self.model.load_state_dict(self.best_model_weights)
        return self.model
    
    def _get_current_lr(self):
        """
        Get the current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def _save_model(self, path, metadata=None):
        """
        Save the model with metadata.
        """
        if metadata is None:
            metadata = {}
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_model_weights': self.best_model_weights,
            'history': self.history,
            'metadata': metadata
        }, path)
        
        return path


# Utility functions
def save_model(model, path, metadata=None):
    """
    Save a model with optional metadata.
    
    Args:
        model: PyTorch model
        path: Path to save the model
        metadata: Dictionary of metadata to save with the model
    """
    if metadata is None:
        metadata = {}
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    
    # Save model state and metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, path)
    
    return path

def load_model(model, path, device=None):
    """
    Load a model from a file.
    
    Args:
        model: PyTorch model to load into
        path: Path to load the model from
        device: Device to load the model to
        
    Returns:
        model: Loaded model
        metadata: Metadata dictionary if available, otherwise None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    metadata = checkpoint.get('metadata', None)
    
    return model, metadata

def resume_training(trainer, path):
    """
    Resume training from a checkpoint.
    
    Args:
        trainer: PyTorchTrainer instance
        path: Path to load the checkpoint from
        
    Returns:
        trainer: Trainer with loaded state
    """
    checkpoint = torch.load(path, map_location=trainer.device)
    
    # Load model and optimizer states
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load best model weights and history
    trainer.best_model_weights = checkpoint['best_model_weights']
    trainer.history = checkpoint['history']
    
    return trainer


# Example callback functions
def lr_schedule_callback(trainer, epoch, factor=0.1, patience=3, threshold=0.01):
    """
    Reduce learning rate when validation loss stops improving.
    
    Args:
        trainer: PyTorchTrainer instance
        epoch: Current epoch
        factor: Factor to reduce learning rate by
        patience: Number of epochs with no improvement before reducing learning rate
        threshold: Threshold for considering an improvement
    """
    # Skip if not enough epochs
    if len(trainer.history['val_loss']) <= patience:
        return
    
    # Check if val_loss has improved
    recent_val_losses = trainer.history['val_loss'][-patience:]
    min_recent_loss = min(recent_val_losses)
    current_loss = trainer.history['val_loss'][-1]
    
    # If current loss is not better than minimum recent loss by threshold
    if current_loss > min_recent_loss - threshold:
        # Reduce learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] *= factor
            print(f"Reducing learning rate to {param_group['lr']}")

def checkpoint_callback(trainer, epoch, save_dir='checkpoints', every=5):
    """
    Save model checkpoints periodically.
    
    Args:
        trainer: PyTorchTrainer instance
        epoch: Current epoch
        save_dir: Directory to save checkpoints
        every: Save checkpoint every this many epochs
    """
    if (epoch + 1) % every != 0:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'checkpoint_epoch{epoch+1}.pth')
    trainer._save_model(save_path, {
        'epoch': epoch,
        'train_loss': trainer.history['train_loss'][-1],
        'val_loss': trainer.history['val_loss'][-1] if trainer.history['val_loss'] else None,
        'train_acc': trainer.history['train_acc'][-1],
        'val_acc': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else None,
    })