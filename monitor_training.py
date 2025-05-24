#!/usr/bin/env python3
import json
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime
import paramiko
import argparse

class RemoteTrainingMonitor:
    def __init__(self, hostname, username, remote_log_dir, local_log_dir="local_logs"):
        self.hostname = hostname
        self.username = username
        self.remote_log_dir = remote_log_dir
        self.local_log_dir = local_log_dir
        os.makedirs(local_log_dir, exist_ok=True)
        
        # Setup SSH connection
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
    def connect(self, password=None, key_path=None):
        """Connect to remote server"""
        try:
            if key_path:
                self.ssh.connect(self.hostname, username=self.username, key_filename=key_path)
            else:
                self.ssh.connect(self.hostname, username=self.username, password=password)
            print(f"Connected to {self.hostname}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def download_file(self, remote_path, local_path):
        """Download file from remote server"""
        try:
            sftp = self.ssh.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def check_training_status(self):
        """Check current training status"""
        try:
            # Download progress file
            remote_progress = f"{self.remote_log_dir}/progress.txt"
            local_progress = os.path.join(self.local_log_dir, "progress.txt")
            
            if self.download_file(remote_progress, local_progress):
                with open(local_progress, 'r') as f:
                    content = f.read()
                print("=== Training Status ===")
                print(content)
                return True
            return False
        except Exception as e:
            print(f"Status check failed: {e}")
            return False
    
    def download_metrics(self):
        """Download and plot training metrics"""
        try:
            # Download current metrics
            remote_metrics = f"{self.remote_log_dir}/current_metrics.json"
            local_metrics = os.path.join(self.local_log_dir, "current_metrics.json")
            
            if self.download_file(remote_metrics, local_metrics):
                with open(local_metrics, 'r') as f:
                    metrics = json.load(f)
                
                self.plot_metrics(metrics)
                return True
            return False
        except Exception as e:
            print(f"Metrics download failed: {e}")
            return False
    
    def plot_metrics(self, metrics):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(metrics['train_loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss')
        if metrics['val_loss']:
            ax1.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, metrics['train_acc'], 'b-', label='Train Accuracy')
        if metrics['val_acc']:
            ax2.plot(epochs, metrics['val_acc'], 'r-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.set_title('Training and Validation Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.local_log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to: {plot_path}")
    
    def monitor_loop(self, interval=60):
        """Continuously monitor training"""
        print(f"Starting continuous monitoring (checking every {interval} seconds)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                print(f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                
                # Check status
                if self.check_training_status():
                    # Download and plot metrics
                    self.download_metrics()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e}")
    
    def download_all_results(self):
        """Download all training results"""
        print("Downloading all training results...")
        
        files_to_download = [
            'felix_classifier.pth',
            'final_metrics.json',
            'training_summary.json',
            'classification_report.json'
        ]
        
        for filename in files_to_download:
            remote_path = f"{self.remote_log_dir}/{filename}"
            local_path = os.path.join(self.local_log_dir, filename)
            
            if self.download_file(remote_path, local_path):
                print(f"Downloaded: {filename}")
            else:
                print(f"Failed to download: {filename}")
    
    def close(self):
        """Close SSH connection"""
        self.ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor remote training')
    parser.add_argument('hostname', help='Remote server hostname/IP')
    parser.add_argument('username', help='SSH username')
    parser.add_argument('--remote-log-dir', default='/home/username/felix_classifier/training_logs',
                       help='Remote log directory path')
    parser.add_argument('--password', help='SSH password (not recommended, use key instead)')
    parser.add_argument('--key-path', help='Path to SSH private key')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--download-results', action='store_true', help='Download all results and exit')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = RemoteTrainingMonitor(args.hostname, args.username, args.remote_log_dir)
    
    # Connect
    if monitor.connect(password=args.password, key_path=args.key_path):
        if args.download_results:
            monitor.download_all_results()
        else:
            monitor.monitor_loop(args.interval)
        monitor.close()