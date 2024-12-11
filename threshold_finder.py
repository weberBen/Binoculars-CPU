import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging
import argparse

class ThresholdModel(nn.Module):
    """
    A differentiable threshold classifier using smooth approximation.
    """
    def __init__(self, initial_threshold: Optional[float] = None, temperature: float = 0.02):
        """
        Initialize the threshold model.
        
        Args:
            initial_threshold (float, optional): Initial threshold value. If None, will be set during forward pass.
            temperature (float): Temperature parameter for sigmoid smoothing. Lower values make transition sharper.
        """
        super().__init__()
        self.temperature = temperature
        
        self.initialize_threshold()
    
    def initialize_threshold(self, threshold=None):
        if threshold is None:
          self.threshold = nn.Parameter(torch.rand(1))
        else:
          self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing smooth thresholding.
        
        Args:
            x (torch.Tensor): Input tensor of scores
            
        Returns:
            torch.Tensor: Predicted probabilities
        """
        return torch.sigmoid((self.threshold - x) / self.temperature)

class ThresholdClassifier:
    """
    Main classifier class handling training and evaluation.
    """
    def __init__(self, 
                 temperature: float = 0.02,
                 batch_size: int = 64,
                 learning_rate: float = 0.01,
                 num_epochs: int = 1000,
                 device: str = "cpu"):
        """
        Initialize the classifier.
        
        Args:
            temperature (float): Temperature parameter for sigmoid smoothing
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimization
            num_epochs (int): Number of training epochs
        """
        self.model = ThresholdModel(temperature=temperature)
        self.model.to(device)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.loss_fn = nn.MSELoss()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, file_path: str) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
        """
        Load and prepare data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            Tuple containing:
                - DataLoader for training data
                - Test scores tensor
                - Test classes tensor
        """
        # Read and process CSV
        columns_to_use = ["score", "class"]
        data = pd.read_csv(file_path, usecols=columns_to_use)
        
        # Split data
        train_scores, test_scores, train_classes, test_classes = train_test_split(
            data["score"].values,
            data["class"].values,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        
        # Convert to tensors
        train_scores = torch.tensor(train_scores, dtype=torch.float32)
        train_classes = torch.tensor(train_classes, dtype=torch.float32)
        test_scores = torch.tensor(test_scores, dtype=torch.float32)
        test_classes = torch.tensor(test_classes, dtype=torch.float32)

        self.logger.info(f"Training dataset size : {train_scores.shape[0]}")
        self.logger.info(f"Testing dataset size : {test_scores.shape[0]}")
        
        # Create DataLoader
        train_dataset = TensorDataset(train_scores, train_classes)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        init_threshold = train_scores.mean().item()
        self.logger.info(f"Init threshold: {init_threshold}")
        self.model.initialize_threshold(init_threshold)
        
        return train_loader, test_scores, test_classes

    def train(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the threshold model.
        
        Args:
            train_loader (DataLoader): DataLoader containing training data
            
        Returns:
            Tuple containing:
                - Final threshold value
                - Lowest MSE loss achieved
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_scores, batch_classes in train_loader:
                optimizer.zero_grad()
                predicted = self.model(batch_scores)
                loss = self.loss_fn(predicted, batch_classes)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / len(train_loader)
                
            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}, "
                               f"Threshold: {self.model.threshold.item():.4f}")
        
        return self.model.threshold.item()

    def evaluate(self, test_scores: torch.Tensor, test_classes: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_scores (torch.Tensor): Test score data
            test_classes (torch.Tensor): True classes for test data
            
        Returns:
            Tuple containing:
                - MSE loss on test data
                - Success rate (accuracy) as percentage
        """
        with torch.no_grad():
            threshold = self.model.threshold.item()
            self.logger.info(f"Testing with threshold {threshold:.16f}")

            # Use hard threshold for evaluation
            predicted = (test_scores < threshold).float()
            mse_loss = self.loss_fn(predicted, test_classes).item()
            correct_predictions = (predicted == test_classes).sum().item()
            success_rate = (correct_predictions / len(test_classes)) * 100
            
            self.logger.info(f"Test MSE Loss: {mse_loss:.4f}")
            self.logger.info(f"Success Rate: {success_rate:.2f}%")
            
            return mse_loss, success_rate

def main():
    """
    README :
        Select a model to use for the Binoculars detector.py
        Generate a CSV file from the experiment folder by following the commands in experiments/job.sh.
        Provide the path to the generated CSV file as an argument to the script.
    """

    parser = argparse.ArgumentParser(description='Generate a CSV file from the experiment folder by following the commands in experiments/job.sh. Provide the path to the generated CSV file as an argument to this script.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature parameter for sigmoid smoothing')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for optimization')
    parser.add_argument('--num_epochs', type=int, default=1000,
                       help='Number of training epochs')
    parser.add_argument('--testing_threshold', type=float, default=None)
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ThresholdClassifier(
        temperature=args.temperature,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    # Prepare data
    train_loader, test_scores, test_classes = classifier.prepare_data(args.file_path)
    
    if args.testing_threshold is None:
      # Train model
      best_threshold = classifier.train(train_loader)
      print(f"\nTraining completed:")
      print(f"Best threshold: {best_threshold:.16f}")
    else:
      print("Skipping training !")
      print(f"\tUsing threshold {args.testing_threshold:.16f} for testing")
      classifier.model.initialize_threshold(float(args.testing_threshold))
    
    # Evaluate model
    test_loss, success_rate = classifier.evaluate(test_scores, test_classes)
    print(f"\nEvaluation results:")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    main()