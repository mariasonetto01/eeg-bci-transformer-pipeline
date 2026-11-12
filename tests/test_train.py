import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from eeg_bci_transformer_pipeline.train import EarlyStopper, EEGTrainer  

class DummyModel(nn.Module):
    """
    A simple dummy model to test training and validation loops.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(3, num_classes)

    def forward(self, x):
        # Reduces (batch_size, num_channels=3, time_points=32) to (batch_size, 3)
        x = x.mean(dim=-1) if x.dim() > 2 else x
        return self.linear(x)

@pytest.fixture
def dummy_dataloader():
    """
    Creates a dummy DataLoader with 16 samples, 3 channels, and 32 temporal samples.
    """

    x = torch.randn(16, 3, 32)
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    return loader

def test_early_stopper():
    """
    Verifies that the Early Stopping logic responds correctly to loss variations.
    """
    
    stopper = EarlyStopper(patience=2, min_delta=0.01)
    
    assert stopper.early_stop(1.0) == False
    assert stopper.early_stop(0.8) == False
    assert stopper.early_stop(0.805) == False
    assert stopper.early_stop(0.9) == False  
    assert stopper.early_stop(0.95) == True  

def test_eeg_trainer_train_step(dummy_dataloader):
    """
    Verifies that the train() method executes an epoch returning valid loss and accuracy.
    """
    
    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = EEGTrainer(model, criterion, optimizer, device=torch.device('cpu'))
    
    epoch_loss, epoch_acc = trainer.train(dummy_dataloader)
    
    # Assert types and value ranges    
    assert isinstance(epoch_loss, float)
    assert isinstance(epoch_acc, float)
    assert 0.0 <= epoch_acc <= 1.0
    assert epoch_loss >= 0.0

def test_eeg_trainer_evaluate(dummy_dataloader):
    """
    Verifies that the evaluate() method computes loss/accuracy and puts the model in eval mode.
    """
    
    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = EEGTrainer(model, criterion, optimizer, device=torch.device('cpu'))
    
    eval_loss, eval_acc = trainer.evaluate(dummy_dataloader)
    
    assert isinstance(eval_loss, float)
    assert isinstance(eval_acc, float)
    assert 0.0 <= eval_acc <= 1.0
    # Ensure the trainer correctly set the model to evaluation mod
    assert not model.training

def test_train_and_validate(dummy_dataloader):
    """
    Verifies the full training and validation lifecycle for a reduced number of epochs (e.g., 3).
    """
    
    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = EEGTrainer(model, criterion, optimizer, device=torch.device('cpu'))
    
    best_val_acc, train_losses, val_losses, val_accuracies = trainer.train_and_validate(
        train_loader=dummy_dataloader,
        val_loader=dummy_dataloader,
        epochs=3
    )
    
    # Verify that historical data is returned for all 3 epochs
    assert isinstance(best_val_acc, float)
    assert len(train_losses) == 3
    assert len(val_losses) == 3
    assert len(val_accuracies) == 3