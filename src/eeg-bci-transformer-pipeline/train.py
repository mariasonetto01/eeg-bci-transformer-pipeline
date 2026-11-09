import torch
import copy


class EarlyStopper:

    """
    Early stopping utility to stop training when validation loss stops improving for a given number of consecutive epochs (patience).

    Note: Following the reference paper's methodology, training is executed for a fixed number of epochs (50). Consequently, `patience` is set to 50 to effectively disable early stopping and observe the full training curve, while keeping the class structure intact for future flexibility.

    """
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Checks whether training should be terminated based on validation loss progression.

        Parameters
        ----------
        validation_loss : float
            The validation loss of the current epoch.

        Returns
        -------
        bool
            True if patience has been exceeded and training should stop, False otherwise.
        """
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EEGTrainer:
    """
    Trainer class designed for EEG deep learning models in PyTorch, supporting training loops, validation, learning rate scheduling, and gradient clipping.
    """

    def __init__(self, model, criterion, optimizer, device=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Automatically select the best available hardware accelerator
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.model.to(self.device)

    def train(self, train_loader):    
        """
        Executes a single training epoch over the training dataset.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader providing mini-batches of training data.

        Returns
        -------
        epoch_loss : float
            Average training loss for the epoch.
        epoch_acc : float
            Average training classification accuracy for the epoch.
        """
        self.model.train() 
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Zero the gradients from the previous step
            self.optimizer.zero_grad() 

            # Forward pass
            outputs = self.model(x_batch) 

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Compute loss
            loss = self.criterion(outputs, y_batch) # Loss

            # Backward pass and optimization step
            loss.backward() 
            self.optimizer.step() 
            
            running_loss += loss.item() 

            # Compute batch accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc


    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation or test dataset.
        
        Computes the average loss and classification accuracy without 
        storing gradients (saving memory and accelerating computation).

        Parameters
        ----------
        val_loader : torch.utils.data.DataLoader
            DataLoader providing mini-batches of validation or test data.

        Returns
        -------
        eval_loss : float
            Average evaluation loss over the dataset.
        eval_acc : float
            Average classification accuracy (ranging from 0.0 to 1.0).
        """

        self.model.eval()  
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

               
                outputs = self.model(x_batch) # Forward

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                    
                loss = self.criterion(outputs, y_batch) # Loss
                running_loss += loss.item()

                
                preds = torch.argmax(outputs, dim=1) # Accuracy
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        eval_loss = running_loss / len(val_loader)
        eval_acc = correct / total

        return eval_loss, eval_acc

  
    
    def train_and_validate(self, train_loader, val_loader, epochs, scheduler=None):
        train_losses = []
        val_losses   = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        
        early_stopper = EarlyStopper(patience=200, min_delta=1e-4)

        for epoch in range(epochs):
            # TRAIN 
            train_loss, train_acc = self.train(train_loader)
    
            # VALIDATION 
            val_loss, val_acc = self.evaluate(val_loader)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc) 
                else:
                    scheduler.step()       
            
            # Tentativo sicuro di recuperare il learning rate corrente senza far crashare il codice
            try:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:02d}/{epochs:02d}: train_loss={train_loss:.5f}, train_acc={train_acc * 100:.2f}%, val_loss={val_loss:.5f}, val_acc={val_acc * 100:.2f}% | LR: {current_lr:.6f}")
            except Exception:
                current_lr = 0.001  # Fallback
                        
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Salviamo i pesi se l'Accuracy di Validazione migliora
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())
    
            if early_stopper.early_stop(val_loss):
                print(f"Early stopping all'epoca {epoch+1}")
                break
          
        self.model.load_state_dict(best_model_weights) # Re-inietta i pesi migliori

        return best_val_acc, train_losses, val_losses, val_accuracies


    def train_gradient(self, train_loader, max_grad_norm=1.0):    
        """Train the model with gradient clipping and compute training loss and accuracy."""
        self.model.train() 
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad() # Azzero i gradienti

            outputs = self.model(x_batch) # Forward

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.criterion(outputs, y_batch) # Loss

            loss.backward() # Backpropagation

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            self.optimizer.step() # Update dei pesi
            
            running_loss += loss.item() 

            # Calcolo accuracy di training per il batch
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def train_and_validate_gradient(self, train_loader, val_loader, epochs, scheduler=None, max_grad_norm=1.0):
        train_losses = []
        val_losses   = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        
        early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

        for epoch in range(epochs):
            # TRAIN 
            train_loss, train_acc = self.train_gradient(train_loader, max_grad_norm=max_grad_norm)
    
            # VALIDATION 
            val_loss, val_acc = self.evaluate(val_loader)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)  # ReduceLROnPlateau esige la validation loss
                else:
                    scheduler.step()       
            
            # Tentativo sicuro di recuperare il learning rate corrente senza far crashare il codice
            try:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:02d}/{epochs:02d}: train_loss={train_loss:.5f}, train_acc={train_acc * 100:.2f}%, val_loss={val_loss:.5f}, val_acc={val_acc * 100:.2f}% | LR: {current_lr:.6f}")
            except Exception:
                current_lr = 0.001  # Fallback
        
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # Salviamo i pesi se l'Accuracy di Validazione migliora
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())
    
            if early_stopper.early_stop(val_loss):
                print(f"Early stopping all'epoca {epoch+1}")
                break
      
        self.model.load_state_dict(best_model_weights) # iniettare nuovamente i pesi "congelati" nel modello 

        return best_val_acc, train_losses, val_losses, val_accuracies