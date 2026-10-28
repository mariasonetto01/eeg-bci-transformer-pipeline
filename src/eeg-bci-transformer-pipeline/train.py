import torch
import copy


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EEGTrainer:

    def __init__(self,model, criterion, optimizer, device=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

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

    def train(self, train_loader, max_grad_norm=1.0):    
        """Train the model."""

        #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #conviene definirlo poi nel notebook

        self.model.train() 

        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad() # Azzero i gradienti

            outputs = self.model(x_batch) # Forward , restituisce una tupla

            if isinstance(outputs, tuple):
                outputs = outputs[0]
     

            loss = self.criterion(outputs, y_batch) # Loss

            loss.backward() # Backpropagation
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)

            #controllo i gradienti
            #for name, param in self.model.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name}: grad norm = {param.grad.norm().item()}")
            #    else:
            #        print(f"{name}: grad è None!")

            self.optimizer.step() # Update
            
            running_loss += loss.item() 

        epoch_loss = running_loss / len(train_loader)

        return epoch_loss

    

    def evaluate(self, val_loader):
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

                #print("Tipo di output:", type(outputs))

                #if isinstance(outputs, tuple):
                 #   print(f"Il modello restituisce una tupla con {len(outputs)} elementi.")
                  #  for i, item in enumerate(outputs):
                      #  print(f" - Elemento {i}: tipo {type(item)}")
                      #  if hasattr(item, 'shape'):
                     #       print(f"   Shape: {item.shape}")
                    #    else:
                    #        print("   (Non ha una shape)")
               # else:
              #      print("Shape dell'output:", outputs.shape)
                    
                loss = self.criterion(outputs, y_batch) # Loss
                running_loss += loss.item()

                
                preds = torch.argmax(outputs, dim=1) # Accuracy
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        eval_loss = running_loss / len(val_loader)
        eval_acc = correct / total

        return eval_loss, eval_acc
    
    def train_and_validate(self, train_loader, val_loader, epochs, scheduler=None, max_grad_norm=1.0):

        train_losses = []
        val_losses   = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        
        early_stopper = EarlyStopper(patience= 10 , min_delta=1e-4)

        for epoch in range(epochs):
        
                # TRAIN 
                train_loss = self.train(train_loader, max_grad_norm=max_grad_norm)
        
                #  VALIDATION 
                val_loss, val_acc = self.evaluate(val_loader)

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)  # ReduceLROnPlateau esige la validation loss
                    else:
                        scheduler.step()       
                
                # Tentativo sicuro di recuperare il learning rate corrente senza far crashare il codice
                try:
                    current_lr = self.optimizer.param_groups[0]['lr']
                except Exception:
                    current_lr = 0.001  # Fallback
            
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

               
                # Salviamo i pesi se l'Accuracy di Validazione migliora
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_weights = copy.deepcopy(self.model.state_dict())
            
                
                print(
                    f'Epoch {epoch + 1:02d}/{epochs:02d}: train_loss={train_loss:.5f},'
                    f' val_loss={val_loss:.5f}, val_acc={val_acc * 100:.2f}%'
                )
            
                
        
                if early_stopper.early_stop(val_loss):
                    print(f"Early stopping all'epoca {epoch+1}")
                    break
          
        self.model.load_state_dict(best_model_weights) #iniettare nuovamente i pesi "congelati" nel modello 

        return best_val_acc, train_losses, val_losses, val_accuracies

