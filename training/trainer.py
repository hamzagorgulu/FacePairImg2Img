import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from tqdm import tqdm
from loss_function import PerceptualLoss
import matplotlib.pyplot as plt

class BeardRemovalTrainer:
    def __init__(self, model, train_dataset, val_dataset, device, 
                 learning_rate=0.0002, batch_size=8):
        self.device = device
        self.model = model.to(device)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                     shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                   shuffle=False, num_workers=2)
        
        #self.criterion = nn.L1Loss()
        self.criterion = PerceptualLoss(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        
        # Create directories for checkpoints and samples
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('samples', exist_ok=True)
        
    def save_samples(self, epoch, batch):
        self.model.eval()
        with torch.no_grad():
            # Move input images to device
            input_images = batch['input'].to(self.device)
            target_images = batch['target'].to(self.device)
            generated_images = self.model(input_images)
            
            # Move everything back to CPU for saving
            input_images = input_images.cpu()
            generated_images = generated_images.cpu()
            target_images = target_images.cpu()
            
            # Save sample images
            for i in range(min(3, input_images.size(0))):
                save_image(
                    [input_images[i], generated_images[i], target_images[i]],
                    f'samples/epoch_{epoch}_sample_{i}.png',
                    normalize=True
                )
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()
                
                input_images = batch['input'].to(self.device)
                target_images = batch['target'].to(self.device)
                
                generated_images = self.model(input_images)
                loss = self.criterion(generated_images, target_images)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_images = batch['input'].to(self.device)
                target_images = batch['target'].to(self.device)
                
                generated_images = self.model(input_images)
                loss = self.criterion(generated_images, target_images)
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save samples
            if epoch % 1 == 0:
                sample_batch = next(iter(self.val_loader))
                self.save_samples(epoch, sample_batch)
            
            # Adjust learning rate using the scheduler
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'checkpoints/best_model.pth')

    def plot_loss_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curve')
        plt.legend()
        plt.grid()
        plt.savefig('loss_curve.png')
        plt.show()
