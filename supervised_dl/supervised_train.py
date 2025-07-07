import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from data.fastmri import MRIReconstructionDataset  # Replace with your actual file

from utils.recon_funcs import complex_to_magnitude,estimate_object_mask, CGDataConsistency

from utils.models import UNet

def log_images(writer, inputs, outputs, targets, step, max_images=4):
    """
    Log input, output, target magnitude images to TensorBoard.
    """
    inputs = complex_to_magnitude(inputs)[:max_images]
    outputs = complex_to_magnitude(outputs)[:max_images]
    targets = complex_to_magnitude(targets)[:max_images]

    writer.add_images("Input/combined_us", inputs, step)
    writer.add_images("Output/reconstruction", outputs, step)
    writer.add_images("Target/combined_full", targets, step)


# Evaluation
@torch.no_grad()
def evaluate(model, val_loader, device, writer=None, epoch=None, loss_type ='post'):
    model.eval()
    val_loss_l2 = 0.0;val_loss_l1 = 0.0
    first_batch_logged = False
    for batch in val_loader:
        combined_us = batch['combined_us'].to(device)
        combined_full = batch['combined_full'].to(device)
        us_coil = batch['us_coil'].to(device)
        sensitivity = batch['sensitivity'].to(device)
        mask = batch['mask'].unsqueeze(1).to(device)

        x = combined_us
        for _ in range(iterations):
            x_pre = model(x) 
            x  = cg_layer(x_pre, us_coil, sensitivity, mask)
            x = x * estimate_object_mask(sensitivity)
        if loss_type=='pre':
            l1 = torch.norm(x_pre- combined_full,p=1)/torch.norm(combined_full,p=1)
            l2 = torch.norm(x_pre- combined_full,p=2)/torch.norm(combined_full,p=2)
            val_loss_l1 += l1.item()
            val_loss_l2 += l2.item()
        else:
            l1 = torch.norm(x- combined_full,p=1)/torch.norm(combined_full,p=1)
            l2 = torch.norm(x- combined_full,p=2)/torch.norm(combined_full,p=2)
            val_loss_l1 += l1.item()
            val_loss_l2 += l2.item()
        # Log images from the first batch
        if not first_batch_logged and writer is not None and epoch is not None:
            log_images(writer, combined_us, x, combined_full, epoch)
            first_batch_logged = True
    return (val_loss_l1 / len(val_loader), val_loss_l2 / len(val_loader))

# Training 
data_dir = '/auto/data2/salman/datasets/fastMRI_brain_20coils_T2_640x320_training/uniform/'
exp_dir = ''
exp = 'exp_1'
iterations = 5
loss_type= 'post'
train_dataset = MRIReconstructionDataset(data_root=data_dir, split='train')
val_dataset = MRIReconstructionDataset(data_root=data_dir, split='val')

model = UNet()
cg_layer = CGDataConsistency(max_iter=10)
all_params = list(model.parameters()) + list(cg_layer.parameters())
print(model)
print(cg_layer)
device='cuda'

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,num_workers=4)
optimizer = optim.Adam(all_params, lr=1e-4)
loss_l1 = nn.L1Loss() 
loss_l2 = nn.MSELoss()
writer = SummaryWriter(log_dir=os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S")))

model.to(device)
epochs = 200

for epoch in range(1, epochs + 1):


    model.train()
    epoch_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        combined_us = batch['combined_us'].to(device)
        combined_full = batch['combined_full'].to(device)
        us_coil = batch['us_coil'].to(device)
        sensitivity = batch['sensitivity'].to(device)
        mask = batch['mask'].unsqueeze(1).to(device)

        x = combined_us
        for _ in range(iterations):
            x_pre = model(x)
            x  = cg_layer(x_pre, us_coil, sensitivity, mask)
            x = x * estimate_object_mask(sensitivity)
        if loss_type=='pre':
            l1 = torch.norm(x_pre- combined_full,p=1)/torch.norm(combined_full,p=1)
            l2 = torch.norm(x_pre- combined_full,p=2)/torch.norm(combined_full,p=2)
            loss = l1+l2
        else:
            l1 = torch.norm(x- combined_full,p=1)/torch.norm(combined_full,p=1)
            l2 = torch.norm(x- combined_full,p=2)/torch.norm(combined_full,p=2)
            loss = l1+l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 1 == 0:
        val_loss_l1,val_loss_l2 = evaluate(model, val_loader, device, writer, epoch,loss_type = loss_type)
        writer.add_scalar("Loss/val_l1", val_loss_l1, epoch)
        writer.add_scalar("Loss/val_l2", val_loss_l2, epoch)
        writer.add_scalar("Loss/val", val_loss_l1+val_loss_l2, epoch)
        print(f"Epoch {epoch}, Validation Loss L1: {val_loss_l1:.6f}")
        print(f"Epoch {epoch}, Validation Loss L2: {val_loss_l2:.6f}")
        writer.add_scalar("Lambda/value", cg_layer.lam.item(), epoch)


    if epoch % 5 == 0:
        os.makedirs(exp_dir+exp, exist_ok=True)
        model_path = os.path.join(exp_dir+exp, "recon_model_"+str(epoch)+".pth")
        cg_path = os.path.join(exp_dir+exp, "cg_"+str(epoch)+".pth")
        torch.save(model.state_dict(), model_path)
        torch.save(cg_layer.state_dict(), cg_path)
        print(f"Model saved at {model_path}")    
    avg_train_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.6f}")



# Save final model
os.makedirs(exp_dir+exp, exist_ok=True)
model_path = os.path.join(exp_dir+exp, "recon_model_final.pth")
cg_path = os.path.join(exp_dir+exp, "cg_final.pth")
torch.save(model.state_dict(), model_path)
torch.save(cg_layer.state_dict(), cg_path)
print(f"Model saved at {model_path}")
writer.close()