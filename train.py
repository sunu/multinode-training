import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L


# Generate a synthetic dataset
def generate_data(num_samples):
    # Generate random integers
    data = torch.randint(0, 1000, (num_samples, 1)).float()
    # Labels are 1 if the number is odd, 0 if even
    labels = data % 2
    return TensorDataset(data, labels)


class EvenOddDataset(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = generate_data(800)
            self.val_dataset = generate_data(200)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class SimpleClassifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y.long().squeeze())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y.long().squeeze())
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# Create the model
model = SimpleClassifier()
dm = EvenOddDataset()

# CPU
if os.environ.get("USE_CPU") == "1":
    trainer = L.Trainer(max_epochs=10, accelerator="cpu", num_nodes=1)
elif torch.cuda.is_available():
    if os.environ.get("MULTI_GPU") == "1":
        trainer = L.Trainer(max_epochs=10, accelerator="auto", devices="auto", num_nodes=1)
    elif os.environ.get("MULTI_NODE") == "1":
        trainer = L.Trainer(max_epochs=10, accelerator="auto", devices="auto", num_nodes=2, strategy="ddp")
    else:
        trainer = L.Trainer(max_epochs=10, accelerator="gpu", num_nodes=1)
else:
    print("No GPU available")
trainer.fit(model, dm)
        
# trainer = L.Trainer(max_epochs=10, accelerator="cpu", num_nodes=1)

# Multi GPU
# trainer_gpu = L.Trainer(max_epochs=10, accelerator="auto", devices="auto", num_nodes=1)

# # MultiNode
# trainer_gpu = L.Trainer(max_epochs=1, accelerator="auto", devices="auto", num_nodes=2, strategy="ddp")

# trainer_gpu.fit(model, dm)