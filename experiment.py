from src.trainers.dm import Trainer
from src.datasets.dm import DAMNIST
from src.models.model import Model
import torch.utils.data as data

model = Model()
dataset = DAMNIST()
data_loader = data.DataLoader(dataset, batch_size=12, shuffle=True)

trainer = Trainer(
        model=model,
        data_loader=data_loader,
        n_epoch=100,
        )
trainer.train()
