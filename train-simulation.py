from dataset.preprocess_v2 import *
from dataset.simulation_v2 import *

trainer = pl.Trainer(max_epochs=1000)
trainer.fit(sec_back_t_model, sec_back_t_train_dataloader, sec_back_t_val_dataloader)
