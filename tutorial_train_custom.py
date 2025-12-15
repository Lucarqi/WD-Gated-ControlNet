import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

from share import *

import torch
torch.cuda.empty_cache()
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MedicalNiiDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, compare_weights
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='ControlNet Training with Custom Dataset')

parser.add_argument('--data_root', type=str, default='data/Dataset501_RENJISLICE', 
                    help='Root directory of the dataset (default: data/Dataset501_RENJISLICE)')
parser.add_argument('--cls', type=str, default='cardiac MRI', 
                    help='Class name for prompt (default: cardiac MRI)')
parser.add_argument('--classes', type=int, default=3, 
                    help='Number of classes in mask (default: 3)')
parser.add_argument('--size', type=int, default=192, 
                    help='Image size after resizing (default: 192)')
parser.add_argument("--logdir_name", type=str, default='RENJI')

parser.add_argument('--resume_ckpt', type=str, default=None, 
                    help='Path to resume checkpoint (auto-find latest if not specified)')
args = parser.parse_args()
image_logdir = os.path.join('image_log', args.logdir_name)
ckpt_save_dir = os.path.join('checkpoints', args.logdir_name)

pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


resume_path = './stable-diffusion-v1-5/control_sd15.ckpt'
batch_size = 4 # default 4
logger_freq = 300
learning_rate = 1e-5 # default 1e-5
sd_locked = False
only_mid_control = False


model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


dataset = MedicalNiiDataset(
    root=args.data_root,
    cls=args.cls,
    classes=args.classes,
    size=args.size
)

checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_save_dir,          
    filename='model-{step:04d}',    
    every_n_train_steps=500,        
    save_top_k=-1,                  
    save_last=True,                 
)

dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)


logger = ImageLogger(batch_frequency=logger_freq, log_dir_name=image_logdir)
trainer = pl.Trainer(
    strategy=DeepSpeedStrategy(
        offload_optimizer = True,
        ), # default deepspeed_stage_2
    accelerator="gpu",
    devices=3,
    callbacks=[logger, checkpoint_callback],
    deterministic=True,
    max_steps=3000, # default 3000
)

resume_from_checkpoint = args.resume_ckpt

trainer.fit(
    model,
    dataloader,
    ckpt_path=resume_from_checkpoint,  
    val_dataloaders=None

)
