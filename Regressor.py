import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
from torchmetrics import MeanSquaredError
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
import timm
import albumentations as A
import pytorch_lightning_spells as pls
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import sys
import warnings
import timm
import cv2
import os
from os.path import join
from torchvision import models
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image
from tqdm import tqdm
from torchmetrics import Metric
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import models
from efficientnet_pytorch import EfficientNet
warnings.simplefilter("ignore", DeprecationWarning)

mmodel = timm.create_model('resnet18', pretrained = True, num_classes=1)

cfg = {
    "num_classes": 1,
    "learning_rate": 3e-4,
    "epochs": 10,
    'picture_size': 224,
    "train_batch_size": 16,
    "test_batch_size": 8,
    "model_name": 'tf_efficientnet_b3',
    "freeze": True,
    "unfreeze_layers": ["_fc", "_conv_head", "_bn1", "_avg_pooling", "_swish"],
    "unfreeze_tf_layers": ['classifier'],
    "unfreeze_resnet_layers": ["fc", 'global_pool'],
    'StepLR' : {
        'step_size' : 3,
        'gamma': 0.5
    },
    'num_gpu': 1,
    'optimizer': 'AdamW',
    'train_workers': 0,
    'test_workers': 0,
    'threshold_step': 0.05,
    'beta': 0.6,
    'weights_path': r'C:\Users\User\Desktop\testing\comp'
}


def read_image(image_path: str):
    with open(image_path, 'rb') as f:
        nparr = np.fromstring(f.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image


class CSVDataset(Dataset):

    def __init__(self, csv_path: str, image_path_='', transform=None, return_image_type=False):

        self.image_path = image_path_
        self.transform = transform
        self.dt = pd.read_csv(csv_path)
        self.image_type = return_image_type

    def __len__(self):
        return len(self.dt)

    def read_image__(self, idx, debug=False):
        row = self.dt.iloc[idx].values
        image_path = row[0].split('/')[0] if self.image_path == '' else join(self.image_path, row[0].split('/')[1])
        image = read_image(image_path)
        if self.transform and not debug:
            try:
                image = self.transform(image=image)['image']
            except Exception as e:
                raise ValueError

        label = row[1:].astype('int32')[0]
        if not self.image_type:
            return image, label

    def __getitem__(self, idx):
        return self.read_image__(idx)

    def debug(self, idx):
        return self.read_image__(idx, debug=True)


train_transform = A.Compose([
    A.LongestMaxSize(max_size = cfg['picture_size']),
    A.PadIfNeeded(cfg['picture_size'], cfg['picture_size'], border_mode = 0, value = cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_transform = A.Compose([
    A.LongestMaxSize(max_size = cfg['picture_size']),
    A.PadIfNeeded(cfg['picture_size'], cfg['picture_size'], border_mode = 0, value = cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_csv, test_csv = r"C:\Users\n.shamankov\train_ed.csv", r"C:\Users\n.shamankov\test_ed.csv"
train_data, test_data = r"C:\Users\n.shamankov\Downloads\train_data", r"C:\Users\n.shamankov\Downloads\test_data"

train_dataset = CSVDataset(train_csv, train_data, train_transform)
test_dataset = CSVDataset(test_csv, test_data, test_transform)


train_dataloader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=cfg["train_batch_size"],
                              num_workers=cfg['train_workers'])
test_dataloader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=cfg["test_batch_size"],
                             num_workers=cfg['test_workers'])


def get_optim(net, key, lr = cfg["learning_rate"]):

    Optim = {'Adam': optim.Adam(list(net.parameters()), lr=lr),
            'AdamW': optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False),
             'SGD': optim.SGD(net.parameters(), lr=lr, momentum=0.999)
             }
    return Optim[key]


def create_torchvision_model(name, num_classes):
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    return model


def create_tf_efficientnet_model(name, num_classes):
    try:
        model = timm.create_model(name, pretrained=True, num_classes=num_classes)
        return model
    except Exception:
        raise ValueError('Bad name of model {}.'.format(name))


def freeze(model, layer_names=[]):
    for param in model.parameters():
        param.requires_grad = False
    for layer_name in layer_names:
        params = eval('model.{}.parameters()'.format(layer_name))
        for param in params:
            param.requires_grad = True
    return model


def create_model(model_name, num_classes):
    if model_name.find('tf_efficientnet') > -1:
        return create_tf_efficientnet_model(model_name, num_classes)
    else:
        return create_torchvision_model(model_name, num_classes)


class Learner(pl.LightningModule):

    def __init__(
            self,
            num_classes=cfg['num_classes'],
            learning_rate=cfg['learning_rate'],
            epochs=cfg['epochs'],
    ):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'epochs')
        self.model = freeze(mmodel, cfg["unfreeze_resnet_layers"])
#         if cfg['freeze']:
#             self.model = freeze(self.model, cfg["unfreeze_tf_layers"])
        self.epochs = epochs
        self.sigmoid = nn.Sigmoid()
        self.loss_f = nn.MSELoss()
        self.train_rmse = MeanSquaredError()
        self.validate_rmse = MeanSquaredError()

    def forward(self, x):
        model_output = self.model(x)
        #out, idxs = torch.max(model_output, dim = 1)
        return model_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_f(output, y.float())
        self.log('train_loss', loss)
        self.train_rmse.update(output.flatten(), y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_f(output, y.float())
        self.log('validate_loss', loss)
        Tvalue = self.validate_rmse(output.flatten(), y.float())
        print(f'RMSE for batch is {Tvalue}')
        return loss

    def training_epoch_end(self, training_step_outputs):
        rmse_tvalue  = self.train_rmse.compute()
        self.train_rmse.reset()

    def validation_epoch_end(self, validation_step_outputs):
        Tvalue = self.validate_rmse.compute()
        self.validate_rmse.reset()


    def configure_optimizers(self):
        optimizer = get_optim(self.model, cfg['optimizer'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg['StepLR']['step_size'],
                                              gamma=cfg['StepLR']['gamma'])
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = cfg['epochs']-1)
        return [optimizer], [scheduler1]


checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
    monitor='validate_loss',
    dirpath=cfg["weights_path"],
    filename='efiicientnent_b0-loss-{epoch:02d}-{validate_loss:.2f}',
    save_top_k = 1,
    mode='min',
    save_weights_only = True,
)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='validate_loss',
    min_delta=0.00,
    patience=3,
    mode='min'
)

lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

learner_cfg = {key: cfg[key] for key in ["num_classes", "learning_rate", "epochs"]}
model = Learner(**learner_cfg)

if __name__ == '__main__':

    trainer = pl.Trainer(
        gpus=cfg['num_gpu'], num_nodes=1, accelerator='dp', sync_batchnorm=True, auto_lr_find=False,
        callbacks=[early_stop_callback,
                   checkpoint_callback_loss,
                   lr_monitor],
        max_epochs=cfg["epochs"],
    )

    trainer.tune(model, train_dataloader, test_dataloader)

    trainer.fit(model, train_dataloader, test_dataloader)
