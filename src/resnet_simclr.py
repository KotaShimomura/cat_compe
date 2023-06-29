import math
import pandas as pd
import numpy as np
import os
import random
import gc

gc.enable()
pd.set_option("display.max_columns", None)


from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    CosineAnnealingLR,
)

import torch
import timm
import lightly
from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads
import torch.nn as nn
import torch.nn.functional as F
import PIL
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

import os
import gc
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm import tqdm
import time
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from sklearn.metrics import f1_score


class CFG:
    exp_name = "ex1"
    datapath = "/workspace/hdd_data/cat/cat_ssl/"
    output_dir = "/workspace/hdd_data/cat/output/" + exp_name + "/"
    seed = 42
    model_name = "tf_efficientnet_b0"
    batch_size = 16
    epochs = 10
    gpu = [0]
    max_lr = 1e-3
    weight_decay = 1e-4
    n_fold = 3
    trn_fold = [0]
    apex = True
    num_workers = 0
    print_freq = 50

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu[0]}")
        print(device)
    else:
        device = torch.device("cpu")
        print("does not use GPU")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def make_dir(CFG):
    if not os.path.exists(CFG.output_dir):
        os.makedirs(CFG.output_dir)


def get_csv():
    cat_categories = []
    path = CFG.datapath
    for directory in os.listdir(path):
        if "." not in directory:
            cat_categories.append(directory)

    image_directory = {}
    for i in cat_categories:
        image_directory[i] = [
            os.path.join(path, i, j) for j in os.listdir(os.path.join(path, i))
        ]

    file_category = []
    file_name = []
    for i in image_directory.keys():
        for j in image_directory[i]:
            file_category.append(i)
            file_name.append(j)

    data = {"file_name": file_name, "categories": file_category}

    return cat_categories, pd.DataFrame(data)


class CatDataset(Dataset):
    def __init__(self, cfg, data, transform=None):
        self.data = data
        self.transform = transform
        self.str_to_int = {
            "ragdoll": 0,
            "siamese": 1,
            "domestic_shorthair": 2,
            "bengal": 3,
            "maine_coon": 4,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data["file_name"][idx]
        image = PIL.Image.open(image_path)
        category_name = self.data["categories"][idx]
        label = self.str_to_int[category_name]

        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class SimCLR(torch.nn.Module):
    def __init__(self, backbone, backbone_features):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=backbone_features,
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def _fit_train(train_loader, model, criterion, optimizer, epoch, CFG):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    loss_meter = AverageMeter()
    start = end = time.time()
    t = tqdm(train_loader, total=len(train_loader))

    for step, ((view0, view1), targets, filenames) in enumerate(t):
        image1 = view0.to(CFG.device)
        image2 = view1.to(CFG.device)
        targets = targets.to(CFG.device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            output1 = model(image1)
            output2 = model(image2)
        loss = criterion(output1, output2)
        loss_meter.update(loss.data)
        t.set_description("[ loss: {:.4f} ]".format(loss_meter.avg))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return loss_meter.avg


def train_loop():
    # ====================================================
    # loader
    # ====================================================

    transform = transforms.SimCLRTransform(input_size=224, cj_prob=0.5)
    dataset = LightlyDataset(input_dir=CFG.datapath, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    backbone_model = timm.create_model(CFG.model_name, pretrained=False, num_classes=0)
    in_features = backbone_model.num_features

    model = SimCLR(backbone_model, in_features)
    model.to(CFG.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)
    criterion = loss.NTXentLoss(temperature=0.5)

    best_loss = 10

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        avg_loss = _fit_train(train_loader, model, criterion, optimizer, epoch, CFG)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_loss:.4f} Model")
            torch.save(
                model.state_dict(),
                CFG.output_dir + f"{CFG.model_name.replace('/', '-')}_simCLR_best.pth",
            )

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    make_dir(CFG)
    seed_everything(CFG.seed)
    LOGGER = get_logger(CFG.output_dir + "train")

    train_loop()
