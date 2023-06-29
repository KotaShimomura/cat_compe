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
from sklearn import svm
import joblib
import torch
import timm
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
    datapath = "/workspace/hdd_data/cat/cat_compe/train/"
    output_dir = "/workspace/hdd_data/cat/output/" + exp_name + "/"
    seed = 42
    model_name = "tf_efficientnet_b0"
    batch_size = 16
    epochs = 2
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


class CustomModel(nn.Module):
    def __init__(self, cfg=CFG, n_class=5, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name, pretrained=pretrained, num_classes=0
        )

        self.in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(self.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.Linear(512, n_class),
        )

    def feature(self, x):
        features = self.backbone(x)
        return features

    def forward(self, x):
        backbone_features = self.feature(x)
        x = self.head(backbone_features)
        output = x
        return output


def get_score(y_true, y_pred):
    score = f1_score(y_true, y_pred, average="macro")
    return score


def get_features(
    loader,
    model,
):
    model.eval()
    features = []
    labels = []
    t = tqdm(loader, total=len(loader))
    for _, data in enumerate(t):
        image, label = data

        image = image.to(CFG.device)
        with torch.no_grad():
            feature = model.feature(image)
        features.append(feature.to("cpu").numpy())
        labels.append(label.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    return features, labels


def train_svc(features, labels):
    model = svm.SVC(C=10, kernel="linear")
    model.fit(features, labels)

    return model


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds["fold"] != fold].reset_index(drop=True)
    valid_folds = folds[folds["fold"] == fold].reset_index(drop=True)
    valid_labels = valid_folds["categories"].values

    train_dataset = CatDataset(CFG, train_folds, transform=train_aug)
    valid_dataset = CatDataset(CFG, valid_folds, transform=val_aug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = CustomModel(CFG, pretrained=False)
    model.load_state_dict(
        torch.load(CFG.output_dir + CFG.model_name + "_fold" + str(fold) + "_best.pth")
    )
    model.to(CFG.device)

    features, labels = get_features(train_loader, model)
    svc_model = train_svc(features, labels)

    val_features, _ = get_features(valid_loader, model)
    val_preds = svc_model.predict(val_features)
    valid_folds["pred"] = val_preds

    joblib.dump(
        svc_model,
        CFG.output_dir + f"{CFG.model_name.replace('/', '-')}_SVC_fold{fold}_best.pkl",
    )

    return valid_folds


if __name__ == "__main__":
    make_dir(CFG)
    seed_everything(CFG.seed)
    LOGGER = get_logger(CFG.output_dir + "train")

    cat_categories, train_df = get_csv()
    str_to_int = {
        "ragdoll": 0,
        "siamese": 1,
        "domestic_shorthair": 2,
        "bengal": 3,
        "maine_coon": 4,
    }

    Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(
        Fold.split(train_df, train_df["categories"])
    ):
        train_df.loc[val_index, "fold"] = int(n)
    train_df["fold"] = train_df["fold"].astype(int)

    train_df["fold"].value_counts()

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_aug = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_aug = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def get_result(oof_df):
        labels = oof_df["categories"]
        pred = oof_df["pred"]
        score = get_score(labels, pred)
        LOGGER.info(f"Score: {score:<.4f}")

    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(train_df, fold)
            oof_df = pd.concat([oof_df, _oof_df])
    oof_df = oof_df.reset_index(drop=True)
    oof_df = oof_df.replace(str_to_int)

    get_result(oof_df)
    oof_df.to_pickle(CFG.output_dir + f"{CFG.exp_name}_oof_df.pkl")
