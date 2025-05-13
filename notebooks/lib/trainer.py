import warnings
import os
import typing
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
from torchvision.io import read_video
from torchvision.transforms import v2
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

warnings.filterwarnings("ignore")


@dataclass
class HParams:

    # Base parameters

    dataset_dir: Path = Path("../data/UCF101")
    """Dataset directory path"""
    train_meta: Path = Path("../data/UCF101/test.csv")
    """Path to meta csv file inforamation for Train Loop"""
    test_meta: Path = Path("../data/UCF101/test.csv")
    """Path to meta csv file inforamation for Test Loop"""
    validation_meta: Path = Path("../data/UCF101/val.csv")
    """Path to meta csv file inforamation for Validation Loop"""
    output_dir: Path = Path("saved_models/")
    """Path to save all output information"""

    # Dataset parameters

    size: typing.Tuple[int, int] = (224, 224)
    """Image size [H, W]"""
    mean: typing.Tuple[float, float, float] = (0.485, 0.456, 0.406)
    """Image normalization parameter: mean"""
    std: typing.Tuple[float, float, float] = (0.229, 0.224, 0.225)
    """Image normalization parameter: std"""
    clip_len: int = 32
    """Video frame count [T]"""
    clip_format: str = "CTHW"
    """Final frame shape like: "TCHW" """
    batch_size: int = 8
    """Batch size [B]"""
    num_workers: int = 2
    """Workers number"""

    # Model parameters

    arch: str = "timesformer"
    """Archetecture model name (meta info)"""
    n_classes: int = 101
    """Number of classes"""
    freeze: bool = True
    """Set requires_grad to False for part of the model, to learn only model head"""
    lr: float = 1e-3
    """Learning rate"""
    ls: float = 0.4
    """Label smoothing"""
    weight_decay: float = 0.0
    """Optimizer param weight_decay"""
    num_epoch: int = 10
    """Number of epoch"""


class VideoDataset(Dataset):

    def __init__(
        self,
        dir: Path,
        meta: Path,
        clip_len: int,
        transform: v2.Transform = None,
        output_format: str = "TCHW",
    ) -> None:
        """Dataset class to load UCF101

        Args:
            dir (Path): Path to the directory with video files.
            meta (Path): Path to file with information of video [clip_name, clip_path, label] in csv format
            clip_len (int): The number of frames per video
            transform (Transform, optional): Optional transform to be applied on a sample
            output_format (str, optional): The format of the output video tensors. Can be either "TCHW" (default) or differ combination.
        """

        self.dir = dir
        self.clip_len = clip_len
        self.transform = transform
        self.output_format = output_format

        df = pd.read_csv(meta)

        labels = sorted(df["label"].unique())

        self._map_label2idx = {l: i for i, l in enumerate(labels)}
        self._map_idx2label = {i: l for i, l in enumerate(labels)}

        self.labels = df["label"].to_numpy()
        self.paths = df["clip_path"].to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def _clip_sampler(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[0] < self.clip_len:
            padding_size = self.clip_len - frames.shape[0]
            last_frame = frames[-1].unsqueeze(0)
            padded_video = torch.cat(
                [frames, last_frame.repeat(padding_size, 1, 1, 1)], dim=0
            )
            return padded_video
        else:
            padding_size = frames.shape[0] - self.clip_len
            start_idx = np.random.randint(0, padding_size + 1)
            return frames[start_idx : start_idx + self.clip_len]

    def _clip_format(self, frames: torch.Tensor) -> torch.Tensor:
        f_idx = {"T": 0, "C": 1, "H": 2, "W": 3}
        transpose_idx = [f_idx[i] for i in self.output_format]
        return frames.permute(*transpose_idx)

    def __getitem__(self, idx) -> typing.Tuple[torch.Tensor, int, int]:
        label = self.labels[idx]
        path = self.paths[idx][1:]

        frames, *_ = read_video(os.path.join(self.dir, path), output_format="TCHW")
        frames = frames.float() / 255
        frames = self._clip_sampler(frames)

        if self.transform is not None:
            frames = self.transform(frames)

        frames = self._clip_format(frames)

        return frames, self._map_label2idx[label], idx


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, params: HParams) -> None:
        super().__init__()
        self.params = params

        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=params.size),
                v2.Normalize(mean=params.mean, std=params.std),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train = VideoDataset(
            self.params.dataset_dir,
            self.params.train_meta,
            self.params.clip_len,
            self.transform,
            self.params.clip_format,
        )

        self.test = VideoDataset(
            self.params.dataset_dir,
            self.params.test_meta,
            self.params.clip_len,
            self.transform,
            self.params.clip_format,
        )

        self.validation = VideoDataset(
            self.params.dataset_dir,
            self.params.validation_meta,
            self.params.clip_len,
            self.transform,
            self.params.clip_format,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=True,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=False,
            pin_memory=False,
        )


class train_model(pl.LightningModule):
    def __init__(self, model: nn.Module = None, params: HParams = None):
        super().__init__()
        self.save_hyperparameters(
            params.__dict__,
            ignore=("dataset_dir", "train_meta", "test_meta", "validation_meta"),
        )
        self.params = params
        self.model = model

        self.accuracy = F1Score(
            task="multiclass", num_classes=params.n_classes, average="micro"
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=params.ls)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc": acc,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log_dict(
            {
                "val_loss": loss,
                "val_acc": acc,
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.accuracy(logits, y)

        self.log_dict(
            {
                "test_loss": loss,
                "test_acc": acc,
            },
            on_step=True,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.params.lr, momentum=0.9, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]

    def on_before_optimizer_step(self, optimizer):
        norm_order = 2.0
        norms = grad_norm(self, norm_type=norm_order)
        self.log(
            "grad_norm",
            norms[f"grad_{norm_order}_norm_total"],
            on_step=True,
            on_epoch=False,
        )
