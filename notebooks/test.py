import torch
import typing
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm.notebook import tqdm


class OptimizerConfig(typing.TypedDict):
    optimizer: Optimizer
    scheduler: typing.Optional[LRScheduler]


BatchType = typing.Union[typing.Tuple[torch.Tensor, ...], torch.Tensor]


class Logger:
    pass


class DataModule:

    def setup(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def valid_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

    def init(self) -> None:
        self.setup()
        self.prepare()


class Agent(Logger, torch.nn.Module):

    def training_augmentation_step(
        self, batch: BatchType, batch_idx: int
    ) -> typing.Tuple[BatchType, int]:
        """Function `training_augmentation_step`.

        You can rewrite this function to add augmentation to your training data

        Args:
            batch (BatchType): Dataloader batch, it can be simple tensor or tuple with tensors
            batch_idx (int): Dataloader batch index

        Returns:
            batch (BatchType): Dataloader augmented batch, it can be simple tensor or tuple with tensors.

            batch_idx (int): Dataloader batch index
        """
        return batch, batch_idx

    def training_step(
        self, batch: BatchType, batch_idx: int
    ) -> typing.Union[
        torch.Tensor, typing.Dict[str, typing.Union[torch.Tensor, typing.Any]]
    ]:
        raise NotImplementedError("function `training_step` should to be implemented")

    def validation_step(
        self, batch: BatchType, batch_idx: int
    ) -> typing.Union[
        torch.Tensor, typing.Dict[str, typing.Union[torch.Tensor, typing.Any]]
    ]:
        raise NotImplementedError("function `validation_step` should to be implemented")

    def test_step(
        self, batch: BatchType, batch_idx: int
    ) -> typing.Union[
        torch.Tensor, typing.Dict[str, typing.Union[torch.Tensor, typing.Any]]
    ]:
        raise NotImplementedError("function `test_step` should to be implemented")

    def configure_optimizer(self) -> OptimizerConfig:
        pass


class Trainer:

    def __init__(
        self,
        max_epochs: int,
        device: typing.Literal["cpu", "gpu", "auto"] = "auto",
        include_test: bool = False,
    ) -> None:

        self.device = self.__select_device(device)
        self.max_epochs = max_epochs
        self.include_test = include_test

        self.__train_step = 0
        self.__validation_step = 0
        self.__test_step = 0
        self.__epoch = 0

    def __select_device(
        self, t_device: typing.Literal["cpu", "gpu", "auto"]
    ) -> torch.device:
        if t_device == "cpu":
            return torch.device("cpu")
        elif t_device == "gpu":
            return torch.device("cuda")
        elif t_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError(f"The device {t_device} is not 'gpu', 'cpu', or 'auto'")

    def _reset(self) -> None:
        self.__train_step = 0
        self.__validation_step = 0
        self.__test_step = 0
        self.__epoch = 0

    def __batch_hook(
        self, batch: BatchType, batch_idx: int
    ) -> typing.Tuple[BatchType, int]:
        if isinstance(batch, tuple):
            t_batch = [el.to(self.device) for el in batch]
            return tuple(t_batch), batch_idx
        return batch.to(self.device), batch_idx

    def _train_epoch(
        self, model: Agent, loader: DataLoader, epoch: int
    ) -> torch.Tensor:
        model.train()

        loop = tqdm(
            iterable=loader, desc=f"Training {epoch}", total=len(loader), leave=True
        )
        batch_size = loader.batch_size or 1
        torch.no_grad()
        for batch_idx, batch in enumerate(loop):
            self.__train_step += batch_size
            batch, batch_idx = self.__batch_hook(batch, batch_idx)
            batch, batch_idx = model.training_augmentation_step(batch, batch_idx)
            # response = model.training_step(batch, batch_idx)

    def _valid_epoch(
        self, model: Agent, loader: DataLoader, epoch: int
    ) -> torch.Tensor:
        model.eval()

    def _test_epoch(self, model: Agent, loader: DataLoader, epoch: int) -> torch.Tensor:
        model.eval()

    def _loop(self) -> None:
        pass

    def fit(self, model: Agent, data: DataModule) -> None:
        self._reset()
        data.init()

        for epoch in range(self.max_epochs):
            self.__epoch = epoch

            # train_loss = self._train_epoch(model, data.train_dataloader(), epoch)

            # valid_loss = self._valid_epoch(model, data.valid_dataloader(), epoch)

            # test_loss = self._test_epoch(model, data.test_dataloader(), epoch)


class AttentionFusion3D(nn.Module):
    def __init__(self, x_channels: int, y_channels: int, embed_channels: int):
        super(AttentionFusion3D, self).__init__()

        # Project both paths to same shape if needed
        self.align = (
            nn.Conv3d(y_channels, x_channels, kernel_size=(5, 1, 1), stride=(8, 1, 1))
            if y_channels != x_channels
            else nn.Identity()
        )

        # Attention mechanism (Conv3D version)
        self.attention = nn.Sequential(
            nn.Conv3d(x_channels * 2, embed_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(
                embed_channels, 2, kernel_size=1
            ),  # Output: attention map for x and y
            nn.Softmax(dim=1),  # Softmax over 2 branches (not channels)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        y: [B, C' or C, T, H, W]
        """
        y_aligned = self.align(y)  # Ensure both have same channel dim

        # Concatenate along channel dimension: [B, 2C, T, H, W]
        concat = torch.cat([x, y_aligned], dim=1)

        # Compute attention weights: [B, 2, T, H, W]
        weights = self.attention(concat)

        # Split weights: [B, 1, T, H, W] for x and y
        wx = weights[:, 0:1, :, :, :]
        wy = weights[:, 1:2, :, :, :]

        # Weighted sum
        fused = x * wx + y_aligned * wy
        return fused
