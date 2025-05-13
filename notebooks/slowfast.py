import torch
import torch.nn as nn
import typing as t

"""

Fast [8, 8, 32, 244, 244] (input)
Slow [8, 64, 4, 244, 244] (output)

"""


def init_weights(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm3d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            torch.nn.init.constant_(m.bias, 0)


class BlockConfig(t.TypedDict):
    kernel: t.Optional[t.Union[int, t.Tuple[int, int, int]]]
    stride: t.Optional[t.Union[int, t.Tuple[int, int, int]]]
    padding: t.Optional[t.Union[int, t.Tuple[int, int, int]]]


class Fusion3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = None,
        stride: int = None,
        **kwargs
    ):
        """Fusion 3D tensors

        Fusion tensor with different channel and resolution sizes. When we call it, the `x` is source tensor [B, T, C, H, W], which should to
        be projected to `y` tensor shape [B, T', C', H, W].

        Args:
            in_channels (int): The source channel number
            out_channels (int): The target channel number
            kernel (int, optional): The kernel size to projected the tensors resolutions
            stride (int, optional): The stride to projected the tensors resolutions
        """
        super(Fusion3D, self).__init__()

        self.proj_channels = (
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.proj_resolution = (
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(kernel, 1, 1),
                stride=(stride, 1, 1),
            )
            if kernel is not None and stride is not None
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fusion 3D tensors

        Fusion tensor with different channel and resolution sizes. When we call it, the `x` is source tensor [B, T, C, H, W], which should to
        be projected to `y` tensor shape [B, T', C', H, W].

        Args:
            x (Tensor): The source tensor with shape [B, T, C, H, W]
            y (Tensor): The target tensor with shape [B, T', C', H, W]

        Returns:
            out (Tuple[Tensor, Tensor]): The concated tensors with shape [B, T', 2C', H, W], and projectiled source tensor with shape [B, T', C', H, W]
        """
        projected = self.proj_channels(x)
        projected = self.proj_resolution(projected)

        z = torch.cat([y, projected], dim=1)

        return z, projected


class AttentionFusion3D(Fusion3D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed: int,
        kernel: int = None,
        stride: int = None,
        **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel, stride)

        self.attention = nn.Sequential(
            nn.Conv3d(out_channels * 2, embed, kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(embed, 2, kernel_size=1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z, projected = super().forward(x, y)
        weights = self.attention(z)

        # Split weights: [B, 1, T, H, W] for x and y
        wx = weights[:, 0:1, :, :, :]
        wy = weights[:, 1:2, :, :, :]

        # Weighted sum
        fused = y * wx + projected * wy
        return fused


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: list[BlockConfig],
        depth: int = 1,
        inplace: bool = False,
    ):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self._consruct(config, depth, inplace)

    def _consruct(self, config: list[BlockConfig], depth: int, inplace: bool) -> None:
        layers = []

        for i in range(depth):
            for j, params in enumerate(config):
                layers.extend(
                    [
                        nn.Conv3d(
                            in_channels=self.in_dim,
                            out_channels=(
                                self.in_dim if j != len(config) - 1 else self.out_dim
                            ),
                            kernel_size=params.get("kernel", 1),
                            stride=params.get("stride", 1),
                            padding=params.get("padding", 0),
                            bias=False,
                        ),
                        nn.BatchNorm3d(
                            num_features=(
                                self.in_dim if j != len(config) - 1 else self.out_dim
                            )
                        ),
                        nn.ReLU(inplace=inplace),
                    ]
                )

            if i != depth - 1:
                layers.append(
                    nn.Conv3d(self.out_dim, self.in_dim, kernel_size=1, stride=1),
                )
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


class SlowFastNet(nn.Module):
    def __init__(self, n_outputs: int, fusion: t.Union[t.Type[Fusion3D], t.Type[AttentionFusion3D]]):
        super(SlowFastNet, self).__init__()
        fusion()


def slowfast_r18() -> SlowFastNet:
    pass
