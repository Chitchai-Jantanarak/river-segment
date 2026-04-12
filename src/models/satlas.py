from __future__ import annotations

import collections
from io import BytesIO

import requests
import satlaspretrain_models
import torch
import torch.nn as nn
import torchvision
from loguru import logger


def _adjust_prefix(state_dict: dict, needed: str, prefix: str | None = None, prefix_allowed_count: int | None = None) -> dict:
    out = {}
    for key, value in state_dict.items():
        if needed not in key:
            continue
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)
        out[key] = value
    return out


def _adjust_prefix_modified(state_dict: dict, needed: str, prefix: str | None = None, prefix_allowed_count: int | None = None) -> dict:
    out = {}
    for key, value in state_dict.items():
        if (needed in key) or (needed == "upsample"):
            if prefix is not None:
                while key.count(prefix) > prefix_allowed_count:
                    if needed == "fpn":
                        key = key.replace(prefix, "", 1)
                        key = "fpn." + key
                    elif needed == "upsample":
                        key = key.replace(prefix, "upsample.", 1)
        out[key] = value
    return out


class FPN(nn.Module):
    def __init__(self, backbone_channels: list, out_channels: int) -> None:
        super().__init__()
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict([("feat{}".format(i), el) for i, el in enumerate(x)])
        return list(self.fpn(inp).values())


class Upsample(nn.Module):
    def __init__(self, backbone_channels: list) -> None:
        super().__init__()
        self.in_channels = backbone_channels
        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels
        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch // 2, out_channels)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            ch = next_ch
            depth //= 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return [self.layers(x[0])] + x


class SwinBackbone(nn.Module):
    def __init__(self, num_channels: int, arch: str) -> None:
        super().__init__()
        if arch == "swinb":
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [[4, 128], [8, 256], [16, 512], [32, 1024]]
        elif arch == "swint":
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [[4, 96], [8, 192], [16, 384], [32, 768]]
        else:
            raise ValueError(f"Unsupported arch: {arch}")
        self.backbone.features[0][0] = nn.Conv2d(
            num_channels,
            self.backbone.features[0][0].out_channels,
            kernel_size=(4, 4),
            stride=(4, 4),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class ResnetBackbone(nn.Module):
    def __init__(self, num_channels: int, arch: str = "resnet50") -> None:
        super().__init__()
        if arch == "resnet50":
            self.resnet = torchvision.models.resnet50(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == "resnet152":
            self.resnet = torchvision.models.resnet152(weights=None)
            ch = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported arch: {arch}")
        self.resnet.conv1 = nn.Conv2d(num_channels, self.resnet.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.out_channels = [[4, ch[0]], [8, ch[1]], [16, ch[2]], [32, ch[3]]]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        l1 = self.resnet.layer1(x)
        l2 = self.resnet.layer2(l1)
        l3 = self.resnet.layer3(l2)
        l4 = self.resnet.layer4(l3)
        return [l1, l2, l3, l4]


class ProxySatlas(nn.Module):
    def __init__(self, model_identifier: str) -> None:
        super().__init__()
        self.model_identifier = model_identifier
        if model_identifier == "Sentinel2_Resnet50_SI_RGB":
            self.backbone = ResnetBackbone(num_channels=3)
        elif model_identifier == "Sentinel2_SwinB_SI_RGB":
            self.backbone = SwinBackbone(num_channels=3, arch="swinb")
            out_channels = [[4, 128], [8, 256], [16, 512], [32, 1024]]
            self.fpn = FPN(out_channels, 128)
            self.upsample = Upsample(self.fpn.out_channels)
        elif model_identifier == "Sentinel2_SwinT_SI_RGB":
            self.backbone = SwinBackbone(num_channels=3, arch="swint")
        else:
            raise ValueError(f"Unknown model: {model_identifier}")

    def forward(self, x: torch.Tensor):
        if self.model_identifier == "Sentinel2_SwinB_SI_RGB":
            return self.upsample(self.fpn(self.backbone(x)))
        return self.backbone(x)


class SatlasModel(nn.Module):
    def __init__(self, num_inp_feats: int = 6, fpn: bool = True, model_name: str = "Sentinel2_SwinB_SI_RGB") -> None:
        super().__init__()
        self.first = nn.Conv2d(num_inp_feats, 3, 1) if num_inp_feats != 3 else nn.Identity()

        weights_manager = satlaspretrain_models.Weights()

        if model_name == "Sentinel2_SwinB_SI_RGB":
            try:
                logger.debug("Loading Satlas SwinB weights")
                self.backbone = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=fpn)
                self.backbone_channels = self.backbone.upsample.layers[-1][-2].out_channels
            except Exception as e:
                logger.warning(f"weights_manager failed: {e}")
                model = ProxySatlas(model_identifier=model_name)
                weights_url = "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth?download=true"
                response = requests.get(weights_url)
                if response.status_code == 200:
                    weights_file: str | BytesIO = BytesIO(response.content)
                else:
                    logger.warning(f"Using local file. Download failed: {weights_url}")
                    weights_file = "checkpoints/satlas/sentinel2_swinb_si_rgb.pth"
                weights = torch.load(weights_file, map_location="cpu")
                fpn_sd = _adjust_prefix_modified(weights, "fpn", "intermediates.0.", 0)
                fpn_sd = _adjust_prefix_modified(fpn_sd, "upsample", "intermediates.1.", 0)
                model.load_state_dict(fpn_sd, strict=False)
                self.backbone = model
                self.backbone_channels = 128

        elif model_name == "Sentinel2_SwinT_SI_RGB":
            try:
                model = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=False)
            except Exception as e:
                logger.warning(f"weights_manager failed: {e}")
                model = ProxySatlas(model_identifier=model_name)
                local_weights = torch.load("checkpoints/satlas/sentinel2_swint_si_rgb.pth", map_location="cpu")
                model.load_state_dict(local_weights, strict=False)
            out_channels = [[4, 96], [8, 192], [16, 384], [32, 768]]
            model_fpn = FPN(out_channels, 128)
            if fpn:
                weights_url = "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true"
                response = requests.get(weights_url)
                weights_file = BytesIO(response.content) if response.status_code == 200 else "checkpoints/satlas/sentinel2_swint_si_rgb.pth"
                weights = torch.load(weights_file, map_location="cpu")
                model_fpn.load_state_dict(_adjust_prefix(weights, "fpn", "intermediates.0.", 0), strict=True)
            self.backbone = nn.Sequential(model, model_fpn, Upsample(model_fpn.out_channels))
            self.backbone_channels = 128

        elif model_name == "Sentinel2_Resnet50_SI_RGB":
            try:
                model = weights_manager.get_pretrained_model(model_identifier=model_name, fpn=False)
                model.backbone.freeze_bn = False
            except Exception as e:
                logger.warning(f"weights_manager failed: {e}")
                model = ProxySatlas(model_identifier=model_name)
                local_weights = torch.load("checkpoints/satlas/sentinel2_resnet50_si_rgb.pth", map_location="cpu")
                model.load_state_dict(local_weights, strict=False)
            out_channels = [[4, 256], [8, 512], [16, 1024], [32, 2048]]
            model_fpn = FPN(out_channels, 128)
            if fpn:
                weights_url = "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true"
                response = requests.get(weights_url)
                weights_file = BytesIO(response.content) if response.status_code == 200 else "checkpoints/satlas/sentinel2_resnet50_si_rgb.pth"
                weights = torch.load(weights_file, map_location="cpu")
                model_fpn.load_state_dict(_adjust_prefix(weights, "fpn", "intermediates.0.", 0), strict=True)
            self.backbone = nn.Sequential(model, model_fpn, Upsample(model_fpn.out_channels))
            self.backbone_channels = 128

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        x = self.backbone(x)
        return x[0]


class SatlasHead(nn.Module):
    def __init__(self, backbone_channels: int, out_channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(1):
            layers.append(nn.Sequential(nn.Conv2d(backbone_channels, backbone_channels, 3, padding=1), nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(backbone_channels, out_channels, 3, padding=1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
