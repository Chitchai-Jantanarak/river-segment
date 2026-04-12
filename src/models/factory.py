from __future__ import annotations

from types import SimpleNamespace

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from loguru import logger

from src.models.satlas import FPN, SatlasHead, SatlasModel, ProxySatlas, Upsample
from src.models.base import MultiTaskModel, SingleTaskModel


_SATLAS_CHANNELS = 128

_SMP_BACKBONES = {
    "resnet50", "mobilenet_v2",
    "resnet50_mocov3", "resnet50_seco",
    "swint", "swinb",
    "satlas_si_swinb", "satlas_si_swint", "satlas_si_resnet50",
    "satlas_mi_swinb",
}


class ModelwithAdaptor(nn.Module):
    def __init__(self, adaptor: str, backbone: nn.Module, num_inp_feats: int = 4, out_channels: int = 3) -> None:
        super().__init__()
        assert adaptor in ("linear", "drop", "no_init")
        if adaptor == "linear" and ((num_inp_feats != 3) or (out_channels != num_inp_feats)):
            logger.debug("Using Conv2D adaptor")
            self.adaptor: nn.Module = nn.Conv2d(num_inp_feats, out_channels, 1)
        else:
            logger.debug("Using identity adaptor")
            self.adaptor = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.adaptor(x))


def _get_head(head: str, backbone_channels: int | None, task_output: int) -> nn.Module:
    if head == "satlas_head":
        assert backbone_channels is not None
        return SatlasHead(backbone_channels=backbone_channels, out_channels=task_output)
    if head in ("unet_head", "no_head"):
        return nn.Identity()
    raise ValueError(f"Unknown head: {head}")


def _build_satlas_backbone(backbone: str, num_inp_feats: int, pretrained: bool) -> tuple[nn.Module, int]:
    name_map = {
        "satlas_si_swinb": "Sentinel2_SwinB_SI_RGB",
        "satlas_mi_swinb": "Sentinel2_SwinB_MI_RGB",
        "satlas_si_swint": "Sentinel2_SwinT_SI_RGB",
        "satlas_si_resnet50": "Sentinel2_Resnet50_SI_RGB",
    }
    model_name = name_map[backbone]
    if pretrained:
        model = SatlasModel(num_inp_feats=num_inp_feats, model_name=model_name)
    else:
        logger.warning(f"Building {backbone} architecture without SatlasPretrain weights")
        model = _build_proxy_satlas(model_name, num_inp_feats)
    return model, _SATLAS_CHANNELS


def _build_proxy_satlas(model_name: str, num_inp_feats: int) -> nn.Module:
    proxy = ProxySatlas(model_identifier=model_name)
    first = nn.Conv2d(num_inp_feats, 3, 1) if num_inp_feats != 3 else nn.Identity()

    class _Wrapped(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = first
            self.backbone = proxy

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(self.first(x))[0]

    return _Wrapped()


def get_model(
    args: SimpleNamespace,
    tasks_outputs: dict[str, int],
    num_inp_feats: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    backbone_name: str = getattr(args, "backbone", "resnet50")
    head_name: str = getattr(args, "head", "no_head")
    segment_model: str = getattr(args, "segment_model", "fpn")
    adaptor: str = getattr(args, "adaptor", "linear")
    method: str = getattr(args, "method", "vanilla")

    backbone_channels: int | None = None
    segmodel_in_channels = 3

    if adaptor == "no_init":
        segmodel_in_channels = num_inp_feats
        pretrained = False
    if adaptor == "drop":
        num_inp_feats = 3

    logger.debug(f"num_inp_feats={num_inp_feats} backbone={backbone_name} head={head_name} seg={segment_model}")

    if segment_model in ("deeplabv3", "deeplabv3plus", "unet", "fpn"):
        weights = "imagenet" if pretrained else None

        if backbone_name == "resnet50":
            smp_model = smp.create_model(arch=segment_model, encoder_name="resnet50", encoder_weights=weights, in_channels=segmodel_in_channels, classes=1)
            backbone: nn.Module = ModelwithAdaptor(adaptor=adaptor, backbone=smp_model, num_inp_feats=num_inp_feats)
            head_name = "no_head"

        elif backbone_name == "resnet50_mocov3":
            smp_model = smp.create_model(arch=segment_model, encoder_name="resnet50", encoder_weights="imagenet", in_channels=segmodel_in_channels, classes=1)
            sd = torch.load("checkpoints/moco_v3/r-50-100ep.pth.tar")["state_dict"]
            new_sd = {k[len("module.base_encoder."):]: v for k in sd if k.startswith("module.base_encoder") and not k.startswith("module.base_encoder.fc")}
            smp_model.encoder.load_state_dict(new_sd, strict=True)
            backbone = ModelwithAdaptor(adaptor=adaptor, backbone=smp_model, num_inp_feats=num_inp_feats)
            head_name = "no_head"

        elif backbone_name == "mobilenet_v2":
            smp_model = smp.create_model(arch=segment_model, encoder_name="mobilenet_v2", encoder_weights=weights, in_channels=segmodel_in_channels, classes=1)
            backbone = ModelwithAdaptor(adaptor=adaptor, backbone=smp_model, num_inp_feats=num_inp_feats)
            head_name = "no_head"

        elif backbone_name == "swint":
            smp_model = smp.create_model(arch=segment_model, encoder_name="tu-swin_s3_tiny_224.ms_in1k", encoder_weights=weights, in_channels=segmodel_in_channels, classes=1)
            backbone = ModelwithAdaptor(adaptor=adaptor, backbone=smp_model, num_inp_feats=num_inp_feats)
            head_name = "no_head"

        elif backbone_name == "swinb":
            smp_model = smp.create_model(arch=segment_model, encoder_name="tu-swin_s3_base_224.ms_in1k", encoder_weights=weights, in_channels=segmodel_in_channels, classes=1)
            backbone = ModelwithAdaptor(adaptor=adaptor, backbone=smp_model, num_inp_feats=num_inp_feats)
            head_name = "no_head"

        elif backbone_name in ("satlas_si_swinb", "satlas_mi_swinb", "satlas_si_swint", "satlas_si_resnet50"):
            if segment_model in ("deeplabv3plus", "unet"):
                raise NotImplementedError(f"{backbone_name} does not support {segment_model}")
            backbone, backbone_channels = _build_satlas_backbone(backbone_name, num_inp_feats, pretrained)

        else:
            raise ValueError(f"Unknown backbone for {segment_model}: {backbone_name}")

    elif segment_model == "dpt":
        weights = "imagenet" if pretrained else None
        _dpt_map = {
            "vitb": "tu-vit_base_patch16_224.orig_in21k",
            "vitb_dino": "tu-vit_base_patch16_224.dino",
            "vitb_clip": "tu-vit_base_patch16_clip_224.laion2b",
            "vitl": "tu-vit_large_patch16_224.orig_in21k",
        }
        enc_name = _dpt_map.get(backbone_name)
        if enc_name is None:
            raise ValueError(f"Unknown DPT backbone: {backbone_name}")
        smp_model = smp.create_model(arch="dpt", encoder_name=enc_name, encoder_weights=weights, in_channels=segmodel_in_channels, classes=1)
        backbone = ModelwithAdaptor(adaptor=adaptor, backbone=smp_model, num_inp_feats=num_inp_feats)
        head_name = "no_head"

    else:
        if backbone_name in ("satlas_si_swinb", "satlas_mi_swinb", "satlas_si_swint", "satlas_si_resnet50"):
            backbone, backbone_channels = _build_satlas_backbone(backbone_name, num_inp_feats, pretrained)
        else:
            raise ValueError(f"Unknown segment_model: {segment_model}")

    tasks: list[str] = getattr(args, "tasks", ["water_mask"])

    if method == "single-task":
        task = getattr(args, "task", tasks[0])
        head = _get_head(head_name, backbone_channels, tasks_outputs[task])
        return SingleTaskModel(backbone, head, task)

    selected = {t: tasks_outputs[t] for t in tasks if t in tasks_outputs}
    heads = nn.ModuleDict({t: _get_head(head_name, backbone_channels, out) for t, out in selected.items()})
    return MultiTaskModel(backbone, heads, list(selected.keys()))
