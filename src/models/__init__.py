from types import SimpleNamespace

import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(
    opts: SimpleNamespace,
    tasks_outputs: dict[str, int],
    num_inp_feats: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    backbone_name = getattr(opts, "backbone", "resnet50")
    getattr(opts, "head", "unet")
    segment_model = getattr(opts, "segment_model", "unet")
    getattr(opts, "resize_size", 512)
    num_classes = tasks_outputs.get("water_mask", 1)

    weights = "imagenet" if pretrained else None

    if segment_model == "fpn":
        model = smp.FPN(
            encoder_name=backbone_name, encoder_weights=weights, in_channels=num_inp_feats, classes=num_classes
        )
    elif segment_model == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name=backbone_name, encoder_weights=weights, in_channels=num_inp_feats, classes=num_classes
        )
    elif segment_model == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=backbone_name, encoder_weights=weights, in_channels=num_inp_feats, classes=num_classes
        )
    else:
        model = smp.Unet(
            encoder_name=backbone_name, encoder_weights=weights, in_channels=num_inp_feats, classes=num_classes
        )

    return model


__all__ = ["get_model"]
