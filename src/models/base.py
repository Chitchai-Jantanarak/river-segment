import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.out_fns import get_outfns


class SingleTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: nn.Module, task: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.task = task
        self.outfns = get_outfns([task])

    def forward(self, x: torch.Tensor, feat: bool = False):
        out_size = x.size()[2:]
        feats = self.backbone(x)
        if isinstance(feats, list):
            feats = feats[-1]
        if feats.size()[2:] == out_size:
            out = self.decoder(feats)
        else:
            out = F.interpolate(self.decoder(feats), out_size, mode="bilinear", align_corners=True)
        result = {self.task: self.outfns[self.task](out)}
        if feat:
            return result, feats
        return result

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MultiTaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list[str]) -> None:
        super().__init__()
        assert set(decoders.keys()) == set(tasks)
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.outfns = get_outfns(tasks)

    def forward(self, x: torch.Tensor, feat: bool = False):
        out_size = x.size()[2:]
        shared = self.backbone(x)
        feats = shared
        if isinstance(shared, list):
            feats = shared
            shared = shared[-1]

        if shared.size()[2:] == out_size:
            result = {t: self.outfns[t](self.decoders[t](shared)) for t in self.tasks}
        else:
            result = {
                t: self.outfns[t](
                    F.interpolate(self.decoders[t](shared), out_size, mode="bilinear", align_corners=True)
                )
                for t in self.tasks
            }
        if feat:
            return result, feats
        return result
