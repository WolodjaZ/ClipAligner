import torch
import torch.nn as nn

from typing import List, Any, Tuple

from .base import BaseImageCaptionModel, BaseImageModel, BaseCaptionModel

class MLPAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x)

class ConvAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self._conv = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self._conv(x)
        x = x.squeeze(2)
        return x


class ResAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self._res = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self._downsample = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_x = self._res(x)
        return self._downsample(res_x + x)


class ClipAligner(BaseImageCaptionModel):
    def __init__(self, 
                 vision_model: BaseImageModel, 
                 caption_model: BaseCaptionModel,
                 vision_layer: List[Tuple[int, str]],
                 caption_layer: List[Tuple[int, str]],
                 init_logit_scale: float = torch.log(torch.tensor(1 / 0.07)),
                 *args, **kwargs) -> None:
        super().__init__(vision_model, caption_model, *args, **kwargs)
        if vision_layer[-1][0] != caption_layer[-1][0]:
            raise ValueError("The last layer of vision and caption must have the same dimension.")

        self._vision_aligner = _build_aligner_module(vision_layer, vision_model.output_dim)
        self._caption_aligner = _build_aligner_module(caption_layer, caption_model.output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self._output_dim = vision_layer[-1][0]

    def get_basic_transformations(self) -> (List[Any], Any):
        return self._vision_model.get_basic_transformations(), self._caption_model.get_tokenizer()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, image: torch.Tensor, caption: torch.Tensor | str | dict) -> torch.Tensor:
        image_embeddings = self._vision_model(image)
        caption_embeddings = self._caption_model(caption)
        image_aligned_embeddings = self._vision_aligner(image_embeddings)
        caption_aligned_embeddings = self._caption_aligner(caption_embeddings)
        return image_aligned_embeddings, caption_aligned_embeddings, self.logit_scale.exp()


def _build_aligner_module(vision_layer: List[Tuple[int, str]], input_dim: int) -> nn.Module:
    prev_dim = input_dim
    layers = []
    for dim, aligner in vision_layer:
        aligner = aligner.lower()
        if aligner == "mlp":
            layers.append(MLPAligner(prev_dim, dim, dropout=0.0, hidden_dim=prev_dim))
        elif aligner == "conv":
            layers.append(ConvAligner(prev_dim, dim, dropout=0.0, hidden_dim=prev_dim))
        elif aligner == "res":
            layers.append(ResAligner(prev_dim, dim, dropout=0.0, hidden_dim=prev_dim))
        else:
            raise ValueError(f"Aligner {aligner} not supported for {ClipAligner.__name__}.")
        prev_dim = dim
    return nn.Sequential(*layers)
        