import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Any, Dict

from .base import BaseImageCaptionModel, BaseImageModel, BaseCaptionModel, activation_resolver

class MLPAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, activation: str = 'relu', normalize: bool = False, *args, **kwargs) -> None:
        super().__init__()
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        activation = activation_resolver(activation)
        normalize = nn.BatchNorm1d(output_dim) if normalize else nn.Identity()
        self._mlp = nn.Sequential(
            dropout.
            nn.Linear(input_dim, output_dim),
            normalize,
            activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
        return self._mlp(x)

class ConvAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, activation: str = 'relu', normalize: bool = False, *args, **kwargs) -> None:
        super().__init__()
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        activation = activation_resolver(activation)
        normalize = nn.BatchNorm1d(output_dim) if normalize else nn.Identity()
        self._conv = nn.Sequential(
            dropout,
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1, stride=1),
            normalize,
            activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self._conv(x)
        
        if self.pooling:
            x = F.avg_pool1d(x, x.size()[2])
        
        return x


class ResAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, activation: str = 'relu', normalize: bool = False, pooling: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation_resolver(activation)
        self.normalize = nn.BatchNorm1d(output_dim) if normalize else nn.Identity()
        self.pooling = pooling
        
        self._linear1 = nn.Linear(input_dim, output_dim)
        self._linear2 = nn.Linear(output_dim, output_dim)
        
        self._match_dimensions = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
    
        x = self.dropout(x)
        identity = x

        out = self.activation(self.normalize(self._linear1(x)))
        out = self.normalize(self._linear2(out))

        identity = self._match_dimensions(identity)

        out += identity
        out = self.activation(out)

        return out

class ResConvAligner(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, activation: str = 'relu', normalize: bool = False, pooling: bool = False, *args, **kwargs) -> None:
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.activation = activation_resolver(activation)
        self.normalize = nn.BatchNorm1d(output_dim) if normalize else nn.Identity()
        self.pooling = pooling
        
        self._conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1, stride=1),
        self._conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1, stride=1),
        
        # Adjusting the skip connection
        self._match_dimensions = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else nn.Identity()


    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Save the original input for the skip connection
        x = self.dropout(x)
        identity = x

        # First Conv1D -> BatchNorm -> ReLU
        out = self.activation(self.normalize(self._conv1(x)))

        # Second Conv1D -> BatchNorm
        out = self.normalize(self._conv2(out))

        # Adjusting the dimensions of the skip connection if needed
        identity = self._match_dimensions(identity)

        # Adding the skip connection
        out += identity
        out = self.activation(out)
        
        # Pooling
        if self.pooling:
            x = F.avg_pool1d(x, x.size()[2])

        return out

class ClipAligner(BaseImageCaptionModel):
    def __init__(self, 
                 vision_model: BaseImageModel, 
                 caption_model: BaseCaptionModel,
                 vision_layer: List[Dict[str, Any]],
                 caption_layer: List[Dict[str, Any]],
                 init_logit_scale: float = torch.log(torch.tensor(1 / 0.07)),
                 *args, **kwargs) -> None:
        super().__init__(vision_model, caption_model, *args, **kwargs)
        if vision_layer[-1]["output"] != caption_layer[-1]["output"]:
            raise ValueError("The last layer of vision and caption must have the same dimension.")

        self._vision_aligner = _build_aligner_module(vision_layer, vision_model.output_dim)
        self._caption_aligner = _build_aligner_module(caption_layer, caption_model.output_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        self._output_dim = vision_layer[-1]["output"]

    def get_transformations(self) -> (Any, Any):
        return {
            "image": self._vision_model.get_transformations()['image'],
            "caption": self._caption_model.get_transformations()['caption']
        }

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, image: torch.Tensor, caption: torch.Tensor | str | dict) -> torch.Tensor:
        image_embeddings = self._vision_model(image)
        caption_embeddings = self._caption_model(caption)

        image_aligned_embeddings = self._vision_aligner(image_embeddings)
        if len(image_aligned_embeddings.shape) == 3:
            image_aligned_embeddings = torch.flatten(image_aligned_embeddings, start_dim=1)

        caption_aligned_embeddings = self._caption_aligner(caption_embeddings)
        if len(caption_aligned_embeddings.shape) == 3:
            caption_aligned_embeddings = torch.flatten(caption_aligned_embeddings, start_dim=1)

        return image_aligned_embeddings, caption_aligned_embeddings, self.logit_scale.exp()


def _build_aligner_module(layers: List[Dict[str, Any]], input_dim: int) -> nn.Module:
    prev_dim = None
    
    layers = []
    for layer_args in layers:
        aligner = layer_args.pop("name")
        output = layer_args.pop("output")
        if aligner == "mlp":
            prev_dim = input_dim if prev_dim is None else prev_dim
            layers.append(MLPAligner(prev_dim, output, **layer_args))
        elif aligner == "conv":
            layers.append(ConvAligner(prev_dim, output, **layer_args))
        elif aligner == "res":
            layers.append(ResAligner(prev_dim, output, **layer_args))
        elif aligner == "res_conv":
            layers.append(ResConvAligner(prev_dim, output, **layer_args))
        else:
            raise ValueError(f"Aligner {aligner} not supported for {ClipAligner.__name__}.")
        prev_dim = output
    return nn.Sequential(*layers)
        