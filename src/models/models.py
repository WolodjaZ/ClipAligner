import torch
from torchvision import transforms
from loguru import logger
from typing import Dict, Any

from .base import BaseImageModel, BaseCaptionModel

try:
    import transformers # type: ignore
except ImportError:
    transformers = None


class DinoVisionModel(BaseImageModel):
    SHAPES = {
        "dinov2_vits14": (384, 256),
        "dinov2_vitb14": (768, 256),
        "dinov2_vitl14": (1024, 256),
        "dinov2_vitg14": (1536, 256),
    }
    
    def __init__(self, checkpoint: str, pooling: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = torch.hub.load('facebookresearch/dinov2', checkpoint)
        self._pooler, self._output_dim = self._get_pooler_fn(pooling, checkpoint)
        self._transform = [
            transforms.Resize(224, antialias=True), transforms.CenterCrop(224), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def get_transformations(self) -> Dict[str, Any]:
        return {"image": self._transform}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._model.forward_features(x)
        x = self._pooler(output)
        return x
    
    @staticmethod
    def _get_pooler_fn(pooler: str, checkpoint: str) -> callable:
        if pooler == "cls":
            return lambda x: x['x_norm_clstoken'], DinoVisionModel.SHAPES[checkpoint][0]
        elif pooler == "mean":
            return lambda x: x['x_norm_patchtokens'].mean(dim=-1), DinoVisionModel.SHAPES[checkpoint][1]
        elif pooler == "max":
            return lambda x: x['x_norm_patchtokens'].max(dim=-1)[0], DinoVisionModel.SHAPES[checkpoint][1]
        elif pooler == "mean_max":
            return lambda x: torch.cat([x['x_norm_patchtokens'].mean(dim=-1), x['x_norm_patchtokens'].max(dim=-1)[0]], dim=-1), DinoVisionModel.SHAPES[checkpoint][1] * 2
        elif pooler == "cls_mean":
            return lambda x: torch.cat([x['x_norm_clstoken'], x['x_norm_patchtokens'].mean(dim=-1)], dim=-1), DinoVisionModel.SHAPES[checkpoint][0] + DinoVisionModel.SHAPES[checkpoint][1]
        elif pooler == "cls_max":
            return lambda x: torch.cat([x['x_norm_clstoken'], x['x_norm_patchtokens'].max(dim=-1)[0]], dim=-1), DinoVisionModel.SHAPES[checkpoint][0] + DinoVisionModel.SHAPES[checkpoint][1]
        elif pooler == "cls_mean_max":
            return lambda x: torch.cat([x['x_norm_clstoken'], x['x_norm_patchtokens'].mean(dim=-1), x['x_norm_patchtokens'].max(dim=-1)[0]], dim=-1), DinoVisionModel.SHAPES[checkpoint][0] + DinoVisionModel.SHAPES[checkpoint][1] * 2
        else:
            raise ValueError(f"Pooling {pooler} is not supported for {DinoVisionModel.__name__}.")


class RobertaCaptionModel(BaseCaptionModel):
    SHAPES = {
        "roberta-base": 768,
        "roberta-large": 1024,
        "xlm-roberta-base": 250002,
    }
    
    def __init__(self, checkpoint: str, pooling: str, add_pooling_layer: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if transformers is None:
            raise ImportError("Please install transformers library.")
        
        if checkpoint.startswith("xlm"):
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
            self._model = transformers.AutoModelForMaskedLM.from_pretrained(checkpoint)
        else:
            config = transformers.AutoConfig.from_pretrained(checkpoint)
            self._tokenizer = transformers.RobertaTokenizer.from_pretrained(checkpoint)
            self._model = transformers.RobertaModel(config, add_pooling_layer=add_pooling_layer)
        logger.info("Sssssh don't listen to the warning above, Robert can be grumpy.")
        self._pooler, self._output_dim = self._get_pooler_fn(pooling, checkpoint)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_transformations(self) -> Dict[str, Any]:
        return {"caption": self._tokenizer}
    
    def forward(self, x: torch.Tensor | str | dict) -> torch.Tensor:
        if isinstance(x, str):
            x = self._tokenizer(x, return_tensors="pt", padding=True, truncation=True)

        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)

        output = self._model(**x)
        x = self._pooler(output)
        return x

    @staticmethod
    def _get_pooler_fn(pooler: str, checkpoint: str) -> callable:
        key = "logits" if checkpoint.startswith("xlm") else "last_hidden_state"
        if pooler == "cls":
            return lambda x: x[key][:, 0, :], RobertaCaptionModel.SHAPES[checkpoint]
        elif pooler == "mean":
            return lambda x: x[key][:, 1:, :].mean(dim=-1).reshape(x[key].shape[0], -1), RobertaCaptionModel.SHAPES[checkpoint]
        elif pooler == "max":
            return lambda x: (x[key][:, 1:, :].max(dim=1)[0]).reshape(x[key].shape[0], -1), RobertaCaptionModel.SHAPES[checkpoint]
        elif pooler == "mean_max":
            return lambda x: torch.cat([x[key][:, 1:, :].mean(dim=1).reshape(x[key].shape[0], -1), (x[key][:, 1:, :].max(dim=1)[0]).reshape(x[key].shape[0], -1)], dim=-1), RobertaCaptionModel.SHAPES[checkpoint] * 2
        elif pooler == "cls_mean":
            return lambda x: torch.cat([x[key][:, 0, :], x[key][:, 1:, :].mean(dim=1).reshape(x[key].shape[0], -1)], dim=-1), RobertaCaptionModel.SHAPES[checkpoint] * 2
        elif pooler == "cls_max":
            return lambda x: torch.cat([x[key][:, 0, :], (x[key][:, 1:, :].max(dim=1)[0]).reshape(x[key].shape[0], -1)], dim=-1), RobertaCaptionModel.SHAPES[checkpoint] * 2
        elif pooler == "cls_mean_max":
            return lambda x: torch.cat([x[key][:, 0, :], x[key][:, 1:, :].mean(dim=1).reshape(x[key].shape[0], -1), (x[key][:, 1:, :].max(dim=1)[0]).reshape(x[key].shape[0], -1)], dim=-1), RobertaCaptionModel.SHAPES[checkpoint] * 3
        else:
            raise ValueError(f"Pooling {pooler} is not supported for {RobertaCaptionModel.__name__}.")