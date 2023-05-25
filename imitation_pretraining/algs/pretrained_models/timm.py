"""Encode features with timm."""
from typing import Dict
import jax.numpy as jnp
import numpy as np
from PIL import Image

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TIMMLearner:
    """Pre-trained timm representations."""

    def __init__(self, config, batch) -> None:
        del batch
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = timm.create_model(
            config["model_name"],
            pretrained=True,
            num_classes=0,  # Gives us flattened/pooled output features
        )
        self.model.eval()
        self.model.to(self.device)

        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)

    def encode(self, observation: Dict) -> jnp.ndarray:
        """Encode observation using the feature extractor."""
        images = np.array(observation["pixels"])

        # Process images one at a time, this is suboptimal, but we only do it once
        embeddings = []
        for im in images:
            image = self.transform(Image.fromarray(im.astype(np.uint8)))
            image = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embed = self.model(image).cpu().numpy()
                embeddings.append(embed)
        embeddings = np.concatenate(embeddings, axis=0)
        if embeddings.shape[0] != 1:
            print("Successful embedding: ", embeddings.shape, embeddings.dtype)
        return jnp.array(embeddings, dtype=jnp.float32)
