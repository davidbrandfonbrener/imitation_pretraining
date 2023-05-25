"""Encode features with r3m."""
from typing import Dict
import jax.numpy as jnp

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from r3m import load_r3m


class R3MLearner:
    """Pre-trained R3M representations."""

    def __init__(self, config, batch) -> None:
        del batch
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = load_r3m(config["model_name"])  # resnet18, resnet34, resnet50
        self.model.eval()
        self.model.to(self.device)

        self.transforms = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),  # ToTensor() divides by 255
            ]
        )

    def encode(self, observation: Dict) -> jnp.ndarray:
        """Encode observation using the feature extractor."""
        images = np.array(observation["pixels"])

        # Process images one at a time, this is suboptimal, but we only do it once
        embeddings = []
        for im in images:
            image = self.transforms(Image.fromarray(im.astype(np.uint8)))
            image = image.reshape(-1, 3, 224, 224).to(self.device)
            image = image * 255.0  # R3M expects image input to be [0-255]
            with torch.no_grad():
                embed = self.model(image).cpu().numpy()
                embeddings.append(embed)
        embeddings = np.concatenate(embeddings, axis=0)
        if embeddings.shape[0] != 1:
            print("Successful embedding: ", embeddings.shape, embeddings.dtype)
        return jnp.array(embeddings, dtype=jnp.float32)
