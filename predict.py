"""
download the models to ./weights
wget https://openaipublic.azureedge.net/main/point-e/base_40m_imagevec.pt -O base40M-imagevec.pt
wget https://openaipublic.azureedge.net/main/point-e/base_40m_textvec.pt  -O base40M-textvec.pt
wget https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt  -O upsample_40m.pt
wget https://openaipublic.azureedge.net/main/point-e/base_40m.pt -O base40M.pt
"""

from PIL import Image
import os
from typing import Any, List

import torch
from tqdm.auto import tqdm
from cog import BasePredictor, Input, Path, BaseModel

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint

from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud


class ModelOutput(BaseModel):
    samples: Any
    figure: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("creating base model...")
        base_name = "base40M-textvec"
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print("creating img2pointcloud base model...")
        base_name_img = "base40M"  # use base300M or base1B for better results
        base_model_img = model_from_config(MODEL_CONFIGS[base_name_img], device)
        base_model_img.eval()
        base_diffusion_img = diffusion_from_config(DIFFUSION_CONFIGS[base_name_img])

        print("creating upsample model...")
        upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

        print("downloading base checkpoint...")
        base_model_img.load_state_dict(
            torch.load(f"weights/{base_name_img}.pt", map_location=device)
        )

        base_model.load_state_dict(
            torch.load(f"weights/{base_name}.pt", map_location=device)
        )

        print("downloading upsampler checkpoint...")
        upsampler_model.load_state_dict(
            torch.load("weights/upsample_40m.pt", map_location=device)
        )

        self.sampler_text = PointCloudSampler(
            device=device,
            models=[base_model, upsampler_model],
            diffusions=[base_diffusion, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=["R", "G", "B"],
            guidance_scale=[3.0, 0.0],
            model_kwargs_key_filter=(
                "texts",
                "",
            ),  # Do not condition the upsampler at all
        )

        self.sampler_img = PointCloudSampler(
            device=device,
            models=[base_model_img, upsampler_model],
            diffusions=[base_diffusion_img, upsampler_diffusion],
            num_points=[1024, 4096 - 1024],
            aux_channels=["R", "G", "B"],
            guidance_scale=[3.0, 0.0],
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt.",
            default=None,
        ),
        image: Path = Input(
            description="Input image. When prompt is set, the model will disregard the image and generate pointcloud based on the prompt",
            default=None,
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        assert (
            prompt is not None or image is not None
        ), "Please provide either a prompt or an image for generating pointcloud"

        sampler = self.sampler_text if prompt is not None else self.sampler_img

        # Produce a sample from the model.
        samples = None
        for x in tqdm(
            sampler.sample_batch_progressive(
                batch_size=1,
                model_kwargs=dict(texts=[prompt])
                if prompt is not None
                else dict(images=[Image.open(str(image))]),
            )
        ):
            samples = x

        pc = sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=3)

        out_path = f"/tmp/out.png"
        fig.savefig(str(out_path))

        samples_list = samples.tolist()

        return ModelOutput(samples=samples_list, figure=Path(out_path))
