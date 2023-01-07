"""
download the models to ./weights
wget https://openaipublic.azureedge.net/main/point-e/base_40m_imagevec.pt -O base40M-imagevec.pt
wget https://openaipublic.azureedge.net/main/point-e/base_40m_textvec.pt  -O base40M-textvec.pt
wget https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt  -O upsample_40m.pt
wget https://openaipublic.azureedge.net/main/point-e/base_40m.pt -O base40M.pt
"""

import os
import numpy as np
from typing import Any, List, Optional
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from cog import BasePredictor, Input, Path, BaseModel

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler, PointCloud
from point_e.models.download import load_checkpoint
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud


class ModelOutput(BaseModel):
    pointcloud_json: Any
    pointcloud_npz: Optional[Path]
    figure: Optional[Path]
    annimation: Optional[Path]


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
        save_npz: bool = Input(
            description="If set true, the pointcloud will be saved as a .npz file.",
            default=False,
        ),
        generate_pc_plot: bool = Input(
            description="If set true, the point cloud will be rendered as a plot with 9 views.",
            default=False,
        ),
        generate_annimation: bool = Input(
            description="If set true, a gif file with the annimated pointcloud will be generated.",
            default=False,
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

        pc_json = {"coords": pc.coords, "channels": pc.channels}

        if save_npz:
            pointcloud_out_path = f"/tmp/pointcloud.npz"
            PointCloud.save(pc, pointcloud_out_path)

            samples_list = samples.tolist()

        if generate_pc_plot:
            fig = plot_point_cloud(pc, grid_size=3)
            out_path = f"/tmp/out.png"
            fig.savefig(str(out_path))

        if generate_annimation:
            print(
                "Generating annimation of the pointcloud, this may take a few minutes..."
            )
            gif_out_path = f"/tmp/out.gif"
            save_gif(pc, gif_out_path)

        return ModelOutput(
            pointcloud_json=pc_json,
            pointcloud_npz=Path(pointcloud_out_path) if save_npz else None,
            figure=Path(out_path) if generate_pc_plot else None,
            annimation=Path(gif_out_path) if generate_annimation else None,
        )


def save_gif(pc, out_path, fig_size=8):
    colors = np.stack([pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1)

    fig = plt.figure(figsize=[fig_size, fig_size])
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.scatter(
        pc.coords[:, 0],
        pc.coords[:, 1],
        pc.coords[:, 2],
        c=colors,
        clip_on=False,
        vmax=2 * pc.coords[:, 1].max(),
    )
    ax.elev = -5
    ax.axis("off")

    # make sure equal scale is used across all axes. From Stackoverflow
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    max_range = np.array([np.diff(xlim), np.diff(ylim), np.diff(zlim)]).max() / 2.0

    mid_x = np.mean(xlim)
    mid_y = np.mean(ylim)
    mid_z = np.mean(zlim)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # build animated loop
    def rotate_view(frame, azim_delta=1):
        ax.azim = -20 - azim_delta * frame

    animation = FuncAnimation(fig, rotate_view, frames=360, interval=15)

    writer = PillowWriter(fps=40)
    print("Saving the annimation...")
    animation.save(out_path, writer=writer, dpi=100)
