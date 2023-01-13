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
    json_file: Optional[Any]
    animation: Optional[Path]


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
            description="Input image. When prompt is set, the model will disregard the image and generate point cloud based on the prompt",
            default=None,
        ),
        output_format: str = Input(
            description='Choose the format of the output, either an animation or a json file of the point cloud. The json format is: { "coords": [...], "colors": [...] }, where "coords" is an [N x 3] array of (X,Y,Z) point coordinates, and "colors" is an [N x 3] array of (R,G,B) color values.',
            default="animation",
            choices=["animation", "json_file"],
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        assert (
            prompt is not None or image is not None
        ), "Please provide either a prompt or an image for generating point cloud"

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
        colors = np.stack(
            [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
        )

        if output_format == "json_file":
            pc_json = {"coords": pc.coords, "colors": colors}
            return ModelOutput(json_file=pc_json)

        print(
            "Generating the animation of the point cloud, this may take a few minutes..."
        )
        gif_out_path = f"/tmp/out.gif"
        save_gif(pc.coords, colors, gif_out_path)
        return ModelOutput(animation=Path(gif_out_path))


def save_gif(coords, colors, out_path, fig_size=8):
    fig = plt.figure(figsize=[fig_size, fig_size])
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        clip_on=False,
        vmax=2 * coords[:, 1].max(),
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
    print("Saving the animation...")
    animation.save(out_path, writer=writer, dpi=100)
