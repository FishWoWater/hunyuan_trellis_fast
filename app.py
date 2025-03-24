#!/usr/bin/env python
# coding=utf-8

import os
import uuid
import time
from typing import Literal
import random
import gradio as gr
import torch
from PIL import Image
from pipe import HunyuanTrellisImageTo3D
from gradio_litmodel3d import LitModel3D

# Constants
MAX_SEED = 1e7
EXAMPLE_IMAGES_DIR = "assets/example_image"


def randomize_seed(seed: int, randomize_seed: bool) -> int:
    """Randomize seed if requested"""
    if randomize_seed:
        seed = random.randint(0, int(MAX_SEED))
    return seed


class ImageTo3DApp:
    def __init__(self):
        self.pipeline = None
        self.current_hunyuan_mini = False
        self.current_trellis_mesh = False
        self.cached_geo_image_hash = None
        self.cached_raw_mesh = None
        self.cached_geo_path = None
        self._initialize_pipeline()

    def _compute_image_hash(self, image: Image.Image) -> str:
        """Compute a hash of the image content for comparison.
        Converts to grayscale and small size to be more robust to minor changes."""
        # Convert to small grayscale image to be more robust to minor changes
        small_img = image.convert("L").resize((64, 64))
        return hash(small_img.tobytes())

    def _initialize_pipeline(self, use_hunyuan_mini=False, use_trellis_mesh=False):
        # Check if we need to reinitialize
        if (
            self.pipeline is not None
            and use_hunyuan_mini == self.current_hunyuan_mini
            and use_trellis_mesh == self.current_trellis_mesh
        ):
            return

        # Cleanup old pipeline if it exists
        if self.pipeline is not None:
            del self.pipeline.geometry_pipeline
            del self.pipeline.tex_pipeline
            del self.pipeline.rembg
            del self.pipeline
            torch.cuda.empty_cache()

        # Create new pipeline with updated settings
        self.pipeline = HunyuanTrellisImageTo3D(
            use_hunyuan_mini=use_hunyuan_mini,
            use_trellis_mesh=use_trellis_mesh,
            save_gs_video=False,
        )

        # Update current settings
        self.current_hunyuan_mini = use_hunyuan_mini
        self.current_trellis_mesh = use_trellis_mesh

    def process_image(
        self,
        image: Image.Image,
        image_tex: Image.Image | None,
        use_hunyuan_mini: bool,
        use_trellis_mesh: bool,
        # seed: int,
        # randomize_seed_checkbox: bool,
        inference_steps: int,
        guidance_scale: float,
        bake_mode: Literal["fast", "opt"],
        fill_mesh_holes: bool,
        get_srgb_texture: bool,
    ) -> tuple:
        """Process image and return geometry and textured mesh"""

        # Initialize pipeline with current settings
        self._initialize_pipeline(use_hunyuan_mini, use_trellis_mesh)

        tik_pipe = time.time()
        torch.cuda.reset_peak_memory_stats()

        # Randomize seed if requested
        # actual_seed = randomize_seed(seed, randomize_seed_checkbox)

        savedir = f"./results/server/{uuid.uuid4()}"
        os.makedirs(savedir, exist_ok=True)

        # Check if we can reuse cached geometry
        current_image_hash = self._compute_image_hash(image)
        if (
            current_image_hash == self.cached_geo_image_hash
            and self.cached_raw_mesh is not None
        ):
            yield None, None, "Reusing cached geometry..."
            raw_mesh = self.cached_raw_mesh
            geo_path = self.cached_geo_path
        else:
            # Generate new geometry
            yield None, None, "Generating geometry..."
            raw_mesh, geo_path = self.pipeline.image_to_3d_geo(
                image,
                savedir=savedir,  # seed=actual_seed
            )
            # Cache the results
            self.cached_geo_image_hash = current_image_hash
            self.cached_raw_mesh = raw_mesh
            self.cached_geo_path = geo_path

        # Update UI with geometry mesh
        yield geo_path, None, f"Generated geometry mesh. Generating texture..."  # Seed: {actual_seed}"

        # Then generate texture
        # yield geo_path, None, "60%", "Generating texture..."
        # Use geometry image for texture if no texture image provided
        texture_image = image_tex if image_tex is not None else image
        _, tex_path = self.pipeline.image_to_3d_tex(
            raw_mesh,
            texture_image,
            num_trellis_steps=inference_steps,
            trellis_cfg=guidance_scale,
            trellis_bake_mode=bake_mode,
            fill_mesh_holes=fill_mesh_holes,
            savedir=savedir,
            get_srgb_texture=get_srgb_texture,
            # seed=actual_seed,
        )

        tok = time.time()
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        meta = (
            f"Completed!  TimeCost: {tok - tik_pipe:.1f}s, PeakMemory: {max_mem:.1f}GB"
        )
        # Final update with both meshes
        yield geo_path, tex_path, meta

        return (
            geo_path,
            tex_path,
            meta,
            # f"Seed used: {actual_seed}",
        )


def create_demo():
    app = ImageTo3DApp()

    with gr.Blocks(title="Image to 3D Pipeline Demo") as demo:
        gr.Markdown("# Image to 3D Pipeline Demo")
        gr.Markdown("Generate 3D models from images using Hunyuan3D and Trellis")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    # Geometry input
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="pil",
                            label="Geometry Input Image",
                            height=300,
                        )
                        examples = gr.Examples(
                            examples=[
                                f"assets/example_image/{image}"
                                for image in os.listdir("assets/example_image")[:8]
                            ],
                            inputs=[input_image],
                            outputs=[input_image],
                            fn=lambda x: x,
                            run_on_click=True,
                            examples_per_page=64,
                            label="Example Images (Geometry)",
                        )

                    # Texture input (optional)
                    with gr.Column(scale=1):
                        texture_image = gr.Image(
                            type="pil",
                            label="Texture Input Image (Optional)",
                            height=300,
                        )
                        texture_examples = gr.Examples(
                            examples=[
                                f"assets/example_image/{image}"
                                for image in os.listdir("assets/example_image")[:8]
                            ],
                            inputs=[texture_image],
                            outputs=[texture_image],
                            fn=lambda x: x,
                            run_on_click=True,
                            examples_per_page=64,
                            label="Example Images (Texture)",
                        )

                with gr.Accordion("Model Settings", open=True):
                    use_hunyuan_mini = gr.Checkbox(
                        value=False,
                        label="Use Hunyuan Mini",
                        info="Use smaller Hunyuan model. Faster but may reduce geometry quality.",
                    )
                    use_trellis_mesh = gr.Checkbox(
                        value=False,
                        label="Use Trellis Geometry",
                        info="Use Trellis decoded mesh as final geometry. Better quality but requires more VRAM.",
                    )
                    # seed = gr.Number(value=42, label="Seed", precision=0)
                    # randomize_seed_checkbox = gr.Checkbox(
                    #     value=True, label="Randomize Seed"
                    # )
                    inference_steps = gr.Slider(
                        minimum=5,
                        maximum=25,
                        value=12,
                        step=1,
                        label="Inference Steps",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.5,
                        step=0.1,
                        label="Guidance Scale",
                    )
                    bake_mode = gr.Radio(
                        choices=["fast", "opt"],
                        value="fast",
                        label="Texture Bake Mode",
                        info="Fast mode is quicker but may have lower quality textures",
                    )

                    fill_mesh_holes = gr.Checkbox(value=False, label="Fill Mesh Holes")
                    get_srgb_texture = gr.Checkbox(
                        value=False, label="Get SRGB Texture"
                    )

            with gr.Column(scale=1):
                # Progress tracking
                # with gr.Row():
                #     progress_text = gr.Textbox(
                #         label="Progress", value="Ready", interactive=False
                #     )

                # Output displays
                geometry_model = LitModel3D(
                    label="Geometry (Mesh)",
                    height=400,
                )
                textured_model = LitModel3D(
                    label="Textured Model",
                    height=400,
                )
                output_info = gr.Textbox(label="Generation Info", interactive=False)

        # Set up processing button
        process_btn = gr.Button("Generate 3D Model", variant="primary")
        process_btn.click(
            fn=app.process_image,
            inputs=[
                input_image,
                texture_image,  # Now passing texture_image instead of duplicating input_image
                use_hunyuan_mini,
                use_trellis_mesh,
                # seed,
                # randomize_seed_checkbox,
                inference_steps,
                guidance_scale,
                bake_mode,
                fill_mesh_holes,
                get_srgb_texture,
            ],
            outputs=[
                geometry_model,
                textured_model,
                # progress_text,
                output_info,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
