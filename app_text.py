#!/usr/bin/env python
# coding=utf-8

import os
import uuid
import time
import random
import gradio as gr
import torch
from typing import Literal
from pipe_text import HunyuanTrellisTextTo3D
from gradio_litmodel3d import LitModel3D

# Constants
MAX_SEED = 1e7
LOW_VRAM = False
EXAMPLE_PROMPTS = open("assets/example_prompt.txt", "r").read().strip().split("\n")


def randomize_seed(seed: int, randomize_seed: bool) -> int:
    """Randomize seed if requested"""
    if randomize_seed:
        seed = random.randint(0, int(MAX_SEED))
    return seed


class TextTo3DApp:
    def __init__(self):
        self.pipeline = None
        self.current_hunyuan_mini = False
        self.current_trellis_mesh = False
        self._initialize_pipeline()

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
        self.pipeline = HunyuanTrellisTextTo3D(
            use_hunyuan_mini=use_hunyuan_mini,
            use_trellis_mesh=use_trellis_mesh,
            save_gs_video=False,
            low_vram=LOW_VRAM,
        )

        # Update current settings
        self.current_hunyuan_mini = use_hunyuan_mini
        self.current_trellis_mesh = use_trellis_mesh

    def process_text(
        self,
        prompt: str,
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
        """Process text prompt and return geometry and textured mesh"""

        # Initialize pipeline with current settings
        self._initialize_pipeline(use_hunyuan_mini, use_trellis_mesh)

        tik_pipe = time.time()
        torch.cuda.reset_peak_memory_stats()

        # Randomize seed if requested
        # actual_seed = randomize_seed(seed, randomize_seed_checkbox)

        savedir = f"./results/server/text23d/{uuid.uuid4()}"
        os.makedirs(savedir, exist_ok=True)

        # Generate new geometry
        yield None, None, None, "Generating geometry from text prompt..."
        image, raw_mesh, geo_path = self.pipeline.text_to_3d_geo(
            prompt,
            savedir=savedir,  # seed=actual_seed
        )

        # Update UI with geometry mesh and generated image
        yield geo_path, None, image, f"Generated geometry mesh. Generating texture..."  # Seed: {actual_seed}"

        # Then generate texture
        _, tex_path = self.pipeline.text_to_3d_tex(
            raw_mesh,
            image,
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
        yield geo_path, tex_path, image, meta

        return (
            geo_path,
            tex_path,
            image,
            meta,
            # f"Seed used: {actual_seed}",
        )


def create_demo():
    app = TextTo3DApp()

    with gr.Blocks(title="Hunyuan3D and Trellis Text to 3D Pipeline Demo") as demo:
        gr.Markdown("# Hunyuan3D and Trellis Text to 3D Pipeline Demo")
        gr.Markdown("Generate 3D models from text prompts using Hunyuan3D and Trellis")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    # Text input
                    with gr.Column(scale=1):
                        input_text = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter a text prompt",
                            lines=5,
                        )
                        examples = gr.Examples(
                            examples=EXAMPLE_PROMPTS,
                            inputs=[input_text],
                            outputs=[input_text],
                            fn=lambda x: x,
                            run_on_click=True,
                            examples_per_page=64,
                            label="Example Text Prompts",
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
                image_output = gr.Image(label="Generated Image")
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
        process_btn = gr.Button("Generate 3D Model from Text", variant="primary")
        process_btn.click(
            fn=app.process_text,
            inputs=[
                input_text,
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
                image_output,
                # progress_text,
                output_info,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
