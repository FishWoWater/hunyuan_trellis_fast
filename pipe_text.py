import os, os.path as osp
import sys

sys.path.insert(0, osp.join(os.getcwd(), "TRELLIS"))
os.environ["ATTN_BACKEND"] = "flash-attn"

import torch
from typing import Literal

from torchvision import transforms

# LCM requires a from-source installation
from diffusers import (
    DiffusionPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    AutoPipelineForText2Image,
)
from transformers import pipeline, AutoModelForImageSegmentation  # for rembg
from pipe import HunyuanTrellisImageTo3D  # for image-to-3d

# predefined common settings
SETTINGS = {
    # 1st setting (should require less memory(~8GB) and is faster)
    "fast": {
        "use_hunyuan_mini": False,
        "use_trellis_mesh": False,
    },
    # 2nd setting (should have higher quality but require 15G+ memory)
    "quality": {
        "use_hunyuan_mini": True,
        "use_trellis_mesh": True,
    },
}


# NOTICE THE `LOW_VRAM` parameter
# In the low-vram mode(14GB-), we will use smaller version of models (e.g. bria-rmbg-1.4 & sd-turbo)
# Otherwise(24GB+), we will use the full version (e.g. bria-rmbg-2.0, 4GB+ & lcm+sdxl 8GB+ memory)


class HunyuanTrellisTextTo3D:
    def __init__(
        self,
        use_hunyuan_mini: bool,
        use_trellis_mesh: bool,
        save_gs_video: bool = False,
        low_vram: bool = True,
        device: str = "cuda",
    ):
        self.use_hunyuan_mini = use_hunyuan_mini
        self.use_trellis_mesh = use_trellis_mesh
        self.save_gs_video = save_gs_video
        self.low_vram = low_vram
        self.device = device

        # need a t2i pipe
        if self.low_vram:
            self._t2i_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
            )
            self._t2i_pipe.to(self.device)
        else:
            unet = UNet2DConditionModel.from_pretrained(
                "latent-consistency/lcm-sdxl",
                torch_dtype=torch.float16,
                variant="fp16",
            )
            self._t2i_pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                unet=unet,
                torch_dtype=torch.float16,
            )
            self._t2i_pipe.scheduler = LCMScheduler.from_config(
                self._t2i_pipe.scheduler.config
            )
            self._t2i_pipe.to(self.device)

        # if self.low_vram:
        if True:
            # need a background remover
            # notice that bria rembg 1.4 is vram-efficient
            # and better than rembg package
            self._rembg = pipeline(
                "image-segmentation",
                model="briaai/RMBG-1.4",
                trust_remote_code=True,
                device=self.device,
            )
            self._rembg_transform_image = None
        else:
            self._rembg = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2.0", trust_remote_code=True
            )
            # torch.set_float32_matmul_precision(["high", "highest"][0])
            self._rembg.to("cuda")
            self._rembg.eval()

            # Data settings
            image_size = (1024, 1024)
            self._rembg_transform_image = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        self._image_to_3d_pipe = HunyuanTrellisImageTo3D(
            use_hunyuan_mini=self.use_hunyuan_mini,
            use_trellis_mesh=self.use_trellis_mesh,
            save_gs_video=self.save_gs_video,
            require_rembg=False, 
            device=self.device,
        )

    @torch.no_grad()
    def text_to_image(self, prompt):
        generator = torch.manual_seed(0)
        image = self._t2i_pipe(
            prompt=prompt, num_inference_steps=4, generator=generator, guidance_scale=8.0
        ).images[0]
        return image

    @torch.no_grad()
    def text_to_3d_geo(self, prompt: str, savedir: str = "", seed=2025):
        image = self.text_to_image(prompt)
        if savedir:
            image.save(osp.join(savedir, "t2i_image.png"))
        # remove the background
        image = self.remove_bg(image)
        if savedir:
            image.save(osp.join(savedir, "t2i_image_wobg.png"))
        raw_mesh, rawpath = self._image_to_3d_pipe.image_to_3d_geo(
            image, savedir=savedir, seed=seed
        )
        return image, raw_mesh, rawpath

    @torch.no_grad()
    def remove_bg(self, image):
        # if self.low_vram:
        if True:
            image = self._rembg(image)
        else:
            if self._rembg_transform_image is not None:
                input_images = self._rembg_transform_image(image)[None].to(self.device)
            preds = self._rembg(input_images)[-1].sigmoid().cpu()
            pred_pil = transforms.ToPILImage()(preds[0].squeeze())
            mask = pred_pil.resize(image.size)
            image.putalpha(mask)
        return image

    @torch.no_grad()
    def text_to_3d_tex(
        self,
        image,
        raw_mesh,
        savedir="",
        num_trellis_steps: int = 12,
        trellis_cfg: float = 3.75,
        trellis_bake_mode: Literal["fast", "opt"] = "fast",
        fill_mesh_holes: bool = True,
        get_srgb_texture: bool = False,
        seed=2025,
    ):
        mesh, path = self._image_to_3d_pipe.image_to_3d_tex(
            image,
            raw_mesh,
            savedir=savedir,
            num_trellis_steps=num_trellis_steps,
            trellis_cfg=trellis_cfg,
            trellis_bake_mode=trellis_bake_mode,
            fill_mesh_holes=fill_mesh_holes,
            get_srgb_texture=get_srgb_texture,
            seed=seed,
        )
        return mesh, path

    @torch.no_grad()
    def text_to_3d(
        self,
        prompt: str,
        num_trellis_steps: int = 12,
        trellis_cfg: float = 3.75,
        trellis_bake_mode: Literal["fast", "opt"] = "fast",
        fill_mesh_holes: bool = True,
        savedir: str = "",
        get_mesh: bool = True,
        seed=2025,
    ):

        image, raw_mesh, rawpath = self.text_to_3d_geo(
            prompt, savedir=savedir, seed=seed
        )
        tex_mesh, texpath = self.text_to_3d_tex(
            raw_mesh,
            image,
            num_trellis_steps=num_trellis_steps,
            trellis_cfg=trellis_cfg,
            trellis_bake_mode=trellis_bake_mode,
            fill_mesh_holes=fill_mesh_holes,
            savedir=savedir,
            seed=seed,
        )
        if get_mesh:
            return raw_mesh, tex_mesh
        return rawpath, texpath
