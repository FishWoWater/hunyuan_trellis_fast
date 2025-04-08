import os, os.path as osp
import time
import sys

sys.path.insert(0, osp.join(os.getcwd(), "TRELLIS"))
os.environ["ATTN_BACKEND"] = "flash-attn"

import multiprocessing as mp
import imageio
import trimesh
import numpy as np
import torch
from typing import Optional, Literal
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils, render_utils
from utils import voxelize, decimate_and_parameterize_process, decimate_process

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


class HunyuanTrellisImageTo3D:
    def __init__(
        self,
        use_hunyuan_mini: bool,
        use_trellis_mesh: bool,
        save_gs_video: bool = False,
        require_rembg: bool = True, 
        device: str = "cuda",
    ):
        self.use_hunyuan_mini = use_hunyuan_mini
        self.use_trellis_mesh = use_trellis_mesh
        self.save_gs_video = save_gs_video

        assert not (
            self.use_trellis_mesh and not self.use_hunyuan_mini
        ), "Unnecessary setups"

        self.device = "cuda"
        self.trellis_skip_models = [
            "sparse_structure_decoder",
            "sparse_structure_flow_model",
            "slat_decoder_rf",
        ]
        if not self.use_trellis_mesh:
            self.trellis_skip_models.append("slat_decoder_mesh")

        self.geometry_pipeline = None
        self.tex_pipeline = None
        self.rembg = None
        self.result_queue = mp.Queue()
        self.require_rembg = require_rembg

        self._initialize_models()

    def _initialize_models(self):
        if self.geometry_pipeline is None:
            self.geometry_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                # mini-turbo will NOT be that good, thus we use ori turbo
                (
                    "tencent/Hunyuan3D-2"
                    if not self.use_hunyuan_mini
                    else "tencent/Hunyuan3D-2mini"
                ),
                subfolder=(
                    "hunyuan3d-dit-v2-0-turbo"
                    if not self.use_hunyuan_mini
                    else "hunyuan3d-dit-v2-mini-turbo"
                ),
                use_safetensors=False,
                device=self.device,
            )
            self.geometry_pipeline.enable_flashvdm(topk_mode="merge")

        if self.tex_pipeline is None:
            self.tex_pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large", skip_models=self.trellis_skip_models
            )
            self.tex_pipeline.to(self.device)

        # suggested to use birefnet-general for better bg removal
        if self.rembg is None and self.require_rembg:
            self.rembg = BackgroundRemover()

    def run_geometry(self, image):
        return self.geometry_pipeline(
            image=image,
            num_inference_steps=5,
            octree_resolution=192,
            num_chunks=20000,
            generator=torch.manual_seed(12345),
            output_type="trimesh",
        )[0]

    def run_tex(self, image, binary_voxel, num_steps: int = 13, cfg: float = 3.75):
        outputs = self.tex_pipeline.run_detail_variation(
            binary_voxel,
            image,
            seed=1,
            # more steps, larger cfg
            slat_sampler_params={
                "steps": num_steps,
                "cfg_strength": cfg,
            },
            formats=["gaussian", "mesh"] if self.use_trellis_mesh else ["gaussian"],
        )
        return outputs

    def image_to_3d_geo(self, image: str, savedir: str = "", seed=2025):
        if image.mode == "RGB" and self.require_rembg:
            tik = time.time()
            image = self.rembg(image)
            if savedir:
                image.save(osp.join(savedir, "image_rembg.png"))
            print("rembg finished in {:.3f}s".format(time.time() - tik))

        tik = time.time()
        mesh = self.run_geometry(image)
        material = trimesh.visual.material.PBRMaterial(
            roughnessFactor=1.0,
            baseColorFactor=np.array([127, 127, 127, 255], dtype=np.uint8),
        )
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices
            @ np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
            faces=mesh.faces,
            process=False,
            material=material,
        )
        print(
            "geometry generation finished in {:.3f}s: {}vertices, {}faces".format(
                time.time() - tik, len(mesh.vertices), len(mesh.faces)
            )
        )
        tik = time.time()
        # rotate to z-up for compatibility with trellis
        # mesh.vertices = (
        #     mesh.vertices
        #     @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        #     @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        # )
        # also fix some problems in the mesh
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        savepath = ""
        if savedir:
            # save before we rotate as gradio expects a y-up mesh
            savepath = osp.join(savedir, "geo.glb")
            mesh.export(savepath)

        # rotate to z-up for compatibility with trellis
        mesh.vertices = (
            mesh.vertices
            @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            # @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        )
        return mesh, savepath

    def image_to_3d_tex(
        self,
        mesh,
        image_tex,
        num_trellis_steps: int = 13,
        trellis_cfg: float = 3.75,
        trellis_bake_mode: Literal["fast", "opt"] = "fast",
        fill_mesh_holes: bool = True,
        get_srgb_texture: bool = False,
        savedir: str = "",
        seed=2025,
    ):
        binary_voxel = voxelize(mesh)

        if not self.use_trellis_mesh:
            tik = time.time()
            mesh_process = mp.Process(
                target=(
                    decimate_and_parameterize_process
                    if not fill_mesh_holes
                    else decimate_process
                ),
                args=(mesh, self.result_queue),
            )
            mesh_process.start()

            # Run the pipeline
            outputs = self.run_tex(
                image_tex, binary_voxel, num_steps=num_trellis_steps, cfg=trellis_cfg
            )

            print(
                "[Hunyuan3D-Mesh] texture generation finished in {}s".format(
                    time.time() - tik
                )
            )

            # Wait for the decimate_and_parameterize process to finish and collect results
            mesh_result = self.result_queue.get()
            assert mesh_result[0] == "mesh"  # Ensure we got the right result
            # if we need to fill mesh holes, uvs should come after it
            if fill_mesh_holes:
                vertices, faces = mesh_result[1]
                uvs = None
            else:
                vertices, faces, uvs = mesh_result[1]

            # Ensure the process has finished
            mesh_process.join()
            extra_kwargs = {"vertices": vertices, "faces": faces, "uvs": uvs}
        else:
            tik = time.time()
            # Run the pipeline
            outputs = self.run_tex(image_tex, binary_voxel)
            print(
                "[TRELLIS-MESH] texture generation finished in {:.3f}s".format(
                    time.time() - tik
                )
            )
            extra_kwargs = {}

        if self.save_gs_video and savedir:
            torch.cuda.empty_cache()
            # Render the outputs
            video = render_utils.render_video(outputs["gaussian"][0])["color"]
            imageio.mimsave(os.path.join(savedir, f"gs.mp4"), video, fps=30)

        tik = time.time()
        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_trimesh(
            outputs["gaussian"][0],
            outputs["mesh"][0] if self.use_trellis_mesh else mesh,
            # Optional parameters
            simplify=0.95,  # Ratio of triangles to remove in the simplification process
            texture_size=1024,  # Size of the texture used for the GLB
            texture_bake_mode=trellis_bake_mode,
            get_srgb_texture=get_srgb_texture,
            fill_holes=fill_mesh_holes,
            debug=False,
            verbose=False,
            forward_rot=False,
            **extra_kwargs,
        )

        print("postprocessing finished in {:.3f}s".format(time.time() - tik))

        savepath = ""
        if savedir:
            savepath = osp.join(savedir, "textured_mesh.glb")
            glb.export(savepath)

        return glb, savepath

    def image_to_3d(
        self,
        image,
        image_tex,
        num_trellis_steps: int = 13,
        trellis_cfg: float = 3.75,
        trellis_bake_mode: Literal["fast", "opt"] = "fast",
        fill_mesh_holes: bool = True,
        savedir: str = "",
        get_mesh: bool = True,
        seed=2025,
    ):

        raw_mesh, rawpath = self.image_to_3d_geo(image, savedir=savedir, seed=seed)
        tex_mesh, texpath = self.image_to_3d_tex(
            raw_mesh,
            image_tex,
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
