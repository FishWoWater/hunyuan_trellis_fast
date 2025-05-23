#!/usr/bin/env python
# coding=utf-8

import os, os.path as osp
import time
import json
import numpy as np
import torch
from pipe_text import HunyuanTrellisTextTo3D

if __name__ == "__main__":
    use_hunyuan_mini = False
    use_trellis_mesh = False
    trellis_fast_bake = True
    fill_mesh_holes = False
    num_trellis_steps = 12
    save_gs_video = False
    low_vram = False

    pipe = HunyuanTrellisTextTo3D(
        use_hunyuan_mini=use_hunyuan_mini,
        use_trellis_mesh=use_trellis_mesh,
        save_gs_video=save_gs_video,
        low_vram=low_vram,
    )

    # Get images to test and build up saving directory
    # example_image_paths = glob.glob("./assets/example_image/*")
    example_prompts = open("assets/example_prompt.txt", "r").read().strip().split("\n")
    # Test image path and saving directory
    saveroot = "results/text23d/hunyuan={}_geo={}_bake={}_fill={}".format(
        "mini" if use_hunyuan_mini else "normal",
        "trellis" if use_trellis_mesh else "hunyuan",
        "fast" if trellis_fast_bake else "opt",
        "yes" if fill_mesh_holes else "no",
    )

    time_cost, mem_info = {}, {}

    ### testing the image-to-3d pipeline
    for example_prompt in example_prompts:
        print("processing the prompt", example_prompt)
        torch.cuda.reset_peak_memory_stats()
        instance_name = osp.basename(example_prompt).split(".")[0]
        savedir = os.path.join(saveroot, instance_name)
        os.makedirs(savedir, exist_ok=True)

        tik_pipe = time.time()
        rawglb, glb = pipe.text_to_3d(
            example_prompt,
            savedir=savedir,
            num_trellis_steps=num_trellis_steps,
            trellis_bake_mode="fast" if trellis_fast_bake else "opt",
            fill_mesh_holes=fill_mesh_holes,
            get_mesh=True,
        )
        time_cost_pipe = time.time() - tik_pipe
        print(f"all pipeline finished in {time_cost_pipe:.3f}s")

        glb.export(osp.join(savedir, "ans.glb"))
        max_mem = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        print(f"Maximum memory used: {max_mem:.2f} GB")

        time_cost[instance_name] = time_cost_pipe
        mem_info[instance_name] = max_mem

    json.dump(time_cost, open(osp.join(saveroot, "text_time_cost.json"), "w"))
    json.dump(mem_info, open(osp.join(saveroot, "text_mem_info.json"), "w"))

    time_costs = np.array(list(time_cost.values()))
    mem_infos = np.array(list(mem_info.values()))
    print(
        f"min/max/mean time cost: {time_costs.min()}/{time_costs.max()}/{time_costs.mean()}"
    )
    print(
        f"min/max/mean mem info: {mem_infos.min()}/{mem_infos.max()}/{mem_infos.mean()}"
    )
