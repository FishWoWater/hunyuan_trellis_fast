## Hunyuan + TRELLIS -> Fast and Memory-Efficient Textured Mesh Generation

### Main features:
* **About 8 seconds and 8GB VRAM** to generate a textured mesh
* Support using **different** images for geometry and texture generation 

### Techniques Behind:
> no any magic here, simply reusing hunyuan3d-turbo, + trellis as the texture generator
* Image -> Hunyuan3D-2 mini geometry generation -> Trellis texture generation + mesh decimation -> baking texture -> glb export 
* Asynchronous texture generation and mesh decimation(when the trellis runs DiT to generate the texture, perform decimation on the hunyuan 3d geometry mesh)

### Demo
![video_demo](assets/video_demo.gif)

### Setup 
``` shell 
# clone the repo 
git clone --recurse-modules https://github.com/FishWoWater/hunyuan_trellis_fast

cd hunyuan_trellis_fast 
conda create -n hunyuan_trellis python=3.10
conda activate hunyuan_trellis
# install trellis depdendencies (write into existing env)
cd TRELLIS 
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast

# install hunyuan3d dependencies
cd ../Hunyuan3D-2  
pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

### Usage
``` shell
# run cli demo  
python demo.py 

# run gradio app 
python app.py
```


### Parameters 
| Parameters  | Description | Default | 
| :-- |:--| :--| 
| fill_holes | whether to fill holes in the generated mesh, will also **remove some isolated components**, **slightly slower** | True |
| use_hunyuan_mini | use the mini-turbo version of Hunyuan3D2, **slightly less VRAM, almost same speed** | False |
| use_trellis_mesh | use the mesh from trellis mesh decoder, instead of Hunyuan3D-2. Set to true will **require more memory(~3G+), but higher quality** | False |
| inference_steps |  number of inference steps in trellis texture generation  | 12 |
| guidance_scale |  classifier-guided scale in the trellis texture diffusion | 3.5 | 
| bake_mode |  'fast' or 'opt' mode in trellis texture baking, 'opt' mode will **consumes much more memory(~2G+)** | 'fast' |
| get_srgb_texture | get texture in the sRGB space | False |


### Comparison Against TRELLIS/Hunyuan3D-2/Spar3D
1. [Trellis](https://github.com/microsoft/TRELLIS) is **a bit slower** (or require **more memory** (14GB+) to be faster). 
2. [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) recently released the turbo/mini-turbo version, but the texture generation is still **slow** and **a bit over-saturated** 
3. [Spar3D](https://github.com/Stability-AI/stable-point-aware-3d) is very fast (~1 second) but the quality **is NOT good**, require 10G memory by default (or can sacrifice a bit speed to save memory)

Refer to this [thread](https://github.com/microsoft/TRELLIS/issues/139) for some comparison experiements  
