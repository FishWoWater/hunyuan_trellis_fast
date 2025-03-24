import open3d as o3d
import numpy as np
import time
import xatlas
from trellis.utils import postprocessing_utils


def decimate_and_parameterize_process(mesh, result_queue):
    result = decimate_and_parameterize(mesh, simplify_ratio=0.95, verbose=True)
    result_queue.put(("mesh", result))


def decimate_process(mesh, result_queue):
    result = decimate(mesh, simplify_ratio=0.95, verbose=True)
    result_queue.put(("mesh", result))


def decimate(mesh, simplify_ratio=0.95, verbose=False):
    tik = time.time()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        mesh.vertices,
        mesh.faces,
        postprocess_mode="simplify",
        simplify_ratio=simplify_ratio,
        fill_holes=False,
        verbose=verbose,
    )
    print(
        "decimate finished in {}s, {}vertices and {}faces".format(
            time.time() - tik, vertices.shape, faces.shape
        )
    )
    return vertices, faces


def decimate_and_parameterize(mesh, simplify_ratio=0.95, verbose=False):
    tik = time.time()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        mesh.vertices,
        mesh.faces,
        postprocess_mode="simplify",
        simplify_ratio=simplify_ratio,
        fill_holes=False,
        verbose=verbose,
    )
    tik_uv = time.time()
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    print("uv unwarping finished in {}s".format(time.time() - tik_uv))

    vertices = vertices[vmapping]
    faces = indices
    print(
        "decimate_and_process finished in {}s, {}vertices and {}faces".format(
            time.time() - tik, vertices.shape, faces.shape
        )
    )
    return vertices, faces, uvs


def voxelize(mesh, resolution: int = 64):
    try:
        mesh = mesh.as_open3d
    except:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    binary_voxel = np.zeros((resolution, resolution, resolution), dtype=bool)
    binary_voxel[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = True
    return binary_voxel
