"""
This code is from https://github.com/allenai/objaverse-rendering

Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import math
import os
import random
import sys
import time
import urllib.request
from typing import Tuple

import bpy
from mathutils import Vector
from PIL import Image
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--base_dir", type=str, default="/data/qys/objectverse-lvis/Lvis_rendering_cube_fixdistance_highreso/")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--resolution", type=int, default=768) #512
parser.add_argument("--camera_dist", type=int, default=(0.5 / np.tan(np.radians(30/2))))
# parser.add_argument("--camera_dist", type=int, default=4)


argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)
args.output_dir = os.path.join(args.base_dir, "images")

context = bpy.context
scene = context.scene
render = scene.render

def set_background_color(color=(1, 1, 1)):
    """Set the world background color in Blender."""
    bpy.data.worlds["World"].use_nodes = True
    bg = bpy.data.worlds["World"].node_tree.nodes.get('Background')
    if not bg:
        bg = bpy.data.worlds["World"].node_tree.nodes.new('ShaderNodeBackground')
    bg.inputs[0].default_value = (*color, 1)  # RGBA, where A is mostly ignored



render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
# instant mesh 和zero123++都是320*320的
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 500
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

# Set the device_type
cycles_preferences = bpy.context.preferences.addons["cycles"].preferences
cycles_preferences.compute_device_type = "CUDA"  # or "OPENCL"
cuda_devices = cycles_preferences.get_devices_for_type("CUDA")
for device in cuda_devices:
    device.use = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

# 这是之前渲染的方案
# def add_lighting() -> None:
#     # delete the default light
#     bpy.data.objects["Light"].select_set(True)
#     bpy.ops.object.delete()
#     # add a new light
#     bpy.ops.object.light_add(type="AREA")
#     light2 = bpy.data.lights["Area"]
#     light2.energy = 30000
#     bpy.data.objects["Area"].location[2] = 0.5
#     bpy.data.objects["Area"].scale[0] = 100
#     bpy.data.objects["Area"].scale[1] = 100
#     bpy.data.objects["Area"].scale[2] = 100

def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 1
    bpy.data.objects["Area"].location[2] = 0.5
    
    bpy.data.objects["Area"].scale[0] = 200
    bpy.data.objects["Area"].scale[1] = 200
    bpy.data.objects["Area"].scale[2] = 200


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    # print("cube")


def normalize_scene_to_sphere():
    bbox_min, bbox_max = scene_bbox()
    
    # 计算包围盒的中心和最大半径（从中心到包围盒角点的最大距离）
    bbox_center = (bbox_min + bbox_max) / 2
    max_distance = max((bbox_max - bbox_center).length, (bbox_center - bbox_min).length)
    
    # 计算缩放因子，使得最大距离为0.5
    scale = 0.5 / max_distance
    
    # 对每个对象应用缩放
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    
    # 更新场景以应用缩放
    bpy.context.view_layer.update()
    
    # 重新计算包围盒
    bbox_min, bbox_max = scene_bbox()
    bbox_center = (bbox_min + bbox_max) / 2
    
    # 计算偏移量，使得对象中心位于原点
    offset = -bbox_center
    
    # 对每个对象应用偏移
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    
    # 取消所有对象的选中状态
    bpy.ops.object.select_all(action="DESELECT")
    # print("sphere")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 0.5 / np.tan(np.radians(30/2)), 0)
    cam.data.lens = 55.42 #35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


# def process_image(image_path):
#     """ Process the image to fill alpha=1 with white and save as RGB. """
#     with Image.open(image_path) as img:
#         # Convert to RGBA if not already
#         img = img.convert("RGBA")

#         # Create a new image with white background
#         background = Image.new('RGBA', img.size, (255, 255, 255, 255))
#         # Composite the two images together
#         combined = Image.alpha_composite(background, img)
#         # Convert to RGB
#         rgb_image = combined.convert("RGB")
        
#         # Save the image
#         # rgb_image.save(image_path.replace('.png', '_rgb.png'))
#         rgb_image.save(image_path)

def process_image(image_path):  #其实没啥差别
    """ Process the image to fill alpha=1 with white and save as RGB. """
    with Image.open(image_path) as img:
        # Convert to RGBA if not already
        img = img.convert("RGBA")
        img_data = np.array(img, dtype=np.float32) / 255.  # 归一化

        # 分离颜色通道和Alpha通道
        rgb_channels = img_data[:, :, :3]
        alpha_channel = img_data[:, :, 3:]

        # 使用加权合成方法，将透明区域填充为白色
        color = [255, 255, 255]
        color = np.array(color, dtype=np.float32) / 255.  # 归一化指定的填充颜色
        composite_image = rgb_channels * alpha_channel + color * (1 - alpha_channel)

        # 转换回0-255范围的图像数据
        composite_image = (composite_image * 255).astype(np.uint8)

        # 将处理后的图像保存为RGB格式
        rgb_image = Image.fromarray(composite_image, 'RGB')
        rgb_image.save(image_path)

# def process_image_newpath(image_path, new_image_path):
#     """ Process the image to fill alpha=1 with white and save as RGB. """
#     with Image.open(image_path) as img:
#         # Convert to RGBA if not already
#         img = img.convert("RGBA")

#         # Create a new image with white background
#         background = Image.new('RGBA', img.size, (255, 255, 255, 255))
#         # Composite the two images together
#         combined = Image.alpha_composite(background, img)
#         # Convert to RGB
#         rgb_image = combined.convert("RGB")
        
#         # Save the image
#         # rgb_image.save(image_path.replace('.png', '_rgb.png'))
#         rgb_image.save(new_image_path)

def process_image_newpath(image_path, new_image_path):
    """ Process the image to fill alpha=1 with white and save as RGB. """
    with Image.open(image_path) as img:
        # Convert to RGBA if not already
        img = img.convert("RGBA")
        img_data = np.array(img, dtype=np.float32) / 255.  # 归一化

        # 分离颜色通道和Alpha通道
        rgb_channels = img_data[:, :, :3]
        alpha_channel = img_data[:, :, 3:]

        # 使用加权合成方法，将透明区域填充为白色
        color = [255, 255, 255] 
        color = np.array(color, dtype=np.float32) / 255.  # 归一化指定的填充颜色
        composite_image = rgb_channels * alpha_channel + color * (1 - alpha_channel)
        
        # 转换回0-255范围的图像数据
        composite_image = (composite_image * 255).astype(np.uint8)

        # 将处理后的图像保存为RGB格式
        rgb_image = Image.fromarray(composite_image, 'RGB')
        rgb_image.save(new_image_path)
            
        

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    # normalize_scene()
    normalize_scene_to_sphere()
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    for i in range(args.num_images):
        # set the camera position
        theta = (i / args.num_images) * math.pi * 2
        phi = math.radians(60)
        point = (
            args.camera_dist * math.sin(phi) * math.cos(theta),
            args.camera_dist * math.sin(phi) * math.sin(theta),
            args.camera_dist * math.cos(phi),
        )
        cam.location = point
        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        process_image(render_path)

def save_images_6view(object_file: str, save_query_view: bool = False) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    # normalize_scene_to_sphere()
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    # 固定的方位角和仰角
    azimuths = np.array([30, 90, 150, 210, 270, 330])
    elevations = np.array([20, -10, 20, -10, 20, -10])

    for i in range(len(azimuths)):
        # 将角度转换为弧度
        theta = np.radians(azimuths[i])
        phi = np.radians(elevations[i])
        # 根据球坐标公式计算相机位置
        point = (
            args.camera_dist * np.cos(phi) * np.cos(theta),
            args.camera_dist * np.cos(phi) * np.sin(theta),
            args.camera_dist * np.sin(phi),
        )
        cam.location = point
        # 渲染图像
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        process_image(render_path)
    
    if save_query_view:
        """
            Saves 
                1. query image(random) 
                2. 0 elavation image 
            of the object in the scene.
        """
        
        # 参照InstantMesh的finetune 方案 https://github.com/TencentARC/InstantMesh/issues/114
        
        # 从 [0, 360] 随机取一个 theta 值  # 从 [-20, 45] 随机取一个 phi 值
        theta = np.radians(np.random.uniform(0, 360))
        phi = np.radians(np.random.uniform(-20, 45))
        # 从 [1.25, 2.5] 随机取一个 camera_dist 值， 因为InstantMesh是单位化到[-1,1]的单位圆，而我们是[-0.5,+0.5]所以距离除以2
        # camera_dist = np.random.uniform(1.25, 2.5)
        
        # 因为不需要finetune 只是benchmark 所以无需随机距离
        camera_dist = args.camera_dist    
        
        # 根据球坐标公式计算相机位置
        point = (
            camera_dist * np.cos(phi) * np.cos(theta),
            camera_dist * np.cos(phi) * np.sin(theta),
            camera_dist * np.sin(phi),
        )
        cam.location = point
        render_path = os.path.join(args.output_dir, object_uid, "query_rgba.png")
        render_path_processed = os.path.join(args.output_dir, object_uid, "query.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        process_image_newpath(render_path, render_path_processed)
        
        
        # # 从0旋转角0俯仰角看 渲染一个图
        # theta = np.radians(0)
        # phi = np.radians(0)
        # # 相机距离采用target image的同样距离
        # camera_dist = args.camera_dist
        # # 根据球坐标公式计算相机位置
        # theta = np.radians(theta)
        # phi = np.radians(phi)
        # point = (
        #     camera_dist * np.cos(phi) * np.cos(theta),
        #     camera_dist * np.cos(phi) * np.sin(theta),
        #     camera_dist * np.sin(phi),
        # )
        # cam.location = point
        # render_path = os.path.join(args.output_dir, object_uid, "query0_rgba.png")
        # render_path_processed = os.path.join(args.output_dir, object_uid, "query0.png")
        # scene.render.filepath = render_path
        # bpy.ops.render.render(write_still=True)
        # process_image_newpath(render_path, render_path_processed)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
            
        
        # # 在主函数或适当的位置调用该函数以设置背景颜色
        # set_background_color()  # 默认为白色
        save_images_6view(local_path,  save_query_view=True)
        
        # 保存输入图片和0度图片
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        # if args.object_path.startswith("http"):
        #     os.remove(local_path)
        # Write the completed path to a file
        # print()
        
        with open(os.path.join(args.base_dir, "completed_renders.txt"), "a") as file:
            file.write(local_path + "\n")
            
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
