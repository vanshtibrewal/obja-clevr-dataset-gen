"""
Blender script to generate CLEVR-like scenes with specified objects and positions.
This script loads object models (.glb files) from specified directories and positions them
according to coordinates given in a JSON file.

Run with:
    blender --background --python render_specified_scenes.py -- [arguments]
"""

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile, csv, glob, re
from datetime import datetime as dt
from collections import Counter

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
    print("Using utils from:", utils.__file__)
  except ImportError as e:
    print("\nERROR")
    print("Running render_specified_scenes.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining materials and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "the \"materials\" field contains material definitions.")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--input_scenes_dir', required=True,
    help="Directory containing subdirectories, each representing a scene with .glb files")
parser.add_argument('--coords_json', required=True,
    help="JSON file containing scene definitions with object coordinates and rotations")

# Settings for object placement
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

def main(args):
    random.seed(args.start_idx + int(dt.now().timestamp()))
    
    # Create output directories if they don't exist
    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir)
    
    # Load the JSON file containing scene definitions
    print(f"Loading scene definitions from {args.coords_json}")
    with open(args.coords_json, 'r') as f:
        scenes_data = json.load(f)
    
    # Check if the JSON has the expected structure
    if 'scenes' not in scenes_data:
        print(f"Error: JSON file {args.coords_json} must contain a 'scenes' key")
        sys.exit(1)
    
    all_scene_paths = []
    num_digits = 6
    output_index = args.start_idx
    
    # Iterate through each scene in the JSON file
    for i, scene_data in enumerate(scenes_data['scenes']):
        if 'key' not in scene_data or 'objects' not in scene_data:
            print(f"Warning: Scene {i} is missing 'key' or 'objects' field. Skipping.")
            continue
        
        scene_key = scene_data['key']
        scene_dir = os.path.join(args.input_scenes_dir, scene_key)
        
        # Check if the scene directory exists
        if not os.path.isdir(scene_dir):
            print(f"Warning: Scene directory {scene_dir} not found. Skipping scene {scene_key}.")
            continue
        
        # Create output filenames
        prefix = scene_key
        img_path = os.path.join(args.output_image_dir, f"{prefix}.png")
        scene_path = os.path.join(args.output_scene_dir, f"{prefix}.json")
        all_scene_paths.append(scene_path)
        
        blend_path = None
        if args.save_blendfiles == 1:
            blend_path = os.path.join(args.output_blend_dir, f"{prefix}.blend")
        
        # Get the .glb files in the scene directory
        glb_files = sorted(glob.glob(os.path.join(scene_dir, '*.glb')))
        
        # Check if we have enough .glb files for the objects in the scene
        if len(glb_files) != len(scene_data['objects']):
            print(f"Warning: Mismatch between number of .glb files ({len(glb_files)}) and number of objects in JSON ({len(scene_data['objects'])}) for scene {scene_key}. Skipping.")
            continue
        
        print(f"Rendering scene {scene_key} ({output_index})")
        
        # Render the scene
        render_scene(
            args,
            glb_files=glb_files,
            object_coords=scene_data['objects'],
            output_index=output_index,
            output_split=args.split,
            output_image=img_path,
            output_scene=scene_path,
            output_blendfile=blend_path,
            scene_name=scene_key
        )
        
        output_index += 1
    
    # # After rendering all images, combine the JSON files for each scene into a single JSON file
    # all_scenes = []
    # for scene_path in all_scene_paths:
    #     if os.path.exists(scene_path):
    #         with open(scene_path, 'r') as f:
    #             all_scenes.append(json.load(f))
    #     else:
    #         print(f"Warning: Scene file {scene_path} not found, skipping in combined output.")
    
    # output = {
    #     'info': {
    #         'date': args.date,
    #         'version': args.version,
    #         'split': args.split,
    #         'license': args.license,
    #     },
    #     'scenes': all_scenes
    # }
    
    # print(f"Writing combined scene data to {args.output_scene_file}")
    # with open(args.output_scene_file, 'w') as f:
    #     json.dump(output, f)

def render_scene(args,
                 glb_files=[],
                 object_coords=[],
                 output_index=0,
                 output_split='none',
                 output_image='render.png',
                 output_scene='render_json',
                 output_blendfile=None,
                 scene_name='scene'):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    scene = bpy.context.scene
    render_args = scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    scene.cycles.tile_size = args.render_tile_size
    scene.cycles.samples = args.render_num_samples
    scene.cycles.transparent_min_bounces = args.render_min_bounces
    scene.cycles.transparent_max_bounces = args.render_max_bounces
    
    scene.cycles.sample_clamp_direct = 0.0   # leave direct light unclamped
    scene.cycles.sample_clamp_indirect = 10.0  # clamp very bright indirect paths

    if args.use_gpu:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.refresh_devices()
        for d in cycles_prefs.devices:
            if d.type in {'CUDA', 'OPTIX'}:
                d.use = True
        for sc in bpy.data.scenes:
            sc.cycles.device = 'GPU'

    # ---------------------------------------------------------------------
    # Scene layout metadata
    # ---------------------------------------------------------------------
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
        'scene_name': scene_name,
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(size=10)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    cam = scene.objects['Camera']
    if args.camera_jitter:
        cam.location.x += rand(args.camera_jitter)
        cam.location.y += rand(args.camera_jitter)
        cam.location.z += rand(args.camera_jitter)

    # Local axes on ground plane
    plane_normal = plane.data.vertices[0].normal
    quat = cam.matrix_world.to_quaternion()
    cam_behind = (quat @ Vector((0, 0, -1))).normalized()
    cam_left = (quat @ Vector((-1, 0, 0))).normalized()
    cam_up = (quat @ Vector((0, 1, 0))).normalized()

    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Clean up helper plane
    utils.delete_object(plane)

    # Store directions
    scene_struct['directions'] = {
        'behind': tuple(plane_behind),
        'front': tuple(-plane_behind),
        'left': tuple(plane_left),
        'right': tuple(-plane_left),
        'above': tuple(plane_up),
        'below': tuple(-plane_up),
    }

    # Jitter lights (object names come from base_scene.blend)
    for light_name, jitter in [("Lamp_Key", args.key_light_jitter),
                              ("Lamp_Fill", args.fill_light_jitter),
                              ("Lamp_Back", args.back_light_jitter)]:
        if jitter:
            obj = scene.objects.get(light_name)
            if obj:
                obj.location.x += rand(jitter)
                obj.location.y += rand(jitter)
                obj.location.z += rand(jitter)

    # Populate scene with specified objects
    objs, blender_objs = add_specified_objects(scene_struct, glb_files, object_coords, args, cam)

    # Check if we successfully placed all objects
    if objs is None or blender_objs is None:
        print(f"Warning: Failed to place objects for scene {scene_name}. Skipping.")
        return

    # Render final image
    scene_struct['objects'] = objs
    scene_struct['relationships'] = compute_all_relationships(scene_struct)
    bpy.ops.render.render(write_still=True)

    # Save per‑image JSON
    with open(output_scene, "w") as f:
        json.dump(scene_struct, f, indent=2)

    # Optionally save the .blend
    if output_blendfile:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

def add_specified_objects(scene_struct, glb_files, object_coords, args, camera):
    """Place objects at specified positions from the provided glb files."""
    objects, blender_objects = [], []
    
    # Make sure we have the same number of .glb files and object coordinates
    if len(glb_files) != len(object_coords):
        print(f"Error: Mismatch between number of .glb files ({len(glb_files)}) and object coordinates ({len(object_coords)})")
        return None, None
    
    # Place each object at its specified position
    for i, (glb_file, obj_data) in enumerate(zip(glb_files, object_coords)):
        if '3d_coords' not in obj_data:
            print(f"Warning: Object {i} is missing '3d_coords' field. Skipping.")
            continue
        
        coords = obj_data['3d_coords']
        if len(coords) < 3:
            print(f"Warning: Object {i} has insufficient coordinates: {coords}. Skipping.")
            continue
        
        # Extract coordinates from the object data
        x, y, theta = coords[0], coords[1], coords[2]
        
        # Add the object at the specified position
        try:
            utils.add_object_glb(glb_file, (x, y), theta=theta)
            
            obj = bpy.context.object
            blender_objects.append(obj)
            
            # Record metadata for this object
            pix = utils.get_camera_coords(camera, obj.location)
            objects.append({
                'shape': os.path.basename(glb_file),
                '3d_coords': [x, y, 0.0],
                '3d_coords_transformed': tuple(obj.location),
                'rotation': theta,
                'pixel_coords': pix,
            })
        except Exception as e:
            print(f"Error placing object from {glb_file}: {e}")
            # Clean up any objects placed so far
            for o in blender_objects:
                utils.delete_object(o)
            return None, None
    
    # Visibility test
    if not check_visibility(blender_objects, args.min_pixels_per_object):
        print("Warning: Visibility test failed. Some objects may be occluded or too small.")
        # Instead of recreating the scene like in random placement, we'll log the warning and continue
    
    return objects, blender_objects

def compute_all_relationships(scene_struct, eps=0.2):
    rels = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name in ('above', 'below'):
            continue
        rels[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = [j for j, obj2 in enumerate(scene_struct['objects'])
                       if i != j and sum((obj2['3d_coords'][k]-coords1[k])*direction_vec[k] for k in range(3)) > eps]
            rels[name].append(related)
    return rels

def check_visibility(blender_objects, min_pixels):
    """Check if all objects in the scene are sufficiently visible."""
    if not blender_objects:
        return False
        
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    colors = render_shadeless(blender_objects, path)
    img = bpy.data.images.load(path)
    pixels = list(img.pixels)
    os.remove(path)

    step = 4  # RGBA
    counts = Counter(tuple(pixels[i:i+4]) for i in range(0, len(pixels), step))
    if len(counts) != len(blender_objects) + 1:
        return False
    return all(c >= min_pixels for c in counts.values())

def render_shadeless(blender_objects, path):
    """Flat-shade render that counts visible pixels for each *logical* object."""
    scene = bpy.context.scene
    render_args = scene.render

    # Backup settings we will touch
    old_filepath = render_args.filepath
    old_samples = scene.cycles.samples

    render_args.filepath = path
    scene.cycles.samples = 1  # speed up dummy render

    # Hide lights & ground
    hidden_objs = []
    for name in ('Lamp_Key', 'Lamp_Fill', 'Lamp_Back', 'Ground'):
        obj = scene.objects.get(name)
        if obj and not obj.hide_render:
            obj.hide_render = True
            hidden_objs.append(obj)

    # Assign unique emission mats
    object_colors = set()
    old_slots_map = []      # list[ list[bpy.types.Material] ]

    for i, obj in enumerate(blender_objects):
        # remember the full slot → material mapping
        old_slots_map.append([slot.material for slot in obj.material_slots])

        # build a flat-emission material in a unique colour
        flat = bpy.data.materials.new(name=f"Flat_{i}")
        flat.use_nodes = True
        nodes = flat.node_tree.nodes
        links = flat.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emit = nodes.new("ShaderNodeEmission")

        while True:
            r, g, b = random.random(), random.random(), random.random()
            if (r, g, b) not in object_colors:
                break
        emit.inputs["Color"].default_value = (r, g, b, 1.0)
        links.new(emit.outputs["Emission"], output.inputs["Surface"])
        object_colors.add((r, g, b, 1.0))

        # swap – **do not clear** slots, just repoint them
        if not obj.material_slots:                         # mesh had no slots
            obj.data.materials.append(flat)
        for slot in obj.material_slots:
            slot.material = flat

    # Render
    bpy.ops.render.render(write_still=True)

    # Restore original material pointers
    for obj, old_mats in zip(blender_objects, old_slots_map):
        # add slots if the mesh originally had more than it does now
        while len(obj.material_slots) < len(old_mats):
            obj.data.materials.append(None)

        for slot, original in zip(obj.material_slots, old_mats):
            slot.material = original

    # Un-hide lights/ground and restore render settings
    for obj in hidden_objs:
        obj.hide_render = False

    # Restore settings
    render_args.filepath = old_filepath
    scene.cycles.samples = old_samples

    return object_colors

if __name__ == '__main__':
    if not INSIDE_BLENDER:
        print("This script must be run from within Blender 3.x:")
        print("  blender --background --python render_specified_scenes.py -- [args]")
        sys.exit(1)

    argv = utils.extract_args()  # strips the "--" Blender passes before our args
    args = parser.parse_args(argv)
    main(args) 