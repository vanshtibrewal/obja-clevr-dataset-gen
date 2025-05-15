"""
Blender script to generate CLEVR-like scenes from directories each containing a single .glb file.
Each .glb file is placed at the origin for rendering.

Run with:
    blender --background --python render_single_glb_scenes.py -- [arguments]
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
    print("Running render_single_glb_scenes.py from Blender and cannot import utils.py.")
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

# Define help string for input_scenes_root_dir
input_scenes_help_text = f"Root directory with scene subdirectories, each having one .glb for origin placement."
parser.add_argument(
    '--input_scenes_root_dir',
    required=True,
    help=input_scenes_help_text
)
# Removed --coords_json as coordinates are not used.

# Settings for object placement (min_dist and margin are less relevant for a single object at origin)
# parser.add_argument('--min_dist', default=0.25, type=float, # Not used for single object
#     help="The minimum allowed distance between object centers")
# parser.add_argument('--margin', default=0.4, type=float, # Not used for single object
#     help="Along all cardinal directions (left, right, front, back), all " +
#          "objects will be at least this distance apart. This makes resolving " +
#          "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.") # Still relevant for the single object

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--filename_prefix', default='CLEVR_SINGLE', # Changed prefix
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new_single', # Changed split name
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output_single/images/', # Changed output dir
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output_single/scenes/', # Changed output dir
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
# parser.add_argument('--output_scene_file', default='../output_single/CLEVR_scenes.json', # Combined JSON not primary focus now
#     help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output_single/blendfiles', # Changed output dir
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0_single', # Changed version
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
parser.add_argument('--key_light_jitter', default=0.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=0.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=0.0, type=float,
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
parser.add_argument('--light_energy_multiplier', type=float, default=1.0,
    help="Multiply the energy of the Key, Fill, and Back lights by this factor.")
parser.add_argument('--exposure_offset', type=float, default=0.0,
    help="Adjust the scene exposure value in color management (positive values increase brightness).")

def main(args):
    random.seed(args.start_idx + int(dt.now().timestamp()))
    
    if not os.path.isdir(args.input_scenes_root_dir):
        print(f"Error: Input scenes root directory {args.input_scenes_root_dir} not found.")
        sys.exit(1)

    # Create output directories if they don't exist
    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir, exist_ok=True)
    
    output_index = args.start_idx
    
    # Iterate through each subdirectory in the input_scenes_root_dir
    for scene_key in sorted(os.listdir(args.input_scenes_root_dir)):
        scene_dir_path = os.path.join(args.input_scenes_root_dir, scene_key)
        
        if not os.path.isdir(scene_dir_path):
            print(f"Skipping {scene_dir_path}, not a directory.")
            continue
            
        # Find the .glb file in the scene directory
        glb_files = glob.glob(os.path.join(scene_dir_path, '*.glb'))
        if not glb_files:
            print(f"Warning: No .glb file found in directory {scene_dir_path}. Skipping scene {scene_key}.")
            continue
        if len(glb_files) > 1:
            print(f"Warning: Multiple .glb files found in {scene_dir_path}. Using the first one: {glb_files[0]}. Skipping others.")
        
        single_glb_file = glb_files[0]
        
        # Create scene-specific output subdirectories
        scene_img_dir = os.path.join(args.output_image_dir, scene_key)
        scene_json_dir = os.path.join(args.output_scene_dir, scene_key)
        os.makedirs(scene_img_dir, exist_ok=True)
        os.makedirs(scene_json_dir, exist_ok=True)

        blend_path = None
        if args.save_blendfiles == 1:
            scene_blend_dir = os.path.join(args.output_blend_dir, scene_key)
            os.makedirs(scene_blend_dir, exist_ok=True)
            blend_path = os.path.join(scene_blend_dir, f"{scene_key}.blend")

        # Define the single object to be placed at the origin
        # Shape will be filled by add_single_object
        object_to_place = {
            '3d_coords': [0.0, 0.0, 0.0], # Placed at origin
            'rotation': 0.0, # No rotation by default
            'shape': '' # Will be set from glb filename
        }
        
        print(f"Rendering scene {scene_key} ({output_index}) using {os.path.basename(single_glb_file)}")
        
        render_scene(
            args,
            glb_file_to_render=single_glb_file,
            object_data_for_json=[object_to_place], # Pass as a list with one item
            output_index=output_index,
            output_split=args.split,
            base_output_image_dir=args.output_image_dir,
            base_output_scene_dir=args.output_scene_dir,
            output_blendfile=blend_path,
            scene_name=scene_key
        )
        
        output_index += 1

def render_scene(args,
                 glb_file_to_render=None, # Path to the single .glb file
                 object_data_for_json=None, # List containing the single object's data
                 output_index=0,
                 output_split='none',
                 base_output_image_dir='../output/images/',
                 base_output_scene_dir='../output/scenes/',
                 output_blendfile=None,
                 scene_name='scene'):
    """Renders a single scene (one object) from multiple camera angles."""

    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    ground_obj = bpy.data.objects.get("Ground") 
    if ground_obj:
        scale_factor = 2.0 # Keep ground scaled if desired
        print(f"  Scaling ground plane object '{{ground_obj.name}}' found in base scene by {{scale_factor}}x on X and Y.")
        ground_obj.scale.x *= scale_factor 
        ground_obj.scale.y *= scale_factor 
        bpy.context.view_layer.update()

        dark_floor_mat = bpy.data.materials.get("DarkFloor")
        if dark_floor_mat is None:
            dark_floor_mat = bpy.data.materials.new(name="DarkFloor")
            dark_floor_mat.use_nodes = True
            nodes = dark_floor_mat.node_tree.nodes
            links = dark_floor_mat.node_tree.links
            nodes.clear()
            pbsdf  = nodes.new("ShaderNodeBsdfPrincipled")
            pbsdf.inputs["Base Color"].default_value = (0.18, 0.18, 0.18, 1) # (0.3, 0.3, 0.3, 1)  # medium grey
            pbsdf.inputs["Roughness"].default_value = 0.95                   # very matte
            pbsdf.inputs["Specular"].default_value  = 0.00                   # no hot spots
            out = nodes.new("ShaderNodeOutputMaterial")
            links.new(pbsdf.outputs["BSDF"], out.inputs["Surface"])
        ground_obj.data.materials.clear()
        ground_obj.data.materials.append(dark_floor_mat)

        # dark_floor_mat = bpy.data.materials.get("DarkFloor")
        # if dark_floor_mat is None:
        #     dark_floor_mat = bpy.data.materials.new(name="DarkFloor")
        #     dark_floor_mat.use_nodes = True
        #     nodes = dark_floor_mat.node_tree.nodes
        #     links = dark_floor_mat.node_tree.links
        #     nodes.clear()
        #     pbsdf  = nodes.new("ShaderNodeBsdfPrincipled")
        #     pbsdf.inputs["Base Color"].default_value = (0.15, 0.15, 0.15, 1)  # dark grey
        #     pbsdf.inputs["Roughness"].default_value = 0.95                   # very matte
        #     pbsdf.inputs["Specular"].default_value  = 0.00                   # no hot spots
        #     out = nodes.new("ShaderNodeOutputMaterial")
        #     links.new(pbsdf.outputs["BSDF"], out.inputs["Surface"])
        # ground_obj.data.materials.clear()
        # ground_obj.data.materials.append(dark_floor_mat)
    else:
        print("  Warning: Could not find object named 'Ground' to scale in base scene.") 

    utils.load_materials(args.material_dir)

    scene = bpy.context.scene
    render_args = scene.render
    render_args.engine = "CYCLES"
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    scene.cycles.tile_size = args.render_tile_size
    scene.cycles.samples = args.render_num_samples
    scene.cycles.transparent_min_bounces = args.render_min_bounces
    scene.cycles.transparent_max_bounces = args.render_max_bounces
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 10.0

    if args.use_gpu:
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        # ... (rest of GPU setup identical to original script)
        cycles_prefs.refresh_devices()
        for d in cycles_prefs.devices:
            if d.type in {'CUDA', 'OPTIX'}:
                d.use = True
        for sc in bpy.data.scenes:
            sc.cycles.device = 'GPU'


    scene_img_dir = os.path.join(base_output_image_dir, scene_name)
    scene_json_dir = os.path.join(base_output_scene_dir, scene_name)

    base_scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'scene_name': scene_name,
        'objects': [], # Populated by add_single_object
        'directions': {},
        'relationships': {}
    }

    bpy.ops.mesh.primitive_plane_add(size=10) # For cardinal directions, as in original
    # plane = bpy.context.object # Not explicitly used beyond this function scope in original

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    cam = scene.objects['Camera']
    initial_cam_location = cam.location.copy()
    initial_cam_rotation_euler = cam.rotation_euler.copy()

    bpy.ops.object.empty_add(location=(0, 0, 0))
    target_empty = bpy.context.object
    target_empty.name = "Camera_Target"
    target_empty.location = (0.0, 0.0, 0.0)
    
    if args.exposure_offset != 0.0:
        print(f"  Applying exposure offset: {args.exposure_offset}")
        scene.view_settings.exposure = args.exposure_offset

    light_names = ["Lamp_Key", "Lamp_Fill", "Lamp_Back"]
    light_jitters = {
        "Lamp_Key": args.key_light_jitter,
        "Lamp_Fill": args.fill_light_jitter,
        "Lamp_Back": args.back_light_jitter
    }
    initial_light_energies = {}

    for light_name in light_names:
        light_obj = scene.objects.get(light_name)
        if light_obj and light_obj.data:
            initial_light_energies[light_name] = light_obj.data.energy
            if args.light_energy_multiplier != 1.0:
                light_obj.data.energy *= args.light_energy_multiplier
                print(f"  Adjusted {light_name} energy to {light_obj.data.energy:.2f}")
            jitter = light_jitters.get(light_name, 0.0)
            if jitter > 0:
                light_obj.location.x += rand(jitter)
                light_obj.location.y += rand(jitter)
                light_obj.location.z += rand(jitter)

    # Populate scene with the single specified object
    # add_single_object will modify object_data_for_json[0]['shape']
    objs_info, blender_obj_list = add_single_object(
        base_scene_struct, # Pass to populate 'objects' list within it
        glb_file_to_render, 
        object_data_for_json[0], # Pass the single object's data dictionary
        args
    )

    if objs_info is None or blender_obj_list is None:
        print(f"Warning: Failed to place object for scene {scene_name}. Skipping.")
        utils.delete_object(target_empty)
        return
    
    # 'objects' list in base_scene_struct is already populated by add_single_object
    # base_scene_struct['objects'] = objs_info # This line is now redundant

    if output_blendfile:
        print(f"  Saving blend file to {output_blendfile}")
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    # --- Render 'angle_base' (original camera position) ---
    print(f"    Angle base")
    cam.location = initial_cam_location
    cam.rotation_euler = initial_cam_rotation_euler
    for c in list(cam.constraints):
        if c.type == 'TRACK_TO':
            cam.constraints.remove(c)
    bpy.context.view_layer.update()

    angle_base_scene_struct = json.loads(json.dumps(base_scene_struct)) # Deep copy
    angle_base_prefix = f"{scene_name}_angle_base"
    angle_base_img_path = os.path.join(scene_img_dir, f"{angle_base_prefix}.png")
    angle_base_scene_path = os.path.join(scene_json_dir, f"{angle_base_prefix}.json")

    render_args.filepath = angle_base_img_path
    angle_base_scene_struct['image_filename'] = os.path.basename(angle_base_img_path)
    angle_base_scene_struct['camera_angle_degrees'] = 'base'
    angle_base_scene_struct['camera_location'] = tuple(cam.location)
    angle_base_scene_struct['camera_rotation_euler'] = tuple(cam.rotation_euler)

    # Update pixel_coords for the single object
    if angle_base_scene_struct['objects']: # Should be one object
        blender_obj = blender_obj_list[0] # Only one object
        try:
            angle_base_scene_struct['objects'][0]['pixel_coords'] = utils.get_camera_coords(cam, blender_obj.location)
        except Exception as e:
            print(f"Error getting pixel coords for object at angle_base: {e}")
            angle_base_scene_struct['objects'][0]['pixel_coords'] = [-1,-1,-1]
    
    # Cardinal directions for 'angle_base'
    bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,-0.1))
    temp_plane_obj = bpy.context.object
    plane_normal_vec = Vector((0.0, 0.0, 1.0))
    quat_base = cam.matrix_world.to_quaternion()
    cam_forward_base = (quat_base @ Vector((0, 0, -1))).normalized()
    cam_left_base = (quat_base @ Vector((-1, 0, 0))).normalized()
    dir_front_base = (cam_forward_base - cam_forward_base.project(plane_normal_vec)).normalized()
    dir_left_base = (cam_left_base - cam_left_base.project(plane_normal_vec)).normalized()
    
    angle_base_scene_struct['directions'] = {
        'front': tuple(dir_front_base), 'behind': tuple(-dir_front_base),
        'left': tuple(dir_left_base), 'right': tuple(-dir_left_base),
    }
    utils.delete_object(temp_plane_obj)

    angle_base_scene_struct['relationships'] = compute_all_relationships(angle_base_scene_struct)
    bpy.ops.render.render(write_still=True)
    with open(angle_base_scene_path, "w") as f:
        json.dump(angle_base_scene_struct, f, indent=2)

    # --- Render 4 Cardinal Angles --- 
    print(f"  Setting up tracking constraint for cardinal angles...")
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target_empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    cam_dist = 7.5
    cam_height = 6.0
    angles_deg = [0, 90, 180, 270]
    angles_rad = [math.radians(d) for d in angles_deg]

    for angle_idx, angle_rad in enumerate(angles_rad):
        print(f"    Angle {angle_idx} ({angles_deg[angle_idx]} deg)")
        cam_x = cam_dist * math.cos(angle_rad)
        cam_y = cam_dist * math.sin(angle_rad)
        cam.location = (cam_x, cam_y, cam_height)
        bpy.context.view_layer.update()

        angle_scene_struct = json.loads(json.dumps(base_scene_struct)) # Deep copy
        angle_prefix = f"{scene_name}_angle_{angle_idx}"
        angle_img_path = os.path.join(scene_img_dir, f"{angle_prefix}.png")
        angle_scene_path = os.path.join(scene_json_dir, f"{angle_prefix}.json")

        render_args.filepath = angle_img_path
        angle_scene_struct['image_filename'] = os.path.basename(angle_img_path)
        angle_scene_struct['camera_angle_degrees'] = angles_deg[angle_idx]
        angle_scene_struct['camera_location'] = tuple(cam.location)
        angle_scene_struct['camera_rotation_euler'] = tuple(cam.rotation_euler)

        if angle_scene_struct['objects']: # Should be one object
            blender_obj = blender_obj_list[0] # Only one object
            angle_scene_struct['objects'][0]['pixel_coords'] = utils.get_camera_coords(cam, blender_obj.location)

        bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,-0.1))
        # plane = bpy.context.object # Unused variable
        plane_normal = Vector((0.0, 0.0, 1.0))
        quat = cam.matrix_world.to_quaternion()
        cam_forward = (quat @ Vector((0, 0, -1))).normalized()
        cam_left = (quat @ Vector((-1, 0, 0))).normalized()
        plane_front = (cam_forward - cam_forward.project(plane_normal)).normalized()
        plane_left_dir = (cam_left - cam_left.project(plane_normal)).normalized()
        
        angle_scene_struct['directions'] = {
            'front': tuple(plane_front), 'behind': tuple(-plane_front),
            'left': tuple(plane_left_dir), 'right': tuple(-plane_left_dir),
        }
        # utils.delete_object(plane) # This temp plane for cardinal directions was cleaned up per angle in original - good practice

        angle_scene_struct['relationships'] = compute_all_relationships(angle_scene_struct)
        bpy.ops.render.render(write_still=True)
        with open(angle_scene_path, "w") as f:
            json.dump(angle_scene_struct, f, indent=2)
        # Ensure temp plane is deleted if logic changes where it's created.
        # The current loop creates and implicitly leaves the plane.
        # A robust way is to get reference and delete:
        temp_plane_for_cardinal_dir = bpy.context.object # The plane just added
        utils.delete_object(temp_plane_for_cardinal_dir)


    utils.delete_object(target_empty)
    # Original script deleted a global plane at the end - this script uses temp planes.

def add_single_object(scene_struct, glb_file, object_data, args):
    """Places a single object from the provided .glb file at specified coords (usually origin)."""
    
    coords = object_data['3d_coords']
    theta = object_data['rotation']
    
    x, y, z_offset = coords[0], coords[1], 0.0 # z is typically 0 for ground placement
    
    try:
        # utils.add_object_glb now takes (filepath, (x,y), theta)
        # The original add_object_glb seems to handle only x,y for location and then applies z based on bounding box.
        # We'll assume it places object on ground (z=0 effectively before its internal logic)
        utils.add_object_glb_scale(glb_file, (x, y, -0.025), theta=theta, scale=12) # TODO: change
        
        obj = bpy.context.object # The newly added object
        
        # Update object_data with actual info
        object_data['shape'] = os.path.basename(glb_file) # Use .glb filename as shape
        object_data['3d_coords'] = [x, y, 0.0] # Specified coords (z assumed 0 on plane)
        object_data['3d_coords_transformed'] = tuple(obj.location) # Actual placed Blender coords
        # 'pixel_coords' removed here, will be added per angle

        # Update the scene_struct's objects list
        scene_struct['objects'] = [object_data] # Replace with list containing this single object
        
        blender_objects_list = [obj]

        # Visibility test for the single object
        if not check_visibility(blender_objects_list, args.min_pixels_per_object):
            print(f"Warning: Initial visibility test failed for {object_data['shape']}. Object may be occluded or too small from the initial camera pose.")
            # Continue rendering even if visibility check fails for a single object scenario

        return scene_struct['objects'], blender_objects_list
    
    except Exception as e:
        print(f"Error placing object from {glb_file}: {e}")
        return None, None


def compute_all_relationships(scene_struct, eps=0.2):
    """Computes spatial relationships. For a single object, this will result in empty relationship lists."""
    rels = {}
    if not scene_struct['objects']: # No objects, no relationships
        return rels
        
    for name, direction_vec in scene_struct['directions'].items():
        if name in ('above', 'below'): # Assuming these are not primary focus or handled differently
            continue
        rels[name] = []
        # If there's only one object, the inner loop `if i != j` will prevent any relationships.
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords_transformed']
            # For a single object, 'related' will always be an empty list.
            related = [j for j, obj2 in enumerate(scene_struct['objects'])
                       if i != j and sum((Vector(obj2['3d_coords_transformed'])[k]-Vector(coords1)[k])*direction_vec[k] for k in range(3)) > eps]
            rels[name].append(related)
    return rels

def check_visibility(blender_objects, min_pixels):
    """Check if all objects in the scene are sufficiently visible."""
    if not blender_objects: # Should not happen if add_single_object succeeded
        return True # Or False, depending on desired behavior for "no objects"
        
    # Ensure there's a camera in the scene
    if not bpy.context.scene.camera:
        print("Error in check_visibility: No camera found in the scene.")
        return False # Cannot check visibility without a camera

    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    # render_shadeless expects a list of blender objects
    colors = render_shadeless(blender_objects, path) 
    
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"Error in check_visibility: Rendered image {path} is empty or not found.")
        if os.path.exists(path): os.remove(path)
        return False # Render failed or produced no output

    img = bpy.data.images.load(path)
    pixels = list(img.pixels)
    bpy.data.images.remove(img) # Clean up loaded image
    os.remove(path)

    if not pixels:
        print("Error in check_visibility: Image has no pixel data.")
        return False

    step = 4  # RGBA
    counts = Counter(tuple(pixels[i:i+4]) for i in range(0, len(pixels), step))
    
    # In render_shadeless, each object gets a unique color. Background is another color.
    # So, len(counts) should be len(blender_objects) + 1 (for background)
    # However, if an object is completely occluded, its color won't appear.
    
    # Count how many of the assigned object colors appear in the render
    num_visible_objects = 0
    # `colors` is a set of (r,g,b,a) tuples from render_shadeless
    # `counts.keys()` are also (r,g,b,a) tuples from the rendered image pixels
    for obj_color in colors:
        if obj_color in counts and counts[obj_color] >= min_pixels:
            num_visible_objects +=1
        elif obj_color in counts: # Object is visible but below threshold
            # This print can be verbose if many objects are small
            # print(f"  Visibility warning: An object rendered with color {obj_color} has only {counts[obj_color]} pixels (min: {min_pixels}).")
            pass


    if num_visible_objects < len(blender_objects):
        # This means one or more objects didn't get any pixels or not enough.
        # print(f"  Visibility check: {num_visible_objects} objects visible >= {min_pixels}px, expected {len(blender_objects)}.")
        return False
        
    return True # All objects assigned a color in shadeless render are visible with enough pixels

def render_shadeless(blender_objects, path):
    """Flat-shade render that counts visible pixels for each *logical* object."""
    scene = bpy.context.scene
    render_args = scene.render

    old_filepath = render_args.filepath
    old_engine = scene.render.engine
    old_samples = scene.cycles.samples if old_engine == 'CYCLES' else None # Store Cycles samples if it's the engine
    old_use_nodes_world = scene.world.use_nodes if scene.world else False
    old_material_override = scene.view_layers[0].material_override if scene.view_layers else None


    render_args.filepath = path
    # Use EEVEE for speed if available and appropriate for shadeless, or basic BLENDER_WORKBENCH
    # For simple shadeless emission, CYCLES with 1 sample is also fine and was original.
    # Let's stick to original logic to minimize changes unless performance is an issue.
    scene.render.engine = 'CYCLES' 
    scene.cycles.samples = 1

    # Disable world lighting for shadeless by unsetting use_nodes or using a black background
    if scene.world:
        scene.world.use_nodes = False


    hidden_objs_render_state = []
    for name in ('Lamp_Key', 'Lamp_Fill', 'Lamp_Back', 'Ground'): # Include 'Ground'
        obj = scene.objects.get(name)
        if obj:
            hidden_objs_render_state.append((obj, obj.hide_render))
            obj.hide_render = True
    
    # For shadeless, also ensure no material override is globally set that would interfere
    if scene.view_layers:
        scene.view_layers[0].material_override = None


    object_colors = set()
    old_slots_map = []

    for i, obj in enumerate(blender_objects):
        old_slots_map.append([slot.material for slot in obj.material_slots])
        flat_material_name = f"Flat_Emission_{i}"
        flat = bpy.data.materials.get(flat_material_name)
        if not flat:
            flat = bpy.data.materials.new(name=flat_material_name)
        flat.use_nodes = True
        nodes = flat.node_tree.nodes
        links = flat.node_tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emit = nodes.new("ShaderNodeEmission")

        # Generate unique color
        while True:
            # Ensure good separation for color counting, avoid very dark/light colors
            # that might blend with aliasing or background if not careful.
            r, g, b = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
            # Ensure distinctness by checking against already used colors, simple check
            color_tuple = (round(r,3), round(g,3), round(b,3)) # Reduce precision for comparison
            is_unique = True
            for c_r, c_g, c_b, _ in object_colors: # Compare only RGB
                if abs(c_r-r) < 0.05 and abs(c_g-g) < 0.05 and abs(c_b-b) < 0.05: # Simple proximity check
                    is_unique = False
                    break
            if is_unique:
                break
        
        emit.inputs["Color"].default_value = (r, g, b, 1.0) # Full Alpha
        links.new(emit.outputs["Emission"], output.inputs["Surface"])
        
        # Store the exact color tuple used for the material
        current_obj_color = (emit.inputs["Color"].default_value[0],
                             emit.inputs["Color"].default_value[1],
                             emit.inputs["Color"].default_value[2],
                             emit.inputs["Color"].default_value[3])
        object_colors.add(current_obj_color)


        if not obj.material_slots:
            obj.data.materials.append(flat)
        else:
            for slot in obj.material_slots:
                slot.material = flat
    
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print(f"Error during shadeless render: {e}")
        # Fall through to cleanup

    for obj, old_mats in zip(blender_objects, old_slots_map):
        # Ensure obj.data exists
        if not obj.data:
            print(f"Warning: Object {obj.name} has no data block, cannot restore materials.")
            continue
        
        # Add material slots if mesh originally had more than it does now
        # (e.g. if they were cleared instead of overwritten)
        # This shouldn't be necessary if we just overwrite slot.material
        # while len(obj.material_slots) < len(old_mats):
        #     obj.data.materials.append(None) # This adds a new slot with no material

        for i, slot in enumerate(obj.material_slots):
            if i < len(old_mats):
                slot.material = old_mats[i]
            # else: if new slots were added beyond original count, this could be an issue
                # This case should ideally not happen if we only replace existing slots.

    for obj, hidden_state in hidden_objs_render_state:
        if obj: # Object might have been deleted elsewhere
            obj.hide_render = hidden_state
    
    render_args.filepath = old_filepath
    scene.render.engine = old_engine # Restore engine
    if old_engine == 'CYCLES' and old_samples is not None:
        scene.cycles.samples = old_samples
    
    if scene.world: # Restore world settings
        scene.world.use_nodes = old_use_nodes_world
    if scene.view_layers and old_material_override: # Restore material override
         scene.view_layers[0].material_override = old_material_override


    return object_colors


if __name__ == '__main__':
    if not INSIDE_BLENDER:
        print("This script must be run from within Blender 3.x or newer:")
        print("  blender --background --python render_single_glb_scenes.py -- [args]")
        sys.exit(1)

    if bpy.app.version < (3,0,0):
        print("Warning: This script is designed for Blender 3.x or newer. You are using:", bpy.app.version_string)
        # sys.exit(1) # Could exit, or just warn.

    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args) 