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
    
    # Create output directories if they don't exist
    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir, exist_ok=True)
    
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
        
        # Create scene-specific output subdirectories
        scene_img_dir = os.path.join(args.output_image_dir, scene_key)
        scene_json_dir = os.path.join(args.output_scene_dir, scene_key)
        os.makedirs(scene_img_dir, exist_ok=True)
        os.makedirs(scene_json_dir, exist_ok=True)

        blend_path = None
        if args.save_blendfiles == 1:
            # Save blend file per scene key in its own subdir within blend dir
            scene_blend_dir = os.path.join(args.output_blend_dir, scene_key)
            os.makedirs(scene_blend_dir, exist_ok=True)
            blend_path = os.path.join(scene_blend_dir, f"{scene_key}.blend")

        # Get the .glb files in the scene directory
        glb_files = sorted(glob.glob(os.path.join(scene_dir, '*.glb')))
        
        # Check if we have enough .glb files for the objects in the scene
        if len(glb_files) != len(scene_data['objects']):
            print(f"Warning: Mismatch between number of .glb files ({len(glb_files)}) and number of objects in JSON ({len(scene_data['objects'])}) for scene {scene_key}. Skipping.")
            continue
        
        print(f"Rendering scene {scene_key} ({output_index})")
        
        # Render the scene from multiple angles
        render_scene(
            args,
            glb_files=glb_files,
            object_coords=scene_data['objects'],
            output_index=output_index,
            output_split=args.split,
            base_output_image_dir=args.output_image_dir, # Pass base dirs
            base_output_scene_dir=args.output_scene_dir,
            output_blendfile=blend_path,
            scene_name=scene_key
        )
        
        output_index += 1
    
    # # Combined JSON generation is commented out as output structure changed
    # all_scenes = []
    # for scene_path in all_scene_paths:
    #     if os.path.exists(scene_path):
    #         with open(scene_path, 'r') as f:
    #             all_scenes.append(json.load(f))
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
                 object_coords=[], # noqa: B006
                 output_index=0,
                 output_split='none',
                 base_output_image_dir='../output/images/', # Renamed arg
                 base_output_scene_dir='../output/scenes/', # Renamed arg
                 output_blendfile=None,
                 scene_name='scene'):
    """Renders a single scene from 4 different camera angles."""

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Attempt to scale the ground plane object if it exists
    ground_obj = bpy.data.objects.get("Ground") 
    if ground_obj:
        scale_factor = 2.0
        print(f"  Scaling ground plane object '{{ground_obj.name}}' found in base scene by {{scale_factor}}x on X and Y.")
        ground_obj.scale.x *= scale_factor 
        ground_obj.scale.y *= scale_factor 
        bpy.context.view_layer.update() # Apply scale change
    else:
        print("  Warning: Could not find object named 'Ground' to scale in base scene.") 

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    scene = bpy.context.scene
    render_args = scene.render
    render_args.engine = "CYCLES"
    # render_args.filepath set per angle later
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

    # Calculate scene-specific output paths
    scene_img_dir = os.path.join(base_output_image_dir, scene_name)
    scene_json_dir = os.path.join(base_output_scene_dir, scene_name)
    # Directories are already created in main()

    # ---------------------------------------------------------------------
    # Scene layout metadata
    # ---------------------------------------------------------------------
    base_scene_struct = { # Base structure, angle-specific details added later
        'split': output_split,
        'image_index': output_index, # Index of the scene (group of angles)
        'scene_name': scene_name,
        'objects': [], # Object info (except pixel coords) filled by add_specified_objects
        'directions': {}, # Calculated per angle
        'relationships': {} # Calculated per angle
    } # noqa: E501

    # Put a plane on the ground so we can compute cardinal directions later
    bpy.ops.mesh.primitive_plane_add(size=10)
    plane = bpy.context.object # Keep reference for cleanup

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    cam = scene.objects['Camera']
    # Camera jitter is removed for predictable angles
    # if args.camera_jitter: ...

    # Store initial camera state for 'angle_base'
    initial_cam_location = cam.location.copy()
    initial_cam_rotation_euler = cam.rotation_euler.copy()

    # Create an empty object at the origin for the camera to track (for the 4 cardinal angles)
    bpy.ops.object.empty_add(location=(0, 0, 0))
    target_empty = bpy.context.object
    target_empty.name = "Camera_Target"
    
    # Reverted: Cardinal cameras simply track the origin (0,0,0)
    target_empty.location = (0.0, 0.0, 0.0)
    
    # Apply exposure offset
    if args.exposure_offset != 0.0:
        print(f"  Applying exposure offset: {args.exposure_offset}")
        scene.view_settings.exposure = args.exposure_offset

    # Jitter lights (object names come from base_scene.blend)
    light_names = ["Lamp_Key", "Lamp_Fill", "Lamp_Back"]
    light_jitters = { # Map name to jitter arg
        "Lamp_Key": args.key_light_jitter,
        "Lamp_Fill": args.fill_light_jitter,
        "Lamp_Back": args.back_light_jitter
    }
    initial_light_energies = {} # Store original energies

    for light_name in light_names:
        light_obj = scene.objects.get(light_name)
        if light_obj and light_obj.data:
            # Store initial energy before applying multiplier
            initial_light_energies[light_name] = light_obj.data.energy
            # Apply energy multiplier
            if args.light_energy_multiplier != 1.0:
                light_obj.data.energy *= args.light_energy_multiplier
                print(f"  Adjusted {light_name} energy to {light_obj.data.energy:.2f}")
            
            # Apply jitter
            jitter = light_jitters.get(light_name, 0.0)
            if jitter > 0:
                light_obj.location.x += rand(jitter)
                light_obj.location.y += rand(jitter)
                light_obj.location.z += rand(jitter)

    # Populate scene with specified objects
    # Pass base_scene_struct - objects list will be populated
    # Do not pass camera - pixel coords calculated per angle
    objs, blender_objs = add_specified_objects(base_scene_struct, glb_files, object_coords, args) # noqa: E501

    # Check if we successfully placed all objects
    if objs is None or blender_objs is None:
        print(f"Warning: Failed to place objects for scene {scene_name}. Skipping.")
        utils.delete_object(target_empty) # Clean up empty
        return

    # Store placed object data (without pixel coords yet) in base struct
    base_scene_struct['objects'] = objs

    # Optionally save the .blend file *before* the camera angle loop
    # Saves the scene with objects placed, camera constrained but at its initial position
    if output_blendfile:
        print(f"  Saving blend file to {output_blendfile}")
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    # --- Render 'angle_base' (original camera position) ---
    print(f"    Angle base")
    cam.location = initial_cam_location
    cam.rotation_euler = initial_cam_rotation_euler
    # Remove 'Track To' constraint if it exists (e.g., from base scene or previous logic)
    for c in list(cam.constraints): # Iterate over a copy
        if c.type == 'TRACK_TO':
            cam.constraints.remove(c)
    bpy.context.view_layer.update()

    angle_base_scene_struct = json.loads(json.dumps(base_scene_struct))
    angle_base_prefix = f"{scene_name}_angle_base"
    angle_base_img_path = os.path.join(scene_img_dir, f"{angle_base_prefix}.png")
    angle_base_scene_path = os.path.join(scene_json_dir, f"{angle_base_prefix}.json")

    render_args.filepath = angle_base_img_path
    angle_base_scene_struct['image_filename'] = os.path.basename(angle_base_img_path)
    angle_base_scene_struct['camera_angle_degrees'] = 'base' # Indicate base angle
    angle_base_scene_struct['camera_location'] = tuple(cam.location)
    angle_base_scene_struct['camera_rotation_euler'] = tuple(cam.rotation_euler)

    for i, obj_info in enumerate(angle_base_scene_struct['objects']):
        blender_obj = blender_objs[i]
        try:
            obj_info['pixel_coords'] = utils.get_camera_coords(cam, blender_obj.location)
        except Exception as e:
            print(f"Error getting pixel coords for object {i} at angle_base: {e}")
            obj_info['pixel_coords'] = [-1,-1,-1]

    # Cardinal directions for 'angle_base' (original method)
    bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,-0.1)) # Temp plane
    temp_plane_obj = bpy.context.object
    # The plane added by primitive_plane_add is at world origin, Z up.
    plane_normal_vec = Vector((0.0, 0.0, 1.0)) # Safer assumption for flat ground

    quat_base = cam.matrix_world.to_quaternion()
    cam_forward_base = (quat_base @ Vector((0, 0, -1))).normalized()
    cam_left_base = (quat_base @ Vector((-1, 0, 0))).normalized()

    # Project onto the ground plane (XY plane for 'front', 'left')
    dir_front_base = (cam_forward_base - cam_forward_base.project(plane_normal_vec)).normalized()
    dir_left_base = (cam_left_base - cam_left_base.project(plane_normal_vec)).normalized()
    
    angle_base_scene_struct['directions'] = {
        'front': tuple(dir_front_base),
        'behind': tuple(-dir_front_base),
        'left': tuple(dir_left_base),
        'right': tuple(-dir_left_base),
    }
    utils.delete_object(temp_plane_obj)

    angle_base_scene_struct['relationships'] = compute_all_relationships(angle_base_scene_struct)
    bpy.ops.render.render(write_still=True)
    with open(angle_base_scene_path, "w") as f:
        json.dump(angle_base_scene_struct, f, indent=2)
    # --- End Render 'angle_base' ---

    # --- Render 4 Cardinal Angles --- 
    # Now, set up Track To constraint for the 4 cardinal angles
    print(f"  Setting up tracking constraint for cardinal angles...")
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target_empty # This should now work
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    # Define camera positions (radius, height, angles) for the 4 cardinal views
    cam_dist = 7.5 # Default distance in base scene
    cam_height = 6.0 # Reverted to original value as ground plane is now larger.
    angles_deg = [0, 90, 180, 270]
    angles_rad = [math.radians(d) for d in angles_deg]

    for angle_idx, angle_rad in enumerate(angles_rad):
        print(f"    Angle {angle_idx} ({angles_deg[angle_idx]} deg)")
        # Position camera
        cam_x = cam_dist * math.cos(angle_rad)
        cam_y = cam_dist * math.sin(angle_rad)
        cam.location = (cam_x, cam_y, cam_height)
        bpy.context.view_layer.update() # Force constraint update for correct rotation

        # Create a deep copy of the base structure for this angle
        # Using json loads/dumps for simplicity with nested dicts/lists
        angle_scene_struct = json.loads(json.dumps(base_scene_struct))

        # Define output paths for this angle
        angle_prefix = f"{scene_name}_angle_{angle_idx}"
        angle_img_path = os.path.join(scene_img_dir, f"{angle_prefix}.png")
        angle_scene_path = os.path.join(scene_json_dir, f"{angle_prefix}.json")

        # Update render path
        render_args.filepath = angle_img_path

        # Update scene struct with angle-specific info
        angle_scene_struct['image_filename'] = os.path.basename(angle_img_path)
        angle_scene_struct['camera_angle_degrees'] = angles_deg[angle_idx]
        angle_scene_struct['camera_location'] = tuple(cam.location)
        angle_scene_struct['camera_rotation_euler'] = tuple(cam.rotation_euler)

        # Calculate pixel coordinates for objects from this angle
        # We need the actual blender objects to get updated locations if they were moved/snapped
        for i, obj_info in enumerate(angle_scene_struct['objects']):
            blender_obj = blender_objs[i] # Assumes blender_objs order matches objs
            obj_info['pixel_coords'] = utils.get_camera_coords(cam, blender_obj.location)

        # Calculate cardinal directions based on current camera view
        # Re-create a temporary plane to calculate ground-projected directions
        bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,-0.1)) # Temp plane below origin
        plane = bpy.context.object
        # Use a fixed plane normal assuming flat ground
        plane_normal = Vector((0.0, 0.0, 1.0))
        # Get camera orientation vectors
        quat = cam.matrix_world.to_quaternion()
        cam_forward = (quat @ Vector((0, 0, -1))).normalized()
        cam_left = (quat @ Vector((-1, 0, 0))).normalized()
        # Project onto the ground plane (XY plane)
        # Scene 'front' is direction camera looks projected onto ground
        plane_front = (cam_forward - cam_forward.project(plane_normal)).normalized() # plane_normal is (0,0,1)
        plane_left_dir = (cam_left - cam_left.project(plane_normal)).normalized() # Renamed to avoid conflict
        
        # Store directions 
        angle_scene_struct['directions'] = {
            'front': tuple(plane_front),
            'behind': tuple(-plane_front),
            'left': tuple(plane_left_dir), # Use renamed variable
            'right': tuple(-plane_left_dir), # Use renamed variable
            # 'above': tuple(plane_normal), # Assuming world Z is up
            # 'below': tuple(-plane_normal), # Assuming world Z is up
        }
        # utils.delete_object(plane) # Clean up temp plane # This was for the old per-angle plane - REMOVED

        # Calculate relationships for this angle using the calculated directions
        angle_scene_struct['relationships'] = compute_all_relationships(angle_scene_struct)

        # Render final image for this angle
        bpy.ops.render.render(write_still=True)

        # Save per-angle JSON
        with open(angle_scene_path, "w") as f:
            json.dump(angle_scene_struct, f, indent=2)

    # Clean up the tracking empty AFTER all rendering is done
    utils.delete_object(target_empty)

    # Restore initial light energies (optional, good practice if script reused scene)
    # for light_name, initial_energy in initial_light_energies.items():
    #     light_obj = scene.objects.get(light_name)
    #     if light_obj and light_obj.data:
    #         light_obj.data.energy = initial_energy

def add_specified_objects(scene_struct, glb_files, object_coords, args): # Removed camera param
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
            # Pixel coords calculated later per angle in render_scene loop
            objects.append({
                'shape': os.path.basename(glb_file),
                '3d_coords': [x, y, 0.0], # Original specified coords
                '3d_coords_transformed': tuple(obj.location), # Actual placed coords
                'rotation': theta,
                # 'pixel_coords': pix, # Removed here
            }) # noqa: E501
        except Exception as e:
            print(f"Error placing object from {glb_file}: {e}")
            # Clean up any objects placed so far
            for o in blender_objects:
                utils.delete_object(o)
            return None, None
    
    # Visibility test (runs only based on initial camera pose before looping)
    if not check_visibility(blender_objects, args.min_pixels_per_object):
        print("Warning: Initial visibility test failed. Some objects may be occluded or too small from the initial camera pose.") # noqa: E501
        # Instead of recreating the scene like in random placement, we'll log the warning and continue
    
    return objects, blender_objects

def compute_all_relationships(scene_struct, eps=0.2):
    rels = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name in ('above', 'below'):
            continue # Skip vertical relationships if not needed, or handle differently
        rels[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords_transformed'] # Use transformed coords for relationships
            related = [j for j, obj2 in enumerate(scene_struct['objects'])
                       if i != j and sum((Vector(obj2['3d_coords_transformed'])[k]-Vector(coords1)[k])*direction_vec[k] for k in range(3)) > eps] # noqa: E501
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