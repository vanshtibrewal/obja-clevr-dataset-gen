# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Blender 3.x‑compatible rewrite of the CLEVR image generator.
Only the minimal API changes needed for Blender 3.0+ have been applied:
  • removed legacy render settings (tile_x/tile_y, use_antialiasing, BLENDER_RENDER)
  • updated GPU preference calls
  • swapped Quaternion * Vector for @ operator
  • replaced Object.select / scene.objects.active with select_set / view_layer.objects.active
  • switched primitive_plane_add radius→size
  • added flat‑shaded (emission) material creation instead of removed use_shadeless flag
  • replaced layer switching with hide_render toggles
The script is **not** backward‑compatible with Blender 2.7x.
Run with:

    blender --background --python render_images_blender3.py -- [arguments]
"""

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
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
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
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
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
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
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
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
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  all_scene_paths = []
  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)
    render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)



def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
  
  scene.cycles.sample_clamp_direct   = 0.0   # leave direct light unclamped
  scene.cycles.sample_clamp_indirect = 10.0  # clamp very bright indirect paths

  if args.use_gpu:
      prefs     = bpy.context.preferences
      cycles_prefs = prefs.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'
      for d in cycles_prefs.devices:
          d.use = True  # enable all CUDA devices
      scene.cycles.device = 'GPU'

  # ---------------------------------------------------------------------
  # Scene layout metadata
  # ---------------------------------------------------------------------
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
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
  quat         = cam.matrix_world.to_quaternion()
  cam_behind   = (quat @ Vector((0, 0, -1))).normalized()
  cam_left     = (quat @ Vector((-1, 0, 0))).normalized()
  cam_up       = (quat @ Vector((0, 1, 0))).normalized()

  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left   = (cam_left   - cam_left.project(plane_normal)).normalized()
  plane_up     =  cam_up.project(plane_normal).normalized()

  # Clean up helper plane
  utils.delete_object(plane)

  # Store directions
  scene_struct['directions'] = {
      'behind': tuple( plane_behind),
      'front':  tuple(-plane_behind),
      'left':   tuple( plane_left),
      'right':  tuple(-plane_left),
      'above':  tuple( plane_up),
      'below':  tuple(-plane_up),
  }

  # Jitter lights (object names come from base_scene.blend)
  for light_name, jitter in [("Lamp_Key",  args.key_light_jitter),
                              ("Lamp_Fill", args.fill_light_jitter),
                              ("Lamp_Back", args.back_light_jitter)]:
      if jitter:
          obj = scene.objects.get(light_name)
          if obj:
              obj.location.x += rand(jitter)
              obj.location.y += rand(jitter)
              obj.location.z += rand(jitter)

  # Populate scene with random objects
  objs, blender_objs = add_random_objects(scene_struct, num_objects, args, cam)

  # Render final image
  scene_struct['objects']       = objs
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  bpy.ops.render.render(write_still=True)

  # Save per‑image JSON
  with open(output_scene, "w") as f:
      json.dump(scene_struct, f, indent=2)

  # Optionally save the .blend
  if output_blendfile:
      bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

# -----------------------------------------------------------------------------
# Object placement helpers
# -----------------------------------------------------------------------------

def add_random_objects(scene_struct, num_objects, args, camera):
    """Place non‑intersecting random objects into the current scene."""

    with open(args.properties_json) as f:
        props = json.load(f)
    color_name_to_rgba = {name: [c / 255.0 for c in rgb] + [1.0] for name, rgb in props['colors'].items()}
    material_mapping   = [(v, k) for k, v in props['materials'].items()]
    object_mapping     = [(v, k) for k, v in props['shapes'   ].items()]
    size_mapping       = list(props['sizes'].items())

    shape_color_combos = None
    if args.shape_color_combos_json:
        with open(args.shape_color_combos_json) as f:
            shape_color_combos = list(json.load(f).items())

    positions, objects, blender_objects = [], [], []

    for _ in range(num_objects):
        size_name, r = random.choice(size_mapping)
        # size_name, r = "one", 1

        # Spatial rejection‑sampling
        for attempt in range(args.max_retries):
            x, y = random.uniform(-3, 3), random.uniform(-3, 3)
            if all(math.hypot(x-xx, y-yy) - r - rr >= args.min_dist for xx, yy, rr in positions):
                break
        else:
            # Could not place – reset scene and retry recursively
            for o in blender_objects:
                utils.delete_object(o)
            return add_random_objects(scene_struct, num_objects, args, camera)

        # Choose random shape & colour
        # if shape_color_combos is None:
        #     obj_name, obj_name_out = random.choice(object_mapping)
        #     color_name, rgba       = random.choice(list(color_name_to_rgba.items()))
        # else:
        #     obj_name_out, colors   = random.choice(shape_color_combos)
        #     color_name             = random.choice(colors)
        #     obj_name               = next(k for k, v in object_mapping if v == obj_name_out)
        #     rgba                   = color_name_to_rgba[color_name]
        obj_name, obj_name_out = random.choice(object_mapping)

        # Cube diagonal adjustment
        # if obj_name == 'Cube':
        #     r /= math.sqrt(2)

        # theta = 360.0 * random.random()
        # sample radian
        theta = random.uniform(-math.pi, math.pi)


        # Add mesh
        utils.add_object_glb(args.shape_dir, obj_name, r, (x, y), theta=theta)
        # utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Random material
        # mat_name, mat_name_out = random.choice(material_mapping)
        # utils.add_material(mat_name, Color=rgba)

        # Record metadata
        pix = utils.get_camera_coords(camera, obj.location)
        objects.append({
            'shape':       obj_name_out,
            'size':        size_name,
            # 'material':    mat_name_out,
            '3d_coords':   tuple(obj.location),
            'rotation':    theta,
            'pixel_coords':pix,})
            # 'color':       color_name})

    # Visibility test
    if not check_visibility(blender_objects, args.min_pixels_per_object):
        for o in blender_objects:
            utils.delete_object(o)
        return add_random_objects(scene_struct, num_objects, args, camera)

    return objects, blender_objects

# -----------------------------------------------------------------------------
# Relationship computation
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Visibility check helpers (flat‑shaded render)
# -----------------------------------------------------------------------------

def check_visibility(blender_objects, min_pixels):
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    colors = render_shadeless(blender_objects, path)
    img    = bpy.data.images.load(path)
    pixels = list(img.pixels)
    os.remove(path)

    step = 4  # RGBA
    counts = Counter(tuple(pixels[i:i+4]) for i in range(0, len(pixels), step))
    if len(counts) != len(blender_objects) + 1:
        return False
    return all(c >= min_pixels for c in counts.values())


def render_shadeless(blender_objects, path):
    """Render without lights using per‑object emission materials."""
    scene       = bpy.context.scene
    render_args = scene.render

    # Backup settings we will touch
    old_filepath = render_args.filepath
    old_samples  = scene.cycles.samples

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
    old_mats      = []
    for i, obj in enumerate(blender_objects):
        old_mats.append(obj.data.materials[0])
        mat = bpy.data.materials.new(name=f"Flat_{i}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        emit   = nodes.new('ShaderNodeEmission')
        # unique random colour
        while True:
            r, g, b = random.random(), random.random(), random.random()
            if (r, g, b) not in object_colors:
                break
        emit.inputs['Color'].default_value = (r, g, b, 1)
        links.new(emit.outputs['Emission'], output.inputs['Surface'])
        object_colors.add((r, g, b, 1))
        obj.data.materials[0] = mat

    # Render
    bpy.ops.render.render(write_still=True)

    # Restore mats
    for obj, mat in zip(blender_objects, old_mats):
        obj.data.materials[0] = mat

    # Un‑hide lights & ground
    for obj in hidden_objs:
        obj.hide_render = False

    # Restore settings
    render_args.filepath = old_filepath
    scene.cycles.samples = old_samples

    return object_colors

# -----------------------------------------------------------------------------
# Script entry‑point inside Blender
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if not INSIDE_BLENDER:
        print("This script must be run from within Blender 3.x:")
        print("  blender --background --python render_images_blender3.py -- [args]")
        sys.exit(1)

    argv = utils.extract_args()  # strips the "--" Blender passes before our args
    args = parser.parse_args(argv)
    main(args)
