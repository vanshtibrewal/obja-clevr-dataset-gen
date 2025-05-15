# Copyright 2017‑present, Facebook, Inc.
# All rights reserved.
#
# Licensed under the BSD‑style license found in the LICENSE file in the root
# directory of this source tree. An additional grant of patent rights can be
# found in the PATENTS file in the same directory.
"""Utility helpers updated for **Blender 3.x**.
Only minimal API changes were made so the original CLEVR generator runs under
modern Blender builds (2.93 LTS, 3.0+).
- `Object.select`   ➜ `select_set()`
- `scene.objects.active` ➜ `view_layer.objects.active`
- Layer switching API removed; `set_layer` now toggles `hide_render`.
- `primitive_plane_add` call remains in caller; no layer masks here.
- Object appending rewritten with explicit `directory` & `filename` args.
"""

from __future__ import annotations
import sys, os, random
import bpy, bpy_extras
from mathutils import Vector, Matrix
import math

# -----------------------------------------------------------------------------
# Argument helpers (unchanged)
# -----------------------------------------------------------------------------

def extract_args(input_argv=None):
    """Return argv elements appearing after the first "--"."""
    if input_argv is None:
        input_argv = sys.argv
    return input_argv[input_argv.index('--') + 1:] if '--' in input_argv else []

def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))

# -----------------------------------------------------------------------------
# Scene / object helpers
# -----------------------------------------------------------------------------

def delete_object(obj: bpy.types.Object) -> None:
    """Delete *obj* from the current scene."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.delete()


def get_camera_coords(cam: bpy.types.Object, pos: Vector):
    """Return (px, py, pz) pixel coords of *pos* from *cam*."""
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale   = scene.render.resolution_percentage / 100.0
    w       = int(scale * scene.render.resolution_x)
    h       = int(scale * scene.render.resolution_y)
    return int(round(x * w)), int(round(h - y * h)), z


def set_layer(obj: bpy.types.Object, layer_idx: int) -> None:
    """Legacy layer toggle – now approximated with hide_render."""
    # Layer 0 (idx 0) => visible; anything else => hide from render
    obj.hide_render = layer_idx != 0

# -----------------------------------------------------------------------------
# Asset loading / placement
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# helper --------------------------------------------------------------------
def add_object_glb(name: str, loc, *, theta=0):
    """Load *.glb* object *name* from *object_dir* and place it."""
    count = sum(o.name.startswith(name) for o in bpy.data.objects)

    glb_path = name

    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.import_scene.gltf(filepath=glb_path)

    new_objects = list(bpy.context.selected_objects)

    if not new_objects:
        raise RuntimeError(f"No objects selected after import of {name}. Assuming none were added.")

    obj = None
    if len(new_objects) > 1:
        mesh_objects = []
        for o in new_objects:
            if o.type == 'MESH':
                mesh_objects.append(o)
            else:
                o.select_set(False)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    elif len(new_objects) == 1:
        obj = new_objects[0]
        bpy.context.view_layer.objects.active = obj

    new_name = f"{name}_{count}"
    obj.name = new_name

    # ——— set up for a Z rotation ———
    obj.rotation_mode    = 'XYZ'

    # ------------------------------------------------------------
    # 2.  world-Z spin  (matrix-world multiplication → children safe)
    # ------------------------------------------------------------
    theta_rad   = theta # math.radians(theta)
    Rz_world    = Matrix.Rotation(theta_rad, 4, 'Z')
    obj.matrix_world = Rz_world @ obj.matrix_world

    # Transform
    max_dim = max(obj.dimensions) if obj.dimensions else 0.0
    norm_scale = 2.5 / max_dim if max_dim > 0 else 1.0
    obj.scale = (norm_scale, norm_scale, norm_scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()

    x, y = loc
    # ——— absolute drop-to-ground in world space ———
    # 1) find lowest world‐Z of the mesh:
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox_world)

    # 2) build the new desired world‐location:
    world_old = obj.matrix_world.to_translation()
    world_new = Vector((x, y, world_old.z - min_z))

    # 3) if parented, convert world_new → local coords before assigning:
    if obj.parent:
        obj.location = obj.parent.matrix_world.inverted() @ world_new
    else:
        obj.location = world_new


import math
import bpy
from mathutils import Matrix, Vector

def add_object_glb_scale(
        name: str,
        loc,
        *,
        theta: float = 0.0,     # rotation about world‑Z in degrees
        scale: float = 1.0      # uniform scale factor
    ):
    """
    Import a *.glb* file, merge meshes, rotate, scale, and drop so its
    lowest point sits at the given (x, y, z).

    Parameters
    ----------
    name   : str
        Path (absolute or relative) to the *.glb* file.
    loc    : tuple[float, float, float]
        (x, y, z) world‑space position where the mesh’s lowest point
        should touch the Z plane.
    theta  : float, optional
        Rotation around world‑Z in degrees.  Default 0.
    scale  : float, optional
        Uniform scale multiplier.  Default 1.
    """
    # ------------------------------------------------------------
    # 1.  import the .glb, merge any mesh children
    # ------------------------------------------------------------
    count = sum(o.name.startswith(name) for o in bpy.data.objects)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=name)

    new_objects = list(bpy.context.selected_objects)
    if not new_objects:
        raise RuntimeError(f"No objects selected after import of {name}")

    if len(new_objects) > 1:
        mesh_objects = [o for o in new_objects if o.type == 'MESH']
        for o in new_objects:
            if o not in mesh_objects:
                o.select_set(False)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = f"{name}_{count}"

    # ------------------------------------------------------------
    # 2.  rotate about world‑Z
    # ------------------------------------------------------------
    obj.rotation_mode = 'XYZ'
    Rz_world = Matrix.Rotation(math.radians(theta), 4, 'Z')
    obj.matrix_world = Rz_world @ obj.matrix_world

    # ------------------------------------------------------------
    # 3.  uniform scale
    # ------------------------------------------------------------
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()

    # ------------------------------------------------------------
    # 4.  drop so lowest vertex sits on requested Z level
    # ------------------------------------------------------------
    x, y, z_target = loc
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min(v.z for v in bbox_world)

    world_old = obj.matrix_world.to_translation()
    world_new = Vector((x, y, z_target + (world_old.z - min_z)))

    if obj.parent:
        obj.location = obj.parent.matrix_world.inverted() @ world_new
    else:
        obj.location = world_new

    return obj  # handy if you want a reference later


def add_object(object_dir: str, name: str, scale: float, loc, *, theta=0):
    """Append *name* mesh from *object_dir* and place it in the scene."""
    # Count existing duplicates so we can give this instance a unique name
    count = sum(o.name.startswith(name) for o in bpy.data.objects)

    blend_path   = os.path.join(object_dir, f"{name}.blend")
    obj_dir      = os.path.join(blend_path, "Object")  # section inside .blend

    # Append the object
    bpy.ops.wm.append(directory=obj_dir, filename=name)

    # Rename & activate
    new_name = f"{name}_{count}"
    obj      = bpy.data.objects[name]
    obj.name = new_name

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Transform
    x, y = loc
    obj.rotation_euler[2] = theta               # radians expected
    obj.scale             = (scale, scale, scale)
    obj.location          = (x, y, scale)      # z = scale to sit on ground


# -----------------------------------------------------------------------------
# Material helpers
# -----------------------------------------------------------------------------

def load_materials(material_dir: str) -> None:
    """Append all .blend NodeTree materials found in *material_dir*."""
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'):
            continue
        name      = os.path.splitext(fn)[0]
        ntree_dir = os.path.join(material_dir, fn, 'NodeTree')
        bpy.ops.wm.append(directory=ntree_dir, filename=name)


def add_material(name: str, **properties):
    """Create a new material instance from pre‑loaded *name* group.

    The caller should have loaded NodeTree "name" via :func:`load_materials`.
    Any keyword that matches a group input (e.g. ``Color=(r,g,b,a)``) will be
    forwarded to that socket.
    """
    mat = bpy.data.materials.new(name=f"Material_{len(bpy.data.materials)}")
    mat.use_nodes = True

    # Active object must exist
    obj = bpy.context.active_object
    if not obj:
        raise RuntimeError("add_material() called with no active object")
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    group  = nodes.new('ShaderNodeGroup')
    group.node_tree = bpy.data.node_groups[name]

    # Forward provided property values into matching inputs
    for inp in group.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    links.new(group.outputs['Shader'], output.inputs['Surface'])
