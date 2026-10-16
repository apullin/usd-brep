#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for _omni_solid.tessellate_fast, the opt-in fast tessellator, on bundled models."""

import array
import gc
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - the repo's Python has no numpy
    np = None

if sys.platform == "win32":
    for p in os.environ.get("PATH", "").split(";"):
        if p and os.path.isdir(p):
            os.add_dll_directory(p)

import _omni_solid as sm

_TEST_FILES = Path(__file__).resolve().parents[3] / "TestFiles"
# Planar, curved, seamed, periodic, pole, trimmed, holed and filleted faces; open, closed and
# non-manifold Breps.
_SMB_MODELS = (
    "pt_TestFiles/box_with_large_fillet.smb", "pt_TestFiles/Mouse.smb", "pt_TestFiles/sphere.smb",
    "pt_TestFiles/torus.smb", "pt_TestFiles/toruswvy.smb", "pt_TestFiles/cone.smb", "pt_TestFiles/disk.smb",
    "pt_TestFiles/plane_circ.smb", "pt_TestFiles/ellips_ts2.smb", "pt_TestFiles/surf_rev.smb",
    "pt_TestFiles/vase_rev.smb", "pt_TestFiles/Endcap.smb", "pt_TestFiles/Boolean/BoltMBox1.smb",
    "pt_TestFiles/Boolean/extrudeHoles1.smb", "pt_TestFiles/Defeature/brep1.smb",
    "pt_TestFiles/Defeature/defillet.smb", "pt_TestFiles/Defeature/flathead_screw_top.smb",
    "pt_TestFiles/Defeature/inset_cube_3x1fillet.smb", "pt_TestFiles/Defeature/LShape_I.smb",
    "pt_TestFiles/Defeature/Platte.smb", "usd_TestFiles/boxes_all.smb",
    "usd_TestFiles/box_with_second_box_with_one_loop_vertex.smb", "usd_TestFiles/box_with_detached_sheet.smb",
)
_USD_MODELS = (
    "brep_validator/CubeFaceMaterials.usda", "brep_validator/CubeBrepArray.usda",
    "brep_validator/Cone_SmConnect_Brep.usda", "brep_validator/cubeWithVoid.usda",
    "brep_validator/nonManifoldCubes.usda", "usd_TestFiles/sphere_nurbs.usda",
)


def _bundled_breps():
    """(name, Brep) for every Brep of the bundled models, and a few primitives."""
    sm.usd.ensure_plugin_registered()
    breps = [("box", sm.create_box((0, 0, 0), 10, 20, 30)), ("sphere", sm.create_sphere((0, 0, 0), 5)),
             ("cylinder", sm.create_cylinder((0, 0, 0), 5, 10)), ("cone", sm.create_cone((0, 0, 0), 5, 0, 10)),
             ("torus", sm.create_torus((0, 0, 0), 10, 3))]
    for model in _SMB_MODELS:
        breps.append((model, sm.read_brep_from_file(str(_TEST_FILES / model))))
    for model in _USD_MODELS:
        for index, brep in enumerate(sm.usd.import_breps(str(_TEST_FILES / model), heal=False)):
            breps.append((f"{model}[{index}]", brep))
    return breps


def _triangles(arrays):
    faces = arrays["faces"]
    assert len(faces) % 4 == 0 and all(count == 3 for count in faces[0::4]), "not a triangle stream"
    return list(zip(faces[1::4], faces[2::4], faces[3::4]))


def _polygons(stream):
    polygons, index = [], 0
    while index < len(stream):
        polygons.append(stream[index + 1:index + 1 + stream[index]])
        index += stream[index] + 1
    return polygons


def _edge_uses(polygons):
    """Directed edge -> number of polygons using it."""
    return Counter((p[k], p[(k + 1) % len(p)]) for p in polygons for k in range(len(p)))


def _edge_defects(polygons):
    """Directed edges used more than once, and directed edges without their opposite."""
    uses = _edge_uses(polygons)
    return sum(count > 1 for count in uses.values()), sum((b, a) not in uses for a, b in uses)


def _is_watertight(polygons):
    """Every edge used by exactly two polygons, once in each direction."""
    uses = _edge_uses(polygons)
    return all(count == 1 and uses.get((b, a)) == 1 for (a, b), count in uses.items())


def _is_unit(vector, tolerance=1e-5):
    length = math.sqrt(sum(value * value for value in vector))
    return math.isfinite(length) and abs(length - 1.0) < tolerance


class TestTessellateFastLayout(unittest.TestCase):
    """The arrays have PolyBrep.to_mesh_arrays()'s keys, types and conventions."""

    def test_same_layout_as_to_mesh_arrays(self):
        sphere = sm.create_sphere((0, 0, 0), 5)
        for face_varying in (False, True):
            with self.subTest(face_varying_normals=face_varying):
                fast = sm.tessellate_fast(sphere, face_varying_normals=face_varying)
                reference = sm.tessellate(sphere).to_mesh_arrays(face_varying_normals=face_varying)
                self.assertIsInstance(fast, dict)
                self.assertEqual(sorted(fast), sorted(reference))
                for key in fast:
                    self.assertIsInstance(fast[key], list)
                for key in ("points", "normals", "face_normals"):
                    for row in fast[key]:
                        self.assertIsInstance(row, tuple)
                        self.assertEqual(len(row), 3)
                        self.assertTrue(all(isinstance(value, float) for value in row))
                self.assertTrue(all(isinstance(value, int) for value in fast["faces"]))
                triangles = _triangles(fast)
                self.assertEqual(len(fast["face_normals"]), len(triangles))
                self.assertEqual(len(fast["normals"]), 3 * len(triangles) if face_varying else len(fast["points"]))

    def test_values_are_float32(self):
        arrays = sm.tessellate_fast(sm.create_sphere((0.1, 0.2, 0.3), 5.0))
        packed = __import__("array").array("f", [value for row in arrays["points"] for value in row])
        self.assertEqual(list(packed), [value for row in arrays["points"] for value in row])

    def test_box(self):
        box = sm.create_box((0, 0, 0), 10, 20, 30)
        arrays, report = sm.tessellate_fast(box, face_varying_normals=True, return_report=True)
        triangles = _triangles(arrays)
        self.assertEqual((len(arrays["points"]), len(triangles)), (8, 12))
        self.assertEqual(report["face_ids"], [face for face in range(6) for _ in range(2)])
        self.assertEqual(report["face_kinds"], ["fast"] * 6)
        self.assertEqual((report["fast_face_count"], report["fallback_face_count"], report["failed_face_count"]),
                         (6, 0, 0))
        self.assertEqual(report["failed_faces"], [])
        # Counterclockwise seen from outside: the corner normals are the triangles' own normals.
        for index, normal in enumerate(arrays["face_normals"]):
            self.assertEqual(arrays["normals"][3 * index:3 * index + 3], [normal] * 3)

    def test_sphere_normals(self):
        arrays = sm.tessellate_fast(sm.create_sphere((0, 0, 0), 5))
        for point, normal in zip(arrays["points"], arrays["normals"]):
            radius = math.sqrt(sum(value * value for value in point))
            self.assertAlmostEqual(radius, 5.0, places=5)
            for p, n in zip(point, normal):
                self.assertAlmostEqual(p / radius, n, places=5)

    def test_quality_controls(self):
        sphere = sm.create_sphere((0, 0, 0), 5)
        default = sm.tessellate_fast(sphere)
        self.assertEqual(default, sm.tessellate_fast(sphere, **sm.tessellation_defaults()))
        for parameter, value in (("surface_angle_tolerance_deg", 10.0), ("chord_height_tolerance", 0.01),
                                 ("max_edge_length", -2.0)):
            with self.subTest(parameter=parameter):
                finer = sm.tessellate_fast(sphere, **{parameter: value})
                self.assertGreater(len(finer["faces"]), len(default["faces"]))
                self.assertTrue(_is_watertight(_triangles(finer)))

    def test_errors(self):
        box = sm.create_box((0, 0, 0), 10, 10, 10)
        with self.assertRaises(ValueError):
            sm.tessellate_fast(box, allow_partial=True)
        with self.assertRaises(RuntimeError):
            sm.tessellate_fast(None)
        # No faces: no mesh, as for tessellate.
        wire = sm.read_brep_from_file(str(_TEST_FILES / "usd_TestFiles" / "wire_edge.smb"))
        with self.assertRaises(RuntimeError):
            sm.tessellate(wire)
        with self.assertRaises(RuntimeError):
            sm.tessellate_fast(wire)

    def test_failed_faces(self):
        # Two of its faces fail in tessellate too; partial output keeps the others.
        sm.usd.ensure_plugin_registered()
        (brep,) = sm.usd.import_breps(str(_TEST_FILES / "usd_TestFiles" / "RevitWallHealerCrash.usda"), heal=False)
        _, failures = sm.tessellate(brep, allow_partial=True, return_failures=True)
        with self.assertRaises(RuntimeError) as raised:
            sm.tessellate_fast(brep)
        self.assertIn("failed to tessellate", str(raised.exception))
        arrays, report = sm.tessellate_fast(brep, allow_partial=True, return_report=True)
        self.assertEqual(report["failed_faces"], sorted(failure["face_index"] for failure in failures))
        self.assertEqual(report["failed_face_count"], len(report["failed_faces"]))
        self.assertGreater(len(_triangles(arrays)), 0)
        self.assertFalse(set(report["face_ids"]) & set(report["failed_faces"]))
        for face in report["failed_faces"]:
            self.assertEqual(report["face_kinds"][face], "failed")


class TestTessellateFastBundledModels(unittest.TestCase):
    """Invariants on every bundled model, against tessellate() as the reference."""

    @classmethod
    def setUpClass(cls):
        cls.breps = _bundled_breps()

    def test_invariants(self):
        for name, brep in self.breps:
            with self.subTest(model=name):
                arrays, report = sm.tessellate_fast(brep, return_report=True)
                points, triangles = arrays["points"], _triangles(arrays)
                face_count = brep.face_count()
                # Counts and indices.
                self.assertEqual(len(arrays["normals"]), len(points))
                self.assertEqual(len(arrays["face_normals"]), len(triangles))
                used = set()
                for triangle in triangles:
                    self.assertEqual(len(set(triangle)), 3)
                    self.assertTrue(all(0 <= index < len(points) for index in triangle))
                    used.update(triangle)
                self.assertEqual(len(used), len(points))
                # Finite unit normals.
                for normal in arrays["normals"] + arrays["face_normals"]:
                    self.assertTrue(_is_unit(normal), normal)
                for normal in sm.tessellate_fast(brep, face_varying_normals=True)["normals"]:
                    self.assertTrue(_is_unit(normal), normal)
                # Faces: the per-face subsets cover every triangle once, each within its face.
                self.assertEqual(len(report["face_ids"]), len(triangles))
                self.assertEqual(len(report["face_kinds"]), face_count)
                self.assertEqual(report["failed_faces"], [])
                kinds = Counter(report["face_kinds"])
                self.assertEqual((kinds["fast"], kinds["fallback"]),
                                 (report["fast_face_count"], report["fallback_face_count"]))
                self.assertEqual(kinds["fast"] + kinds["fallback"], face_count)
                low, high = brep.bounding_box()
                slack = 1e-4 * max(1.0, math.dist(low, high))
                faces = brep.faces()
                for face, triangle in zip(report["face_ids"], triangles):
                    self.assertTrue(0 <= face < face_count)
                    face_low, face_high = faces[face].bounding_box()
                    for index in triangle:
                        for axis in range(3):
                            self.assertGreaterEqual(points[index][axis], face_low[axis] - slack)
                            self.assertLessEqual(points[index][axis], face_high[axis] + slack)
                # Watertight wherever the reference is, and no more edges used twice in one direction
                # (misoriented or non-manifold) or without their opposite (open) than in the reference.
                reference = _polygons(sm.tessellate(brep).to_mesh_arrays()["faces"])
                if _is_watertight(reference):
                    self.assertTrue(_is_watertight(triangles))
                fast_defects, reference_defects = _edge_defects(triangles), _edge_defects(reference)
                self.assertLessEqual(fast_defects[0], reference_defects[0])
                self.assertLessEqual(fast_defects[1], reference_defects[1])

    def test_deterministic(self):
        for name, brep in self.breps:
            with self.subTest(model=name):
                self.assertEqual(sm.tessellate_fast(brep, return_report=True),
                                 sm.tessellate_fast(brep, return_report=True))


_BUFFER_TYPES = {"points": ("d", 8), "normals": ("f", 4), "triangles": ("i", 4), "face_ids": ("i", 4),
                 "face_kinds": ("b", 1)}
_KIND_CODES = {"failed": 0, "fast": 1, "fallback": 2}


def _flat(rows):
    return [value for row in rows for value in row]


class TestTessellateFastBuffers(unittest.TestCase):
    """buffers=True: flat typed array.array buffers with the values of the default output."""

    @classmethod
    def setUpClass(cls):
        cls.breps = _bundled_breps()

    def test_layout(self):
        sphere = sm.create_sphere((0, 0, 0), 5)
        for face_varying in (False, True):
            with self.subTest(face_varying_normals=face_varying):
                buffers = sm.tessellate_fast(sphere, face_varying_normals=face_varying, buffers=True)
                self.assertEqual(sorted(buffers), sorted(_BUFFER_TYPES))
                for key, (typecode, itemsize) in _BUFFER_TYPES.items():
                    self.assertIsInstance(buffers[key], array.array)
                    self.assertEqual((buffers[key].typecode, buffers[key].itemsize), (typecode, itemsize))
                vertices, triangles = len(buffers["points"]) // 3, len(buffers["triangles"]) // 3
                self.assertEqual(len(buffers["points"]), 3 * vertices)
                self.assertEqual(len(buffers["triangles"]), 3 * triangles)
                self.assertEqual(len(buffers["normals"]), 9 * triangles if face_varying else 3 * vertices)
                self.assertEqual(len(buffers["face_ids"]), triangles)
                self.assertEqual(list(buffers["face_kinds"]), [_KIND_CODES["fast"]] * sphere.face_count())
                # The (V, 3) layout without numpy: a 2D view of the same memory.
                view = memoryview(buffers["points"]).cast("B").cast("d", [vertices, 3])
                self.assertEqual(view.shape, (vertices, 3))
                self.assertEqual(view[1, 2], buffers["points"][5])

    def test_values_equal_default(self):
        for name, brep in self.breps:
            for face_varying in (False, True):
                with self.subTest(model=name, face_varying_normals=face_varying):
                    arrays, report = sm.tessellate_fast(brep, face_varying_normals=face_varying, return_report=True)
                    buffers, buffers_report = sm.tessellate_fast(brep, face_varying_normals=face_varying,
                                                                 return_report=True, buffers=True)
                    self.assertEqual(buffers_report, report)
                    # Points are the kernel's doubles; the default rounds them to float32.
                    self.assertEqual(array.array("f", buffers["points"]).tolist(), _flat(arrays["points"]))
                    self.assertEqual(buffers["normals"].tolist(), _flat(arrays["normals"]))
                    faces = arrays["faces"]
                    self.assertEqual(buffers["triangles"].tolist(),
                                     [v for k in range(0, len(faces), 4) for v in faces[k + 1:k + 4]])
                    self.assertEqual(buffers["face_ids"].tolist(), report["face_ids"])
                    self.assertEqual(buffers["face_kinds"].tolist(), [_KIND_CODES[k] for k in report["face_kinds"]])

    @unittest.skipIf(np is None, "numpy not installed")
    def test_numpy_views_without_copies(self):
        buffers = sm.tessellate_fast(sm.create_sphere((0, 0, 0), 5), buffers=True)
        points = np.frombuffer(buffers["points"], dtype=np.float64).reshape(-1, 3)
        triangles = np.frombuffer(buffers["triangles"], dtype=np.int32).reshape(-1, 3)
        self.assertEqual(points.shape, (len(buffers["points"]) // 3, 3))
        self.assertEqual(int(triangles.max()), points.shape[0] - 1)
        buffers["points"][0] = 123.0  # a view, not a copy
        self.assertEqual(points[0, 0], 123.0)

    def test_partial(self):
        # Two of its faces fail; with buffers, face_kinds tells which, so no report is needed.
        sm.usd.ensure_plugin_registered()
        (brep,) = sm.usd.import_breps(str(_TEST_FILES / "usd_TestFiles" / "RevitWallHealerCrash.usda"), heal=False)
        _, report = sm.tessellate_fast(brep, allow_partial=True, return_report=True)
        buffers = sm.tessellate_fast(brep, allow_partial=True, buffers=True)
        failed = [face for face, kind in enumerate(buffers["face_kinds"]) if kind == _KIND_CODES["failed"]]
        self.assertEqual(failed, report["failed_faces"])
        self.assertTrue(failed)
        self.assertFalse(set(buffers["face_ids"]) & set(failed))
        with self.assertRaises(RuntimeError):
            sm.tessellate_fast(brep, buffers=True)

    def test_python_owned(self):
        box = sm.create_box((0, 0, 0), 10, 20, 30)
        first, second = (sm.tessellate_fast(box, buffers=True) for _ in range(2))
        first["points"][0] = -1.0  # independent copies
        self.assertNotEqual(second["points"][0], -1.0)
        del box
        gc.collect()
        self.assertEqual(len(second["triangles"]), 36)  # outlive the Brep
        self.assertEqual(max(second["triangles"]), 7)


class TestFastTessellationSwitch(unittest.TestCase):
    """SMLIB_FAST_TESSELLATION=1 makes the USD tessellation (here usd.tessellate_file) use tessellate_fast."""

    _SCRIPT = """
import os
import sys

dll_directories = []
if sys.platform == "win32":
    for path in os.environ.get("PATH", "").split(";"):
        if path and os.path.isdir(path):
            dll_directories.append(os.add_dll_directory(path))

import _omni_solid as sm

print(sm.usd.tessellate_file(sys.argv[1], sys.argv[2])[0].point_count)
"""

    def _point_count(self, source, output, switch):
        environment = dict(os.environ)
        environment.pop("SMLIB_FAST_TESSELLATION", None)
        if switch is not None:
            environment["SMLIB_FAST_TESSELLATION"] = switch
        completed = subprocess.run([sys.executable, "-c", self._SCRIPT, source, output], env=environment,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return int(completed.stdout.split()[-1])

    def test_switch(self):
        mouse = sm.read_brep_from_file(str(_TEST_FILES / "pt_TestFiles" / "Mouse.smb"))
        with tempfile.TemporaryDirectory(prefix="smpy_tessellate_fast_") as tmpdir:
            source = os.path.join(tmpdir, "mouse.usda")
            output = os.path.join(tmpdir, "out.usda")
            sm.usd.export_brep(mouse, source)
            (brep,) = sm.usd.import_breps(source, heal=False)
            fast = len(sm.tessellate_fast(brep)["points"])
            reference = len(sm.tessellate(brep).to_mesh_arrays()["points"])
            self.assertNotEqual(fast, reference)  # so that the meshes tell the tessellators apart
            counts = {switch: self._point_count(source, output, switch) for switch in (None, "0", "1")}
        self.assertEqual(counts, {None: reference, "0": reference, "1": fast})


if __name__ == "__main__":
    unittest.main()
