#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time BRep tessellation and check meshes bit for bit against a saved baseline.

Models are USD files with BrepArray prims, OCCT .brep files (converted with
occt_to_usd), or directories of either. Each model is imported once, then
tessellated with the default parameters in one warmup round and --repetitions
measured rounds, in a seeded random order. Timings cover tessellate() only.
Every round hashes the meshes, with points and face normals at double
precision; --save records hashes and timings, and --compare reports speedups
and fails if any mesh changed. --fast times the opt-in fast tessellator,
tessellate_fast(buffers=True), instead, including copying its arrays into
Python buffers, and hashes those buffers, with the points at double precision.
"""

import argparse
from array import array
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
USD_SUFFIXES = (".usda", ".usdc", ".usd")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_occt_to_usd():
    override = os.environ.get("OCCT_TO_USD")
    if override:
        return Path(override)
    candidates = sorted((REPO_ROOT / "_build").glob("*/release/occt_to_usd"))
    return candidates[0] if candidates else None


def collect_inputs(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.iterdir() if p.suffix == ".brep" or p.suffix in USD_SUFFIXES)
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"No such model: {path}")
    by_name = {}
    for path in files:
        if path.stem in by_name:
            raise SystemExit(f"Duplicate model name: {by_name[path.stem]} and {path}")
        by_name[path.stem] = path
    return by_name


def loaded_kernel():
    """Path and SHA-256 of the libSMLib in this process, on Linux."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return None
    for line in maps.read_text().splitlines():
        if line.endswith("/libSMLib.so"):
            path = Path(line.split()[-1]).resolve()
            return {"path": str(path), "sha256": sha256_file(path)}
    return None


def describe_machine():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
            "python": platform.python_version()}


def tessellate(sm, breps, parameters, fast=False):
    """Tessellate every Brep of a model; returns elapsed ns and meshes (exceptions for failures).

    With fast, the meshes are tessellate_fast()'s buffers, which the timing includes copying.
    """
    meshes = []
    started = time.perf_counter_ns()
    for brep in breps:
        try:
            if fast:
                meshes.append(sm.tessellate_fast(brep, buffers=True, **parameters))
            else:
                meshes.append(sm.tessellate(brep, **parameters))
        except RuntimeError as error:
            meshes.append(error)
    return time.perf_counter_ns() - started, meshes


def mesh_signature(meshes):
    """SHA-256 of the meshes, plus triangle counts or error messages.

    Points and face normals are hashed as the kernel's doubles. to_mesh_arrays()
    rounds to float32, so only its vertex normals, which have no double-precision
    accessor, and its face indices are taken from there.
    """
    digest = hashlib.sha256()
    summary = []
    for mesh in meshes:
        if isinstance(mesh, RuntimeError):
            digest.update(b"tessellation failed")
            summary.append(str(mesh))
            continue
        if isinstance(mesh, dict):  # tessellate_fast() buffers: points as the kernel's doubles
            for key in ("points", "normals", "triangles", "face_ids", "face_kinds"):
                digest.update(mesh[key].tobytes())
            summary.append(len(mesh["triangles"]) // 3)
            continue
        digest.update(array("d", itertools.chain.from_iterable(mesh.get_points())).tobytes())
        digest.update(array("d", itertools.chain.from_iterable(face.get_normal() for face in mesh.get_faces())).tobytes())
        arrays = mesh.to_mesh_arrays()
        digest.update(array("d", itertools.chain.from_iterable(arrays["normals"])).tobytes())
        digest.update(array("q", arrays["faces"]).tobytes())
        summary.append(len(arrays["face_normals"]))
    return digest.hexdigest(), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", type=Path, help="USD or OCCT .brep files, or directories of them")
    parser.add_argument("--repetitions", type=int, default=5, help="measured rounds after one warmup round")
    parser.add_argument("--seed", type=int, default=20261013, help="seed for the per-round model order")
    parser.add_argument("--save", type=Path, help="write hashes and timings to this JSON file")
    parser.add_argument("--compare", type=Path, help="baseline JSON from --save; fail if any mesh differs")
    parser.add_argument("--fast", action="store_true", help="time the opt-in fast tessellator, tessellate_fast()")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    tessellator = "fast" if args.fast else "smlib"
    if baseline and baseline.get("tessellator", "smlib") != tessellator:
        parser.error(f"--compare baseline was made with the {baseline.get('tessellator', 'smlib')} tessellator")

    try:
        import _omni_solid as sm
    except ImportError as error:
        raise SystemExit(f"Cannot import _omni_solid ({error}); set up the environment described in README.md")
    sm.usd.ensure_plugin_registered()
    parameters = sm.tessellation_defaults()

    inputs = collect_inputs(args.models)
    models = {}
    with tempfile.TemporaryDirectory(prefix="tessellation-benchmark-") as scratch:
        converter = None
        for name, path in inputs.items():
            record = {"input": path.name, "input_sha256": sha256_file(path)}
            models[name] = record
            usd_path = path
            if path.suffix == ".brep":
                converter = converter or find_occt_to_usd()
                if not converter or not converter.exists():
                    raise SystemExit("occt_to_usd is not built; build it or set OCCT_TO_USD")
                usd_path = Path(scratch) / (name + ".usda")
                result = subprocess.run([str(converter), str(path), str(usd_path)], capture_output=True, text=True)
                if result.returncode:
                    record["error"] = "occt_to_usd: " + (result.stderr.strip() or f"exit {result.returncode}")
                    continue
            try:
                record["breps"] = sm.usd.import_breps(str(usd_path), heal=False)
            except RuntimeError as error:
                record["error"] = "import: " + str(error)

        runnable = [name for name, record in models.items() if "breps" in record]
        rng = random.Random(args.seed)
        for round_index in range(args.repetitions + 1):
            order = list(runnable)
            rng.shuffle(order)
            for name in order:
                record = models[name]
                elapsed, meshes = tessellate(sm, record["breps"], parameters, args.fast)
                signature, summary = mesh_signature(meshes)
                if record.setdefault("mesh_sha256", signature) != signature:
                    record["unstable"] = True
                record["triangles"] = summary
                if round_index:
                    record.setdefault("times_ms", []).append(elapsed / 1e6)

    for record in models.values():
        record.pop("breps", None)
    report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "repetitions": args.repetitions, "seed": args.seed, "tessellator": tessellator,
              "tessellation": parameters, "kernel": loaded_kernel(), "machine": describe_machine(), "models": models}

    failed = False
    print(f"{'model':<30}{'triangles':>10}{'min ms':>10}{'mean ms':>10}{'std':>8}"
          + (f"{'baseline':>10}{'change':>9}  mesh" if baseline else ""))
    for name, record in models.items():
        times = record.get("times_ms")
        if times:
            triangles = sum(t for t in record["triangles"] if isinstance(t, int))
            std = statistics.stdev(times) if len(times) > 1 else 0.0
            line = f"{name:<30}{triangles:>10}{min(times):>10.2f}{statistics.mean(times):>10.2f}{std:>8.2f}"
        else:
            line = f"{name:<30}{'':>38}"
        if baseline:
            previous = baseline["models"].get(name)
            if previous is None:
                status = "new"
            elif previous["input_sha256"] != record["input_sha256"]:
                status, failed = "input differs", True
            elif (previous.get("mesh_sha256") != record.get("mesh_sha256")
                  or bool(previous.get("error")) != bool(record.get("error"))):
                status, failed = "DIFFERENT", True
            else:
                status = "same"
            if times and previous and previous.get("times_ms"):
                before = statistics.mean(previous["times_ms"])
                line += f"{before:>10.2f}{(statistics.mean(times) / before - 1) * 100:>8.1f}%  {status}"
            else:
                line += f"{'':>19}  {status}"
        if record.get("unstable"):
            failed = True
            line += "  mesh changed between rounds"
        failures = sum(isinstance(t, str) for t in record.get("triangles", []))
        if failures:
            line += f"  {failures} of {len(record['triangles'])} Breps failed to tessellate"
        if record.get("error"):
            line += "  " + record["error"]
        print(line)
    kernel = report["kernel"]
    if kernel:
        print(f"Kernel: {kernel['path']} (sha256 {kernel['sha256'][:16]})")
    if args.save:
        args.save.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {args.save}")
    if baseline:
        print("Meshes differ from the baseline" if failed else "All meshes are bit-identical to the baseline")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
