#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build an isolated -O3 SMLib kernel, preferring Clang/ThinLTO with GCC fallback.

Dependencies must already be staged by the ordinary build. Compiler names are
not trusted: compile/link probes check C++17, strict FP flags, and LTO support.
Explicit CXX/--cxx selections are honored; source build failures never trigger
a silent compiler fallback. --activate preserves the previous kernel first.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, timeout=60, **kwargs)


def compiler_candidates(explicit):
    if explicit:
        compiler = shutil.which(explicit)
        if not compiler:
            raise ValueError(f"Requested C++ compiler is not executable: {explicit}")
        return [os.path.abspath(compiler)]
    directories = list(dict.fromkeys(os.get_exec_path()))
    candidates = []
    for name in ("clang++", "g++", "c++"):
        for directory in directories:
            folder = Path(directory)
            paths = [folder / name]
            if name != "c++":
                pattern = re.escape(name) + r"-\d+"
                versioned = [p for p in folder.glob(name + "-*") if re.fullmatch(pattern, p.name)]
                paths += sorted(versioned, key=lambda p: int(p.name.rsplit("-", 1)[1]), reverse=True)
            for path in paths:
                if path.is_file() and os.access(path, os.X_OK):
                    candidates.append(str(path.absolute()))
    return list(dict.fromkeys(candidates))


def probe(compiler, compile_flags, link_flags, system):
    # Use two translation units so the probe exercises a real LTO link.
    with tempfile.TemporaryDirectory(prefix="usd-brep-compiler-") as temporary:
        folder = Path(temporary)
        sources = [folder / "entry.cpp", folder / "helper.cpp"]
        sources[0].write_text('extern int helper();\nextern "C" int probe() { return helper(); }\n')
        sources[1].write_text('#include <string>\nint helper() { return std::string(3, \'x\').size(); }\n')
        objects = []
        for source in sources:
            obj = source.with_suffix(".o")
            result = run([compiler, *compile_flags, "-c", str(source), "-o", str(obj)])
            if result.returncode:
                return False, result.stderr.strip()
            objects.append(str(obj))
        shared = "-dynamiclib" if system == "Darwin" else "-shared"
        result = run([compiler, *objects, shared, *link_flags, "-o", str(folder / "probe.so")])
        return result.returncode == 0, result.stderr.strip()


def select_profile(args, system):
    arch_flags = []
    if system == "Darwin":
        arch_flags = ["-arch", "arm64"]
        if os.environ.get("SMLIB_MACOS_ARM64_ONLY", "0") in ("", "0"):
            arch_flags += ["-arch", "x86_64"]
    elif args.platform == "linux-x86_64":
        arch_flags = ["-m64"]
    failures = []
    for compiler in compiler_candidates(args.cxx):
        try:
            macros = run([compiler, "-dM", "-E", "-x", "c++", "-"], input="")
            if macros.returncode:
                reason = macros.stderr.strip().splitlines() or ["compiler macro probe failed"]
                failures.append(f"{compiler}: {reason[0]}")
                continue
            family = "clang" if "__clang__" in macros.stdout else "gcc" if "__GNUC__" in macros.stdout else None
            if family is None:
                failures.append(f"{compiler}: requires a Clang or GCC compatible compiler")
                continue
            toolchain_flags = []
            if args.gcc_install_dir and family == "clang":
                toolchain_flags = ["--gcc-install-dir=" + str(args.gcc_install_dir.resolve())]
            base_flags = [*arch_flags, *toolchain_flags, "-O3", "-fPIC", "-std=c++17", "-ffp-contract=off"]
            usable, reason = probe(compiler, base_flags, [*arch_flags, *toolchain_flags], system)
            if not usable:
                failures.append(f"{compiler}: {reason}")
                continue
            modes = ["thin", "full", "off"] if family == "clang" else ["full", "off"]
            if args.lto != "auto":
                modes = [args.lto] if args.lto in modes else []
            for mode in modes:
                lto_flags = [] if mode == "off" else ["-flto=thin" if mode == "thin" else "-flto"]
                linkers = ["lld", "system"] if family == "clang" and system == "Linux" and lto_flags else ["system"]
                for linker in linkers:
                    linker_flags = ["-fuse-ld=lld"] if linker == "lld" else []
                    usable, reason = probe(compiler, [*base_flags, *lto_flags],
                                           [*arch_flags, *toolchain_flags, *lto_flags, *linker_flags], system)
                    if usable:
                        cc_name = Path(compiler).name.replace("clang++", "clang").replace("g++", "gcc")
                        if cc_name == "c++":
                            cc_name = "cc"
                        cc = Path(compiler).with_name(cc_name)
                        return {
                            "cxx": compiler, "cc": args.cc or (str(cc) if cc.is_file() else compiler),
                            "compiler_version": run([compiler, "--version"]).stdout.strip(),
                            "compiler_family": family, "platform": args.platform, "lto": mode, "linker": linker,
                            "gcc_install_dir": str(args.gcc_install_dir.resolve()) if toolchain_flags else None,
                            "optimization": "-O3", "floating_point": "strict; -ffp-contract=off; no fast math",
                            "skipped_compilers": failures,
                        }
            failures.append(f"{compiler}: requested LTO mode is unsupported")
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{compiler}: {error}")
    raise ValueError("No usable performance compiler:\n" + "\n".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cxx", default=os.environ.get("CXX"), help="C++ compiler executable; otherwise prefer Clang")
    parser.add_argument("--cc", default=os.environ.get("CC"), help="Optional C compiler executable")
    parser.add_argument("--lto", choices=("auto", "thin", "full", "off"), default="auto")
    parser.add_argument("--gcc-install-dir", type=Path, help="Optional Clang/Linux libstdc++ toolchain selection")
    parser.add_argument("--platform", choices=("linux-x86_64", "linux-aarch64", "macos-universal"))
    parser.add_argument("--premake", type=Path, help="Premake executable; defaults to the staged host dependency")
    parser.add_argument("--premake-file", type=Path, default=ROOT / "premake5.lua", help="Optional local Premake entry")
    parser.add_argument("-j", "--jobs", type=int, default=int(os.environ.get("USD_BREP_BUILD_JOBS", "4")))
    parser.add_argument("--activate", action="store_true",
                        help="Atomically install the candidate after a successful build")
    parser.add_argument("--baseline-dir", type=Path, help="Where to preserve the previous kernel before activation")
    parser.add_argument("--probe-only", action="store_true", help="Print the detected profile without building")
    args = parser.parse_args()
    system = platform.system()
    if system not in ("Linux", "Darwin"):
        parser.error("This GNU Make performance helper supports Linux and macOS")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    machine = platform.machine().lower()
    host_platform = "macos-universal" if system == "Darwin" else (
        "linux-aarch64" if machine in ("aarch64", "arm64") else "linux-x86_64")
    args.platform = args.platform or host_platform
    if args.platform != host_platform:
        parser.error("Use the platform for the current host; cross compilation is not supported")
    if args.gcc_install_dir and system != "Linux":
        parser.error("--gcc-install-dir is a Linux-only Clang option")
    if args.cc:
        cc = shutil.which(args.cc)
        if not cc:
            parser.error(f"Requested C compiler is not executable: {args.cc}")
        args.cc = os.path.abspath(cc)
    if os.environ.get("SMLIB_NO_DEBUG_GFX_CHECK") == "1":
        parser.error("Use the ordinary release configuration for this performance profile")
    try:
        profile = select_profile(args, system)
    except ValueError as error:
        parser.error(str(error))
    if args.probe_only:
        print(json.dumps(profile, indent=2))
        return 0
    premake = args.premake or ROOT / "_build/host-deps/premake/premake5"
    if not premake.is_file() or not os.access(premake, os.X_OK):
        parser.error("Premake is missing; stage dependencies with the ordinary build or pass --premake")
    if not args.premake_file.is_file():
        parser.error(f"Premake entry is missing: {args.premake_file}")
    definitions = (Path(__file__), ROOT / "premake5.lua", ROOT / "premake5-utils.lua",
                   ROOT / "premake5-libraries.lua", ROOT / "source/SMLib/premake5.lua", args.premake_file)
    profile["build_definition_sha256"] = {str(path.resolve()): sha256(path) for path in definitions}
    profile["build_environment"] = {
        name: os.environ[name] for name in ("SMLIB_MACOS_ARM64_ONLY", "CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS")
        if name in os.environ
    }
    identity = {key: value for key, value in profile.items() if key != "skipped_compilers"}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    variant = ROOT / "_build/perf" / key
    variant.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SMLIB_PERF_BUILD_DIR=str(variant), SMLIB_PERF_LTO=profile["lto"],
               SMLIB_PERF_LINKER=profile["linker"])
    env.pop("SMLIB_PERF_GCC_INSTALL_DIR", None)
    if profile["gcc_install_dir"]:
        env["SMLIB_PERF_GCC_INSTALL_DIR"] = profile["gcc_install_dir"]
    config = "release_" + args.platform.split("-", 1)[1]
    commands = [
        [str(premake.resolve()), "--file=" + str(args.premake_file.resolve()),
         "--cc=" + profile["compiler_family"], "gmake2"],
        ["make", "--directory=" + str(variant / "gmake2/SMLib"), "config=" + config,
         "CC=" + shlex.quote(profile["cc"]), "CXX=" + shlex.quote(profile["cxx"]), "-j" + str(args.jobs)],
    ]
    if system == "Linux":
        commands[1].append("--output-sync=target")
    metadata = {**profile, "commands": commands, "build_directory": str(variant), "complete": False}
    record_path = variant / "build.json"
    record_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Building {profile['compiler_family']} {profile['optimization']}, LTO={profile['lto']}, "
          f"linker={profile['linker']}: {profile['cxx']}", flush=True)
    if profile["lto"] == "off" and args.lto == "auto":
        print("LTO is unavailable; using the explicitly recorded -O3 profile without LTO.", flush=True)
    started = time.monotonic()
    log_path = variant / "build.log"
    with log_path.open("w") as log:
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                metadata.update(build_elapsed_seconds=time.monotonic() - started, returncode=result.returncode)
                record_path.write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"Build failed; see {log_path}. No compiler fallback after a source build failure.")
                return result.returncode
    candidate = variant / "lib" / ("libSMLib.dylib" if system == "Darwin" else "libSMLib.so")
    metadata.update(build_elapsed_seconds=time.monotonic() - started, candidate_sha256=sha256(candidate), complete=True)
    record_path.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.activate:
        active = ROOT / "_build" / args.platform / "release" / candidate.name
        baseline_dir = args.baseline_dir or ROOT / "_build/perf/baseline" / args.platform
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline = baseline_dir / candidate.name
        if not baseline.exists() and active.exists():
            shutil.copy2(active, baseline)
        temporary = active.with_name(active.name + ".perf-new")
        shutil.copy2(candidate, temporary)
        temporary.replace(active)
        activation = {"active": str(active), "active_sha256": sha256(active), "baseline": str(baseline),
                      "baseline_sha256": sha256(baseline) if baseline.exists() else None}
        (variant / "activation.json").write_text(json.dumps(activation, indent=2) + "\n")
        print(f"Activated: {active}\nPreserved baseline: {baseline}")
    else:
        print(f"Built: {candidate}")
    print(f"Build record: {variant / 'build.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
