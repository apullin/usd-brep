-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

project "SMLib"
    kind "SharedLib"
    location (workspace_dir.."/%{prj.name}")
    use_usd_tbb()
    
    -- Build PCH on Windows
    filter { "system:windows" }
        pchheader "StdAfx.h"
        pchsource "src/StdAfx.cpp"
        files { "src/StdAfx.cpp" }
        removeflags{ "NoPCH" }
    filter {}

    -- Regular build files
    files { "src/*.cpp", "inc/*.h" }
    includedirs {
        root.."/source/SMLib/inc",
    }
    vpaths { [''] = "*.cpp" }
    
    -- Compiler flags
    filter { "system:windows" }
        defines { "NL_DEBUG_CODE" }
        buildoptions {}
    filter {"configurations:debug", "system:windows"}
        links { "tbbmalloc_debug","tbb_debug"}
    filter {"configurations:release", "system:windows"}
        links { "tbbmalloc","tbb"}
    filter {"configurations:debug", "system:linux"}
        links { "tbbmalloc_debug","tbb_debug"}
    filter {"configurations:release", "system:linux"}
        links { "tbbmalloc","tbb"}
    filter { "system:macosx" }
        buildoptions (macos_common_buildoptions())
        linkoptions (macos_common_linkoptions())
    filter {}

    -- Opt-in kernel profile selected by tools/scripts/build_kernel_perf.py after
    -- probing the actual compiler and linker. Keep the ordinary build unchanged.
    local perf_build_dir = os.getenv("SMLIB_PERF_BUILD_DIR")
    if perf_build_dir then
        local perf_lto = assert(os.getenv("SMLIB_PERF_LTO"), "Performance build requires a probed LTO mode")
        assert(perf_lto == "thin" or perf_lto == "full" or perf_lto == "off", "Invalid performance LTO mode")
        filter { "configurations:release", "system:linux or macosx" }
            targetdir (path.join(perf_build_dir, "lib"))
            optimize "Speed"
            -- Preserve strict floating-point semantics, including multiply/add rounding.
            buildoptions { "-ffp-contract=off" }
            if perf_lto ~= "off" then
                local lto_flag = perf_lto == "thin" and "-flto=thin" or "-flto"
                buildoptions { lto_flag }
                linkoptions { lto_flag }
            end
        filter { "configurations:release", "system:linux", "toolset:clang" }
            removebuildoptions { "-Wshadow=local" } -- GCC-only warning spelling.
            buildoptions (macos_warning_suppressions()) -- Legacy SM_COMMON declarations, also on Linux.
            buildoptions { "-Wno-missing-braces" } -- Same legacy aggregate initializers as the macOS build.
            local gcc_install_dir = os.getenv("SMLIB_PERF_GCC_INSTALL_DIR")
            if gcc_install_dir then
                buildoptions { "--gcc-install-dir=" .. gcc_install_dir }
                linkoptions { "--gcc-install-dir=" .. gcc_install_dir }
            end
            if os.getenv("SMLIB_PERF_LINKER") == "lld" then
                linkoptions { "-fuse-ld=lld" }
            end
        filter { "configurations:release", "system:linux" }
            -- These paths remain valid after installing the isolated candidate.
            linkoptions { "-Wl,-rpath,'$$ORIGIN'", "-Wl,-rpath,'$$ORIGIN/../../target-deps/usd/release/lib'" }
        filter {}
    end





