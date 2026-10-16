// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Bridges: SmApiBrep.h (SmApiTessellate — exact Brep NURBS to PolyBrep mesh;
//                       SmApiTessellateFast — opt-in fast tessellation to mesh arrays;
//                       SmApiTessellateBoundaries — trimmed edge curves to polylines).

#include "SmPyCommon.h"

#include <SmApiBrep.h>
#include <SmEdge.h>

#include <cmath>

static py::dict tessellation_report_to_dict(const SmTessellationReport& report)
{
    py::dict result;
    result["failures"] = tessellation_failures_to_list(report.failures);
    result["lamina_edge_count"] = report.laminaEdges;
    result["spine_edge_count"] = report.spineEdges;
    result["input_manifold"] = bool(report.inputManifold);
    result["output_manifold"] = bool(report.outputManifold);
    result["has_mesh"] = bool(report.hasMesh);
    return result;
}

static void check_tessellation_status(SmStatus status, const SmTessellationReport& report)
{
    if (status == SM_SUCCESS) return;
    try
    {
        SmPyRaise(status, "SmApiTessellate", __FILE__, __LINE__);
    }
    catch (const std::runtime_error& error)
    {
        py::object type = py::module_::import("_omni_solid").attr("TessellationError");
        py::object exception = type(error.what());
        exception.attr("report") = tessellation_report_to_dict(report);
        exception.attr("failures") = exception.attr("report")["failures"];
        exception.attr("status") = status;
        exception.attr("status_name") = sm_status_name(status);
        PyErr_SetObject(type.ptr(), exception.ptr());
        throw py::error_already_set();
    }
}


py::object smpy_tessellate(SmBrep* brep, double chord_height_tol, double curve_angle_tol_deg,
                          double surface_angle_tol_deg, double max_edge_length, double max_aspect_ratio,
                          bool allow_partial, bool return_failures, bool return_report)
{
    SmPolyBrep* r = nullptr;
    if (allow_partial && !return_failures && !return_report)
        throw py::value_error("allow_partial=True requires return_failures=True or return_report=True");
    SmTessellationReport report;
    SmPyErrorTrail::clear();
    SmTessellationParams params;
    params.dChordHeightTol     = chord_height_tol;
    params.dCurveAngleTolDeg   = curve_angle_tol_deg;
    params.dSurfaceAngleTolDeg = surface_angle_tol_deg;
    params.dMaxEdgeLength      = max_edge_length;
    params.dMaxAspectRatio     = max_aspect_ratio;
    const SmStatus status = SmApiTessellate(brep, r, params, allow_partial, nullptr, &report);
    check_tessellation_status(status, report);
    py::object mesh = py::cast(r, py::return_value_policy::reference);
    if (return_report) return py::make_tuple(mesh, tessellation_report_to_dict(report));
    if (return_failures)
        return py::make_tuple(mesh, tessellation_failures_to_list(report.failures));
    return mesh;
}

// A list of (x, y, z) tuples of Python floats, from 3 values per tuple.
static py::list float3_list(const std::vector<float>& values)
{
    const size_t count = values.size() / 3;
    py::list result(count);
    for (size_t i = 0; i < count; ++i)
    {
        PyObject* tuple = PyTuple_New(3);
        if (!tuple) throw py::error_already_set();
        PyList_SET_ITEM(result.ptr(), static_cast<py::ssize_t>(i), tuple);
        for (size_t k = 0; k < 3; ++k)
        {
            PyObject* value = PyFloat_FromDouble(values[3 * i + k]);
            if (!value) throw py::error_already_set();
            PyTuple_SET_ITEM(tuple, static_cast<py::ssize_t>(k), value);
        }
    }
    return result;
}

static py::list int_list(const std::vector<long>& values)
{
    py::list result(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        PyObject* value = PyLong_FromLong(values[i]);
        if (!value) throw py::error_already_set();
        PyList_SET_ITEM(result.ptr(), static_cast<py::ssize_t>(i), value);
    }
    return result;
}

static SmVector3d unit_or(const SmVector3d& vector, const SmVector3d& fallback)
{
    const double length = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    return std::isfinite(length) && length > 1e-30 ? vector / length : fallback;
}

// SmApiTessellateFast's mesh as PolyBrep.to_mesh_arrays() returns a mesh: float32 values, polygon
// stream, vertex or corner normals, and face normals as SmPolyFace::GetNormal() computes them.
static py::dict fast_mesh_arrays(const SmFastTessellation& mesh, bool faceVaryingNormals)
{
    const ULONG nVerts = mesh.sPoints.GetSize() / 3;
    const ULONG nTris = mesh.sTriangles.GetSize() / 3;
    if (mesh.sNormals.GetSize() != 3 * nVerts || (faceVaryingNormals && mesh.sCornerNormals.GetSize() != 9 * nTris))
        throw std::runtime_error("SmApiTessellateFast returned inconsistent mesh arrays");
    std::vector<float> points(3 * static_cast<size_t>(nVerts));
    for (ULONG i = 0; i < 3 * nVerts; ++i)
        points[i] = static_cast<float>(mesh.sPoints[i]);
    std::vector<float> normals;
    std::vector<long> faces;
    std::vector<float> faceNormals;
    faces.reserve(4 * static_cast<size_t>(nTris));
    faceNormals.reserve(3 * static_cast<size_t>(nTris));
    if (faceVaryingNormals)
        normals.reserve(9 * static_cast<size_t>(nTris));
    else
        normals.assign(mesh.sNormals.GetDataArray(), mesh.sNormals.GetDataArray() + mesh.sNormals.GetSize());
    auto point = [&](int v) {
        return SmVector3d(mesh.sPoints[3 * v], mesh.sPoints[3 * v + 1], mesh.sPoints[3 * v + 2]);
    };
    for (ULONG t = 0; t < nTris; ++t)
    {
        const int* corners = &mesh.sTriangles[3 * t];
        faces.insert(faces.end(), { 3L, corners[0], corners[1], corners[2] });
        if (faceVaryingNormals)
        {
            for (ULONG c = 0; c < 3; ++c)
            {
                const float* corner = &mesh.sCornerNormals[9 * t + 3 * c];
                const float* average = &mesh.sNormals[3 * corners[c]];
                const SmVector3d cornerNormal = unit_or(SmVector3d(corner[0], corner[1], corner[2]),
                                                        SmVector3d(average[0], average[1], average[2]));
                normals.insert(normals.end(), { static_cast<float>(cornerNormal.x), static_cast<float>(cornerNormal.y),
                                                static_cast<float>(cornerNormal.z) });
            }
        }
        // As SmPolyFace::GetNormal() for a triangle: (1, 1, 1) when degenerate.
        const SmVector3d start = point(corners[0]);
        SmVector3d normal = (point(corners[1]) - start) * (point(corners[2]) - start);
        if (normal.LengthSquared() > SM_EFF_ZERO_SQ)
            normal.Unitize();
        else
            normal.Set(1, 1, 1);
        normal = unit_or(normal, SmVector3d(0, 0, 1));
        faceNormals.insert(faceNormals.end(), { static_cast<float>(normal.x), static_cast<float>(normal.y), static_cast<float>(normal.z) });
    }
    py::dict out;
    out["points"] = float3_list(points);
    out["normals"] = float3_list(normals);
    out["faces"] = int_list(faces);
    out["face_normals"] = float3_list(faceNormals);
    return out;
}

static py::dict fast_report_to_dict(const SmFastTessellation& mesh)
{
    std::vector<long> faceIds(mesh.sTriangleFaces.GetDataArray(), mesh.sTriangleFaces.GetDataArray() + mesh.sTriangleFaces.GetSize());
    py::list kinds;
    py::list failed;
    for (ULONG i = 0; i < mesh.sFaceKinds.GetSize(); ++i)
    {
        const SmFastFaceKind kind = mesh.sFaceKinds[i];
        kinds.append(kind == SM_FAST_FACE_FAST ? "fast" : kind == SM_FAST_FACE_FALLBACK ? "fallback" : "failed");
        if (kind == SM_FAST_FACE_FAILED)
            failed.append(i);
    }
    py::dict result;
    result["face_ids"] = int_list(faceIds);
    result["face_kinds"] = kinds;
    result["failed_faces"] = failed;
    result["fast_face_count"] = mesh.lFastFaces;
    result["fallback_face_count"] = mesh.lFallbackFaces;
    result["failed_face_count"] = mesh.lFailedFaces;
    return result;
}

static py::object smpy_tessellate_fast(SmBrep* brep, double chord_height_tol, double curve_angle_tol_deg,
                                       double surface_angle_tol_deg, double max_edge_length, double max_aspect_ratio,
                                       bool allow_partial, bool face_varying_normals, bool return_report)
{
    if (allow_partial && !return_report)
        throw py::value_error("allow_partial=True requires return_report=True");
    SmTessellationParams params;
    params.dChordHeightTol     = chord_height_tol;
    params.dCurveAngleTolDeg   = curve_angle_tol_deg;
    params.dSurfaceAngleTolDeg = surface_angle_tol_deg;
    params.dMaxEdgeLength      = max_edge_length;
    params.dMaxAspectRatio     = max_aspect_ratio;
    // The mesh is a local: everything returned is a Python-owned copy.
    SmFastTessellation mesh;
    SmPyErrorTrail::clear();
    const SmStatus status = SmApiTessellateFast(brep, mesh, params, allow_partial ? TRUE : FALSE, face_varying_normals ? TRUE : FALSE);
    if (status != SM_SUCCESS && mesh.lFailedFaces > 0 && !allow_partial)
    {
        std::string faces;
        for (ULONG i = 0; i < mesh.sFaceKinds.GetSize(); ++i)
            if (mesh.sFaceKinds[i] == SM_FAST_FACE_FAILED)
                faces += (faces.empty() ? "" : ", ") + std::to_string(i);
        throw std::runtime_error(std::string("SmApiTessellateFast failed: ") + sm_status_name(status) + " (" +
                                 std::to_string(status) + "). " + std::to_string(mesh.lFailedFaces) +
                                 " face(s) failed to tessellate (indices: " + faces +
                                 "); allow_partial=True with return_report=True keeps the others");
    }
    if (status != SM_SUCCESS)
        SmPyRaise(status, "SmApiTessellateFast", __FILE__, __LINE__);
    py::dict arrays = fast_mesh_arrays(mesh, face_varying_normals);
    if (return_report)
        return py::make_tuple(arrays, fast_report_to_dict(mesh));
    return std::move(arrays);
}

void bind_tessellation(py::module_& m)
{
    m.attr("TessellationError") = py::reinterpret_steal<py::object>(
        PyErr_NewException("_omni_solid.TessellationError", PyExc_RuntimeError, nullptr));

    // inspect.signature() does not work on pybind11 builtins, so the keyword defaults
    // below are otherwise unreadable from Python except by parsing the docstring.
    // Built from SmTessellationDefaults, so it cannot drift from the real defaults.
    m.def(
        "tessellation_defaults",
        []()
        {
            py::dict d;
            d["chord_height_tolerance"] = SmTessellationDefaults::kChordHeightTol;
            d["curve_angle_tolerance_deg"] = SmTessellationDefaults::kCurveAngleTolDeg;
            d["surface_angle_tolerance_deg"] = SmTessellationDefaults::kSurfaceAngleTolDeg;
            d["max_edge_length"] = SmTessellationDefaults::kMaxEdgeLength;
            d["max_aspect_ratio"] = SmTessellationDefaults::kMaxAspectRatio;
            return d;
        },
        "Default tessellation quality as a fresh dict, keyed to match the keyword\n"
        "arguments of ``tessellate``, so it can be splatted and overridden:\n"
        "\n"
        ".. code-block:: python\n"
        "\n"
        "    sm.tessellate(brep, **{**sm.tessellation_defaults(), \"chord_height_tolerance\": 0.001})\n"
        "\n"
        "A new dict each call; mutating it does not change what ``tessellate`` uses.\n"
        "\n"
        "Wraps: SmTessellationDefaults");
    m.def(
        "tessellate_boundaries",
        [](SmBrep& brep, double angle_tolerance_deg, bool allow_partial) {
            SmTArray<SmPoint3d> points;
            SmTArray<ULONG> counts;
            SmTArray<SmEdge*> edges;
            // Not CHECK_STATUS: it stringifies its argument to name the failing entry
            // point, and an expression starting with '(' yields the whole source text
            // instead of a name. It also clears the trail before evaluating, so the
            // call has to sit between the clear and the raise to keep the kernel trail.
            SmPyErrorTrail::clear();
            const SmStatus status = SmApiTessellateBoundaries(&brep, angle_tolerance_deg, points, counts, edges);
            // points, not edges: rEdges is filled from GetEdges() before any sampling, so it
            // is non-empty even when every curve failed. A failed curve contributes a zero
            // count and no points, so sampled output is the only evidence of a usable
            // partial result. Without this, an all-failed sample reported success with an
            // empty point list.
            const SmStatus effective =
                allow_partial && status == SM_ERR && points.GetSize() > 0 ? SM_SUCCESS : status;
            if (effective != SM_SUCCESS)
                SmPyRaise(effective, "SmApiTessellateBoundaries", __FILE__, __LINE__);
            py::list pyPoints, pyCounts, pyEdges;
            for (const SmPoint3d& point : points)
                pyPoints.append(point_to_tuple(point));
            for (ULONG count : counts)
                pyCounts.append(count);
            for (SmEdge* edge : edges)
                pyEdges.append(py::cast(edge, py::return_value_policy::reference));
            return py::make_tuple(pyPoints, pyCounts, pyEdges);
        },
        py::arg("brep"), py::kw_only(), py::arg("angle_tolerance_deg") = 5.0, py::arg("allow_partial") = false,
        "Pure: returns boundary samples; inputs unchanged.\n"
        "\n"
        "Args:\n"
        "    brep: Source Brep.\n"
        "    angle_tolerance_deg: Curve tangent tolerance. Must be finite and positive.\n"
        "    allow_partial: Retain successful edges when sampling fails. Defaults to False.\n"
        "\n"
        "Returns:\n"
        "    tuple: Flat point tuples, per-edge vertex counts, and borrowed Edge handles.\n"
        "\n"
        "Notes:\n"
        "    Counts correspond to edges; degenerate curves retain one point.\n"
        "    Missing or failed curves have zero counts when allow_partial=True;\n"
        "    otherwise sampling failures raise RuntimeError. Invalid arguments always raise.\n"
        "    No relative chord criterion is applied. Edge handles belong to the source\n"
        "    Brep and must be reacquired after topology-changing operations.\n"
        "\n"
        "Wraps: SmApiTessellateBoundaries");

    m.def(
        "tessellate",
        &smpy_tessellate,
        py::return_value_policy::reference, py::arg("brep"),
        py::kw_only(),
        py::arg("chord_height_tolerance") = SmTessellationDefaults::kChordHeightTol,
        py::arg("curve_angle_tolerance_deg") = SmTessellationDefaults::kCurveAngleTolDeg,
        py::arg("surface_angle_tolerance_deg") = SmTessellationDefaults::kSurfaceAngleTolDeg,
        py::arg("max_edge_length") = SmTessellationDefaults::kMaxEdgeLength,
        py::arg("max_aspect_ratio") = SmTessellationDefaults::kMaxAspectRatio,
        py::arg("allow_partial") = false, py::arg("return_failures") = false, py::arg("return_report") = false,
        "Pure: returns new PolyBrep; inputs unchanged.\n"
        "\n"
        "Tessellate exact Brep NURBS geometry to a triangular ``PolyBrep`` mesh.\n"
        "\n"
        "Args:\n"
        "    brep: Source Brep. Not modified.\n"
        "    chord_height_tolerance: Maximum allowed distance between the\n"
        "        triangle mesh and the true surface, in modeling units.\n"
        "        Smaller positive values produce a finer mesh.\n"
        "        Defaults to ``0`` (disabled).\n"
        "        Typical ranges: ``0.001`` for high-quality rendering,\n"
        "        ``0.01`` for general visualization, ``0.1`` for preview.\n"
        "        Keyword-only.\n"
        "    curve_angle_tolerance_deg: Curve tangent angular tolerance.\n"
        "        Defaults to ``25``. ``0`` disables the curve angular criterion.\n"
        "        Keyword-only.\n"
        "    surface_angle_tolerance_deg: Surface angular tolerance.\n"
        "        Defaults to ``25``. Smaller positive values refine curved\n"
        "        faces; ``0`` disables this angular criterion. Keyword-only.\n"
        "    max_edge_length: Edge-length subdivision target, in modeling\n"
        "        units. It drives edge sampling and face subdivision and is\n"
        "        not a hard cap: final triangle edges can be somewhat\n"
        "        longer. ``0`` (default) disables it. A negative value\n"
        "        selects an automatic target derived from model size.\n"
        "        Keyword-only.\n"
        "    max_aspect_ratio: Aspect-ratio target for the cells a surface is\n"
        "        subdivided into before triangulation. It does not limit the\n"
        "        aspect ratio of the final triangles, which can still be long\n"
        "        and thin. ``0`` (default) disables it. Keyword-only.\n"
        "    allow_partial: Retain available faces when some faces fail.\n"
        "        Defaults to ``False`` (raise instead). Kernel and output errors\n"
        "        raise in either mode. Requires return_failures or return_report.\n"
        "        when enabled; otherwise raises ValueError. Keyword-only.\n"
        "    return_failures: Return ``(mesh, failures)`` with a list of dictionaries.\n"
        "        Each contains ``face_index`` (input faces() order), ``stage``\n"
        "        (boundary_preparation, face_tessellation or polygon_output),\n"
        "        and kernel ``status`` / ``status_name``. Defaults to ``False``.\n"
        "\n"
        "    return_report: Return (mesh, report) with face failures and nonfatal\n"
        "        edge-incidence diagnostics; defaults to False and takes precedence\n"
        "        over return_failures. These are not full solid-validity checks.\n"
        "\n"
        "Returns:\n"
        "    PolyBrep: triangulated mesh approximating the Brep within\n"
        "    the supplied tolerances, or ``(mesh, failures)`` when\n"
        "    ``return_failures=True`` with the first failure per source face.\n"
        "    Returns ``(mesh, report)`` instead when ``return_report=True``.\n"
        "\n"
        "Raises:\n"
        "    TessellationError: Carries report, failures, status and status_name, including\n"
        "        when strict mode rejects partial output.\n"
        "\n"
        "Notes:\n"
        "    Triangle count grows roughly quadratically as\n"
        "    ``chord_height_tolerance`` is tightened. Pair this with\n"
        "    ``poly_brep_volume_via_api`` for mass-property checks on the\n"
        "    mesh, or with the ``poly_boolean_*`` family for\n"
        "    polygon-mesh CSG.\n"
        "\n"
        "See Also:\n"
        "    poly_brep_is_manifold_via_api: Manifold check on the mesh.\n"
        "    poly_brep_volume_via_api: Volume of the tessellated mesh.\n"
        "    compute_volume: Exact volume of the source Brep.\n"
        "    poly_boolean: Boolean ops on tessellated meshes.\n"
        "    usd.export_mesh: Round-trip the mesh through USD.\n"
        "\n"
        "Wraps: SmApiTessellate");

    m.def(
        "tessellate_fast",
        &smpy_tessellate_fast,
        py::arg("brep"),
        py::kw_only(),
        py::arg("chord_height_tolerance") = SmTessellationDefaults::kChordHeightTol,
        py::arg("curve_angle_tolerance_deg") = SmTessellationDefaults::kCurveAngleTolDeg,
        py::arg("surface_angle_tolerance_deg") = SmTessellationDefaults::kSurfaceAngleTolDeg,
        py::arg("max_edge_length") = SmTessellationDefaults::kMaxEdgeLength,
        py::arg("max_aspect_ratio") = SmTessellationDefaults::kMaxAspectRatio,
        py::arg("allow_partial") = false, py::arg("face_varying_normals") = false, py::arg("return_report") = false,
        "Pure: returns new mesh arrays; inputs unchanged.\n"
        "\n"
        "Tessellate a Brep with the opt-in fast tessellator into mesh arrays.\n"
        "\n"
        "Args:\n"
        "    brep: Source Brep. Not modified.\n"
        "    chord_height_tolerance: As for ``tessellate``. Keyword-only.\n"
        "    curve_angle_tolerance_deg: As for ``tessellate``. Keyword-only.\n"
        "    surface_angle_tolerance_deg: As for ``tessellate``. Keyword-only.\n"
        "    max_edge_length: As for ``tessellate``. Keyword-only.\n"
        "    max_aspect_ratio: As for ``tessellate``. Keyword-only.\n"
        "    allow_partial: Keep the other faces when some faces fail; the failed\n"
        "        ones are listed in the report. Defaults to ``False`` (raise\n"
        "        instead). Requires ``return_report=True``; otherwise raises\n"
        "        ValueError. Keyword-only.\n"
        "    face_varying_normals: Return one normal per triangle corner, the\n"
        "        normal of its source face, instead of one averaged normal per\n"
        "        vertex, as ``PolyBrep.to_mesh_arrays`` does. Keyword-only.\n"
        "    return_report: Return ``(arrays, report)``. The report has\n"
        "        ``face_ids`` (source face of each triangle, in ``faces()``\n"
        "        order), ``face_kinds`` (per face: ``fast``, ``fallback`` or\n"
        "        ``failed``), ``failed_faces`` and the count of each kind.\n"
        "        Defaults to ``False``. Keyword-only.\n"
        "\n"
        "Returns:\n"
        "    dict: The layout of ``PolyBrep.to_mesh_arrays``: ``points``,\n"
        "    ``normals`` and ``face_normals`` as lists of 3-tuples of floats\n"
        "    rounded to float32, and ``faces`` as a flat triangle stream\n"
        "    ``[3, i0, i1, i2, 3, ...]``. ``(arrays, report)`` when\n"
        "    ``return_report=True``.\n"
        "\n"
        "Notes:\n"
        "    The fast tessellator keeps the edge tessellation and quality\n"
        "    controls of ``tessellate``, triangulates faces in parallel, and\n"
        "    triangulates the faces it cannot handle as ``tessellate`` does\n"
        "    (``fallback`` in the report). Faces share the vertices of their\n"
        "    common edges. The mesh is deterministic at any thread count but\n"
        "    differs from ``tessellate``'s, which this does not change. The\n"
        "    arrays are ordinary Python objects owned by the caller.\n"
        "\n"
        "See Also:\n"
        "    tessellate: The reference tessellator, returning a PolyBrep.\n"
        "\n"
        "Wraps: SmApiTessellateFast");
}
