// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmFastFace.h
* PURPOSE: Fast display triangulation of one trimmed face from its
*    boundary loops (SMLib's tessellation Phase 1 edge tessellation):
*    a constrained Delaunay triangulation in a metric-scaled UV space,
*    refined with interior points until the surface tolerances hold.
*    Thread-safe and deterministic, so faces can run in parallel.
**********************************************************************/

#ifndef __SMFASTFACE_H__
#define __SMFASTFACE_H__

#include <SmTypes.h>
#include <SmCoreTypes.h>
#include <SmVector2d.h>
#include <SmVector3d.h>

#include <vector>

class SmFace;
class SmSurface;

/// Tessellation quality controls: the fields of SmTessellationParams (SmApiTessellationParams.h,
/// which SMLib doesn't include), with a negative dMaxEdgeLength already resolved to world units as
/// SmApiTessellate() does. Each control is disabled by 0.0.
struct SmTrimFaceParams
{
  double dChordHeightTol     = 0.0;
  double dCurveAngleTolDeg   = 25.0;
  double dSurfaceAngleTolDeg = 25.0;
  double dMaxEdgeLength      = 0.0;
  double dMaxAspectRatio     = 0.0;
};

/// One vertex of a face boundary loop. Boundary vertices come from SMLib's Phase 1 edge
/// tessellation and are shared with neighboring faces through lId, which makes the result
/// watertight. The triangulator uses them exactly: it never moves or merges them and never adds
/// a vertex on a boundary segment.
struct SmTrimLoopVertex
{
  double     dU = 0.0, dV = 0.0;  ///< parameters on this face's surface
  SmPoint3d  sPoint;              ///< 3D position (the edge's point; used as given)
  long       lId = -1;            ///< global vertex id in the output mesh
};

/// A UV rectangle [dU0, dU1] x [dV0, dV1]: a leaf of a surface refinement tree.
struct SmTrimLeaf
{
  double dU0 = 0.0, dV0 = 0.0, dU1 = 0.0, dV1 = 0.0;
};

struct SmTrimFaceInput
{
  const SmFace    * cpFace    = nullptr;  ///< the face (read only: orientation of its faceuse)
  const SmSurface * cpSurface = nullptr;  ///< its surface (evaluated; must be safe to evaluate concurrently)
  /// Closed loops in UV, without repeating the first vertex; loop 0 is the outer loop. Orientation
  /// as in SMLib's Phase 1 UV PolyBrep (the loops of the face's upward faceuse): the face is to the
  /// left of every loop segment in UV (outer loop CCW, inner loops CW).
  /// A seam edge of a periodic surface appears twice in a loop, at both ends of the period. A pole
  /// (degenerate edge) is a run of distinct UV points that share one lId and one 3D point.
  std::vector<std::vector<SmTrimLoopVertex>> sLoops;
  /// Optional leaves of SMLib's refinement tree for the face (Phase 1 SmTessSrfCache tree), which
  /// resolve the surface's curvature to the tolerances. As in SMLib's tessellation, the corners of
  /// the leaves that lie inside the face without touching its loops become interior vertices.
  std::vector<SmTrimLeaf> sLeaves;
  /// 3D tolerance of the face's Phase 1 polygon edges (SmFace::GetTolerance()). As in SMLib's
  /// SmTessSrfCache::ImplantUVSegment(), a loop segment touches the leaves within 10 times it,
  /// mapped to UV (at most a tenth of the surface's UV domain).
  double dTolerance = 0.0;
  /// Optional further interior UV points. Points outside the face, or too close to its boundary or
  /// to each other, are skipped. The triangulator adds points where the tolerances need them.
  std::vector<SmPoint2d> sInteriorSamples;
  SmTrimFaceParams sParams;
  /// false: triangles and normals are oriented like the face, as SMLib's output: reversed where the
  /// face's outward normal is opposite to the surface normal (the upward faceuse's shell region is
  /// not the Brep's infinite region). true: oriented by the surface alone (triangles CCW in UV,
  /// normals Su x Sv), for callers that reverse faces themselves; cpFace is then not read.
  bool bSurfaceOrientation = false;
  bool bTriangleUVs = false;  ///< fill SmTrimFaceOutput::sTriangleUVs
};

/// Why SmTriangulateTrimmedFace() rejected a face (the face then falls back to SMLib's tessellator).
enum SmTrimFaceFailure
{
  SM_TF_NONE = 0,              ///< success
  SM_TF_BAD_INPUT,             ///< missing surface, a loop with fewer than 3 vertices, non-finite values
  SM_TF_DUPLICATE_POINT,       ///< two different vertex ids at the same UV point
  SM_TF_LOOP_INTERSECTION,     ///< loop segments cross, or a loop vertex lies on another loop segment
  SM_TF_LOOP_ORIENTATION,      ///< the face isn't to the left of its loops (loops that run clockwise in
                               ///< UV, mixed orientations)
  SM_TF_EVALUATION,            ///< a surface evaluation failed
  SM_TF_INVALID_TRIANGLE,      ///< a triangle stays flipped or degenerate in 3D after refinement
  SM_TF_INTERNAL               ///< internal consistency check failed
};

struct SmTrimFaceOutput
{
  SmStatus          eStatus  = SM_ERR;      ///< anything but SM_SUCCESS: the face falls back to SMLib
  SmTrimFaceFailure eFailure = SM_TF_NONE;  ///< why, when eStatus != SM_SUCCESS
  std::vector<SmPoint2d>  sNewUV;           ///< interior vertices added by the triangulator ...
  std::vector<SmPoint3d>  sNewPoints;       ///< ... their 3D points (surface evaluations) ...
  std::vector<SmVector3d> sNewNormals;      ///< ... and unit normals, oriented like the triangles (the
                                            ///< surface normal, negated for a reversed face unless
                                            ///< bSurfaceOrientation, as SMLib's output normals)
  /// Triangles as triples of vertex references: a value >= 0 is a global lId of a loop vertex; a
  /// value < 0 is ~k, the k-th new vertex. Triangles are oriented as SMLib orients the face's
  /// polygons: counterclockwise in UV along the loops, reversed for a reversed face (unless
  /// bSurfaceOrientation), so that their geometric normal agrees with sNewNormals. Triangles that
  /// would be degenerate in 3D because two corners share an lId (at poles) are left out.
  std::vector<long> sTriangles;
  /// When SmTrimFaceInput::bTriangleUVs: the UV of each triangle corner, 3 per triangle (where a loop
  /// vertex occurs at several UVs, at a seam or pole, the one the triangle uses).
  std::vector<SmPoint2d> sTriangleUVs;
  /// Unit normals at the loop vertices, in the order of SmTrimFaceInput::sLoops, oriented like
  /// sNewNormals (computed anyway; saves the caller evaluating them again).
  std::vector<std::vector<SmVector3d>> sLoopNormals;
  /// Pairs of lIds: loop segments next to triangles that stay flipped or degenerate because the
  /// segment is too coarse for the surface (SMLib's tessellation has the same triangles unless its
  /// Phase 2 splits the edge). Splitting these edges in both adjacent faces and triangulating them
  /// again removes the triangles.
  std::vector<long> sSplitSegments;
  bool  bTolerancesMet = true;  ///< false when refinement stopped at its budget with triangles left to refine
  ULONG lSamplesUsed  = 0;      ///< interior vertices from sLeaves and sInteriorSamples
  ULONG lRefinePoints = 0;      ///< points added by refinement
};

/// Triangulates one face. Deterministic (the same output for the same input, on any thread) and
/// thread-safe: it only evaluates the surface and reads the face's orientation, and uses no SMLib
/// marks, notifications or context allocation.
SM_EXPORT SmStatus SmTriangulateTrimmedFace(const SmTrimFaceInput & crIn, SmTrimFaceOutput & rOut);

/// Name of a failure category, for reports.
SM_EXPORT const char * SmTrimFaceFailureName(SmTrimFaceFailure eFailure);

#ifdef SM_FASTFACE_DEBUG
#include <array>
/// Test hook (builds with SM_FASTFACE_DEBUG only): the last triangulation on the calling thread.
/// Vertices: u v x y px py pz nx ny nz ref pole singular; triangles: v0 v1 v2 label state constraints.
struct SmTrimFaceDebug
{
  double                              dMap[5] = {0, 0, 0, 0, 0};
  std::vector<std::array<double, 13>> sVerts;
  std::vector<std::array<int, 6>>     sTris;
  long aReasons[8] = {0, 0, 0, 0, 0, 0, 0, 0};  ///< refinement triggers: seam span, degenerate, flipped,
                                                ///< corner angle, max edge, centroid angle, chord, deviation
};
SM_EXPORT SmTrimFaceDebug & SmTrimFaceDebugState();
#include <string>
/// Test hook: sets a tuning parameter by name (see Tuning in SmFastFace.cpp); false when unknown.
SM_EXPORT bool SmTrimFaceDebugTuning(const std::string & crName, double dValue);
#endif

#endif // __SMFASTFACE_H__
