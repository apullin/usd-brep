// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmFastEdges.h
* PURPOSE: Fast face boundaries for the opt-in fast tessellation path:
*    what SMLib's tessellation Phase 1 (SmTess::Phase1SetupBrep) gives a
*    face triangulator, computed without copying or editing the Brep and
*    without UV trim curves.
*
*    Every edge is discretized once in 3D with SMLib's curve criteria
*    (SmCurve::TessellateByBisection()), split where an adjacent face's
*    refinement tree needs it (SMLib's IsoEdgeSubdivide() and
*    SplitFaceEdgesToUVSize() rules), and each discretization point gets
*    its UV on each adjacent face by point inversion, so both faces of an
*    edge share its points and vertex ids. Faces and edges run in
*    parallel; the result is identical at any thread count. It is not
*    bit-identical to SMLib's Phase 1, which stays unchanged.
**********************************************************************/

#ifndef __SMFASTEDGES_H__
#define __SMFASTEDGES_H__

#include <SmCoreTypes.h>
#include <SmVector2d.h>
#include <SmVector3d.h>
#include "../../SM_API/inc/SmApiTessellationParams.h"
#include <memory>
#include <vector>

class SmBrep;
class SmFace;
class SmEdge;
class SmTessSrfCache;
class SmSurfaceTessDriver;

/// One vertex of a face boundary loop (the layout of SmFastLoopVertex and SmTrimLoopVertex).
struct SmFastBoundaryVertex
{
  double     dU, dV;     ///< parameters on this face's surface
  SmPoint3d  sPoint;     ///< 3D position: the Brep vertex's point, or the edge curve's point
  long       lId;        ///< vertex id, shared by every face that uses the point
};

/// A leaf of a face's refinement tree: a UV rectangle.
struct SmFastLeaf
{
  double dU0, dV0, dU1, dV1;
};

/// Why a face has no fast boundary (it then needs SMLib's own Phase 1).
enum SmFastBoundaryFailure
{
  SM_FB_NONE = 0,
  SM_FB_SURFACE,        ///< no surface, or a surface kind this path doesn't handle
  SM_FB_TREE,           ///< the refinement tree could not be built
  SM_FB_TOPOLOGY,       ///< vertex loops, wire edges in loops, or other unsupported topology
  SM_FB_INVERSION,      ///< a point could not be inverted onto the surface
  SM_FB_LOOP_OPEN,      ///< consecutive edges don't meet in UV (and not at a pole)
  SM_FB_LOOP_WINDING,   ///< a loop winds around a periodic direction without a seam edge
  SM_FB_LOOP_ORIENTATION,  ///< the largest (outer) loop isn't counterclockwise, or another loop isn't clockwise
                           ///< (nor all of them the other way round)
  SM_FB_DEGENERATE,     ///< no loop with at least 3 points, or none that encloses any area in UV
  SM_FB_SAMPLING        ///< a point of one of its edges' curves couldn't be evaluated, or the curve didn't tessellate
};

/// The boundary of one face.
struct SmFastFaceBoundary
{
  const SmFace          * cpFace    = NULL;
  SmStatus                eStatus   = SM_ERR;        ///< SM_SUCCESS: sLoops are valid
  SmFastBoundaryFailure   eFailure  = SM_FB_NONE;
  /// Loops of the face's upward faceuse in UV, first point not repeated: loop 0 is the outer loop
  /// (counterclockwise), the others are inner loops (clockwise). A seam edge appears twice, at both
  /// ends of the period. A pole is a run of distinct UV points with one id and one 3D point.
  std::vector<std::vector<SmFastBoundaryVertex>> sLoops;
  /// TRUE: the faceuse's loops ran the other way (clockwise outer loop: the faceuse contradicts the face)
  /// and sLoops has them reversed. SMLib's mesh of such a face follows the loops, as do its neighbors,
  /// so its triangles run opposite to sLoops, while its normals stay the face's (bReversed).
  SmBoolean               bLoopsReversed = FALSE;
  SmTessSrfCache        * pCache    = NULL;          ///< refinement tree (owned by SmFastBoundaries)
  std::vector<SmFastLeaf> sLeaves;                   ///< its leaves, depth first
  SmBoolean               bReversed = FALSE;         ///< the face's outward normal is opposite to Su x Sv
  double                  dTolerance = 0.0;          ///< the face's 3D tolerance
  /// The loops as vertex ids in loop order (no UV), also for faces that failed (when their topology is
  /// usable): what a fallback needs to stay watertight with the faces around it.
  std::vector<std::vector<long>> sIdLoops;
  ULONG                   lInversions = 0;           ///< point inversions done for the face (statistics)
  double                  dTreeMs     = 0.0;         ///< time building the refinement tree (statistics)
};

/// The discretization of one edge.
struct SmFastEdgeBoundary
{
  const SmEdge      * cpEdge = NULL;
  std::vector<double> sParams;   ///< edge curve parameters, increasing, both ends included
  std::vector<long>   sIds;      ///< vertex id per parameter (the ends are the Brep vertices' ids)
};

struct SmFastBoundaries
{
  std::vector<SmPoint3d>          sPoints;   ///< per vertex id: Brep vertices first, then edge points
  std::vector<SmFastFaceBoundary> sFaces;    ///< in SmBrep::GetFaces() order
  std::vector<SmFastEdgeBoundary> sEdges;    ///< in SmBrep::GetEdges() order
  long   lVertices  = 0;                     ///< ids [0, lVertices) are the Brep vertices
  ULONG  lFailedFaces = 0;

  // milliseconds per stage
  double dPrepMs    = 0.0;   ///< topology tables, lazy curve and surface data
  double dTreesMs   = 0.0;   ///< refinement trees and iso-line tests (faces in parallel)
  double dEdgesMs   = 0.0;   ///< curve discretization (edges in parallel)
  double dSplitsMs  = 0.0;   ///< UVs and refinement-size splits (faces in parallel)
  double dMergeMs   = 0.0;   ///< merging the splits of both faces (edges in parallel)
  double dLoopsMs   = 0.0;   ///< remaining UVs and loops (faces in parallel)
  double dTotalMs   = 0.0;

  // (exported member by member: a class-level export would export its std:: members too)
  SM_EXPORT SmFastBoundaries();
  SM_EXPORT ~SmFastBoundaries();       // deletes the refinement trees
  SmFastBoundaries(const SmFastBoundaries &) = delete;
  SmFastBoundaries & operator=(const SmFastBoundaries &) = delete;

  std::unique_ptr<SmSurfaceTessDriver> pDriver;  ///< the trees' surface tessellation driver
  SM_EXPORT void Clear();
};

struct SmFastBoundaryOptions
{
  SmBoolean bParallel = TRUE;   ///< FALSE: everything on the calling thread
  /// Optional surface tessellation driver for the refinement trees (NULL: an SmSurfaceTessDriver with
  /// the parameters), e.g. one whose type selects a faster tree builder. Not owned.
  SmSurfaceTessDriver * pDriver = NULL;
  /// TRUE: stop after the first stage in which a face fails (topology, surface, tree or inversion
  /// failures: before the trees, or before the loops) and return SM_ERR_TESS_FAIL_FACE; rOut then only
  /// tells which faces failed so far. Failures of the loops come from the last stage.
  SmBoolean bStopAtFirstFailure = FALSE;
};

/// Computes the boundaries of every face of cpBrep for the given tessellation parameters (a negative
/// dMaxEdgeLength is resolved as SmApiTessellate() does). The Brep is not modified, except for lazily
/// built evaluation data of its curves and surfaces. Returns SM_SUCCESS unless the input is invalid;
/// faces it can't handle have eStatus != SM_SUCCESS.
SM_EXPORT SmStatus SmComputeFastBoundaries(const SmBrep                * cpBrep,
                                           const SmTessellationParams  & crParams,
                                           SmFastBoundaries            & rOut,
                                           const SmFastBoundaryOptions & crOptions = SmFastBoundaryOptions());

/// Name of a failure category, for reports.
SM_EXPORT const char * SmFastBoundaryFailureName(SmFastBoundaryFailure eFailure);

/// Polylines of the edges of a Brep, for display (SmBrep::TessellateBoundaries() semantics, not its points).
struct SmFastEdgePolylines
{
  std::vector<SmPoint3d>      sPoints;    ///< the polylines' points, edge after edge
  std::vector<size_t>         sOffsets;   ///< edge e's points are [sOffsets[e], sOffsets[e + 1]) (one entry more than edges)
  std::vector<const SmEdge *> sEdges;     ///< in SmBrep::GetEdges() order
  ULONG                       lFailed = 0;  ///< edges without a curve, or whose curve didn't sample: no points
};

/// Per edge of cpBrep (SmBrep::GetEdges() order): its curve discretized by crParams' curve angle, chord
/// height and maximum edge length (SmCurveTessDriver, as SmComputeFastBoundaries() discretizes edges,
/// without the faces' refinement splits) between its vertices' points, which end the polyline (a closed
/// edge's polyline is closed); a degenerate edge gives its one point; an edge without a curve, or whose
/// curve fails to tessellate or to evaluate at one of its points, none. Edges in parallel (serially with
/// notification callbacks), deterministic; the Brep isn't modified but for lazily built evaluation data.
/// SM_ERR when an edge failed (the others' are kept).
SM_EXPORT SmStatus SmComputeFastEdgePolylines(const SmBrep               * cpBrep,
                                              const SmTessellationParams & crParams,
                                              SmFastEdgePolylines        & rOut);

#endif // !__SMFASTEDGES_H__
