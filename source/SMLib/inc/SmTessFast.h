// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmTessFast.h
* PURPOSE: Opt-in fast tessellation of a Brep into flat triangle arrays.
*
*   SmTessellateFast() keeps SMLib's Phase 1 (SmTess::Phase1SetupBrep:
*   edge tessellation, face refinement trees, edge splits), then
*   triangulates every face directly into arrays, faces in parallel,
*   using the edge vertices of Phase 1 so that seams stay watertight.
*   A face that no fast triangulator handles goes through SMLib's own
*   Phase 2 and its triangles are copied, with the same boundary
*   vertices. It is deterministic at any thread count; it is not
*   bit-identical to SmApiTessellate(), which stays unchanged.
**********************************************************************/

#ifndef __SMTESSFAST_H__
#define __SMTESSFAST_H__

#include <SmCoreTypes.h>
#include <SmVector2d.h>
#include <SmVector3d.h>
#include "../../SM_API/inc/SmApiTessellationParams.h"
#include <cstdint>
#include <vector>

class SmBrep;
class SmFace;
class SmSurface;
class SmTessSrfCache;

/*******************************************************************//**
   Per-face triangulation interface.

   SmTessellateFast() builds one SmFastFaceInput per face after Phase 1 and
   calls SmTriangulateFastFace(), possibly on worker threads. A face whose
   output status is not SM_SUCCESS goes through SMLib's Phase 2 instead.
***********************************************************************/

// One vertex of a face boundary loop. Boundary vertices come from SMLib's Phase 1 edge tessellation and
// are shared with neighboring faces through lId, which is what makes the result watertight. A
// triangulator must use them exactly: never move or merge them, and never add a vertex on a boundary
// segment.
struct SmFastLoopVertex
{
  double     dU, dV;     // parameters on this face's surface
  SmPoint3d  sPoint;     // 3D position (the edge's point: use it, don't re-evaluate the surface)
  long       lId;        // global vertex id in the output mesh
};

struct SmFastFaceInput
{
  const SmFace    * cpFace    = NULL;
  const SmSurface * cpSurface = NULL;
  // Closed loops in UV, without repeating the first vertex; loop 0 is the outer loop. The loops are
  // those of the face's upward faceuse, as in SMLib's Phase 1 UV PolyBrep: the outer loop runs CCW
  // in UV, inner loops CW. A seam edge of a periodic surface appears twice in a loop, at both ends
  // of the period. A pole (degenerate edge) is a run of distinct UV points that share one lId and
  // one 3D point.
  std::vector<std::vector<SmFastLoopVertex>> sLoops;
  // Optional interior UV points that resolve the surface's curvature to tolerance; the triangulator may
  // add more. SmTessellateFast() leaves it empty: cpCache's refinement tree has them (its leaves).
  std::vector<SmPoint2d> sInteriorSamples;
  SmTessellationParams sParams;  // the same parameters SmApiTessellate takes

  // G1 extensions
  const SmTessSrfCache * cpCache   = NULL;   // the face's Phase 1 cache; its refinement tree is built (read only)
  SmBoolean              bReversed = FALSE;  // TRUE: the pipeline reverses this face's triangles (see below)
  double                 dTolerance = 0.0;   // the face's 3D tolerance
  SmBoolean              bParallelSafe = FALSE;  // TRUE: cpSurface may be evaluated on several threads at once
};

// Why a triangulator rejected a face.
enum SmFastFaceFailure
{
  SM_FF_NONE = 0,  // not rejected
  SM_FF_LOOPS,     // its boundary loops are unusable (crossing, clockwise, duplicate points, bad input): the
                   // pipeline sends the faces that share a loop segment with it to SMLib too
  SM_FF_OTHER      // any other reason
};

struct SmFastFaceOutput
{
  SmStatus          eStatus  = SM_ERR;      // anything but SM_SUCCESS: the face falls back to SMLib
  SmFastFaceFailure eFailure = SM_FF_NONE;  // why, when eStatus != SM_SUCCESS
  std::vector<SmPoint2d>  sNewUV;         // interior vertices added by the triangulator ...
  std::vector<SmPoint3d>  sNewPoints;     // ... their 3D points (surface evaluations) ...
  std::vector<SmVector3d> sNewNormals;    // ... and unit surface normals (Su x Sv, not reversed)
  // Triangles as triples of vertex references: a value >= 0 is a global lId of a loop vertex; a value
  // < 0 is ~k, the k-th new vertex. Triangles run CCW in UV, the orientation of the loops, so their
  // geometric normal follows the surface normal Su x Sv. The pipeline reverses the triangles of faces
  // whose outward normal is opposite (SmFastFaceInput::bReversed), as SMLib's output does.
  std::vector<long> sTriangles;
  // Optional, 3 per triangle: the UV of each corner. Where a loop vertex occurs at several UVs of the face
  // (a seam, a pole, a crease), the pipeline evaluates its corner normals at the UV the triangle uses.
  std::vector<SmPoint2d> sTriangleUVs;
};

// Triangulators must be deterministic and must not touch shared SMLib state (no marks, no Notify, no
// context allocation of topology), so that faces can run on worker threads with sm_ParallelFor().
// The dispatcher: the rectangular-face triangulator, then the trimmed-face one.
SM_EXPORT SmStatus SmTriangulateFastFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut);

// Triangulator for faces whose outer loop is the rectangle of their refinement tree's UV domain
// (natural or rectangular trim, e.g. terrain): one or two triangles per tree leaf.
SM_EXPORT SmStatus SmTriangulateRectangularFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut);

// Triangulator for any face: SmTriangulateTrimmedFace() (SmFastFace.h), a constrained Delaunay
// triangulation of the loops with the refinement tree's leaf corners and refinement points.
SM_EXPORT SmStatus SmTriangulateTrimmedFastFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut);

/*******************************************************************//**
   Output of SmTessellateFast(): one triangle mesh for the whole Brep.
***********************************************************************/
struct SmFastMesh
{
  std::vector<double>  sPoints;         // 3 per vertex
  std::vector<float>   sNormals;        // 3 per vertex: unit sum of the vertex's per-face normals
  std::vector<int32_t> sTriangles;      // 3 vertex indices per triangle, CCW seen from outside
  std::vector<int32_t> sTriangleFaces;  // per triangle: index of its face in SmBrep::GetFaces()
  std::vector<float>   sCornerNormals;  // when asked: 9 per triangle, its face's normal at each corner
  std::vector<int8_t>  sFaceKinds;      // per face: 1 fast triangulator, 2 SMLib fallback, 0 failed

  // statistics
  ULONG  lFaces         = 0;   // faces of the Brep
  ULONG  lFastFaces     = 0;   // faces triangulated by a fast triangulator
  ULONG  lFallbackFaces = 0;   // faces triangulated by SMLib's Phase 2
  ULONG  lFailedFaces   = 0;   // faces without triangles (Phase 1 or Phase 2 failed)
  ULONG  lDemotedFaces  = 0;   // faces a fast triangulator did that went to SMLib (neighbor fallback)
  ULONG  lDroppedTriangles = 0;  // degenerate triangles dropped (repeated vertex)
  double dPhase1Ms      = 0.0; // copy + Phase 1
  double dFacesMs       = 0.0; // input building + fast triangulation
  double dFallbackMs    = 0.0; // SMLib Phase 2 of fallback faces
  double dAssemblyMs    = 0.0; // global arrays and normals
  double dCleanupMs     = 0.0; // deleting the Phase 1 data
};

struct SmFastOptions
{
  SmBoolean bCornerNormals  = FALSE;  // fill SmFastMesh::sCornerNormals
  SmBoolean bFastFaces      = TRUE;   // FALSE: every face falls back to SMLib (tests the pipeline)
  SmBoolean bTrimmedFaces   = TRUE;   // FALSE: only the rectangular-face triangulator (it also is the only
                                      // one with a maximum edge length or aspect ratio)
  SmBoolean bNeighborFallback = TRUE; // faces next to faces with unusable loops, and fast faces whose
                                      // chords other faces' triangles use too, go to SMLib as well
  SmBoolean bAllowPartial   = FALSE;  // TRUE: faces that fail are skipped instead of failing the call
  SmBoolean bFastRefinement = TRUE;   // FALSE: SMLib's own refinement tree build (the same tree, slower)
};

// Tessellates pBrep (not modified) into rMesh. Returns SM_SUCCESS, or an error when Phase 1 fails or,
// unless crOptions.bAllowPartial, when some face could not be tessellated.
SM_EXPORT SmStatus SmTessellateFast(const SmBrep              * cpBrep,
                                    const SmTessellationParams & crParams,
                                    SmFastMesh                 & rMesh,
                                    const SmFastOptions        & crOptions = SmFastOptions());

#endif // !__SMTESSFAST_H__
