// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmFastEdges.cpp
* PURPOSE: Fast face boundaries for the opt-in fast tessellation path
*          (see SmFastEdges.h).
*
* Stages (each one parallel over faces or over edges, so that a surface
* is only used by one task at a time; edge curves are only evaluated):
*   prep   topology tables (coedges = edgeuses of upward faceuses), lazy
*          NURBS data of curves and surfaces built up front;
*   E0     per edge: whether its curve is degenerate;
*   F1     per face: periodicity, refinement tree (SMLib's
*          SmTessSrfCache), and for each coedge whether it is an iso line
*          in UV (SMLib's IsoEdgeSubdivide()) and where the tree's leaves
*          cut it;
*   E1     per edge: iso splits of its first iso coedge, then SMLib's
*          curve tessellation of each piece (TessellateEdge()), clamped
*          at the vertices as ClampAndSplitEdge(); a curve that fails to
*          tessellate or evaluate fails the edge's faces;
*   F2     per face: UV of every edge point on the face by point
*          inversion (Gauss-Newton from the previous point or the nearest
*          leaf center, SmSurface::DropPoint() as a fallback), and
*          bisection of edge segments that are larger in UV than the
*          leaves they cross (SplitFaceEdgesToUVSize()); a split point that
*          can't be evaluated or inverted fails the face;
*   E2     per edge: union of both faces' splits;
*   F3     per face: UVs of the other face's splits, poles, loops.
**********************************************************************/

#include "StdAfx.h"

#include <SmFastEdges.h>
#include <SmTess.h>
#include <SmBrep.h>
#include <SmFace.h>
#include <SmFaceuse.h>
#include <SmLoopuse.h>
#include <SmEdge.h>
#include <SmEdgeuse.h>
#include <SmVertex.h>
#include <SmVertexuse.h>
#include <SmShell.h>
#include <SmRegion.h>
#include <SmCurve.h>
#include <SmCrvOnSurf.h>
#include <SmBSplineCurve.h>
#include <SmBSplineSurface.h>
#include <SmSurface.h>
#include <SmSurfaceCache.h>
#include <SmTree.h>
#include <SmParallel.h>
#include <SmTol.h>
#include <SmContext.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <tuple>
#include <map>

namespace {

typedef std::chrono::steady_clock sm_Clock;

double sm_Ms(sm_Clock::time_point sStart, sm_Clock::time_point sEnd)
{
  return std::chrono::duration<double, std::milli>(sEnd - sStart).count();
}

// flattened refinement tree node
struct sm_Node
{
  double u0, v0, u1, v1;
  int    lChild1, lChild2;   // -1 for leaves
  bool   bTessOnly;
};

struct sm_Face
{
  SmFace         * pFace     = NULL;
  SmSurface      * pSurface  = NULL;
  SmTessSrfCache * pSC       = NULL;
  std::vector<sm_Node>          sTree;
  std::vector<std::vector<int>> sLoops;   // coedges of each loopuse, in loop order
  SmExtent2d sNatural;
  double     aPeriod[2] = {0.0, 0.0};     // > 0 for a periodic direction (u, v)
  double     dTol       = 0.0;            // face tolerance
  SmBoolean  bParallel  = FALSE;          // its surface may be used on a worker thread
  SmBoolean  bFlipped   = FALSE;          // its loops run clockwise: the face lies to their right
  SmFastBoundaryFailure eFailure = SM_FB_NONE;
  ULONG      lInversions = 0;
  ULONG      lFallbacks  = 0;             // inversions that needed SmSurface::DropPoint()
  double     dInvTol     = 0.0;           // accepted distance of a point from the surface
  double     dMaxDist    = 0.0;           // largest distance of an inverted point (beyond dInvTol: SMLib's drop)
  // seeds for inversions without a guess: UV and 3D points of leaf centers (built on first use)
  std::vector<SmPoint2d> sSeedUV;
  std::vector<SmPoint3d> sSeedP;
  SmBoolean  bSeeds      = FALSE;
  const std::vector<SmFastLeaf> * cpLeaves = NULL;   // the tree's leaves (in the output)
};

struct sm_Coedge
{
  int         lFace     = -1;
  int         lEdge     = -1;
  SmEdgeuse * pEU       = NULL;
  bool        bReversed = false;   // runs from the edge's end to its start
  bool        bSeam     = false;   // its face uses the edge twice
  int         iIso      = 0;       // 1: u constant along it, 2: v constant
  std::vector<double>    sIsoT;    // where the face's leaves cut it (edge parameters)
  std::vector<double>    sSplitT;  // refinement splits the face asked for, with points and UVs
  std::vector<SmPoint3d> sSplitP;
  std::vector<SmPoint2d> sSplitUV;
  std::vector<SmPoint2d> sUV;      // UV per edge point (edge order)
  std::vector<char>      sHave;    // UV known
  bool        bFollowed = false;   // a closed edge whose UVs follow its curve around a period (sm_CoedgeUVs())
};

struct sm_Edge
{
  SmEdge        * pEdge   = NULL;
  const SmCurve * cpCurve = NULL;
  double          t0 = 0.0, t1 = 0.0;
  int             lV0 = -1, lV1 = -1;   // Brep vertex indices
  double          dTol = 0.0;           // edge tolerance
  bool            bDegenerate = false;
  bool            bFailed = false;      // E1: a point of its curve couldn't be evaluated, or it didn't tessellate
  std::vector<int>       sCoedges;      // one per face side, in SmEdge::GetEdgeuses() order
  std::vector<double>    sT;            // points, edge order, ends included
  std::vector<SmPoint3d> sP;
  std::vector<long>      sIds;          // per point: vertex id
};

struct sm_Context
{
  const SmBrep              * cpBrep = NULL;
  SmTessellationParams        sParams;
  double                      dBrepTol = 0.0;
  std::vector<sm_Face>        sFaces;
  std::vector<sm_Edge>        sEdges;
  std::vector<sm_Coedge>      sCoedges;
  std::vector<SmVertex *>     sVertices;
  std::vector<SmPoint3d>      sVertexPoints;
  std::vector<double>         sVertexTols;
  SmSurfaceTessDriver       * pDriver = NULL;
};

/*******************************************************************//**
   Refinement tree queries
***********************************************************************/

// flattens the cache's tree depth first (node 0 = root) and collects its leaves
void sm_FlattenTree(const SmTessSrfCache * cpSC, std::vector<sm_Node> & rTree, std::vector<SmFastLeaf> & rLeaves)
{
  rTree.clear();
  rLeaves.clear();
  SmTessSrfCache * pSC = SM_CONST_CAST(SmTessSrfCache *, cpSC);
  if (pSC == NULL || !pSC->HasTree())
    { return; }
  const SmTree     * cpTree = pSC->GetTree();
  const SmTreeNode * cpTop  = cpTree ? cpTree->GetTopNode() : NULL;
  if (cpTop == NULL)
    { return; }
  struct Item { const SmTreeNode * cpNode; int lParent; int lSlot; };
  std::vector<Item> sStack(1, Item{cpTop, -1, 0});
  while (!sStack.empty())
    {
      const Item sItem = sStack.back();
      sStack.pop_back();
      const SmBezierAux2d * cpAux = (const SmBezierAux2d *)sItem.cpNode->m_pData;
      if (cpAux == NULL)
        { continue; }
      const int lIndex = (int)rTree.size();
      const SmExtent2d & crD = cpAux->m_sUVDomain;
      rTree.push_back(sm_Node{crD.GetMin().x, crD.GetMin().y, crD.GetMax().x, crD.GetMax().y, -1, -1,
                              cpAux->m_bIsTessOnly ? true : false});
      if (sItem.lParent >= 0)
        { (sItem.lSlot == 1 ? rTree[sItem.lParent].lChild1 : rTree[sItem.lParent].lChild2) = lIndex; }
      if (sItem.cpNode->m_pChild1 != NULL || sItem.cpNode->m_pChild2 != NULL)
        {
          if (sItem.cpNode->m_pChild2) { sStack.push_back(Item{sItem.cpNode->m_pChild2, lIndex, 2}); }
          if (sItem.cpNode->m_pChild1) { sStack.push_back(Item{sItem.cpNode->m_pChild1, lIndex, 1}); }
        }
      else
        { rLeaves.push_back(SmFastLeaf{crD.GetMin().x, crD.GetMin().y, crD.GetMax().x, crD.GetMax().y}); }
    }
}

// calls fn(node) for the leaves (as SmTess::CheckAgainstSurfaceCache() counts them: a node whose second
// child is tessellation-only counts as a leaf, tessellation-only nodes are skipped) that intersect the box
template <class Fn>
void sm_ForLeaves(const std::vector<sm_Node> & crTree, double u0, double v0, double u1, double v1, const Fn & crFn)
{
  if (crTree.empty())
    { return; }
  int aStack[128];
  int lTop = 0;
  aStack[lTop++] = 0;
  while (lTop > 0)
    {
      const sm_Node & crN = crTree[aStack[--lTop]];
      if (crN.u1 < u0 || crN.u0 > u1 || crN.v1 < v0 || crN.v0 > v1)   // SmExtent2d::AreDisjoint()
        { continue; }
      if (crN.bTessOnly)
        { continue; }
      if (crN.lChild1 >= 0)
        {
          if (crN.lChild2 >= 0 && crTree[crN.lChild2].bTessOnly)
            { crFn(crN); continue; }
          if (lTop + 2 > 128)
            { crFn(crN); continue; }  // too deep: use the node itself
          if (crN.lChild2 >= 0) { aStack[lTop++] = crN.lChild2; }
          aStack[lTop++] = crN.lChild1;
        }
      else
        { crFn(crN); }
    }
}

/*******************************************************************//**
   Point inversion on a face's surface
***********************************************************************/

// point and exact first derivatives (SmSurface::Evaluate1stDerivatives() replaces zero tangents, as at
// poles, with small vectors)
inline SmStatus sm_Eval1(const SmSurface * cpSurface, const SmPoint2d & crUV, SmPoint3d & rS, SmVector3d & rSu, SmVector3d & rSv)
{
  SmVector3d aM[4];
  const SmStatus eStat = cpSurface->Evaluate(crUV, 1, 1, TRUE, TRUE, TRUE, aM, FALSE);
  rS  = aM[0];
  rSv = aM[1];
  rSu = aM[2];
  return eStat;
}

// snaps parameters within rounding of a natural domain bound onto it (boundary points then land exactly
// on the bound, as the refinement tree's boundary nodes do)
inline void sm_SnapToDomain(const sm_Face & crF, SmPoint2d & rUV)
{
  for (int d = 0; d < 2; d++)
    {
      double & rX = (d == 0) ? rUV.x : rUV.y;
      const double dMin = (d == 0) ? crF.sNatural.GetMin().x : crF.sNatural.GetMin().y;
      const double dMax = (d == 0) ? crF.sNatural.GetMax().x : crF.sNatural.GetMax().y;
      const double dEps = 1.0e-11 * (dMax - dMin);
      if (std::fabs(rX - dMin) <= dEps) { rX = dMin; }
      else if (std::fabs(rX - dMax) <= dEps) { rX = dMax; }
    }
}

// wraps a periodic parameter into the natural domain, clamps the others
inline void sm_IntoDomain(const sm_Face & crF, SmPoint2d & rUV)
{
  for (int d = 0; d < 2; d++)
    {
      double & rX = (d == 0) ? rUV.x : rUV.y;
      const double dMin = (d == 0) ? crF.sNatural.GetMin().x : crF.sNatural.GetMin().y;
      const double dMax = (d == 0) ? crF.sNatural.GetMax().x : crF.sNatural.GetMax().y;
      const double dP = crF.aPeriod[d];
      if (dP > 0.0)
        {
          if (rX < dMin || rX > dMax)
            { rX -= dP * std::floor((rX - dMin) / dP); }
          if (rX > dMax) { rX = dMax; }
          if (rX < dMin) { rX = dMin; }
        }
      else
        { rX = std::min(dMax, std::max(dMin, rX)); }
    }
}

// Gauss-Newton closest point on the surface from rUV (the edge points lie on the surface within
// tolerance, where it converges quadratically); rdDist is the final distance
SmBoolean sm_Newton(const sm_Face & crF, const SmPoint3d & crP, SmPoint2d & rUV, double & rdDist)
{
  const double dSizeU = crF.sNatural.XLength(), dSizeV = crF.sNatural.YLength();
  const double dEpsU = 1.0e-13 * (1.0 + dSizeU), dEpsV = 1.0e-13 * (1.0 + dSizeV);
  SmPoint2d sUV = rUV;
  sm_IntoDomain(crF, sUV);
  rdDist = std::numeric_limits<double>::infinity();
  for (int it = 0; it < 40; it++)
    {
      SmPoint3d  sS;
      SmVector3d sSu, sSv;
      if (sm_Eval1(crF.pSurface, sUV, sS, sSu, sSv) != SM_SUCCESS)
        { return FALSE; }
      const SmVector3d sR(sS.x - crP.x, sS.y - crP.y, sS.z - crP.z);
      rdDist = sR.Length();
      const double a11 = sSu.Dot(sSu), a12 = sSu.Dot(sSv), a22 = sSv.Dot(sSv);
      const double b1 = -sSu.Dot(sR), b2 = -sSv.Dot(sR);
      const double dDet = a11 * a22 - a12 * a12;
      double dU = 0.0, dV = 0.0;
      if (dDet > 1.0e-12 * a11 * a22 && dDet > 0.0)
        {
          dU = (b1 * a22 - b2 * a12) / dDet;
          dV = (a11 * b2 - a12 * b1) / dDet;
        }
      else if (a22 >= a11 && a22 > 0.0) { dV = b2 / a22; }   // degenerate in u (a pole): v only
      else if (a11 > 0.0)               { dU = b1 / a11; }
      else
        { return FALSE; }
      // limit wild steps
      dU = std::max(-0.25 * dSizeU, std::min(0.25 * dSizeU, dU));
      dV = std::max(-0.25 * dSizeV, std::min(0.25 * dSizeV, dV));
      const SmPoint2d sPrev = sUV;
      sUV.x += dU;
      sUV.y += dV;
      sm_IntoDomain(crF, sUV);
      if (std::fabs(sUV.x - sPrev.x) <= dEpsU && std::fabs(sUV.y - sPrev.y) <= dEpsV)
        {
          sm_SnapToDomain(crF, sUV);
          rUV = sUV;
          return TRUE;
        }
    }
  rUV = sUV;
  return FALSE;
}

// the seed closest to crP (seeds: the leaves' centers, or a grid for very large trees)
SmPoint2d sm_Seed(sm_Face & rF, const SmPoint3d & crP)
{
  if (!rF.bSeeds)
    {
      rF.bSeeds = TRUE;
      static const std::vector<SmFastLeaf> s_sNone;
      const std::vector<SmFastLeaf> & crLeaves = rF.cpLeaves ? *rF.cpLeaves : s_sNone;
      if (!crLeaves.empty() && crLeaves.size() <= 4096)
        {
          for (const SmFastLeaf & crL : crLeaves)
            { rF.sSeedUV.push_back(SmPoint2d(0.5 * (crL.dU0 + crL.dU1), 0.5 * (crL.dV0 + crL.dV1))); }
        }
      else
        {
          const int n = 32;
          for (int i = 0; i <= n; i++)
            {
              for (int j = 0; j <= n; j++)
                {
                  rF.sSeedUV.push_back(SmPoint2d(rF.sNatural.GetMin().x + rF.sNatural.XLength() * (i + 0.5) / (n + 1),
                                                 rF.sNatural.GetMin().y + rF.sNatural.YLength() * (j + 0.5) / (n + 1)));
                }
            }
        }
      rF.sSeedP.resize(rF.sSeedUV.size());
      for (size_t k = 0; k < rF.sSeedUV.size(); k++)
        {
          if (rF.pSurface->EvaluatePoint(rF.sSeedUV[k], rF.sSeedP[k]) != SM_SUCCESS)
            { rF.sSeedP[k] = SmPoint3d(1.0e300, 1.0e300, 1.0e300); }
        }
    }
  size_t lBest = 0;
  double dBest = std::numeric_limits<double>::infinity();
  for (size_t k = 0; k < rF.sSeedP.size(); k++)
    {
      const double d = rF.sSeedP[k].DistanceBetweenSquared(crP);
      if (d < dBest) { dBest = d; lBest = k; }
    }
  return rF.sSeedP.empty() ? SmPoint2d(0.5 * (rF.sNatural.GetMin().x + rF.sNatural.GetMax().x), 0.5 * (rF.sNatural.GetMin().y + rF.sNatural.GetMax().y)) : rF.sSeedUV[lBest];
}

// The UV of crPoint on the face's surface: Gauss-Newton from the guess (or from the closest seed),
// else SmSurface::DropPoint() (closed form for analytic surfaces, local, then global solves). With a
// guess, a periodic result is shifted by whole periods to the guess.
SmBoolean sm_Invert(sm_Face & rF, const SmPoint3d & crPoint, const SmPoint2d * cpGuess, SmPoint2d & rUV)
{
  rF.lInversions++;
  double    dDist = 0.0;
  SmBoolean bDone = FALSE;
  SmPoint2d sUV = cpGuess ? *cpGuess : sm_Seed(rF, crPoint);
  if (sm_Newton(rF, crPoint, sUV, dDist) && dDist <= rF.dInvTol)
    { bDone = TRUE; }
  else if (cpGuess != NULL)
    {
      sUV = sm_Seed(rF, crPoint);
      if (sm_Newton(rF, crPoint, sUV, dDist) && dDist <= rF.dInvTol)
        { bDone = TRUE; }
    }
  if (!bDone)
    {
      // SMLib's inversion; keep whichever point is closer
      rF.lFallbacks++;
      SmBoolean bSuccess = FALSE, bMulti = FALSE;
      double    dGap = 0.0;
      SmPoint2d sDrop;
      if (   rF.pSurface->DropPoint(crPoint, rF.sNatural, NULL, bSuccess, sDrop, dGap, bMulti, SM_SO_MINIMIZE, NULL, NULL)
             == SM_SUCCESS
          && bSuccess)
        {
          SmPoint3d sQ;
          if (rF.pSurface->EvaluatePoint(sDrop, sQ) == SM_SUCCESS && sQ.DistanceBetween(crPoint) < dDist)
            {
              sUV   = sDrop;
              dDist = sQ.DistanceBetween(crPoint);
            }
        }
      if (!(dDist < std::numeric_limits<double>::infinity()))
        { return FALSE; }
    }
  rF.dMaxDist = std::max(rF.dMaxDist, dDist);
  rUV = sUV;
  if (cpGuess != NULL)
    {
      for (int d = 0; d < 2; d++)
        {
          const double dP = rF.aPeriod[d];
          if (dP <= 0.0)
            { continue; }
          double & rX = (d == 0) ? rUV.x : rUV.y;
          const double dG = (d == 0) ? cpGuess->x : cpGuess->y;
          rX += dP * std::floor((dG - rX) / dP + 0.5);
        }
    }
  return TRUE;
}

// For a UV on a periodic boundary of the natural domain, or within dTol3d of it in 3D (a point of an edge a
// little off the seam line inverts a little off the bound): the side the face is on, from the direction
// into the face (crInward, 3D), as the face lies to the left of its loops. Moves rUV to that side.
void sm_PickSeamSide(const sm_Face & crF, const SmVector3d & crInward, double dTol3d, SmPoint2d & rUV)
{
  for (int d = 0; d < 2; d++)
    {
      const double dP = crF.aPeriod[d];
      if (dP <= 0.0)
        { continue; }
      const double dMin = (d == 0) ? crF.sNatural.GetMin().x : crF.sNatural.GetMin().y;
      const double dMax = (d == 0) ? crF.sNatural.GetMax().x : crF.sNatural.GetMax().y;
      double & rX = (d == 0) ? rUV.x : rUV.y;
      const double dOff = std::min(std::fabs(rX - dMin), std::fabs(rX - dMax));
      if (dOff > 1.0e-3 * dP)
        { continue; }
      SmPoint3d  sP;
      SmVector3d sSu, sSv;
      if (sm_Eval1(crF.pSurface, rUV, sP, sSu, sSv) != SM_SUCCESS)
        { continue; }
      if (dOff > 1.0e-9 * dP && dOff * ((d == 0) ? sSu : sSv).Length() > dTol3d)
        { continue; }
      // UV components of crInward: [Su.Su Su.Sv; Su.Sv Sv.Sv] [a b] = [Su.w Sv.w]
      const double a11 = sSu.Dot(sSu), a12 = sSu.Dot(sSv), a22 = sSv.Dot(sSv);
      const double b1 = sSu.Dot(crInward), b2 = sSv.Dot(crInward);
      const double dDet = a11 * a22 - a12 * a12;
      if (!(std::fabs(dDet) > 0.0))
        { continue; }
      const double dA = (b1 * a22 - b2 * a12) / dDet;
      const double dB = (a11 * b2 - a12 * b1) / dDet;
      const double dComp = (d == 0) ? dA : dB;
      const double dOther = (d == 0) ? dB : dA;
      if (std::fabs(dComp) < 1.0e-3 * std::fabs(dOther))
        { continue; }  // along the seam: undecided
      rX = (dComp < 0.0) ? dMax : dMin;   // face toward smaller values: the high side, else the low side
    }
}

// TRUE when the surface is degenerate at rUV in the u (rbU) or v (rbV) direction: moving in that
// parameter doesn't move the point (a pole or apex line). A vanishing derivative decides; a merely small
// one (a nearly collapsed row of control points) is checked by evaluating along the whole parameter line,
// which must stay within twice the vertex's tolerance.
void sm_Degeneracy(const sm_Face & crF, const SmPoint2d & crUV, double dVertexTol, SmBoolean & rbU, SmBoolean & rbV)
{
  rbU = rbV = FALSE;
  SmPoint2d  sAt = crUV;
  sm_IntoDomain(crF, sAt);   // (a loop's UVs may lie a period outside, where a closed surface isn't defined)
  SmPoint3d  sP;
  SmVector3d sSu, sSv;
  if (sm_Eval1(crF.pSurface, sAt, sP, sSu, sSv) != SM_SUCCESS)
    { return; }
  const double dU = sSu.Length(), dV = sSv.Length();
  if (dU <= 1.0e-9 * dV) { rbU = TRUE; }
  if (dV <= 1.0e-9 * dU) { rbV = TRUE; }
  const double dTol = std::max(2.0 * dVertexTol, 1.0e-9 * (1.0 + sP.GetMaxDimension()));
  auto Collapsed = [&](int d)
    {
      const double dMin = (d == 0) ? crF.sNatural.GetMin().x : crF.sNatural.GetMin().y;
      const double dMax = (d == 0) ? crF.sNatural.GetMax().x : crF.sNatural.GetMax().y;
      for (int k = 0; k <= 4; k++)
        {
          SmPoint2d sUV = sAt;
          ((d == 0) ? sUV.x : sUV.y) = dMin + (dMax - dMin) * k / 4.0;
          SmPoint3d sQ;
          if (crF.pSurface->EvaluatePoint(sUV, sQ) != SM_SUCCESS || sQ.DistanceBetween(sP) > dTol)
            { return FALSE; }
        }
      return TRUE;
    };
  if (!rbU && !rbV && dU < 1.0e-3 * dV && Collapsed(0)) { rbU = TRUE; }
  if (!rbU && !rbV && dV < 1.0e-3 * dU && Collapsed(1)) { rbV = TRUE; }
}

/*******************************************************************//**
   F1: refinement trees and iso-line tests
***********************************************************************/
SmStatus sm_BuildTree(sm_Context & rC, sm_Face & rF, std::vector<SmFastLeaf> & rLeaves)
{
  const SmTessellationParams & crP = rC.sParams;
  SmTessSrfCache * pSC = new (*rF.pFace->GetContext())
    SmTessSrfCache(rF.pFace, NULL, rC.pDriver, crP.dChordHeightTol, crP.dSurfaceAngleTolDeg * SM_PI / 180.0,
                   crP.dMaxEdgeLength, 0.0, 0.001, crP.dMaxAspectRatio);
  if (pSC == NULL)
    { return SM_ERR; }
  pSC->SetTessellationAlgorithm(SM_TA_FASTER_TESSELLATION);
  if (pSC->BuildTree() != SM_SUCCESS)
    {
      delete pSC;
      return SM_ERR;
    }
  rF.pSC = pSC;
  sm_FlattenTree(pSC, rF.sTree, rLeaves);
  return rF.sTree.empty() ? SM_ERR : SM_SUCCESS;
}

// SmTess::IsoEdgeSubdivide() / FindIsoLineSplits() for one coedge: when the coedge is a straight iso
// line in the face's UV, the edge parameters where the leaves' lower bounds cut it (linear mapping)
void sm_IsoTest(sm_Context & rC, sm_Face & rF, sm_Coedge & rCE)
{
  const sm_Edge & crE = rC.sEdges[rCE.lEdge];
  if (crE.bDegenerate || crE.cpCurve == NULL || !(crE.t1 > crE.t0))
    { return; }
  static const double aS[6] = {0.0, 0.25, 0.3456, 0.5, 0.75, 1.0};
  SmPoint2d aUV[6];
  const double dSizeU = rF.sNatural.XLength(), dSizeV = rF.sNatural.YLength();
  const double dTolU = 1.0e-9 * (1.0 + dSizeU), dTolV = 1.0e-9 * (1.0 + dSizeV);
  // the middle, then outward point by point (each guessed from its inner neighbor, so the ends of a
  // closed edge on a periodic surface land on both sides of the period); most coedges are no iso
  // lines, which three points show
  static const int aOrder[6] = {3, 1, 0, 4, 5, 2};
  static const int aGuess[6] = {-1, 3, 1, 3, 4, 1};   // per position in aOrder
  for (int q = 0; q < 6; q++)
    {
      const int k = aOrder[q];
      SmPoint3d sP;
      if (k == 0)      { sP = rC.sVertexPoints[crE.lV0]; }
      else if (k == 5) { sP = rC.sVertexPoints[crE.lV1]; }
      else if (crE.cpCurve->EvaluatePoint(crE.t0 + aS[k] * (crE.t1 - crE.t0), sP) != SM_SUCCESS)
        { return; }
      if (!sm_Invert(rF, sP, aGuess[q] >= 0 ? &aUV[aGuess[q]] : NULL, aUV[k]))
        { return; }
      if (q == 2
          && std::fabs(aUV[0].x - aUV[3].x) > dTolU && std::fabs(aUV[0].y - aUV[3].y) > dTolV)
        { return; }
    }
  SmBoolean bUConst = TRUE, bVConst = TRUE;
  for (int k = 1; k < 6; k++)
    {
      if (std::fabs(aUV[k].x - aUV[0].x) > dTolU) { bUConst = FALSE; }
      if (std::fabs(aUV[k].y - aUV[0].y) > dTolV) { bVConst = FALSE; }
    }
  const double dDU = aUV[5].x - aUV[0].x, dDV = aUV[5].y - aUV[0].y;
  if (bUConst && !bVConst && std::fabs(dDV) > dTolV)      { rCE.iIso = 1; }
  else if (bVConst && !bUConst && std::fabs(dDU) > dTolU) { rCE.iIso = 2; }
  else
    { return; }
  // As SmSurface::DropIsoCurve(), which makes the exactly straight UV trim curves IsoEdgeSubdivide()
  // detects (others are fitted): the curve follows the iso line with the parameterization predicted
  // by the speeds at its start, UV(t) = UV(t0) + (t - t0) |C'(t0)| / |S_dir(UV(t0))|, to within the
  // edge's tolerance (distance from the surface normal line); its start tangent is within a degree
  // of the surface direction.
  SmPoint2d sEndUV = aUV[5];
    {
      SmPoint3d  sS;
      SmVector3d sSu, sSv, aPV[2];
      if (   sm_Eval1(rF.pSurface, aUV[0], sS, sSu, sSv) != SM_SUCCESS
          || crE.cpCurve->Evaluate(crE.t0, 1, TRUE, aPV, TRUE) != SM_SUCCESS)
        { rCE.iIso = 0; return; }
      const SmVector3d & crDir = (rCE.iIso == 1) ? sSv : sSu;
      const double dSign = ((rCE.iIso == 1) ? dDV : dDU) > 0.0 ? 1.0 : -1.0;
      const double dCrvSpeed = aPV[1].Length(), dSrfSpeed = crDir.Length();
      if (!(dCrvSpeed > 0.0) || !(dSrfSpeed > 0.0))
        { rCE.iIso = 0; return; }
      SmVector3d sN = sSu * sSv;
      const double dN = sN.Length();
      if (dN > 0.0)
        {
          sN /= dN;
          const SmVector3d sTan = aPV[1] - sN * aPV[1].Dot(sN);
          const double dCos = dSign * sTan.Dot(crDir) / (sTan.Length() * dSrfSpeed + 1e-300);
          if (dCos < std::cos(SM_PI / 180.0))
            { rCE.iIso = 0; return; }
        }
      const double dScale = dSrfSpeed / dCrvSpeed;
      const double dIvl   = crE.t1 - crE.t0;
      auto Predict = [&](double dStep)
        {
          SmPoint2d sUV = aUV[0];
          ((rCE.iIso == 1) ? sUV.y : sUV.x) += dSign * dStep / dScale;
          sUV.x = std::min(rF.sNatural.GetMax().x, std::max(rF.sNatural.GetMin().x, sUV.x));
          sUV.y = std::min(rF.sNatural.GetMax().y, std::max(rF.sNatural.GetMin().y, sUV.y));
          return sUV;
        };
      for (int q = 1; q <= 4; q++)
        {
          const double dStep = dIvl * q / 4.0;
          SmPoint3d sC, sP;
          SmVector3d sNormal;
          const SmPoint2d sUV = Predict(dStep);
          if (   crE.cpCurve->EvaluatePoint(std::min(crE.t1, crE.t0 + dStep), sC) != SM_SUCCESS
              || rF.pSurface->EvaluatePoint(sUV, sP) != SM_SUCCESS
              || rF.pSurface->EvaluateNormal(sUV, TRUE, TRUE, sNormal) != SM_SUCCESS)
            { rCE.iIso = 0; return; }
          const SmVector3d sD(sC.x - sP.x, sC.y - sP.y, sC.z - sP.z);
          const SmVector3d sOff = sD - sNormal * sD.Dot(sNormal);
          if (sOff.Length() > crE.dTol)
            { rCE.iIso = 0; return; }
        }
      sEndUV = Predict(dIvl);
    }
  const double dLU = sEndUV.x - aUV[0].x, dLV = sEndUV.y - aUV[0].y;
  if (!(std::fabs((rCE.iIso == 1) ? dLV : dLU) > 0.0))
    { rCE.iIso = 0; return; }
  double u0 = std::min(aUV[0].x, sEndUV.x), u1 = std::max(aUV[0].x, sEndUV.x);
  double v0 = std::min(aUV[0].y, sEndUV.y), v1 = std::max(aUV[0].y, sEndUV.y);
  u0 -= SM_EFF_ZERO_SQRT; v0 -= SM_EFF_ZERO_SQRT; u1 += SM_EFF_ZERO_SQRT; v1 += SM_EFF_ZERO_SQRT;
  std::vector<double> & rT = rCE.sIsoT;
  rT.clear();
  sm_ForLeaves(rF.sTree, u0, v0, u1, v1, [&](const sm_Node & crN)
    {
      const double s = (rCE.iIso == 1) ? (crN.v0 - aUV[0].y) / dLV : (crN.u0 - aUV[0].x) / dLU;
      if (s > 0.01 && s < 0.99)
        { rT.push_back(crE.t0 + s * (crE.t1 - crE.t0)); }
    });
  std::sort(rT.begin(), rT.end());
  // as sm_InsertSortedBreaks(): values the same to SM_ARE_SAME are one
  size_t w = 0;
  for (size_t r = 0; r < rT.size(); r++)
    {
      if (w > 0 && SM_ARE_SAME(rT[r], rT[w - 1]))
        { continue; }
      rT[w++] = rT[r];
    }
  rT.resize(w);
}

/*******************************************************************//**
   E1: curve discretization
***********************************************************************/

// SmTess::ClampAndSplitEdge(): drop the ends, and parameters whose points are within the end vertices'
// intersection tolerance (twice their tolerance) of them; appends the rest to rOut
void sm_Clamp(const SmCurve * cpCurve, const std::vector<double> & crParams, const SmPoint3d & crStart, double dStartTol,
              const SmPoint3d & crEnd, double dEndTol, std::vector<double> & rOut)
{
  const size_t n = crParams.size();
  if (n < 3)
    { return; }
  size_t lFirst = 1, lLast = n - 2;   // inclusive range of interior parameters kept
  for (size_t ii = 1; ii + 1 < n; ii++)
    {
      SmPoint3d sP;
      if (cpCurve->EvaluatePoint(crParams[ii], sP) != SM_SUCCESS) { break; }
      if (sP.DistanceBetween(crStart) < 2.0 * dStartTol) { lFirst = ii + 1; }
      else break;
    }
  for (size_t ii = 1; ii + 1 < n; ii++)
    {
      SmPoint3d sP;
      if (cpCurve->EvaluatePoint(crParams[n - 1 - ii], sP) != SM_SUCCESS) { break; }
      if (sP.DistanceBetween(crEnd) < 2.0 * dEndTol) { lLast = n - 2 - ii; }
      else break;
    }
  for (size_t ii = lFirst; ii <= lLast && ii + 1 < n; ii++)
    {
      if (lLast + 1 == 0) break;
      rOut.push_back(crParams[ii]);
    }
}

void sm_DiscretizeEdge(sm_Context & rC, sm_Edge & rE)
{
  rE.sT.clear();
  rE.sP.clear();
  rE.sT.push_back(rE.t0);
  rE.sP.push_back(rC.sVertexPoints[rE.lV0]);
  if (!rE.bDegenerate && rE.cpCurve != NULL && rE.t1 > rE.t0)
    {
      // iso splits of the first iso coedge
      std::vector<double> sBreaks;
      for (int lCE : rE.sCoedges)
        {
          const sm_Coedge & crCE = rC.sCoedges[lCE];
          if (crCE.iIso != 0)
            {
              std::vector<double> sAll;
              sAll.reserve(crCE.sIsoT.size() + 2);
              sAll.push_back(rE.t0);
              sAll.insert(sAll.end(), crCE.sIsoT.begin(), crCE.sIsoT.end());
              sAll.push_back(rE.t1);
              sm_Clamp(rE.cpCurve, sAll, rC.sVertexPoints[rE.lV0], rC.sVertexTols[rE.lV0],
                       rC.sVertexPoints[rE.lV1], rC.sVertexTols[rE.lV1], sBreaks);
              break;
            }
        }
      // curve tessellation of each piece, clamped at the piece's vertices (split vertices have the
      // edge's tolerance)
      SmCurveTessDriver sCrvTess(0, rC.sParams.dChordHeightTol, rC.sParams.dCurveAngleTolDeg, rC.sParams.dMaxEdgeLength,
                                 0.0001, FALSE);
      SmTArray<double> sParams;
      std::vector<double> sPiece;
      double    dA = rE.t0;
      SmPoint3d sA = rC.sVertexPoints[rE.lV0];
      double    dATol = rC.sVertexTols[rE.lV0];
      for (size_t k = 0; k <= sBreaks.size(); k++)
        {
          const bool bLast = (k == sBreaks.size());
          const double dB = bLast ? rE.t1 : sBreaks[k];
          SmPoint3d sB;
          if (bLast) { sB = rC.sVertexPoints[rE.lV1]; }
          else if (rE.cpCurve->EvaluatePoint(dB, sB) != SM_SUCCESS) { rE.bFailed = true; break; }
          const double dBTol = bLast ? rC.sVertexTols[rE.lV1] : rE.dTol;
          if (sCrvTess.TessellateCurve(*rE.cpCurve, SmExtent1d(dA, dB), &sParams) != SM_SUCCESS)
            { rE.bFailed = true; break; }
          sPiece.resize(sParams.GetSize());
          for (ULONG q = 0; q < sParams.GetSize(); q++) { sPiece[q] = sParams[q]; }
          std::vector<double> sKept;
          sm_Clamp(rE.cpCurve, sPiece, sA, dATol, sB, dBTol, sKept);
          for (double t : sKept)
            {
              if (t > rE.sT.back() && t < dB)
                {
                  SmPoint3d sP;
                  if (rE.cpCurve->EvaluatePoint(t, sP) != SM_SUCCESS)
                    { rE.bFailed = true; break; }
                  rE.sT.push_back(t);
                  rE.sP.push_back(sP);
                }
            }
          if (rE.bFailed)
            { break; }
          if (!bLast && dB > rE.sT.back() && dB < rE.t1)
            { rE.sT.push_back(dB); rE.sP.push_back(sB); }
          dA = dB; sA = sB; dATol = dBTol;
        }
    }
  rE.sT.push_back(rE.t1);
  rE.sP.push_back(rC.sVertexPoints[rE.lV1]);
}

/*******************************************************************//**
   F2: UVs and refinement splits
***********************************************************************/

// the 3D tangent of the coedge at edge parameter t (in the coedge's direction)
SmVector3d sm_CoedgeTangent(const sm_Edge & crE, const sm_Coedge & crCE, double t)
{
  SmVector3d aPV[2];
  if (crE.cpCurve == NULL || crE.cpCurve->Evaluate(t, 1, TRUE, aPV, TRUE) != SM_SUCCESS)
    { return SmVector3d(0, 0, 0); }
  return crCE.bReversed ? -aPV[1] : aPV[1];
}

// the direction into the face at a boundary point: N x T for the upward faceuse (face to the left), T x N
// for a faceuse that contradicts its face (bFlipped)
SmVector3d sm_Inward(const sm_Face & crF, const SmPoint2d & crUV, const SmVector3d & crT)
{
  SmVector3d sN;
  if (crF.pSurface->EvaluateNormal(crUV, TRUE, TRUE, sN) != SM_SUCCESS)
    { return SmVector3d(0, 0, 0); }
  return crF.bFlipped ? crT * sN : sN * crT;
}

// UVs of the edge's current points on the face (entries with sHave set are kept): a reference point
// first (the middle point, or the curve's midpoint for an edge of two points), then outward with
// guesses; the seam side from the inward direction
SmBoolean sm_CoedgeUVs(sm_Context & rC, sm_Face & rF, sm_Coedge & rCE)
{
  const sm_Edge & crE = rC.sEdges[rCE.lEdge];
  const int n = (int)crE.sT.size();
  rCE.sUV.resize(n);
  rCE.sHave.resize(n, 0);
  // reference
  SmPoint2d sRef;
  int lRef = -1;
  for (int k = 0; k < n; k++)
    {
      if (rCE.sHave[k]) { lRef = k; break; }
    }
  if (lRef < 0)
    {
      double    dRefT;
      SmPoint3d sRefP;
      if (n > 2)
        {
          lRef  = n / 2;
          dRefT = crE.sT[lRef];
          sRefP = crE.sP[lRef];
        }
      else
        {
          dRefT = 0.5 * (crE.t0 + crE.t1);
          if (crE.bDegenerate || crE.cpCurve == NULL || crE.cpCurve->EvaluatePoint(dRefT, sRefP) != SM_SUCCESS)
            { sRefP = crE.sP[0]; dRefT = crE.sT[0]; lRef = 0; }
        }
      if (!sm_Invert(rF, sRefP, NULL, sRef))
        { return FALSE; }
      if (rF.aPeriod[0] > 0.0 || rF.aPeriod[1] > 0.0)
        {
          const SmVector3d sT = sm_CoedgeTangent(crE, rCE, dRefT);
          sm_PickSeamSide(rF, sm_Inward(rF, sRef, sT), crE.dTol, sRef);
        }
      if (lRef < 0)
        {
          // the reference isn't an edge point: its two points from it; an end at a pole takes the
          // reference's free parameter (as below)
          for (int k = 0; k < n; k++)
            {
              if (!sm_Invert(rF, crE.sP[k], &sRef, rCE.sUV[k])) { return FALSE; }
              rCE.sHave[k] = 1;
              const double dVTol = rC.sVertexTols[(k == 0) ? crE.lV0 : crE.lV1];
              if (crE.sP[k].DistanceBetween(sRefP) > 2.0 * std::max(dVTol, crE.dTol))
                {
                  SmBoolean bU, bV;
                  sm_Degeneracy(rF, rCE.sUV[k], dVTol, bU, bV);
                  if (bU && !bV) { rCE.sUV[k].x = sRef.x; }
                  if (bV && !bU) { rCE.sUV[k].y = sRef.y; }
                }
            }
          return TRUE;
        }
      rCE.sUV[lRef]   = sRef;
      rCE.sHave[lRef] = 1;
    }
  // outward from the reference
  for (int k = lRef + 1; k < n; k++)
    {
      if (rCE.sHave[k]) { continue; }
      if (!sm_Invert(rF, crE.sP[k], &rCE.sUV[k - 1], rCE.sUV[k])) { return FALSE; }
      rCE.sHave[k] = 1;
    }
  for (int k = lRef - 1; k >= 0; k--)
    {
      if (rCE.sHave[k]) { continue; }
      if (!sm_Invert(rF, crE.sP[k], &rCE.sUV[k + 1], rCE.sUV[k])) { return FALSE; }
      rCE.sHave[k] = 1;
    }
  // a closed edge on a periodic surface: with points half a period apart (or none between its ends),
  // continuity alone can't tell whether it winds around (its ends a period apart) or not. Follow it with
  // fine samples of its curve and put each point in the period nearest that path.
  if (n >= 2 && crE.lV0 == crE.lV1 && !crE.bDegenerate && crE.cpCurve != NULL && crE.t1 > crE.t0)
    {
      for (int d = 0; d < 2; d++)
        {
          const double dP = rF.aPeriod[d];
          if (dP <= 0.0)
            { continue; }
          double dMaxStep = 0.0;
          for (int k = 1; k < n; k++)
            {
              dMaxStep = std::max(dMaxStep, std::fabs((d == 0) ? rCE.sUV[k].x - rCE.sUV[k - 1].x
                                                               : rCE.sUV[k].y - rCE.sUV[k - 1].y));
            }
          if (dMaxStep <= 0.25 * dP && n > 2)
            { continue; }
          // the path: 16 steps along the edge from its first point, each inverted from the previous
          const int lSteps = 16;
          std::vector<double>    sPathT(lSteps + 1);
          std::vector<SmPoint2d> sPathUV(lSteps + 1);
          sPathT[0] = crE.sT[0];
          sPathUV[0] = rCE.sUV[0];
          SmBoolean bPath = TRUE;
          for (int q = 1; q <= lSteps && bPath; q++)
            {
              sPathT[q] = crE.sT[0] + (crE.sT[n - 1] - crE.sT[0]) * q / lSteps;
              SmPoint3d sQ = crE.sP[n - 1];
              if (q < lSteps && crE.cpCurve->EvaluatePoint(sPathT[q], sQ) != SM_SUCCESS)
                { bPath = FALSE; }
              else
                { bPath = sm_Invert(rF, sQ, &sPathUV[q - 1], sPathUV[q]); }
            }
          if (!bPath)
            { continue; }
          rCE.bFollowed = true;
          for (int k = 1; k < n; k++)
            {
              int q = (int)std::floor((crE.sT[k] - sPathT[0]) / (sPathT[lSteps] - sPathT[0]) * lSteps + 0.5);
              q = std::max(0, std::min(lSteps, q));
              double & rX = (d == 0) ? rCE.sUV[k].x : rCE.sUV[k].y;
              const double dG = (d == 0) ? sPathUV[q].x : sPathUV[q].y;
              rX += dP * std::floor((dG - rX) / dP + 0.5);
            }
        }
    }
  // an end at a pole: its free parameter is arbitrary after inversion; take the neighbor point's, as
  // the edge's UV curve reaches the pole
  if (n >= 2 && !crE.bDegenerate)
    {
      for (int end = 0; end < 2; end++)
        {
          const int k = (end == 0) ? 0 : n - 1, kn = (end == 0) ? 1 : n - 2;
          SmBoolean bU, bV;
          const int lV = (end == 0) ? crE.lV0 : crE.lV1;
          // (not when the segment itself has no length: then it runs along the collapsed line)
          if (crE.sP[k].DistanceBetween(crE.sP[kn]) <= 2.0 * std::max(rC.sVertexTols[lV], crE.dTol))
            { continue; }
          sm_Degeneracy(rF, rCE.sUV[k], rC.sVertexTols[lV], bU, bV);
          if (bU && !bV) { rCE.sUV[k].x = rCE.sUV[kn].x; }
          if (bV && !bU) { rCE.sUV[k].y = rCE.sUV[kn].y; }
        }
    }
  return TRUE;
}

// SmTess::CheckAgainstSurfaceCache() for an exterior segment, first split only: TRUE when the UV
// segment is larger than 1.1 times a leaf it crosses (and not tiny in 3D, as near a pole)
SmBoolean sm_NeedsSplit(const sm_Context & crC, const sm_Face & crF, const SmPoint2d & crA, const SmPoint2d & crB)
{
  const double dRatio = 1.1;
  const SmPoint2d s1(crA.x + 0.01 * (crB.x - crA.x), crA.y + 0.01 * (crB.y - crA.y));
  const SmPoint2d s2(crA.x + 0.99 * (crB.x - crA.x), crA.y + 0.99 * (crB.y - crA.y));
  const double u0 = std::min(s1.x, s2.x), u1 = std::max(s1.x, s2.x);
  const double v0 = std::min(s1.y, s2.y), v1 = std::max(s1.y, s2.y);
  const double dSU = u1 - u0, dSV = v1 - v0;
  SmBoolean bSplit = FALSE;
  sm_ForLeaves(crF.sTree, u0, v0, u1, v1, [&](const sm_Node & crN)
    {
      if (bSplit) return;
      if (dSU > (crN.u1 - crN.u0) * dRatio || dSV > (crN.v1 - crN.v0) * dRatio)
        { bSplit = TRUE; }
    });
  if (!bSplit)
    { return FALSE; }
  // tiny in 3D (near a pole): don't split
  SmPoint3d sP1, sP2, sPm;
  SmPoint2d sW1 = s1, sW2 = s2;
  sm_IntoDomain(crF, sW1);
  sm_IntoDomain(crF, sW2);
  if (   crF.pSurface->EvaluatePoint(sW1, sP1) != SM_SUCCESS
      || crF.pSurface->EvaluatePoint(sW2, sP2) != SM_SUCCESS)
    { return TRUE; }
  if (sP1.DistanceBetween(sP2) < crC.dBrepTol * 20.0)
    {
      SmPoint2d sMid(0.5 * (s1.x + s2.x), 0.5 * (s1.y + s2.y));
      sm_IntoDomain(crF, sMid);
      if (crF.pSurface->EvaluatePoint(sMid, sPm) == SM_SUCCESS
          && sP1.DistanceBetween(sPm) + sPm.DistanceBetween(sP2) < crC.dBrepTol * 20.0)
        { return FALSE; }
    }
  return TRUE;
}

// bisects one segment of the coedge's edge while the face needs it (SmTess::SplitFaceEdgesToUVSize());
// SM_FB_SAMPLING or SM_FB_INVERSION when a split point can't be evaluated or inverted
SmFastBoundaryFailure sm_SplitSegment(sm_Context & rC, sm_Face & rF, sm_Coedge & rCE, double ta, const SmPoint3d & crPa,
                                      const SmPoint2d & crUVa, double tb, const SmPoint3d & crPb, const SmPoint2d & crUVb,
                                      int lDepth)
{
  if (lDepth > 24 || !sm_NeedsSplit(rC, rF, crUVa, crUVb))
    { return SM_FB_NONE; }
  const sm_Edge & crE = rC.sEdges[rCE.lEdge];
  const double tm = 0.5 * (ta + tb);
  if (!(tm > ta && tm < tb))
    { return SM_FB_NONE; }
  SmPoint3d sPm;
  if (crE.cpCurve->EvaluatePoint(tm, sPm) != SM_SUCCESS)
    { return SM_FB_SAMPLING; }
  // as SmBrep::MakeVertexSplitEdge() refusing splits within tolerance of the ends
  if (sPm.DistanceBetween(crPa) < crE.dTol || sPm.DistanceBetween(crPb) < crE.dTol)
    { return SM_FB_NONE; }
  const SmPoint2d sGuess(0.5 * (crUVa.x + crUVb.x), 0.5 * (crUVa.y + crUVb.y));
  SmPoint2d sUVm;
  if (!sm_Invert(rF, sPm, &sGuess, sUVm))
    { return SM_FB_INVERSION; }
  const SmFastBoundaryFailure eFirst = sm_SplitSegment(rC, rF, rCE, ta, crPa, crUVa, tm, sPm, sUVm, lDepth + 1);
  if (eFirst != SM_FB_NONE)
    { return eFirst; }
  rCE.sSplitT.push_back(tm);
  rCE.sSplitP.push_back(sPm);
  rCE.sSplitUV.push_back(sUVm);
  return sm_SplitSegment(rC, rF, rCE, tm, sPm, sUVm, tb, crPb, crUVb, lDepth + 1);
}

SmBoolean sm_FaceSplits(sm_Context & rC, sm_Face & rF)
{
  for (const std::vector<int> & crLoop : rF.sLoops)
    {
      for (int lCE : crLoop)
        {
          sm_Coedge & rCE = rC.sCoedges[lCE];
          const sm_Edge & crE = rC.sEdges[rCE.lEdge];
          rCE.sHave.assign(crE.sT.size(), 0);
          if (!sm_CoedgeUVs(rC, rF, rCE))
            { rF.eFailure = SM_FB_INVERSION; return FALSE; }
          if (crE.bDegenerate || crE.cpCurve == NULL)
            { continue; }
          for (size_t k = 0; k + 1 < crE.sT.size(); k++)
            {
              const SmFastBoundaryFailure eSplit = sm_SplitSegment(rC, rF, rCE, crE.sT[k], crE.sP[k], rCE.sUV[k],
                                                                   crE.sT[k + 1], crE.sP[k + 1], rCE.sUV[k + 1], 0);
              if (eSplit != SM_FB_NONE)
                { rF.eFailure = eSplit; return FALSE; }
            }
        }
    }
  return TRUE;
}

/*******************************************************************//**
   E2: merging the splits of both faces
***********************************************************************/
void sm_MergeSplits(sm_Context & rC, sm_Edge & rE)
{
  // new points: the coedges' splits (in order within each coedge), deduplicated by parameter
  std::vector<std::pair<double, SmPoint3d>> sNew;
  for (int lCE : rE.sCoedges)
    {
      const sm_Coedge & crCE = rC.sCoedges[lCE];
      for (size_t k = 0; k < crCE.sSplitT.size(); k++)
        { sNew.emplace_back(crCE.sSplitT[k], crCE.sSplitP[k]); }
    }
  if (sNew.empty())
    { return; }
  std::sort(sNew.begin(), sNew.end(), [](const std::pair<double, SmPoint3d> & a, const std::pair<double, SmPoint3d> & b)
            { return a.first < b.first; });
  std::vector<double>    sT;
  std::vector<SmPoint3d> sP;
  sT.reserve(rE.sT.size() + sNew.size());
  sP.reserve(rE.sT.size() + sNew.size());
  size_t i = 0, j = 0;
  while (i < rE.sT.size() || j < sNew.size())
    {
      if (j >= sNew.size() || (i < rE.sT.size() && rE.sT[i] <= sNew[j].first))
        {
          if (sT.empty() || rE.sT[i] != sT.back()) { sT.push_back(rE.sT[i]); sP.push_back(rE.sP[i]); }
          i++;
        }
      else
        {
          if (sT.empty() || sNew[j].first != sT.back()) { sT.push_back(sNew[j].first); sP.push_back(sNew[j].second); }
          j++;
        }
    }
  // the coedges' known UVs at the merged points
  for (int lCE : rE.sCoedges)
    {
      sm_Coedge & rCE = rC.sCoedges[lCE];
      std::vector<SmPoint2d> sUV(sT.size());
      std::vector<char>      sHave(sT.size(), 0);
      size_t a = 0;
      for (size_t k = 0; k < sT.size(); k++)
        {
          while (a < rE.sT.size() && rE.sT[a] < sT[k]) a++;
          if (a < rE.sT.size() && rE.sT[a] == sT[k] && a < rCE.sUV.size() && rCE.sHave[a])
            { sUV[k] = rCE.sUV[a]; sHave[k] = 1; }
        }
      for (size_t s = 0; s < rCE.sSplitT.size(); s++)
        {
          const size_t k = (size_t)(std::lower_bound(sT.begin(), sT.end(), rCE.sSplitT[s]) - sT.begin());
          if (k < sT.size() && sT[k] == rCE.sSplitT[s])
            { sUV[k] = rCE.sSplitUV[s]; sHave[k] = 1; }
        }
      rCE.sUV.swap(sUV);
      rCE.sHave.swap(sHave);
    }
  rE.sT.swap(sT);
  rE.sP.swap(sP);
}

// Edges that coincide (distinct edges whose ends meet and whose curves lie within the edges' tolerance
// of each other), as where two shells touch: they get one discretization, the earlier edge's points with
// the later one's projected onto its curve, and the later edge takes those points, so that the meshes meet
// at the same points along the whole edge (the curves' own points agree only now and then to the last bit,
// and the edges' refinement splits come from different faces). Points move by less than the tolerance.
void sm_SnapCoincidentEdges(sm_Context & rC)
{
  SmExtent3d sBox;
  for (const SmPoint3d & crP : rC.sVertexPoints)
    { sBox.AddPoint3d(crP); }
  const double dSize = rC.sVertexPoints.empty() ? 0.0 : sBox.GetSize().Length();
  const double dCell = 1.0e-7 * dSize;
  if (!(dCell > 0.0) || !std::isfinite(dCell))
    { return; }
  typedef std::tuple<long long, long long, long long> sm_Cell;
  auto Cell = [&](const SmPoint3d & crP)
    {
      return sm_Cell((long long)std::floor(crP.x / dCell), (long long)std::floor(crP.y / dCell),
                     (long long)std::floor(crP.z / dCell));
    };
  std::map<std::pair<sm_Cell, sm_Cell>, std::vector<int>> sByEnds;
  for (int e = 0; e < (int)rC.sEdges.size(); e++)
    {
      const sm_Edge & crE = rC.sEdges[e];
      if (crE.bDegenerate || crE.cpCurve == NULL || crE.sP.size() < 2 || !(crE.sT.back() > crE.sT.front()))
        { continue; }
      sm_Cell a = Cell(crE.sP.front()), b = Cell(crE.sP.back());
      if (b < a) { std::swap(a, b); }
      sByEnds[std::make_pair(a, b)].push_back(e);
    }
  // crQ's parameter on crE's curve, from the nearest polyline segment, refined by Gauss-Newton; FALSE when
  // the curve doesn't come within dTol of it
  auto Project = [](const sm_Edge & crE, const SmPoint3d & crQ, double dTol, double & rdT)
    {
      size_t lBest = 0;
      double dBest = 1.0e300, dFraction = 0.0;
      for (size_t k = 0; k + 1 < crE.sP.size(); k++)
        {
          const SmVector3d sD = crE.sP[k + 1] - crE.sP[k];
          const double dLength2 = sD.LengthSquared();
          const double f = (dLength2 > 0.0) ? std::min(1.0, std::max(0.0, (crQ - crE.sP[k]).Dot(sD) / dLength2)) : 0.0;
          const double d = (crE.sP[k] + sD * f).DistanceBetween(crQ);
          if (d < dBest) { dBest = d; lBest = k; dFraction = f; }
        }
      double t = crE.sT[lBest] + dFraction * (crE.sT[lBest + 1] - crE.sT[lBest]);
      SmVector3d aPD[2];
      for (int iIter = 0; iIter < 12; iIter++)
        {
          if (crE.cpCurve->Evaluate(t, 1, TRUE, aPD, FALSE) != SM_SUCCESS)
            { return false; }
          const double dD2 = aPD[1].LengthSquared();
          if (!(dD2 > 0.0))
            { break; }
          const double dStep = (SmVector3d(aPD[0]) - SmVector3d(crQ.x, crQ.y, crQ.z)).Dot(aPD[1]) / dD2;
          t = std::min(crE.sT.back(), std::max(crE.sT.front(), t - dStep));
          if (std::fabs(dStep) <= 1.0e-15 * (std::fabs(t) + 1.0))
            { break; }
        }
      SmPoint3d sAt;
      if (crE.cpCurve->EvaluatePoint(t, sAt) != SM_SUCCESS || !(sAt.DistanceBetween(crQ) <= dTol))
        { return false; }
      rdT = t;
      return true;
    };
  // the merged discretization of A and B (B reversed or not); FALSE when they don't coincide
  auto Merge = [&](const sm_Edge & crA, const sm_Edge & crB, bool bReversed, double dTol,
                   std::vector<double> & rT, std::vector<SmPoint3d> & rP, std::vector<long> & rFromA, std::vector<double> & rTB)
    {
      const size_t nA = crA.sP.size(), nB = crB.sP.size();
      if (   crA.sP.front().DistanceBetween(crB.sP[bReversed ? nB - 1 : 0]) > dTol
          || crA.sP.back().DistanceBetween(crB.sP[bReversed ? 0 : nB - 1]) > dTol)
        { return false; }
      // B's interior points on A; those within a quarter of A's segment of a point of A (or of another of
      // B's in it) merge into that point, so that no short segment makes slivers
      std::vector<std::pair<double, SmPoint3d>> sOnA;
      for (size_t k = 1; k + 1 < nB; k++)
        {
          double t;
          if (!Project(crA, crB.sP[k], dTol, t))
            { return false; }
          SmPoint3d sPoint;
          if (crA.cpCurve->EvaluatePoint(t, sPoint) != SM_SUCCESS)
            { return false; }
          sOnA.emplace_back(t, sPoint);
        }
      std::sort(sOnA.begin(), sOnA.end(), [](const std::pair<double, SmPoint3d> & a, const std::pair<double, SmPoint3d> & b)
                { return a.first < b.first; });
      std::vector<double> sFromB;
      SmPoint3d sLastB;
      size_t    lLastSegment = (size_t)-1;
      for (const auto & crOnA : sOnA)
        {
          const double t = crOnA.first;
          const size_t lNext = (size_t)(std::upper_bound(crA.sT.begin(), crA.sT.end(), t) - crA.sT.begin());
          if (lNext == 0 || lNext >= nA)
            { continue; }   // at an end: that point
          const double dQuarter = 0.25 * crA.sP[lNext - 1].DistanceBetween(crA.sP[lNext]);
          if (   crOnA.second.DistanceBetween(crA.sP[lNext - 1]) < std::max(dQuarter, dTol)
              || crOnA.second.DistanceBetween(crA.sP[lNext]) < std::max(dQuarter, dTol)
              || (lLastSegment == lNext && crOnA.second.DistanceBetween(sLastB) < std::max(dQuarter, dTol)))
            { continue; }
          sFromB.push_back(t);
          sLastB = crOnA.second;
          lLastSegment = lNext;
        }
      rT.clear(); rP.clear(); rFromA.clear();
      size_t j = 0;
      for (size_t k = 0; k < nA; k++)
        {
          while (k > 0 && j < sFromB.size() && sFromB[j] < crA.sT[k])
            {
              SmPoint3d sPoint;
              if (!(sFromB[j] > rT.back()) || crA.cpCurve->EvaluatePoint(sFromB[j], sPoint) != SM_SUCCESS)
                { return false; }
              rT.push_back(sFromB[j]); rP.push_back(sPoint); rFromA.push_back(-1);
              j++;
            }
          rT.push_back(crA.sT[k]); rP.push_back(crA.sP[k]); rFromA.push_back((long)k);
        }
      if (j != sFromB.size())
        { return false; }
      // B's parameters for those points, in B's direction
      const size_t n = rP.size();
      rTB.assign(n, 0.0);
      rTB[0] = crB.sT.front();
      rTB[n - 1] = crB.sT.back();
      for (size_t k = 1; k + 1 < n; k++)
        {
          if (!Project(crB, rP[bReversed ? n - 1 - k : k], dTol, rTB[k]) || !(rTB[k] > rTB[k - 1]))
            { return false; }
        }
      return n < 2 || rTB[n - 1] > rTB[n - 2];
    };
  auto Snap = [&](int lA, int lB)
    {
      sm_Edge & rA = rC.sEdges[lA];
      sm_Edge & rB = rC.sEdges[lB];
      const double dTol = std::max(std::max(rA.dTol, rB.dTol), dCell);
      std::vector<double>    sT, sTB;
      std::vector<SmPoint3d> sP;
      std::vector<long>      sFromA;
      bool bReversed = false;
      if (!Merge(rA, rB, false, dTol, sT, sP, sFromA, sTB))
        {
          if (!Merge(rA, rB, true, dTol, sT, sP, sFromA, sTB))
            { return false; }
          bReversed = true;
        }
      // the edges' new points; UVs stay known where A keeps its points, and at B's ends
      const size_t n = sP.size(), nB = rB.sP.size();
      auto Remap = [&](sm_Edge & rE, const std::vector<long> & crFrom)
        {
          for (int lCE : rE.sCoedges)
            {
              sm_Coedge & rCE = rC.sCoedges[lCE];
              std::vector<SmPoint2d> sUV(crFrom.size());
              std::vector<char>      sHave(crFrom.size(), 0);
              for (size_t k = 0; k < crFrom.size(); k++)
                {
                  const long a = crFrom[k];
                  if (a >= 0 && (size_t)a < rCE.sUV.size() && (size_t)a < rCE.sHave.size() && rCE.sHave[a])
                    { sUV[k] = rCE.sUV[a]; sHave[k] = 1; }
                }
              rCE.sUV.swap(sUV);
              rCE.sHave.swap(sHave);
            }
        };
      std::vector<long> sFromB(n, -1);
      sFromB[0] = 0;
      sFromB[n - 1] = (long)nB - 1;
      Remap(rA, sFromA);
      Remap(rB, sFromB);
      const SmPoint3d sB0 = rB.sP.front(), sB1 = rB.sP.back();   // B's ends are its own vertices
      rA.sT.swap(sT);
      rA.sP = sP;
      rB.sT.swap(sTB);
      rB.sP.resize(n);
      for (size_t k = 0; k < n; k++)
        { rB.sP[k] = sP[bReversed ? n - 1 - k : k]; }
      rB.sP.front() = sB0;
      rB.sP.back()  = sB1;
      return true;
    };
  std::vector<char> sTaken(rC.sEdges.size(), 0);
  for (const auto & crGroup : sByEnds)
    {
      const std::vector<int> & crEdges = crGroup.second;
      for (size_t i = 0; i < crEdges.size(); i++)
        {
          if (sTaken[crEdges[i]])
            { continue; }
          for (size_t j = i + 1; j < crEdges.size(); j++)
            {
              if (!sTaken[crEdges[j]] && Snap(crEdges[i], crEdges[j]))
                { sTaken[crEdges[j]] = 1; }
            }
        }
    }
}

/*******************************************************************//**
   F3: loops
***********************************************************************/

// one loop point before ids: UV and the point's reference (edge point k of edge e, or Brep vertex v)
struct sm_LoopPoint
{
  SmPoint2d sUV;
  int       lEdge;    // -1: Brep vertex lVertex
  int       lIndex;   // edge point index, or vertex index
};

// TRUE when two UVs of one point are a whole period apart along a periodic direction, where the surface
// isn't degenerate (that is a seam crossed without a seam edge, not a pole)
SmBoolean sm_PeriodApart(const sm_Face & crF, const SmPoint2d & crA, const SmPoint2d & crB, double dVertexTol)
{
  for (int d = 0; d < 2; d++)
    {
      const double dP = crF.aPeriod[d];
      if (dP <= 0.0)
        { continue; }
      const double dD = std::fabs((d == 0) ? crA.x - crB.x : crA.y - crB.y);
      const double dK = std::floor(dD / dP + 0.5);
      if (dK < 1.0 || std::fabs(dD - dK * dP) > 1.0e-6 * dP)
        { continue; }
      SmBoolean bU, bV;
      sm_Degeneracy(crF, crA, dVertexTol, bU, bV);
      if (!((d == 0) ? bU : bV))
        { return TRUE; }
    }
  return FALSE;
}

// the face's loops in UV (points still without ids), or why they can't be made
SmFastBoundaryFailure sm_AssembleLoops(sm_Context & rC, sm_Face & rF, std::vector<std::vector<sm_LoopPoint>> & rLoops)
{
  const double dUVTolU = 1.0e-7 * (rF.sNatural.XLength() + 1.0e-300);
  const double dUVTolV = 1.0e-7 * (rF.sNatural.YLength() + 1.0e-300);
  std::vector<std::vector<sm_LoopPoint>> & sLoops = rLoops;
  sLoops.clear();
  for (const std::vector<int> & crLoop : rF.sLoops)
    {
      const int m = (int)crLoop.size();
      // per coedge: its UV polyline in loop direction, fixed for poles at its ends
      std::vector<std::vector<sm_LoopPoint>> sPolys(m);
      for (int c = 0; c < m; c++)
        {
          sm_Coedge & rCE = rC.sCoedges[crLoop[c]];
          const sm_Edge & crE = rC.sEdges[rCE.lEdge];
          if (!sm_CoedgeUVs(rC, rF, rCE))
            { return SM_FB_INVERSION; }
          const int n = (int)crE.sT.size();
          std::vector<sm_LoopPoint> & rPoly = sPolys[c];
          rPoly.resize(n);
          for (int k = 0; k < n; k++)
            {
              const int ke = rCE.bReversed ? n - 1 - k : k;
              sm_LoopPoint & rLP = rPoly[k];
              rLP.sUV = rCE.sUV[ke];
              if (ke == 0)          { rLP.lEdge = -1; rLP.lIndex = crE.lV0; }
              else if (ke == n - 1) { rLP.lEdge = -1; rLP.lIndex = crE.lV1; }
              else                  { rLP.lEdge = rCE.lEdge; rLP.lIndex = ke; }
            }
          // unwrap periodic jumps inside the polyline (not across the long steps of a followed edge)
          for (int k = 1; k < n && !rCE.bFollowed; k++)
            {
              for (int d = 0; d < 2; d++)
                {
                  const double dP = rF.aPeriod[d];
                  if (dP <= 0.0) continue;
                  double & rX = (d == 0) ? rPoly[k].sUV.x : rPoly[k].sUV.y;
                  const double dPrev = (d == 0) ? rPoly[k - 1].sUV.x : rPoly[k - 1].sUV.y;
                  rX += dP * std::floor((dPrev - rX) / dP + 0.5);
                }
            }
          // ends at a pole: the free parameter of the neighbor point along the polyline
          if (n >= 3 || !crE.bDegenerate)
            {
              for (int end = 0; end < 2; end++)
                {
                  const int k = end == 0 ? 0 : n - 1, kn = end == 0 ? 1 : n - 2;
                  if (kn < 0 || kn >= n || kn == k) continue;
                  SmBoolean bU, bV;
                  const double dVTol = (rPoly[k].lEdge < 0) ? rC.sVertexTols[rPoly[k].lIndex] : 0.0;
                  const SmPoint3d & crPk  = (rPoly[k].lEdge < 0) ? rC.sVertexPoints[rPoly[k].lIndex] : rC.sEdges[rPoly[k].lEdge].sP[rPoly[k].lIndex];
                  const SmPoint3d & crPkn = (rPoly[kn].lEdge < 0) ? rC.sVertexPoints[rPoly[kn].lIndex] : rC.sEdges[rPoly[kn].lEdge].sP[rPoly[kn].lIndex];
                  if (crPk.DistanceBetween(crPkn) <= 2.0 * std::max(dVTol, crE.dTol))
                    { continue; }
                  sm_Degeneracy(rF, rPoly[k].sUV, dVTol, bU, bV);
                  if (bU && !bV) { rPoly[k].sUV.x = rPoly[kn].sUV.x; }
                  if (bV && !bU) { rPoly[k].sUV.y = rPoly[kn].sUV.y; }
                }
            }
        }
      // start with a coedge that is neither a seam nor degenerate (its UVs aren't ambiguous)
      int lStart = 0;
      for (int c = 0; c < m; c++)
        {
          const sm_Coedge & crCE = rC.sCoedges[crLoop[c]];
          if (!crCE.bSeam && !rC.sEdges[crCE.lEdge].bDegenerate) { lStart = c; break; }
        }
      std::vector<sm_LoopPoint> sOut;
      SmPoint2d sPrevEnd;
      bool bAnchored = false;   // a seam (whose side is known) fixed the period of the points so far
      for (int i = 0; i < m; i++)
        {
          const int c = (lStart + i) % m;
          const sm_Coedge & crCE = rC.sCoedges[crLoop[c]];
          std::vector<sm_LoopPoint> & rPoly = sPolys[c];
          const int n = (int)rPoly.size();
          const bool bDegenerate = rC.sEdges[crCE.lEdge].bDegenerate;
          if (i > 0 && crCE.bSeam && !bAnchored)
            {
              // the first seam: the points so far (none of them on a seam) move by whole periods to meet it
              for (int d = 0; d < 2; d++)
                {
                  const double dP = rF.aPeriod[d];
                  if (dP <= 0.0) continue;
                  const double dX = (d == 0) ? rPoly[0].sUV.x : rPoly[0].sUV.y;
                  const double dPrev = (d == 0) ? sPrevEnd.x : sPrevEnd.y;
                  const double dShift = dP * std::floor((dX - dPrev) / dP + 0.5);
                  if (dShift == 0.0) continue;
                  for (sm_LoopPoint & rLP : sOut) { ((d == 0) ? rLP.sUV.x : rLP.sUV.y) += dShift; }
                  ((d == 0) ? sPrevEnd.x : sPrevEnd.y) += dShift;
                }
            }
          if (crCE.bSeam)
            { bAnchored = true; }
          if (i > 0)
            {
              if (bDegenerate)
                {
                  // a degenerate edge runs along the pole from the previous end to the next start
                  rPoly[0].sUV = sPrevEnd;
                  const int cn = (c + 1) % m;
                  const std::vector<sm_LoopPoint> & crNext = sPolys[cn];
                  SmPoint2d sNext = crNext.empty() ? sPrevEnd : crNext[0].sUV;
                  for (int d = 0; d < 2; d++)
                    {
                      const double dP = rF.aPeriod[d];
                      if (dP <= 0.0) continue;
                      double & rX = (d == 0) ? sNext.x : sNext.y;
                      const double dPrev = (d == 0) ? sPrevEnd.x : sPrevEnd.y;
                      rX += dP * std::floor((dPrev - rX) / dP + 0.5);
                    }
                  for (int k = 1; k < n; k++)
                    {
                      const double s = (double)k / (double)(n - 1);
                      rPoly[k].sUV = SmPoint2d(sPrevEnd.x + s * (sNext.x - sPrevEnd.x), sPrevEnd.y + s * (sNext.y - sPrevEnd.y));
                    }
                }
              else if (!crCE.bSeam)
                {
                  // whole periods to meet the previous end
                  for (int d = 0; d < 2; d++)
                    {
                      const double dP = rF.aPeriod[d];
                      if (dP <= 0.0) continue;
                      const double dX = (d == 0) ? rPoly[0].sUV.x : rPoly[0].sUV.y;
                      const double dPrev = (d == 0) ? sPrevEnd.x : sPrevEnd.y;
                      const double dShift = dP * std::floor((dPrev - dX) / dP + 0.5);
                      if (dShift != 0.0)
                        {
                          for (sm_LoopPoint & rLP : rPoly) { ((d == 0) ? rLP.sUV.x : rLP.sUV.y) += dShift; }
                        }
                    }
                }
              // the previous end and this start: one vertex; distinct UVs only at a pole
              const SmPoint2d & crStart = rPoly[0].sUV;
              if (std::fabs(crStart.x - sPrevEnd.x) > dUVTolU || std::fabs(crStart.y - sPrevEnd.y) > dUVTolV)
                {
                  SmPoint3d sA, sB;
                  SmPoint2d sWA = sPrevEnd, sWB = crStart;
                  sm_IntoDomain(rF, sWA);
                  sm_IntoDomain(rF, sWB);
                  const SmBoolean bPole =    rF.pSurface->EvaluatePoint(sWA, sA) == SM_SUCCESS
                                          && rF.pSurface->EvaluatePoint(sWB, sB) == SM_SUCCESS
                                          && sA.DistanceBetween(sB) <= std::max(rF.dTol, 1.0e-9);
                  if (!bPole)
                    { return SM_FB_LOOP_OPEN; }
                  if (rPoly[0].lEdge < 0 && sm_PeriodApart(rF, sPrevEnd, crStart, rC.sVertexTols[rPoly[0].lIndex]))
                    { return SM_FB_LOOP_WINDING; }
                  // the previous coedge's end stays as a point of the pole run (rPoly[0] is the shared
                  // Brep vertex)
                  sm_LoopPoint sEnd = rPoly[0];
                  sEnd.sUV = sPrevEnd;
                  sOut.push_back(sEnd);
                }
            }
          for (int k = 0; k + 1 < n; k++)
            { sOut.push_back(rPoly[k]); }
          sPrevEnd = rPoly[n - 1].sUV;

        }
      // closing: the last end and the first start
      if (!sOut.empty())
        {
          const SmPoint2d & crFirst = sOut[0].sUV;
          const double dGU = sPrevEnd.x - crFirst.x, dGV = sPrevEnd.y - crFirst.y;
          if (std::fabs(dGU) > dUVTolU || std::fabs(dGV) > dUVTolV)
            {
              SmPoint3d sA, sB;
              SmPoint2d sWA = sPrevEnd, sWB = crFirst;
              sm_IntoDomain(rF, sWA);
              sm_IntoDomain(rF, sWB);
              const SmBoolean bPole =    rF.pSurface->EvaluatePoint(sWA, sA) == SM_SUCCESS
                                      && rF.pSurface->EvaluatePoint(sWB, sB) == SM_SUCCESS
                                      && sA.DistanceBetween(sB) <= std::max(rF.dTol, 1.0e-9);
              if (!bPole)
                {
                  // a whole period apart: the loop winds around the surface
                  for (int d = 0; d < 2; d++)
                    {
                      const double dP = rF.aPeriod[d];
                      const double dG = (d == 0) ? dGU : dGV;
                      if (dP > 0.0 && std::fabs(std::fabs(dG) - dP) < 1.0e-6 * dP)
                        { return SM_FB_LOOP_WINDING; }
                    }
                  return SM_FB_LOOP_OPEN;
                }
              if (sOut[0].lEdge < 0 && sm_PeriodApart(rF, sPrevEnd, crFirst, rC.sVertexTols[sOut[0].lIndex]))
                { return SM_FB_LOOP_WINDING; }
              sm_LoopPoint sEnd = sOut[0];
              sEnd.sUV = sPrevEnd;
              sOut.push_back(sEnd);
            }
        }
      // whole periods to put the loop into the natural domain
      if (!sOut.empty())
        {
          for (int d = 0; d < 2; d++)
            {
              const double dP = rF.aPeriod[d];
              if (dP <= 0.0) continue;
              double dLo = std::numeric_limits<double>::infinity(), dHi = -dLo;
              for (const sm_LoopPoint & crLP : sOut)
                {
                  const double x = (d == 0) ? crLP.sUV.x : crLP.sUV.y;
                  dLo = std::min(dLo, x); dHi = std::max(dHi, x);
                }
              const double dMin = (d == 0) ? rF.sNatural.GetMin().x : rF.sNatural.GetMin().y;
              const double dMax = (d == 0) ? rF.sNatural.GetMax().x : rF.sNatural.GetMax().y;
              const double dTol = 1.0e-7 * dP;
              // the shift that puts the loop inside the natural domain, else the one closest to the
              // face's domain
              double dShift = dP * std::ceil((dMin - dTol - dLo) / dP);
              if (dHi + dShift > dMax + dTol)
                {
                  const SmExtent2d sFaceDomain = rF.pFace->GetUVDomain();
                  const double dCenter = 0.5 * ((d == 0) ? sFaceDomain.GetMin().x + sFaceDomain.GetMax().x
                                                         : sFaceDomain.GetMin().y + sFaceDomain.GetMax().y);
                  dShift = dP * std::floor((dCenter - 0.5 * (dLo + dHi)) / dP + 0.5);
                }
              if (dShift != 0.0)
                {
                  for (sm_LoopPoint & rLP : sOut) { ((d == 0) ? rLP.sUV.x : rLP.sUV.y) += dShift; }
                }
              // a loop that doesn't fit into one period crosses the seam without a seam edge
              if (dLo + dShift < dMin - dTol || dHi + dShift > dMax + dTol)
                { return SM_FB_LOOP_WINDING; }
            }
        }
      if (sOut.size() >= 3)
        { sLoops.push_back(std::move(sOut)); }
    }
  return sLoops.empty() ? SM_FB_DEGENERATE : SM_FB_NONE;
}

// 1: the largest loop (rlOuter) counterclockwise, the others clockwise; -1: all of them the other way
// round; 0: neither, or no loop encloses any area (rbDegenerate)
int sm_LoopOrientation(const std::vector<std::vector<sm_LoopPoint>> & crLoops, int & rlOuter, SmBoolean & rbDegenerate)
{
  rlOuter = -1;
  std::vector<double> sAreas(crLoops.size(), 0.0);
  for (size_t l = 0; l < crLoops.size(); l++)
    {
      const std::vector<sm_LoopPoint> & crL = crLoops[l];
      for (size_t k = 0; k < crL.size(); k++)
        {
          const SmPoint2d & a = crL[k].sUV, & b = crL[(k + 1) % crL.size()].sUV;
          sAreas[l] += (a.x - b.x) * (a.y + b.y);
        }
      if (rlOuter < 0 || std::fabs(sAreas[l]) > std::fabs(sAreas[rlOuter])) { rlOuter = (int)l; }
    }
  // without area: relative to the loops' own UV extent (a small face of a large surface is no less a face)
  SmExtent2d sBox;
  for (const std::vector<sm_LoopPoint> & crL : crLoops)
    {
      for (const sm_LoopPoint & crLP : crL)
        { sBox.AddPoint2d(crLP.sUV); }
    }
  const double dExtent = crLoops.empty() ? 0.0 : sBox.XLength() * sBox.XLength() + sBox.YLength() * sBox.YLength();
  rbDegenerate = (rlOuter < 0 || !(std::fabs(sAreas[rlOuter]) > 1.0e-14 * dExtent)) ? TRUE : FALSE;
  if (rbDegenerate)
    { return 0; }
  const double dSign = (sAreas[rlOuter] > 0.0) ? 1.0 : -1.0;
  for (size_t l = 0; l < crLoops.size(); l++)
    {
      if ((int)l != rlOuter && !(dSign * sAreas[l] < 0.0)) { return 0; }
    }
  return (dSign > 0.0) ? 1 : -1;
}

SmBoolean sm_FaceLoops(sm_Context & rC, sm_Face & rF, SmFastFaceBoundary & rOut,
                       const std::vector<long> & crVertexIds)
{
  std::vector<std::vector<sm_LoopPoint>> sLoops;
  SmFastBoundaryFailure eFailure = sm_AssembleLoops(rC, rF, sLoops);
  int       lOuter = -1;
  SmBoolean bDegenerate = FALSE;
  int iOrientation = (eFailure == SM_FB_NONE) ? sm_LoopOrientation(sLoops, lOuter, bDegenerate) : 0;
  if (eFailure == SM_FB_NONE && iOrientation == 0)
    { eFailure = bDegenerate ? SM_FB_DEGENERATE : SM_FB_LOOP_ORIENTATION; }
  // A surface that lies far from the face's edges (farther than its tolerances allow by far, and than a
  // thousandth of the loops' size): triangles on it would be wherever it is. Only a loop without area
  // can be output (fanned, as SMLib does, from the edges' points alone), and only while the surface is
  // nearer than the loops are wide.
  if ((eFailure == SM_FB_NONE || eFailure == SM_FB_DEGENERATE) && rF.dMaxDist > 100.0 * rF.dInvTol)
    {
      SmExtent3d sBox;
      for (const std::vector<sm_LoopPoint> & crL : sLoops)
        {
          for (const sm_LoopPoint & crLP : crL)
            {
              sBox.AddPoint3d((crLP.lEdge < 0) ? rC.sVertexPoints[crLP.lIndex] : rC.sEdges[crLP.lEdge].sP[crLP.lIndex]);
            }
        }
      const double dSize = sBox.GetSize().Length();
      if (eFailure == SM_FB_NONE ? rF.dMaxDist > 1.0e-3 * dSize : !(rF.dMaxDist < dSize))
        { eFailure = SM_FB_INVERSION; }
    }
  // A faceuse whose loops run clockwise contradicts its face (SMLib meshes such faces along their loops,
  // consistently with the faces around them, with the face's normals). Without seams that shows as loops
  // all the other way round; with seams their sides were taken for the face to the left of the loops, so
  // they don't meet: once more with the face to their right.
  if (   (eFailure == SM_FB_LOOP_WINDING || eFailure == SM_FB_LOOP_ORIENTATION)
      && (rF.aPeriod[0] > 0.0 || rF.aPeriod[1] > 0.0))
    {
      rF.bFlipped = TRUE;
      for (const std::vector<int> & crLoop : rF.sLoops)
        {
          for (int lCE : crLoop) { rC.sCoedges[lCE].sHave.assign(rC.sCoedges[lCE].sHave.size(), 0); }
        }
      std::vector<std::vector<sm_LoopPoint>> sAgain;
      int lOuterAgain = -1;
      SmBoolean bDegenerateAgain = FALSE;
      if (   sm_AssembleLoops(rC, rF, sAgain) == SM_FB_NONE
          && sm_LoopOrientation(sAgain, lOuterAgain, bDegenerateAgain) == -1)
        {
          sLoops.swap(sAgain);
          lOuter       = lOuterAgain;
          iOrientation = -1;
          eFailure     = SM_FB_NONE;
        }
      else
        { rF.bFlipped = FALSE; }
    }
  if (iOrientation == -1)
    {
      for (std::vector<sm_LoopPoint> & rL : sLoops) { std::reverse(rL.begin(), rL.end()); }
      rOut.bLoopsReversed = TRUE;
    }
  if (iOrientation != 0 && lOuter > 0)
    { std::swap(sLoops[0], sLoops[lOuter]); }
  // UVs within rounding of a leaf bound snap onto it: points of iso edges then lie exactly on the tree's
  // grid lines, as triangulators that work on the leaves expect (an inverted point is only accurate to
  // rounding)
  if (!rF.sTree.empty())
    {
      // the nearest bound within dEps among the leaves around the point (ties to the smaller one)
      struct sm_Snap
      {
        double x, dEps, dBest, dDist = 0.0;
        bool   bFound = false;
        void Try(double c)
        {
          const double d = std::fabs(c - x);
          if (d <= dEps && (!bFound || d < dDist || (d == dDist && c < dBest))) { dBest = c; dDist = d; bFound = true; }
        }
      };
      const double dEpsU = 1.0e-9 * rF.sNatural.XLength(), dEpsV = 1.0e-9 * rF.sNatural.YLength();
      for (std::vector<sm_LoopPoint> & rL : sLoops)
        {
          for (sm_LoopPoint & rLP : rL)
            {
              sm_Snap sU{rLP.sUV.x, dEpsU, rLP.sUV.x}, sV{rLP.sUV.y, dEpsV, rLP.sUV.y};
              sm_ForLeaves(rF.sTree, sU.x - dEpsU, sV.x - dEpsV, sU.x + dEpsU, sV.x + dEpsV, [&](const sm_Node & crN)
                {
                  sU.Try(crN.u0); sU.Try(crN.u1);
                  sV.Try(crN.v0); sV.Try(crN.v1);
                });
              rLP.sUV.x = sU.dBest;
              rLP.sUV.y = sV.dBest;
            }
        }
    }
  // output with ids and points
  rOut.sLoops.resize(sLoops.size());
  for (size_t l = 0; l < sLoops.size(); l++)
    {
      std::vector<SmFastBoundaryVertex> & rL = rOut.sLoops[l];
      rL.resize(sLoops[l].size());
      for (size_t k = 0; k < sLoops[l].size(); k++)
        {
          const sm_LoopPoint & crLP = sLoops[l][k];
          SmFastBoundaryVertex & rV = rL[k];
          rV.dU = crLP.sUV.x;
          rV.dV = crLP.sUV.y;
          if (crLP.lEdge < 0)
            {
              rV.lId    = crVertexIds[crLP.lIndex];
              rV.sPoint = rC.sVertexPoints[crLP.lIndex];
            }
          else
            {
              const sm_Edge & crE = rC.sEdges[crLP.lEdge];
              rV.lId    = crE.sIds[crLP.lIndex];
              rV.sPoint = crE.sP[crLP.lIndex];
            }
        }
    }
  // (the loops stay in rOut for diagnostics when the face fails here)
  if (eFailure != SM_FB_NONE)
    { rF.eFailure = eFailure; return FALSE; }
  return TRUE;
}

} // end anonymous namespace

/*******************************************************************//**
   SmFastBoundaries
***********************************************************************/
SmFastBoundaries::SmFastBoundaries() {}

SmFastBoundaries::~SmFastBoundaries()
{
  Clear();
}

void SmFastBoundaries::Clear()
{
  for (SmFastFaceBoundary & rF : sFaces)
    {
      if (rF.pCache != NULL)
        {
          delete rF.pCache;
          rF.pCache = NULL;
        }
    }
  sFaces.clear();
  sEdges.clear();
  sPoints.clear();
  lVertices = 0;
  lFailedFaces = 0;
}

const char * SmFastBoundaryFailureName(SmFastBoundaryFailure eFailure)
{
  switch (eFailure)
    {
      case SM_FB_NONE:             return "none";
      case SM_FB_SURFACE:          return "surface";
      case SM_FB_TREE:             return "tree";
      case SM_FB_TOPOLOGY:         return "topology";
      case SM_FB_INVERSION:        return "inversion";
      case SM_FB_LOOP_OPEN:        return "loop_open";
      case SM_FB_LOOP_WINDING:     return "loop_winding";
      case SM_FB_LOOP_ORIENTATION: return "loop_orientation";
      case SM_FB_DEGENERATE:       return "degenerate";
      case SM_FB_SAMPLING:         return "sampling";
    }
  return "unknown";
}

/*******************************************************************//**
PURPOSE: Fast face boundaries (see SmFastEdges.h).
***********************************************************************/
SmStatus SmComputeFastBoundaries
 (const SmBrep                * cpBrep,
  const SmTessellationParams  & crParams,
  SmFastBoundaries            & rOut,
  const SmFastBoundaryOptions & crOptions)
{
  rOut.Clear();
  NER(cpBrep);
  const sm_Clock::time_point sStart = sm_Clock::now();
  SmBrep * pBrep = SM_CONST_CAST(SmBrep *, cpBrep);

  sm_Context sC;
  sC.cpBrep  = cpBrep;
  sC.sParams = crParams;
  if (sC.sParams.dMaxEdgeLength < 0.0)
    {
      SmExtent3d sBox;
      pBrep->CalculateBoundingBox(sBox);
      double dSize = sBox.GetSize().Length();
      if (dSize >= 1.0e15 || dSize <= 0.0) dSize = 1.0;
      sC.sParams.dMaxEdgeLength = -0.025 * dSize * sC.sParams.dMaxEdgeLength;
    }
  sC.dBrepTol = pBrep->GetTolerance();
  if (crOptions.pDriver != NULL)
    { sC.pDriver = crOptions.pDriver; }
  else
    {
      rOut.pDriver.reset(new SmSurfaceTessDriver(sC.sParams.dChordHeightTol, sC.sParams.dSurfaceAngleTolDeg,
                                                 sC.sParams.dMaxEdgeLength, 0.0, 0.001, sC.sParams.dMaxAspectRatio));
      sC.pDriver = rOut.pDriver.get();
    }

  // ---- prep: topology tables
  SmTArray<SmFace *>   sFaces;
  SmTArray<SmEdge *>   sEdges;
  SmTArray<SmVertex *> sVertices;
  pBrep->GetFaces(sFaces);
  pBrep->GetEdges(sEdges);
  pBrep->GetVertices(sVertices);
  const int lFaces = (int)sFaces.GetSize(), lEdges = (int)sEdges.GetSize(), lVertices = (int)sVertices.GetSize();
  std::unordered_map<const SmVertex *, int> sVertexIndex;
  std::unordered_map<const SmEdge *, int>   sEdgeIndex;
  std::unordered_map<const SmEdgeuse *, int> sCoedgeOf;
  sC.sVertices.resize(lVertices);
  sC.sVertexPoints.resize(lVertices);
  sC.sVertexTols.resize(lVertices);
  for (int v = 0; v < lVertices; v++)
    {
      sC.sVertices[v]     = sVertices[v];
      sC.sVertexPoints[v] = sVertices[v]->GetPoint();
      sC.sVertexTols[v]   = sVertices[v]->GetTolerance();
      sVertexIndex.emplace(sVertices[v], v);
    }
  sC.sEdges.resize(lEdges);
  for (int e = 0; e < lEdges; e++)
    {
      sm_Edge & rE = sC.sEdges[e];
      SmEdge  * pEdge = sEdges[e];
      rE.pEdge   = pEdge;
      rE.cpCurve = pEdge->GetCurve();
      const SmExtent1d sIvl = pEdge->GetInterval();
      rE.t0 = sIvl.GetMin();
      rE.t1 = sIvl.GetMax();
      rE.dTol = pEdge->GetTolerance();
      auto i0 = sVertexIndex.find(pEdge->GetStartVertex()), i1 = sVertexIndex.find(pEdge->GetEndVertex());
      rE.lV0 = (i0 != sVertexIndex.end()) ? i0->second : -1;
      rE.lV1 = (i1 != sVertexIndex.end()) ? i1->second : -1;
      rE.bDegenerate = (rE.cpCurve == NULL);   // the curve's own test runs in parallel below
      sEdgeIndex.emplace(pEdge, e);
      // lazily built NURBS data, before curves are evaluated on several threads
      SmCurve * pCurve = SM_CONST_CAST(SmCurve *, rE.cpCurve);
      if (SmBSplineCurve * pBSC = SM_CAST_PTR(SmBSplineCurve, pCurve))
        { pBSC->GetOrCreateGwNurbPointer(); }
      else if (SmCrvOnSurf * pCOS = SM_CAST_PTR(SmCrvOnSurf, pCurve))
        {
          if (SmBSplineCurve * pUV = SM_CAST_PTR(SmBSplineCurve, SM_CONST_CAST(SmCurve *, pCOS->GetUVCurve())))
            { pUV->GetOrCreateGwNurbPointer(); }
          if (SmBSplineSurface * pBS = SM_CAST_PTR(SmBSplineSurface, SM_CONST_CAST(SmSurface *, pCOS->GetBaseSurface())))
            { pBS->GetOrCreateGwNurbPointer(); }
        }
    }
  sC.sFaces.resize(lFaces);
  rOut.sFaces.resize(lFaces);
  for (int f = 0; f < lFaces; f++)
    {
      sm_Face & rF = sC.sFaces[f];
      SmFace  * pFace = sFaces[f];
      rF.pFace    = pFace;
      rF.pSurface = pFace->GetSurface();
      rF.dTol     = pFace->GetTolerance();
      SmFastFaceBoundary & rB = rOut.sFaces[f];
      rB.cpFace     = pFace;
      rB.dTolerance = rF.dTol;
      SmFaceuse * pFU = pFace->GetUpwardFaceuse();
      rB.bReversed = (pFU && pFU->GetShell() && pFU->GetShell()->GetRegion() != pBrep->GetInfiniteRegion()) ? TRUE : FALSE;
      if (pFU == NULL)
        { rF.eFailure = SM_FB_TOPOLOGY; continue; }
      // coedges: the upward faceuse's loops (also for faces this path can't handle, for their id loops)
      SmTArray<SmLoopuse *> sLUs;
      SmTArray<SmEdgeuse *> sEUs;
      pFU->GetLoopuses(sLUs);
      std::unordered_map<const SmEdge *, int> sUses;
      for (ULONG l = 0; l < sLUs.GetSize() && rF.eFailure == SM_FB_NONE; l++)
        {
          if (!sLUs[l]->IsEdgeLoopuse())
            { rF.eFailure = SM_FB_TOPOLOGY; break; }
          sLUs[l]->GetEdgeuses(sEUs);
          std::vector<int> sLoop;
          for (ULONG k = 0; k < sEUs.GetSize(); k++)
            {
              SmEdgeuse * pEU = sEUs[k];
              auto it = sEdgeIndex.find(pEU->GetEdge());
              if (it == sEdgeIndex.end() || pEU->GetCCWEdgeuse() == NULL)
                { rF.eFailure = SM_FB_TOPOLOGY; break; }
              const int e = it->second;
              sm_Edge & rE = sC.sEdges[e];
              if (rE.lV0 < 0 || rE.lV1 < 0)
                { rF.eFailure = SM_FB_TOPOLOGY; break; }
              sm_Coedge sCE;
              sCE.lFace     = f;
              sCE.lEdge     = e;
              sCE.pEU       = pEU;
              sCE.bReversed = (pEU->GetOrientation() == SM_OT_OPPOSITE);
              const int lCE = (int)sC.sCoedges.size();
              sC.sCoedges.push_back(sCE);
              sCoedgeOf.emplace(pEU, lCE);
              sLoop.push_back(lCE);
              sUses[pEU->GetEdge()]++;
            }
          rF.sLoops.push_back(std::move(sLoop));
        }
      for (const std::vector<int> & crLoop : rF.sLoops)
        {
          for (int lCE : crLoop)
            { sC.sCoedges[lCE].bSeam = sUses[sC.sEdges[sC.sCoedges[lCE].lEdge].pEdge] > 1; }
        }
      if (rF.eFailure != SM_FB_NONE)
        { continue; }
      // the surface: SMLib's tessellation caches need an SmBSplineSurface kind
      SmBSplineSurface * pBSS = SM_CAST_PTR(SmBSplineSurface, rF.pSurface);
      if (rF.pSurface == NULL || pBSS == NULL)
        { rF.eFailure = SM_FB_SURFACE; continue; }
      pBSS->GetOrCreateGwNurbPointer();
      rF.bParallel = pBSS->HasAttributes() ? FALSE : TRUE;   // copying attributes isn't thread-safe
      rF.sNatural  = rF.pSurface->GetNaturalUVDomain();   // periodicity is tested in F1, in parallel
    }
  // accepted inversion distance: 10 times the largest tolerance of the face, its edges and vertices
  for (int f = 0; f < lFaces; f++)
    {
      sm_Face & rF = sC.sFaces[f];
      double dMax = std::max(rF.dTol, sC.dBrepTol);
      for (const std::vector<int> & crLoop : rF.sLoops)
        {
          for (int lCE : crLoop)
            {
              const sm_Edge & crE = sC.sEdges[sC.sCoedges[lCE].lEdge];
              dMax = std::max(dMax, std::max(crE.dTol, std::max(sC.sVertexTols[crE.lV0], sC.sVertexTols[crE.lV1])));
            }
        }
      rF.dInvTol = 10.0 * dMax;
    }
  // the coedges of each edge, in SmEdge::GetEdgeuses() order (an edgeuse or its mate)
  for (int e = 0; e < lEdges; e++)
    {
      SmTArray<SmEdgeuse *> sEUs;
      sC.sEdges[e].pEdge->GetEdgeuses(sEUs);
      for (ULONG k = 0; k < sEUs.GetSize(); k++)
        {
          auto it = sCoedgeOf.find(sEUs[k]);
          if (it == sCoedgeOf.end() && sEUs[k]->GetMate())
            { it = sCoedgeOf.find(sEUs[k]->GetMate()); }
          if (it == sCoedgeOf.end())
            { continue; }
          std::vector<int> & rList = sC.sEdges[e].sCoedges;
          if (std::find(rList.begin(), rList.end(), it->second) == rList.end())
            { rList.push_back(it->second); }
        }
    }
  // coedges of a face whose topology failed are dropped from their edges
  for (int e = 0; e < lEdges; e++)
    {
      std::vector<int> & rList = sC.sEdges[e].sCoedges;
      rList.erase(std::remove_if(rList.begin(), rList.end(),
                                 [&](int lCE) { return sC.sFaces[sC.sCoedges[lCE].lFace].eFailure == SM_FB_TOPOLOGY; }),
                  rList.end());
    }
  const sm_Clock::time_point sPrepEnd = sm_Clock::now();
  rOut.dPrepMs = sm_Ms(sStart, sPrepEnd);

  // with bStopAtFirstFailure: once a face has failed, the failures and the trees built so far go to rOut
  auto StopAtFailure = [&]()
    {
      if (!crOptions.bStopAtFirstFailure)
        { return false; }
      bool bFailed = false;
      for (int f = 0; f < lFaces && !bFailed; f++)
        { bFailed = (sC.sFaces[f].eFailure != SM_FB_NONE); }
      if (!bFailed)
        { return false; }
      for (int f = 0; f < lFaces; f++)
        {
          sm_Face & rF = sC.sFaces[f];
          SmFastFaceBoundary & rB = rOut.sFaces[f];
          rB.pCache = rF.pSC;
          rF.pSC = NULL;
          if (rF.eFailure != SM_FB_NONE)
            {
              rB.eFailure = rF.eFailure;
              rOut.lFailedFaces++;
            }
        }
      rOut.dTotalMs = sm_Ms(sStart, sm_Clock::now());
      return true;
    };
  if (StopAtFailure())
    { return SM_ERR_TESS_FAIL_FACE; }

  // runs fn(i) for i in [0, n): faces whose surface may be used on worker threads in parallel, the others
  // on the calling thread
  // worker threads only without notify callbacks (objects are created there), as SMLib's Phase 1 precompute
  SmBoolean bParallel = crOptions.bParallel;
  auto HasCallbacks = [](const SmContext * cpContext)
    {
      SmContext * pContext = SM_CONST_CAST(SmContext *, cpContext);
      return pContext != NULL && (pContext->GetUserNotifyCallback() != NULL || pContext->GetSysNotifyCallback() != NULL);
    };
  if (HasCallbacks(pBrep->GetContext()))
    { bParallel = FALSE; }
  for (int f = 0; f < lFaces && bParallel; f++)
    {
      if (HasCallbacks(sC.sFaces[f].pFace->GetContext())) { bParallel = FALSE; }
    }
  auto ForFaces = [&](const auto & crFn)
    {
      std::vector<int> sPar, sSer;
      for (int f = 0; f < lFaces; f++)
        {
          if (sC.sFaces[f].eFailure != SM_FB_NONE) continue;
          ((bParallel && sC.sFaces[f].bParallel) ? sPar : sSer).push_back(f);
        }
      if (!sPar.empty())
        {
          sm_ParallelFor(tbb::blocked_range<size_t>(0, sPar.size(), 1), [&](const tbb::blocked_range<size_t> & crR)
            {
              for (size_t i = crR.begin(); i != crR.end(); ++i) { crFn(sPar[i]); }
            });
        }
      for (int f : sSer) { crFn(f); }
    };
  auto ForEdges = [&](const auto & crFn)
    {
      if (bParallel)
        {
          sm_ParallelFor(tbb::blocked_range<int>(0, lEdges, 4), [&](const tbb::blocked_range<int> & crR)
            {
              for (int e = crR.begin(); e != crR.end(); ++e) { crFn(e); }
            });
        }
      else
        {
          for (int e = 0; e < lEdges; e++) { crFn(e); }
        }
    };

  // ---- E0: degenerate curves (sampling them is slow enough to run in parallel)
  ForEdges([&](int e)
    {
      sm_Edge & rE = sC.sEdges[e];
      if (!rE.bDegenerate && rE.cpCurve->IsDegenerate()) { rE.bDegenerate = true; }
    });

  // ---- F1: periodicity, refinement trees, iso tests
  ForFaces([&](int f)
    {
      sm_Face & rF = sC.sFaces[f];
      // closed without being smooth across the seam (a B-spline with ends that only meet) counts as
      // periodic too: its seam edges have a side as SMLib's UV curves do
      if (   rF.pSurface->IsPeriodic(rF.sNatural, SM_SP_U)
          || rF.pSurface->IsClosed(rF.sNatural, SM_SP_U)) { rF.aPeriod[0] = rF.sNatural.XLength(); }
      if (   rF.pSurface->IsPeriodic(rF.sNatural, SM_SP_V)
          || rF.pSurface->IsClosed(rF.sNatural, SM_SP_V)) { rF.aPeriod[1] = rF.sNatural.YLength(); }

      const sm_Clock::time_point sTreeStart = sm_Clock::now();
      const SmStatus eTree = sm_BuildTree(sC, rF, rOut.sFaces[f].sLeaves);
      rOut.sFaces[f].dTreeMs = sm_Ms(sTreeStart, sm_Clock::now());
      if (eTree != SM_SUCCESS)
        { rF.eFailure = SM_FB_TREE; return; }
      rF.cpLeaves = &rOut.sFaces[f].sLeaves;
      for (const std::vector<int> & crLoop : rF.sLoops)
        {
          for (int lCE : crLoop)
            { sm_IsoTest(sC, rF, sC.sCoedges[lCE]); }
        }
    });
  const sm_Clock::time_point sTreesEnd = sm_Clock::now();
  rOut.dTreesMs = sm_Ms(sPrepEnd, sTreesEnd);
  if (StopAtFailure())
    { return SM_ERR_TESS_FAIL_FACE; }

  // ---- E1: curve discretization; an edge that couldn't be sampled fails its faces (never a shorter boundary)
  ForEdges([&](int e) { sm_DiscretizeEdge(sC, sC.sEdges[e]); });
  ForFaces([&](int f)
    {
      sm_Face & rF = sC.sFaces[f];
      for (const std::vector<int> & crLoop : rF.sLoops)
        {
          for (int lCE : crLoop)
            {
              if (sC.sEdges[sC.sCoedges[lCE].lEdge].bFailed)
                { rF.eFailure = SM_FB_SAMPLING; return; }
            }
        }
    });
  const sm_Clock::time_point sEdgesEnd = sm_Clock::now();
  rOut.dEdgesMs = sm_Ms(sTreesEnd, sEdgesEnd);

  // ---- F2: UVs and refinement splits
  ForFaces([&](int f) { sm_FaceSplits(sC, sC.sFaces[f]); });
  const sm_Clock::time_point sSplitsEnd = sm_Clock::now();
  rOut.dSplitsMs = sm_Ms(sEdgesEnd, sSplitsEnd);
  if (StopAtFailure())
    { return SM_ERR_TESS_FAIL_FACE; }

  // ---- E2: merge, then ids (vertices first, then each edge's interior points)
  ForEdges([&](int e) { sm_MergeSplits(sC, sC.sEdges[e]); });
  sm_SnapCoincidentEdges(sC);
  std::vector<long> sVertexIds(lVertices);
  for (int v = 0; v < lVertices; v++) { sVertexIds[v] = v; }
  // interior points edge by edge
  long lNextId = lVertices;
  for (int e = 0; e < lEdges; e++)
    {
      sm_Edge & rE = sC.sEdges[e];
      const size_t n = rE.sT.size();
      rE.sIds.assign(n, -1);
      if (n > 0)
        {
          rE.sIds[n - 1] = sVertexIds[rE.lV1];
          rE.sIds[0]     = sVertexIds[rE.lV0];
        }
      for (size_t k = 1; k + 1 < n; k++)
        { rE.sIds[k] = lNextId++; }
    }
  rOut.lVertices = lVertices;
  rOut.sPoints.resize(lNextId);
  for (int v = 0; v < lVertices; v++) { rOut.sPoints[v] = sC.sVertexPoints[v]; }
  rOut.sEdges.resize(lEdges);
  for (int e = 0; e < lEdges; e++)
    {
      const sm_Edge & crE = sC.sEdges[e];
      SmFastEdgeBoundary & rB = rOut.sEdges[e];
      rB.cpEdge  = crE.pEdge;
      rB.sParams = crE.sT;
      rB.sIds    = crE.sIds;
      for (size_t k = 1; k + 1 < crE.sT.size(); k++)
        { rOut.sPoints[crE.sIds[k]] = crE.sP[k]; }
    }
  const sm_Clock::time_point sMergeEnd = sm_Clock::now();
  rOut.dMergeMs = sm_Ms(sSplitsEnd, sMergeEnd);

  // ---- F3: loops
  ForFaces([&](int f)
    {
      sm_Face & rF = sC.sFaces[f];
      if (sm_FaceLoops(sC, rF, rOut.sFaces[f], sVertexIds))
        { rOut.sFaces[f].eStatus = SM_SUCCESS; }
    });
  for (int f = 0; f < lFaces; f++)
    {
      sm_Face & rF = sC.sFaces[f];
      SmFastFaceBoundary & rB = rOut.sFaces[f];
      rB.pCache      = rF.pSC;
      rB.lInversions = rF.lInversions;
      rF.pSC = NULL;
      // vertex id loops from the topology and the edge discretizations
      if (rF.eFailure != SM_FB_TOPOLOGY)
        {
          for (const std::vector<int> & crLoop : rF.sLoops)
            {
              std::vector<long> sIds;
              for (int lCE : crLoop)
                {
                  const sm_Coedge & crCE = sC.sCoedges[lCE];
                  const SmFastEdgeBoundary & crEB = rOut.sEdges[crCE.lEdge];
                  const size_t n = crEB.sIds.size();
                  for (size_t k = 0; k + 1 < n; k++)
                    { sIds.push_back(crEB.sIds[crCE.bReversed ? n - 1 - k : k]); }
                }
              // a vertex repeated next to itself (degenerate edges) once
              sIds.erase(std::unique(sIds.begin(), sIds.end()), sIds.end());
              while (sIds.size() > 1 && sIds.front() == sIds.back()) { sIds.pop_back(); }
              if (sIds.size() >= 3) { rB.sIdLoops.push_back(std::move(sIds)); }
            }
        }
      if (rB.eStatus != SM_SUCCESS)
        {
          rB.eFailure = (rF.eFailure != SM_FB_NONE) ? rF.eFailure : SM_FB_TOPOLOGY;
          rOut.lFailedFaces++;
        }
    }
  const sm_Clock::time_point sEnd = sm_Clock::now();
  rOut.dLoopsMs = sm_Ms(sMergeEnd, sEnd);
  rOut.dTotalMs = sm_Ms(sStart, sEnd);
  return SM_SUCCESS;

} // end SmComputeFastBoundaries

/*******************************************************************//**
PURPOSE: Polylines of a Brep's edges (see SmFastEdges.h).
***********************************************************************/
SmStatus SmComputeFastEdgePolylines
 (const SmBrep               * cpBrep,
  const SmTessellationParams & crParams,
  SmFastEdgePolylines        & rOut)
{
  rOut = SmFastEdgePolylines();
  NER(cpBrep);
  SmTArray<SmEdge *> sEdges;
  cpBrep->GetEdges(sEdges);
  const size_t lEdges = sEdges.GetSize();
  for (size_t e = 0; e < lEdges; e++) { rOut.sEdges.push_back(sEdges[(ULONG)e]); }
  // lazily built NURBS data, before curves are evaluated on several threads
  SmBoolean bParallel = TRUE;
  auto HasCallbacks = [](const SmContext * cpContext)
    {
      SmContext * pContext = SM_CONST_CAST(SmContext *, cpContext);
      return pContext != NULL && (pContext->GetUserNotifyCallback() != NULL || pContext->GetSysNotifyCallback() != NULL);
    };
  if (HasCallbacks(cpBrep->GetContext()))
    { bParallel = FALSE; }
  for (size_t e = 0; e < lEdges; e++)
    {
      SmCurve * pCurve = sEdges[e]->GetCurve();
      if (HasCallbacks(sEdges[e]->GetContext()))
        { bParallel = FALSE; }
      if (SmBSplineCurve * pBSC = SM_CAST_PTR(SmBSplineCurve, pCurve))
        { pBSC->GetOrCreateGwNurbPointer(); }
      else if (SmCrvOnSurf * pCOS = SM_CAST_PTR(SmCrvOnSurf, pCurve))
        {
          if (SmBSplineCurve * pUV = SM_CAST_PTR(SmBSplineCurve, SM_CONST_CAST(SmCurve *, pCOS->GetUVCurve())))
            { pUV->GetOrCreateGwNurbPointer(); }
          if (SmBSplineSurface * pBS = SM_CAST_PTR(SmBSplineSurface, SM_CONST_CAST(SmSurface *, pCOS->GetBaseSurface())))
            { pBS->GetOrCreateGwNurbPointer(); }
        }
    }
  std::vector<std::vector<SmPoint3d>> sPolylines(lEdges);
  std::vector<char>                   sFailed(lEdges, 0);
  auto Polyline = [&](size_t e)
    {
      const SmEdge   * cpEdge  = sEdges[e];
      const SmCurve  * cpCurve = cpEdge->GetCurve();
      const SmVertex * cpV0    = cpEdge->GetStartVertex();
      const SmVertex * cpV1    = cpEdge->GetEndVertex();
      std::vector<SmPoint3d> & rP = sPolylines[e];
      if (cpCurve == NULL || cpV0 == NULL || cpV1 == NULL)
        { sFailed[e] = 1; return; }
      const SmExtent1d sIvl = cpEdge->GetInterval();
      const double t0 = sIvl.GetMin(), t1 = sIvl.GetMax();
      const SmPoint3d sP0 = cpV0->GetPoint(), sP1 = cpV1->GetPoint();
      rP.push_back(sP0);
      if (!(t1 > t0) || cpCurve->IsDegenerate())
        { return; }   // one point
      SmCurveTessDriver sCrvTess(0, crParams.dChordHeightTol, crParams.dCurveAngleTolDeg, crParams.dMaxEdgeLength, 0.0001, FALSE);
      SmTArray<double>    sParams;
      if (sCrvTess.TessellateCurve(*cpCurve, SmExtent1d(t0, t1), &sParams) != SM_SUCCESS)
        { rP.clear(); sFailed[e] = 1; return; }
      // the curve's interior points, unless they coincide with an end (no clamping by the vertices'
      // tolerances, which a sloppy model can make as large as the edge: this is for display)
      const double dSame = 1.0e-12 * (1.0 + sP0.DistanceBetween(SmPoint3d(0, 0, 0)) + sP1.DistanceBetween(SmPoint3d(0, 0, 0)));
      for (ULONG q = 0; q < sParams.GetSize(); q++)
        {
          const double t = sParams[q];
          if (!(t > t0 && t < t1))
            { continue; }
          SmPoint3d sP;
          if (cpCurve->EvaluatePoint(t, sP) != SM_SUCCESS)
            { rP.clear(); sFailed[e] = 1; return; }
          if (sP.DistanceBetween(sP0) > dSame && sP.DistanceBetween(sP1) > dSame)
            { rP.push_back(sP); }
        }
      rP.push_back(sP1);
    };
  if (bParallel)
    {
      sm_ParallelFor(tbb::blocked_range<size_t>(0, lEdges, 8), [&](const tbb::blocked_range<size_t> & crR)
        {
          for (size_t e = crR.begin(); e != crR.end(); ++e) { Polyline(e); }
        });
    }
  else
    {
      for (size_t e = 0; e < lEdges; e++) { Polyline(e); }
    }
  rOut.sOffsets.assign(lEdges + 1, 0);
  for (size_t e = 0; e < lEdges; e++)
    {
      rOut.sOffsets[e + 1] = rOut.sOffsets[e] + sPolylines[e].size();
      rOut.lFailed += sFailed[e] ? 1 : 0;
    }
  rOut.sPoints.reserve(rOut.sOffsets[lEdges]);
  for (const std::vector<SmPoint3d> & crP : sPolylines)
    { rOut.sPoints.insert(rOut.sPoints.end(), crP.begin(), crP.end()); }
  return (rOut.lFailed > 0) ? SM_ERR : SM_SUCCESS;

} // end SmComputeFastEdgePolylines
