// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmTessFast.cpp
* PURPOSE: Opt-in fast tessellation of a Brep into flat triangle arrays
*          (see SmTessFast.h).
**********************************************************************/

#include "StdAfx.h"

#include <SmTessFast.h>
#include <SmFastFace.h>
#include <SmFastEdges.h>
#include <SmFastTessSrfCache.h>
#include <SmTess.h>
#include <SmBrep.h>
#include <SmFace.h>
#include <SmFaceuse.h>
#include <SmShell.h>
#include <SmRegion.h>
#include <SmVertex.h>
#include <SmPoly.h>
#include <SmSurface.h>
#include <SmBSplineSurface.h>
#include <SmSurfaceCache.h>
#include <SmTree.h>
#include <SmParallel.h>
#include <SmIsoCurve.h>
#include <SmNurbsSrf.h>
#include <SmNurbsCrv.h>
#include <SmPseudoBox.h>
#include <nurbs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <atomic>
#include <memory>
#include "tbb/parallel_sort.h"
#include "tbb/enumerable_thread_specific.h"
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

typedef std::chrono::steady_clock sm_Clock;

double sm_Ms(sm_Clock::time_point sStart, sm_Clock::time_point sEnd)
{
  return std::chrono::duration<double, std::milli>(sEnd - sStart).count();
}

// TRUE when SMLib objects may be made and deleted on worker threads: a context without notification
// callbacks, which SMLib calls for every object made or deleted, and callers don't expect concurrently
// (as SMLib's parallel refinement and triangulation planning require)
SmBoolean sm_NoNotifyCallbacks(const SmContext * cpContext)
{
  SmContext * pContext = SM_CONST_CAST(SmContext *, cpContext);
  return (   pContext == NULL
          || (pContext->GetUserNotifyCallback() == NULL && pContext->GetSysNotifyCallback() == NULL)) ? TRUE : FALSE;
}

// Runs the calls for Breps of one context one at a time, for as long as it lives: the context's state isn't
// for concurrent operations (its DoingBoolean flag, which these calls set and restore, as SMLib's own
// tessellation and edge destructors do, would otherwise be restored out of order and stay set, and marks
// and object caches are shared). Calls for different contexts run concurrently. Recursive, for a call
// made again from a notification callback. A call made from a notification callback of another
// context's call doesn't wait: if another thread has its context, it is declined (Acquired() FALSE).
// A thread thus waits only while it holds none of these locks, so two threads whose callbacks
// tessellate each other's contexts can't deadlock.
class sm_ContextSerializer
{
public:
  explicit sm_ContextSerializer(const SmContext * cpContext) : m_cpContext(cpContext)
  {
      {
        std::lock_guard<std::mutex> sGuard(MapMutex());
        Entry *& rpEntry = Map()[cpContext];
        if (rpEntry == NULL)
          { rpEntry = new Entry; }
        rpEntry->lUsers++;
        m_pEntry = rpEntry;
      }
    std::vector<const SmContext *> & rHeld = Held();
    if (rHeld.empty() || std::find(rHeld.begin(), rHeld.end(), cpContext) != rHeld.end())
      {
        m_pEntry->sMutex.lock();
        m_bAcquired = TRUE;
      }
    else
      { m_bAcquired = m_pEntry->sMutex.try_lock() ? TRUE : FALSE; }
    if (m_bAcquired)
      { rHeld.push_back(cpContext); }
  }
  ~sm_ContextSerializer()
  {
    if (m_bAcquired)
      {
        Held().pop_back();   // (nested in the scopes of the ones before)
        m_pEntry->sMutex.unlock();
      }
    std::lock_guard<std::mutex> sGuard(MapMutex());
    if (--m_pEntry->lUsers == 0)
      {
        Map().erase(m_cpContext);
        delete m_pEntry;
      }
  }
  SmBoolean Acquired() const { return m_bAcquired; }
  sm_ContextSerializer(const sm_ContextSerializer &) = delete;
  sm_ContextSerializer & operator=(const sm_ContextSerializer &) = delete;
private:
  struct Entry
  {
    std::recursive_mutex sMutex;
    long                 lUsers = 0;   // callers holding or waiting for sMutex
  };
  static std::mutex & MapMutex() { static std::mutex s_sMutex; return s_sMutex; }
  static std::unordered_map<const SmContext *, Entry *> & Map()
  {
    static std::unordered_map<const SmContext *, Entry *> s_sMap;
    return s_sMap;
  }
  // the contexts whose locks the calling thread holds, innermost last
  static std::vector<const SmContext *> & Held()
  {
    static thread_local std::vector<const SmContext *> t_sHeld;
    return t_sHeld;
  }
  const SmContext * m_cpContext;
  Entry           * m_pEntry    = NULL;
  SmBoolean         m_bAcquired = FALSE;
};

// the same for a Brep's context and its faces' (SmFastEdges.cpp tests them likewise)
SmBoolean sm_BrepThreadsSafe(const SmBrep * cpBrep)
{
  if (!sm_NoNotifyCallbacks(cpBrep->GetContext()))
    { return FALSE; }
  SmTArray<SmFace *> sFaces;
  cpBrep->GetFaces(sFaces);
  for (ULONG ii = 0; ii < sFaces.GetSize(); ii++)
    {
      if (!sm_NoNotifyCallbacks(sFaces[ii]->GetContext()))
        { return FALSE; }
    }
  return TRUE;
}


// surface tessellation driver of SmTessellateFast(): its type selects SmTessSrfCache::BuildTreeFast()
class SmFastSurfaceTessDriver : public SmSurfaceTessDriver
{
public:
  using SmSurfaceTessDriver::SmSurfaceTessDriver;
  virtual SM_TYPE GetType() const { return SmFastSurfaceTessDriver_TYPE; }
};


/*******************************************************************//**
   Surface point and unit normal Su x Sv, the normal as
   SmSurface::EvaluateNormal() computes it (it falls back to that call
   where the cross product degenerates, as at poles).
***********************************************************************/
SmStatus sm_PointAndNormal(const SmSurface * cpSurface, double dU, double dV, SmPoint3d & rPoint, SmVector3d & rNormal)
{
  const SmPoint2d sUV(dU, dV);
  SmVector3d sSu, sSv;
  if (cpSurface->Evaluate1stDerivatives(sUV, TRUE, TRUE, rPoint, sSu, sSv) == SM_SUCCESS)
    {
      const SmVector3d sNormal = sSu * sSv;
      const double     dLength = sNormal.Length();
      if (dLength > SmSurface::GetNormalLimit())
        {
          rNormal = sNormal / dLength;
          return SM_SUCCESS;
        }
    }
  else if (cpSurface->EvaluatePoint(sUV, rPoint) != SM_SUCCESS)
    { return SM_ERR; }
  return cpSurface->EvaluateNormal(sUV, TRUE, TRUE, rNormal);
}

/*******************************************************************//**
   Triangulation of one simple polygon, CCW in UV: the triangulation
   that maximizes its worst triangle's 3D shape quality (dynamic
   programming over sub-polygons), with every triangle strictly CCW in
   UV. Vertices with equal ids are one 3D point (a pole): triangles
   with two of them don't count against the quality (the pipeline
   drops them).
***********************************************************************/
class SmPolygonTriangulator
{
public:
  static constexpr int kMaxDP = 40;   // larger polygons are split along diagonals first

  std::vector<double>    sU, sV;     // UV per vertex
  std::vector<SmPoint3d> sP;         // 3D point per vertex
  std::vector<long long> sKey;       // 3D identity per vertex
  std::vector<unsigned char> sSide;  // boundary sides of the vertex (bit mask): no triangle has 3 on one side
  std::vector<int>       sTriangles; // out: 3 polygon indices per triangle

  void Clear() { sU.clear(); sV.clear(); sP.clear(); sKey.clear(); sSide.clear(); }

  void Add(double dU, double dV, const SmPoint3d & crP, long long llKey, unsigned char cSide = 0)
  {
    sU.push_back(dU); sV.push_back(dV); sP.push_back(crP); sKey.push_back(llKey); sSide.push_back(cSide);
  }

  // FALSE when the polygon has no valid triangulation
  SmBoolean Triangulate()
  {
    const int n = (int)sU.size();
    if (n <= kMaxDP)
      { return TriangulateDP(); }

    // a large polygon: split along valid diagonals, the shortest in 3D among those that about halve
    // it, until the pieces are small enough for TriangulateDP()
    const std::vector<double> aU(sU), aV(sV);
    const std::vector<SmPoint3d> aP(sP);
    const std::vector<long long> aKey(sKey);
    const std::vector<unsigned char> aSide(sSide);
    std::vector<int> sResult;
    std::vector<std::vector<int>> sPieces(1, std::vector<int>(n));
    std::iota(sPieces[0].begin(), sPieces[0].end(), 0);
    SmBoolean bOK = TRUE;
    while (bOK && !sPieces.empty())
      {
        const std::vector<int> sPiece = std::move(sPieces.back());
        sPieces.pop_back();
        const int m = (int)sPiece.size();
        Clear();
        for (int k : sPiece)
          { Add(aU[k], aV[k], aP[k], aKey[k], aSide[k]); }
        if (m <= kMaxDP)
          {
            if (!TriangulateDP()) { bOK = FALSE; break; }
            for (int t : sTriangles) { sResult.push_back(sPiece[t]); }
            continue;
          }
        m_bConvex = IsConvex();
        int    lBestI = -1, lBestJ = -1;
        double dBest  = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; i++)
          {
            const int j = (i + m / 2) % m;
            const int a = std::min(i, j), b = std::max(i, j);
            if (!IsDiagonal(a, b))
              { continue; }
            const double dLength = (sP[b] - sP[a]).LengthSquared();
            if (dLength < dBest) { dBest = dLength; lBestI = a; lBestJ = b; }
          }
        if (lBestI < 0) { bOK = FALSE; break; }
        sPieces.emplace_back(sPiece.begin() + lBestI, sPiece.begin() + lBestJ + 1);
        std::vector<int> sOther(sPiece.begin() + lBestJ, sPiece.end());
        sOther.insert(sOther.end(), sPiece.begin(), sPiece.begin() + lBestI + 1);
        sPieces.push_back(std::move(sOther));
      }
    sU = aU; sV = aV; sP = aP; sKey = aKey; sSide = aSide;
    sTriangles = std::move(sResult);
    return bOK;
  }

  // TRUE when the polygon turns left (or goes straight) at every vertex
  SmBoolean IsConvex() const
  {
    const int n = (int)sU.size();
    for (int i = 0; i < n; i++)
      {
        if (Orient((i + n - 1) % n, i, (i + 1) % n) < 0.0)
          { return FALSE; }
      }
    return TRUE;
  }

  // dynamic programming triangulation of a polygon of at most kMaxDP vertices
  SmBoolean TriangulateDP()
  {
    sTriangles.clear();
    const int n = (int)sU.size();
    if (n < 3 || n > kMaxDP)
      { return FALSE; }
    // the polygon's 3D orientation (Newell normal): triangles must agree with it, so none is flipped in 3D
    // where the UV polygon maps to a curved or boundary-bent shape
    m_sNormal = SmVector3d(0, 0, 0);
    for (int i = 0; i < n; i++)
      {
        const SmPoint3d & a = sP[i];
        const SmPoint3d & b = sP[(i + 1) % n];
        m_sNormal.x += (a.y - b.y) * (a.z + b.z);
        m_sNormal.y += (a.z - b.z) * (a.x + b.x);
        m_sNormal.z += (a.x - b.x) * (a.y + b.y);
      }
    if (n == 3)
      {
        m_bConvex = TRUE;
        if (TriangleValue(0, 1, 2) == -kInf)
          { return FALSE; }
        sTriangles.insert(sTriangles.end(), {0, 1, 2});
        return TRUE;
      }

    m_bConvex = IsConvex();

    if (n == 4 && m_bConvex)
      {
        // the two diagonals of a convex quad
        const double q0 = TriangleValue(0, 1, 2), q1 = TriangleValue(0, 2, 3);
        const double q2 = TriangleValue(1, 2, 3), q3 = TriangleValue(1, 3, 0);
        const double dA = std::min(q0, q1), dB = std::min(q2, q3);
        if (dA == -kInf && dB == -kInf)
          { return FALSE; }
        if (dA >= dB) { sTriangles.insert(sTriangles.end(), {0, 1, 2, 0, 2, 3}); }
        else          { sTriangles.insert(sTriangles.end(), {1, 2, 3, 1, 3, 0}); }
        return TRUE;
      }

    if (!m_bConvex)
      { ComputeValidDiagonals(); }

    // m_sBest[i*n+j]: best worst-quality of a triangulation of sub-polygon i..j (closed by j-i)
    m_sBest.assign((size_t)n * n, -kInf);
    m_sChoice.assign((size_t)n * n, -1);
    for (int i = 0; i + 1 < n; i++)
      { m_sBest[(size_t)i * n + i + 1] = kInf; }
    for (int lLen = 2; lLen < n; lLen++)
      {
        for (int i = 0; i + lLen < n; i++)
          {
            const int j = i + lLen;
            if (!(i == 0 && j == n - 1) && !IsValidEdge(i, j))
              { continue; }
            double dBest   = -kInf;
            int    lChoice = -1;
            for (int m = i + 1; m < j; m++)
              {
                const double dLeft  = m_sBest[(size_t)i * n + m];
                const double dRight = m_sBest[(size_t)m * n + j];
                if (dLeft == -kInf || dRight == -kInf)
                  { continue; }
                const double dValue = std::min(std::min(dLeft, dRight), TriangleValue(i, m, j));
                if (dValue > dBest)
                  {
                    dBest   = dValue;
                    lChoice = m;
                  }
              }
            m_sBest[(size_t)i * n + j]   = dBest;
            m_sChoice[(size_t)i * n + j] = lChoice;
          }
      }
    if (m_sChoice[n - 1] < 0)
      { return FALSE; }

    // collect the triangles
    m_sStack.clear();
    m_sStack.push_back(0);
    m_sStack.push_back(n - 1);
    while (!m_sStack.empty())
      {
        const int j = m_sStack.back(); m_sStack.pop_back();
        const int i = m_sStack.back(); m_sStack.pop_back();
        if (j - i < 2)
          { continue; }
        const int m = m_sChoice[(size_t)i * n + j];
        sTriangles.insert(sTriangles.end(), {i, m, j});
        m_sStack.push_back(i); m_sStack.push_back(m);
        m_sStack.push_back(m); m_sStack.push_back(j);
      }
    return TRUE;
  }

private:
  static constexpr double kInf = std::numeric_limits<double>::infinity();

  SmBoolean           m_bConvex = TRUE;
  SmVector3d          m_sNormal;   // Newell normal of the polygon being triangulated
  std::vector<double> m_sBest;
  std::vector<int>    m_sChoice;
  std::vector<char>   m_sValid;   // valid diagonals of a non-convex polygon
  std::vector<int>    m_sStack;

  double Orient(int a, int b, int c) const
  {
    return (sU[b] - sU[a]) * (sV[c] - sV[a]) - (sV[b] - sV[a]) * (sU[c] - sU[a]);
  }

  // 3D shape quality in (0, 1], 1 for equilateral; -inf when invalid in UV; 2 for pole triangles
  double TriangleValue(int a, int b, int c) const
  {
    if (Orient(a, b, c) <= 0.0)
      { return -kInf; }
    if (!m_bConvex && !IsEmpty(a, b, c))
      { return -kInf; }
    if (sKey[a] == sKey[b] || sKey[b] == sKey[c] || sKey[c] == sKey[a])
      { return 2.0; }
    const SmVector3d sAB = sP[b] - sP[a], sAC = sP[c] - sP[a], sBC = sP[c] - sP[b];
    const SmVector3d sCross = sAB * sAC;
    const double dSum = sAB.LengthSquared() + sAC.LengthSquared() + sBC.LengthSquared();
    if (!(dSum > 0.0))
      { return 0.0; }
    double dQuality = 3.4641016151377544 * sCross.Length() / dSum;  // 4 sqrt(3) area / sum of squares
    // avoided unless the alternative is a several times worse sliver: a triangle flipped against the
    // polygon in 3D, and three vertices on one boundary side (a sliver along the edge)
    if (sCross.Dot(m_sNormal) <= 0.0)
      { dQuality *= 0.2; }
    if (sSide[a] & sSide[b] & sSide[c])
      { dQuality *= 0.25; }
    return dQuality;
  }

  // no other polygon vertex in or on triangle abc
  SmBoolean IsEmpty(int a, int b, int c) const
  {
    const int n = (int)sU.size();
    for (int k = 0; k < n; k++)
      {
        if (k == a || k == b || k == c)
          { continue; }
        if (   Orient(a, b, k) >= 0.0
            && Orient(b, c, k) >= 0.0
            && Orient(c, a, k) >= 0.0)
          { return FALSE; }
      }
    return TRUE;
  }

  SmBoolean IsValidEdge(int i, int j) const
  {
    const int n = (int)sU.size();
    if (j - i == 1 || (i == 0 && j == n - 1))
      { return TRUE; }
    return m_bConvex ? TRUE : (SmBoolean)m_sValid[(size_t)i * n + j];
  }

  // segments ab and cd intersect, touching included
  SmBoolean Intersect(int a, int b, int c, int d) const
  {
    const double d1 = Orient(a, b, c), d2 = Orient(a, b, d);
    const double d3 = Orient(c, d, a), d4 = Orient(c, d, b);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
      { return TRUE; }
    auto OnSegment = [&](int p, int q, int r)  // r on segment pq, given collinear
      {
        return    std::min(sU[p], sU[q]) <= sU[r] && sU[r] <= std::max(sU[p], sU[q])
               && std::min(sV[p], sV[q]) <= sV[r] && sV[r] <= std::max(sV[p], sV[q]);
      };
    return    (d1 == 0.0 && OnSegment(a, b, c)) || (d2 == 0.0 && OnSegment(a, b, d))
           || (d3 == 0.0 && OnSegment(c, d, a)) || (d4 == 0.0 && OnSegment(c, d, b));
  }

  // b strictly inside the cone of the polygon at a
  SmBoolean InCone(int a, int b) const
  {
    const int n  = (int)sU.size();
    const int a0 = (a + n - 1) % n, a1 = (a + 1) % n;
    if (Orient(a, a1, a0) >= 0.0)  // convex (or flat) at a
      { return Orient(a, b, a0) > 0.0 && Orient(b, a, a1) > 0.0; }
    return !(Orient(a, b, a1) >= 0.0 && Orient(b, a, a0) >= 0.0);
  }

  // (a, b) is a valid diagonal: inside the polygon, crossing none of its edges
  SmBoolean IsDiagonal(int a, int b) const
  {
    const int n = (int)sU.size();
    if (b - a < 2 || (a == 0 && b == n - 1))
      { return FALSE; }
    if (!InCone(a, b) || !InCone(b, a))
      { return FALSE; }
    for (int k = 0; k < n; k++)
      {
        const int k1 = (k + 1) % n;
        if (k == a || k == b || k1 == a || k1 == b)
          { continue; }
        if (Intersect(a, b, k, k1))
          { return FALSE; }
      }
    return TRUE;
  }

  void ComputeValidDiagonals()
  {
    const int n = (int)sU.size();
    m_sValid.assign((size_t)n * n, 0);
    for (int i = 0; i < n; i++)
      {
        for (int j = i + 2; j < n; j++)
          {
            if (i == 0 && j == n - 1)
              { continue; }
            SmBoolean bValid = InCone(i, j) && InCone(j, i);
            for (int k = 0; k < n && bValid; k++)
              {
                const int k1 = (k + 1) % n;
                if (k == i || k == j || k1 == i || k1 == j)
                  { continue; }
                if (Intersect(i, j, k, k1))
                  { bValid = FALSE; }
              }
            m_sValid[(size_t)i * n + j] = bValid ? 1 : 0;
          }
      }
  }
};

/*******************************************************************//**
   Rectangular faces: helpers
***********************************************************************/

struct SmLeafRect
{
  double u0, v0, u1, v1;
};

struct SmUVKey
{
  double u, v;
};

inline bool sm_LessVU(const SmUVKey & a, const SmUVKey & b) { return a.v < b.v || (a.v == b.v && a.u < b.u); }

// one side of the face's rectangle as a run of the boundary loop, in loop order
struct SmRectSide
{
  int                 lStart = 0;      // loop index of the side's first vertex (a corner)
  int                 lCount = 0;      // vertices on the side, both corners included
  std::vector<double> sT;              // increasing coordinate along the side per side vertex
  SmBoolean           bPole  = FALSE;  // all its vertices are one 3D point
};

enum { SM_SIDE_BOTTOM = 0, SM_SIDE_RIGHT = 1, SM_SIDE_TOP = 2, SM_SIDE_LEFT = 3 };

} // end anonymous namespace

/*******************************************************************//**
PURPOSE: Triangulate a face whose outer loop is the rectangle of its
   refinement tree's UV domain (natural or rectangular trim).

NOTES: Every tree leaf becomes a polygon: its corners and the corners of
   finer neighbors on its sides (T-junctions), so the leaves conform.
   Leaf corners on the face boundary are not vertices: they snap to the
   nearest boundary vertex of their side (on a pole side they stay as
   UV points of the pole). Each polygon is triangulated without new
   vertices, maximizing its worst triangle in 3D. Returns SM_ERR without
   reporting when the face doesn't qualify or a polygon has no valid
   triangulation.
***********************************************************************/
SmStatus SmTriangulateRectangularFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut)
{
  rOut.eStatus = SM_ERR;
  rOut.sNewUV.clear();
  rOut.sNewPoints.clear();
  rOut.sNewNormals.clear();
  rOut.sTriangles.clear();

  if (crIn.cpCache == NULL || crIn.cpSurface == NULL || crIn.sLoops.size() != 1)
    { return SM_ERR; }
  const std::vector<SmFastLoopVertex> & crLoop = crIn.sLoops[0];
  const int lLoop = (int)crLoop.size();
  if (lLoop < 4)
    { return SM_ERR; }
  const SmTree     * cpTree = crIn.cpCache->GetBuiltTree();
  const SmTreeNode * cpTop  = cpTree ? cpTree->GetTopNode() : NULL;
  if (cpTop == NULL || cpTop->m_pData == NULL)
    { return SM_ERR; }

  const SmExtent2d sRoot = ((const SmBezierAux2d *)cpTop->m_pData)->GetUVDomain();
  const double U0 = sRoot.GetUMin(), U1 = sRoot.GetUMax();
  const double V0 = sRoot.GetVMin(), V1 = sRoot.GetVMax();
  if (!(U1 > U0 && V1 > V0))
    { return SM_ERR; }
  const double dTolU = 1e-6 * (U1 - U0), dTolV = 1e-6 * (V1 - V0);

  // 1. the leaves, depth first
  std::vector<SmLeafRect> sLeaves;
    {
      std::vector<const SmTreeNode *> sStack(1, cpTop);
      while (!sStack.empty())
        {
          const SmTreeNode * cpNode = sStack.back();
          sStack.pop_back();
          if (cpNode->m_pData == NULL)
            { return SM_ERR; }
          if (cpNode->m_pChild1 != NULL)
            {
              if (cpNode->m_pChild2 == NULL)
                { return SM_ERR; }
              sStack.push_back(cpNode->m_pChild2);
              sStack.push_back(cpNode->m_pChild1);
              continue;
            }
          const SmExtent2d sDomain = ((const SmBezierAux2d *)cpNode->m_pData)->GetUVDomain();
          SmLeafRect sLeaf = { sDomain.GetUMin(), sDomain.GetVMin(), sDomain.GetUMax(), sDomain.GetVMax() };
          if (!(sLeaf.u0 >= U0 && sLeaf.u1 <= U1 && sLeaf.v0 >= V0 && sLeaf.v1 <= V1 && sLeaf.u0 < sLeaf.u1 && sLeaf.v0 < sLeaf.v1))
            { return SM_ERR; }
          sLeaves.push_back(sLeaf);
        }
    }

  // SubdivideByBlock() computes the splits of each block on its own, so the leaves of neighbor blocks
  // can disagree about a shared split, in the last bits or (with a maximum edge length: equally spaced
  // splits per block) by a little, which would leave slivers at the T-junctions. Use one value per
  // split: values closer than 1e-12 of the domain size or 1% of the narrowest leaf become the smallest
  // of them (the domain bounds win), and leaves that collapse to zero width are dropped.
    {
      const bool bPar = crIn.bParallelSafe ? true : false;
      auto Canonical = [&](double SmLeafRect::*pLo, double SmLeafRect::*pHi, double dMin, double dMax) -> SmBoolean
        {
          double dMinExtent = dMax - dMin;
          for (const SmLeafRect & crLeaf : sLeaves)
            { dMinExtent = std::min(dMinExtent, crLeaf.*pHi - crLeaf.*pLo); }
          const double dEps = std::max(1e-12 * (dMax - dMin), 0.01 * dMinExtent);
          std::vector<double> sValues;
          sValues.reserve(sLeaves.size() * 2);
          for (const SmLeafRect & crLeaf : sLeaves)
            {
              sValues.push_back(crLeaf.*pLo);
              sValues.push_back(crLeaf.*pHi);
            }
          if (bPar) { tbb::parallel_sort(sValues.begin(), sValues.end()); }
          else      { std::sort(sValues.begin(), sValues.end()); }
          sValues.erase(std::unique(sValues.begin(), sValues.end()), sValues.end());
          std::vector<double> sCanonical(sValues.size());
          SmBoolean bMerged = FALSE;
          for (size_t i = 0; i < sValues.size(); i++)
            {
              sCanonical[i] = (i > 0 && sValues[i] - sValues[i - 1] <= dEps) ? sCanonical[i - 1] : sValues[i];
              bMerged = bMerged || (sCanonical[i] != sValues[i]);
            }
          // runs that reach a domain bound take the bound
          for (size_t i = sValues.size(); i-- > 0; )
            {
              if (dMax - sValues[i] <= dEps) { sCanonical[i] = dMax; bMerged = TRUE; }
              else break;
            }
          for (size_t i = 0; i < sValues.size() && sValues[i] - dMin <= dEps; i++)
            { sCanonical[i] = dMin; }
          if (!bMerged)
            { return FALSE; }
          auto Leaves = [&](size_t l0, size_t l1)
            {
              for (size_t l = l0; l < l1; l++)
                {
                  SmLeafRect & rLeaf = sLeaves[l];
                  rLeaf.*pLo = sCanonical[std::lower_bound(sValues.begin(), sValues.end(), rLeaf.*pLo) - sValues.begin()];
                  rLeaf.*pHi = sCanonical[std::lower_bound(sValues.begin(), sValues.end(), rLeaf.*pHi) - sValues.begin()];
                }
            };
          if (bPar)
            {
              tbb::parallel_for(tbb::blocked_range<size_t>(0, sLeaves.size(), 512),
                                [&](const tbb::blocked_range<size_t> & crR) { Leaves(crR.begin(), crR.end()); });
            }
          else
            { Leaves(0, sLeaves.size()); }
          return TRUE;
        };
      SmBoolean bU = FALSE, bV = FALSE;
      auto CanonicalU = [&] { bU = Canonical(&SmLeafRect::u0, &SmLeafRect::u1, U0, U1); };
      auto CanonicalV = [&] { bV = Canonical(&SmLeafRect::v0, &SmLeafRect::v1, V0, V1); };
      if (bPar) { sm_ParallelInvoke(CanonicalU, CanonicalV); }
      else      { CanonicalU(); CanonicalV(); }
      if (bU || bV)
        {
          sLeaves.erase(std::remove_if(sLeaves.begin(), sLeaves.end(),
                                       [](const SmLeafRect & crLeaf) { return !(crLeaf.u0 < crLeaf.u1 && crLeaf.v0 < crLeaf.v1); }),
                        sLeaves.end());
        }
    }
  const int lLeaves = (int)sLeaves.size();

  // 2. the boundary loop as the four sides of the rectangle
  const double aCornerU[4] = { U0, U1, U1, U0 };  // BL, BR, TR, TL
  const double aCornerV[4] = { V0, V0, V1, V1 };
  int aCorner[4];
  for (int c = 0; c < 4; c++)
    {
      aCorner[c] = -1;
      double dBest = 2.0;
      for (int i = 0; i < lLoop; i++)
        {
          const double du = std::fabs(crLoop[i].dU - aCornerU[c]) / dTolU;
          const double dv = std::fabs(crLoop[i].dV - aCornerV[c]) / dTolV;
          const double d  = std::max(du, dv);
          if (d <= 1.0 && d < dBest)
            {
              dBest      = d;
              aCorner[c] = i;
            }
        }
      if (aCorner[c] < 0)
        { return SM_ERR; }
    }
  auto Steps = [lLoop](int a, int b) { return ((b - a) % lLoop + lLoop) % lLoop; };
  const int lToBR = Steps(aCorner[0], aCorner[1]), lToTR = Steps(aCorner[0], aCorner[2]), lToTL = Steps(aCorner[0], aCorner[3]);
  if (!(0 < lToBR && lToBR < lToTR && lToTR < lToTL))
    { return SM_ERR; }

  SmRectSide aSide[4];
  for (int s = 0; s < 4; s++)
    {
      SmRectSide & rSide = aSide[s];
      rSide.lStart = aCorner[s];
      rSide.lCount = Steps(aCorner[s], aCorner[(s + 1) % 4]) + 1;
      rSide.sT.resize(rSide.lCount);
      rSide.bPole = TRUE;
      for (int k = 0; k < rSide.lCount; k++)
        {
          const SmFastLoopVertex & crV = crLoop[(rSide.lStart + k) % lLoop];
          double dT, dOff;
          switch (s)
            {
              case SM_SIDE_BOTTOM: dT =  crV.dU; dOff = std::fabs(crV.dV - V0) / dTolV; break;
              case SM_SIDE_RIGHT:  dT =  crV.dV; dOff = std::fabs(crV.dU - U1) / dTolU; break;
              case SM_SIDE_TOP:    dT = -crV.dU; dOff = std::fabs(crV.dV - V1) / dTolV; break;
              default:             dT = -crV.dV; dOff = std::fabs(crV.dU - U0) / dTolU; break;
            }
          if (dOff > 1.0)
            { return SM_ERR; }
          if (k > 0 && !(dT > rSide.sT[k - 1]))
            { return SM_ERR; }
          rSide.sT[k] = dT;
          if (crV.lId != crLoop[rSide.lStart].lId)
            { rSide.bPole = FALSE; }
        }
    }

  // the sides each loop vertex lies on (corners: two)
  std::vector<unsigned char> sLoopSides(lLoop, 0);
  for (int s0 = 0; s0 < 4; s0++)
    {
      for (int k = 0; k < aSide[s0].lCount; k++)
        { sLoopSides[(aSide[s0].lStart + k) % lLoop] |= (unsigned char)(1 << s0); }
    }

  // A periodic seam (opposite sides sharing vertices) where the surface creases: one normal per vertex
  // and face, as SMLib's output and the checks have, can't suit both sides; leave such faces to SMLib.
  for (int s0 = 0; s0 < 2; s0++)
    {
      const SmRectSide & crA = aSide[s0];       // bottom, right
      const SmRectSide & crB = aSide[s0 + 2];   // top, left
      if (crA.bPole || crB.bPole)
        { continue; }
      for (int ka = 0; ka < crA.lCount; ka++)
        {
          const SmFastLoopVertex & crVA = crLoop[(crA.lStart + ka) % lLoop];
          for (int kb = 0; kb < crB.lCount; kb++)
            {
              const SmFastLoopVertex & crVB = crLoop[(crB.lStart + kb) % lLoop];
              if (crVA.lId != crVB.lId)
                { continue; }
              SmVector3d sNA, sNB;
              if (   crIn.cpSurface->EvaluateNormal(SmPoint2d(crVA.dU, crVA.dV), TRUE, TRUE, sNA) != SM_SUCCESS
                  || crIn.cpSurface->EvaluateNormal(SmPoint2d(crVB.dU, crVB.dV), TRUE, TRUE, sNB) != SM_SUCCESS
                  || sNA.Dot(sNB) < 0.98480775301220802)  // cos(10 deg)
                {
                  return SM_ERR;
                }
            }
        }
    }

  // 3. the leaf corners: the vertices sorted by (u, v), each leaf's corner vertices, and the next
  //    vertex along the same horizontal and vertical line
  std::vector<SmUVKey> sVerts;
  std::vector<int>     sCornerCount;
  std::vector<int>     sFirstCorner;                       // the first of each vertex's leaf corners in sAll
  std::vector<int>     sLeafCorner((size_t)lLeaves * 4);   // BL, BR, TR, TL vertex of each leaf
  struct SmCornerKey { double u, v; int lc; };
  std::vector<SmCornerKey> sAll((size_t)lLeaves * 4);      // the leaf corners 4 l + c, sorted
    {
      for (int l = 0; l < lLeaves; l++)
        {
          const SmLeafRect & crLeaf = sLeaves[l];
          sAll[4 * l]     = {crLeaf.u0, crLeaf.v0, 4 * l};
          sAll[4 * l + 1] = {crLeaf.u1, crLeaf.v0, 4 * l + 1};
          sAll[4 * l + 2] = {crLeaf.u1, crLeaf.v1, 4 * l + 2};
          sAll[4 * l + 3] = {crLeaf.u0, crLeaf.v1, 4 * l + 3};
        }
      auto Less = [](const SmCornerKey & a, const SmCornerKey & b)
        { return a.u < b.u || (a.u == b.u && (a.v < b.v || (a.v == b.v && a.lc < b.lc))); };
      if (crIn.bParallelSafe) { sm_RunIsolated([&] { tbb::parallel_sort(sAll.begin(), sAll.end(), Less); }); }
      else                    { std::sort(sAll.begin(), sAll.end(), Less); }
      sVerts.reserve(sAll.size() / 2);
      sCornerCount.reserve(sAll.size() / 2);
      sFirstCorner.reserve(sAll.size() / 2);
      for (size_t i = 0; i < sAll.size(); i++)
        {
          if (sVerts.empty() || sAll[i].u != sVerts.back().u || sAll[i].v != sVerts.back().v)
            {
              sVerts.push_back({sAll[i].u, sAll[i].v});
              sCornerCount.push_back(0);
              sFirstCorner.push_back((int)i);
            }
          sCornerCount.back()++;
          sLeafCorner[sAll[i].lc] = (int)sVerts.size() - 1;
        }
    }
  const int lVerts = (int)sVerts.size();
  std::vector<int> sNextH(lVerts, -1), sNextV(lVerts, -1), sPrevH(lVerts, -1);
    {
      std::vector<int> sByV(lVerts);
      std::iota(sByV.begin(), sByV.end(), 0);
      auto LessVU = [&](int a, int b) { return sm_LessVU(sVerts[a], sVerts[b]); };
      if (crIn.bParallelSafe) { sm_RunIsolated([&] { tbb::parallel_sort(sByV.begin(), sByV.end(), LessVU); }); }
      else                    { std::sort(sByV.begin(), sByV.end(), LessVU); }
      for (int k = 0; k + 1 < lVerts; k++)
        {
          if (sVerts[k + 1].u == sVerts[k].u) { sNextV[k] = k + 1; }
          if (sVerts[sByV[k + 1]].v == sVerts[sByV[k]].v)
            {
              sNextH[sByV[k]]     = sByV[k + 1];
              sPrevH[sByV[k + 1]] = sByV[k];
            }
        }
    }

  // 4. SMLib's Phase 2 ends the split of a tree node on a vertex of the node's side, made by the neighbor
  //    across it, within a quarter of the adjacent leaves (PrepareTreeNodeData()), so the near coincident
  //    splits of neighbor blocks become one vertex instead of a sliver. The same here: a T-junction whose
  //    leaves lie right of or above its line (made after the line's other side in SMLib's depth first
  //    order) merges into the next vertex on the line that is a T-junction from the other side, when that
  //    is within a quarter of the leaf between them.
  std::vector<int> sKeep(lVerts, -1);  // the vertex a T-junction merges into, or -1
    {
      auto TJunction = [&](int k, int lMask)  // an interior T-junction of the given leaf corners (bit c: corner c)
        {
          return    k >= 0 && sCornerCount[k] == 2
                 && ((1 << (sAll[sFirstCorner[k]].lc & 3)) | (1 << (sAll[sFirstCorner[k] + 1].lc & 3))) == lMask
                 && sVerts[k].u != U0 && sVerts[k].u != U1 && sVerts[k].v != V0 && sVerts[k].v != V1;
        };
      auto Merge = [&](int k)
        {
          // leaves right of a vertical line (BL of the upper leaf, TL of the lower), merging along v into
          // a vertex whose leaves lie left of it (BR, TR); or leaves above a horizontal line (BL of the
          // right leaf, BR of the left one), merging along u into one with leaves below it (TL, TR)
          const SmBoolean bVertical = TJunction(k, 9) ? TRUE : FALSE;
          if (!bVertical && !TJunction(k, 3))
            { return; }
          const int lOther = bVertical ? 6 : 12;
          double dBest = std::numeric_limits<double>::infinity();
          for (int lc = 0; lc < 2; lc++)
            {
              const int lCorner = sAll[sFirstCorner[k] + lc].lc;
              const SmLeafRect & crLeaf = sLeaves[lCorner / 4];
              // the leaf with its BL corner here lies up (right) of the vertex, the other one down (left)
              const SmBoolean bUp = ((lCorner & 3) == 0) ? TRUE : FALSE;
              const int    lNext   = bVertical ? (bUp ? sNextV[k] : ((k > 0 && sNextV[k - 1] == k) ? k - 1 : -1))
                                                 : (bUp ? sNextH[k] : sPrevH[k]);
              const double dExtent = bVertical ? crLeaf.v1 - crLeaf.v0 : crLeaf.u1 - crLeaf.u0;
              if (!TJunction(lNext, lOther))
                { continue; }
              const double dDist = bVertical ? std::fabs(sVerts[lNext].v - sVerts[k].v) : std::fabs(sVerts[lNext].u - sVerts[k].u);
              if (dDist <= 0.25 * dExtent && (dDist < dBest || (dDist == dBest && !bUp)))
                {
                  dBest    = dDist;
                  sKeep[k] = lNext;
                }
            }
        };
      auto MergeRange = [&](const tbb::blocked_range<int> & crRange)
        {
          for (int k = crRange.begin(); k != crRange.end(); ++k)
            { Merge(k); }
        };
      if (crIn.bParallelSafe && lVerts > 2048) { sm_ParallelFor(tbb::blocked_range<int>(0, lVerts, 1024), MergeRange); }
      else                                     { MergeRange(tbb::blocked_range<int>(0, lVerts)); }
    }

  // 5. polygon references of the vertices: r >= 0 new vertex r (interior), r >= lNew pole point r - lNew,
  //    r < 0 loop vertex ~r (corners and snapped boundary points); a merged T-junction takes the reference
  //    of the vertex it merges into
  std::vector<int> sRef(lVerts);
  std::vector<int> sNewVert;               // vertex index of each new vertex
  std::vector<int> sPoleVert;              // vertex index of each pole point
  std::vector<int> sPoleLoop;              // its pole's loop vertex
  auto Snap = [&](const SmRectSide & crSide, double dT) -> int  // side position of the nearest side vertex
    {
      const int k = (int)(std::lower_bound(crSide.sT.begin(), crSide.sT.end(), dT) - crSide.sT.begin());
      if (k <= 0) return 0;
      if (k >= crSide.lCount) return crSide.lCount - 1;
      return (dT - crSide.sT[k - 1] <= crSide.sT[k] - dT) ? k - 1 : k;
    };
  for (int i = 0; i < lVerts; i++)
    {
      const double u = sVerts[i].u, v = sVerts[i].v;
      const SmBoolean bOnU = (u == U0 || u == U1), bOnV = (v == V0 || v == V1);
      if (!bOnU && !bOnV)
        {
          if (sKeep[i] < 0)
            {
              sRef[i] = (int)sNewVert.size();
              sNewVert.push_back(i);
            }
          continue;
        }
      if (bOnU && bOnV)
        {
          const int c = (v == V0) ? ((u == U0) ? 0 : 1) : ((u == U1) ? 2 : 3);
          sRef[i] = ~aCorner[c];
          continue;
        }
      const int s = bOnV ? ((v == V0) ? SM_SIDE_BOTTOM : SM_SIDE_TOP) : ((u == U1) ? SM_SIDE_RIGHT : SM_SIDE_LEFT);
      const SmRectSide & crSide = aSide[s];
      if (crSide.bPole)
        {
          sRef[i] = -2;  // set below, after the new vertices are counted
          sPoleVert.push_back(i);
          sPoleLoop.push_back(crSide.lStart);
          continue;
        }
      const double dT = (s == SM_SIDE_BOTTOM) ? u : (s == SM_SIDE_RIGHT) ? v : (s == SM_SIDE_TOP) ? -u : -v;
      sRef[i] = ~((crSide.lStart + Snap(crSide, dT)) % lLoop);
    }
  const int lNew = (int)sNewVert.size();
  for (size_t p = 0; p < sPoleVert.size(); p++)
    { sRef[sPoleVert[p]] = lNew + (int)p; }
  for (int i = 0; i < lVerts; i++)
    {
      if (sKeep[i] >= 0)
        { sRef[i] = sRef[sKeep[i]]; }
    }

  // SMLib meshes only the corners of leaves that don't touch the boundary, while this mesh has every
  // interior corner. Where many interior corners touch no interior leaf (small faces, thin strips of
  // leaves), that changes the triangle count a lot: leave such faces to SMLib.
    {
      std::vector<char> sTouched(lVerts, 0);
      for (int l = 0; l < lLeaves; l++)
        {
          const SmLeafRect & crLeaf = sLeaves[l];
          if (crLeaf.u0 != U0 && crLeaf.u1 != U1 && crLeaf.v0 != V0 && crLeaf.v1 != V1)
            {
              for (int c = 0; c < 4; c++)
                { sTouched[sLeafCorner[4 * l + c]] = 1; }
            }
        }
      for (int i = 0; i < lVerts; i++)
        {
          if (sKeep[i] >= 0 && sTouched[i])
            { sTouched[sKeep[i]] = 1; }  // a merged T-junction touches the vertex it merges into
        }
      int lUntouched = 0;
      for (int k = 0; k < lNew; k++)
        { lUntouched += sTouched[sNewVert[k]] ? 0 : 1; }
      if (lUntouched > 0 && (lUntouched == lNew || lUntouched > 2 + lNew / 20))
        {
          return SM_ERR;
        }
    }

  // 6. points and normals of the new vertices
  rOut.sNewUV.resize(lNew);
  rOut.sNewPoints.resize(lNew);
  rOut.sNewNormals.resize(lNew);
  std::vector<char> sEvalOK(lNew, 1);
  const SmSurface * cpSurface = crIn.cpSurface;
  auto Evaluate = [&](const tbb::blocked_range<int> & crRange)
    {
      for (int k = crRange.begin(); k != crRange.end(); ++k)
        {
          const SmUVKey & crUV = sVerts[sNewVert[k]];
          rOut.sNewUV[k] = SmPoint2d(crUV.u, crUV.v);
          if (sm_PointAndNormal(cpSurface, crUV.u, crUV.v, rOut.sNewPoints[k], rOut.sNewNormals[k]) != SM_SUCCESS)
            { sEvalOK[k] = 0; }
        }
    };
  if (crIn.bParallelSafe) { sm_ParallelFor(tbb::blocked_range<int>(0, lNew, 256), Evaluate); }
  else                    { Evaluate(tbb::blocked_range<int>(0, lNew)); }
  for (int k = 0; k < lNew; k++)
    {
      if (!sEvalOK[k])
        { return SM_ERR; }
    }
  // 7. leaf polygons, triangulated in parallel chunks; the T-junction structure is checked on the
  //    way: 4 quarter turns around every interior vertex, 2 at boundary points, 1 at the domain
  //    corners (inexact coordinates would show up here)
  const int lChunks = std::max(1, std::min(256, lLeaves / 64));
  std::vector<std::vector<long>> sChunkTriangles(lChunks);
  std::vector<std::vector<SmPoint2d>> sChunkUVs(lChunks);
  std::vector<int>               sChunkFail(lChunks, 0);
  std::unique_ptr<std::atomic<int>[]> pTurns(new std::atomic<int>[lVerts]);
  for (int i = 0; i < lVerts; i++)
    { pTurns[i].store(sCornerCount[i], std::memory_order_relaxed); }
  auto Triangulate = [&](const tbb::blocked_range<int> & crRange)
    {
      SmPolygonTriangulator sTri;
      std::vector<int>      sPoly, sSide;
      for (int c = crRange.begin(); c != crRange.end(); ++c)
        {
          std::vector<long>      & rTris = sChunkTriangles[c];
          std::vector<SmPoint2d> & rUVs  = sChunkUVs[c];
          const int lFirst = (int)((long long)lLeaves * c / lChunks);
          const int lEnd   = (int)((long long)lLeaves * (c + 1) / lChunks);
          for (int l = lFirst; l < lEnd && !sChunkFail[c]; l++)
            {
              const SmLeafRect & crLeaf = sLeaves[l];
              const int lBL = sLeafCorner[4 * l], lBR = sLeafCorner[4 * l + 1];
              const int lTR = sLeafCorner[4 * l + 2], lTL = sLeafCorner[4 * l + 3];
              sPoly.clear();
              auto Push = [&](int r) { if (sPoly.empty() || sPoly.back() != r) sPoly.push_back(r); };
              auto PushChain = [&](const SmRectSide & crSide, int lFrom, int lTo)  // side positions [lFrom, lTo)
                {
                  for (int k = lFrom; k < lTo; k++)
                    { Push(~((crSide.lStart + k) % lLoop)); }
                };
              auto SidePos = [&](const SmRectSide & crSide, int lVert) -> int  // a corner or snapped boundary point
                { return Steps(crSide.lStart, ~sRef[lVert]); };
              // vertices of a leaf side from lFrom up to lTo (excluded) along the links; FALSE if lTo isn't reached
              auto Walk = [&](int lFrom, int lTo, const std::vector<int> & crNext) -> SmBoolean
                {
                  sSide.clear();
                  for (int x = lFrom; x != lTo; x = crNext[x])
                    {
                      if (x < 0 || sSide.size() > (size_t)lVerts)
                        { return FALSE; }
                      if (x != lFrom)
                        { pTurns[x].fetch_add(2, std::memory_order_relaxed); }
                      sSide.push_back(x);
                    }
                  return TRUE;
                };
              // bottom, left to right
              if (!Walk(lBL, lBR, sNextH)) { sChunkFail[c] = 1; break; }
              if (crLeaf.v0 == V0 && !aSide[SM_SIDE_BOTTOM].bPole)
                { PushChain(aSide[SM_SIDE_BOTTOM], SidePos(aSide[SM_SIDE_BOTTOM], lBL), SidePos(aSide[SM_SIDE_BOTTOM], lBR)); }
              else
                { for (int x : sSide) Push(sRef[x]); }
              // right, bottom to top
              if (!Walk(lBR, lTR, sNextV)) { sChunkFail[c] = 1; break; }
              if (crLeaf.u1 == U1 && !aSide[SM_SIDE_RIGHT].bPole)
                { PushChain(aSide[SM_SIDE_RIGHT], SidePos(aSide[SM_SIDE_RIGHT], lBR), SidePos(aSide[SM_SIDE_RIGHT], lTR)); }
              else
                { for (int x : sSide) Push(sRef[x]); }
              // top, right to left
              if (!Walk(lTL, lTR, sNextH)) { sChunkFail[c] = 1; break; }
              if (crLeaf.v1 == V1 && !aSide[SM_SIDE_TOP].bPole)
                { PushChain(aSide[SM_SIDE_TOP], SidePos(aSide[SM_SIDE_TOP], lTR), SidePos(aSide[SM_SIDE_TOP], lTL)); }
              else
                {
                  Push(sRef[lTR]);
                  for (size_t k = sSide.size(); k-- > 1; ) Push(sRef[sSide[k]]);
                }
              // left, top to bottom
              if (!Walk(lBL, lTL, sNextV)) { sChunkFail[c] = 1; break; }
              if (crLeaf.u0 == U0 && !aSide[SM_SIDE_LEFT].bPole)
                { PushChain(aSide[SM_SIDE_LEFT], SidePos(aSide[SM_SIDE_LEFT], lTL), SidePos(aSide[SM_SIDE_LEFT], lBL)); }
              else
                {
                  Push(sRef[lTL]);
                  for (size_t k = sSide.size(); k-- > 1; ) Push(sRef[sSide[k]]);
                }
              while (sPoly.size() > 1 && sPoly.front() == sPoly.back())
                { sPoly.pop_back(); }

              // triangulate it
              sTri.Clear();
              for (int r : sPoly)
                {
                  if (r < 0)
                    {
                      const SmFastLoopVertex & crV = crLoop[~r];
                      sTri.Add(crV.dU, crV.dV, crV.sPoint, (long long)crV.lId, sLoopSides[~r]);
                    }
                  else if (r < lNew)
                    {
                      const SmUVKey & crUV = sVerts[sNewVert[r]];
                      sTri.Add(crUV.u, crUV.v, rOut.sNewPoints[r], -1 - (long long)r);
                    }
                  else
                    {
                      const SmUVKey          & crUV   = sVerts[sPoleVert[r - lNew]];
                      const SmFastLoopVertex & crPole = crLoop[sPoleLoop[r - lNew]];
                      sTri.Add(crUV.u, crUV.v, crPole.sPoint, (long long)crPole.lId, sLoopSides[sPoleLoop[r - lNew]]);
                    }
                }
              if (!sTri.Triangulate())
                {
                  sChunkFail[c] = 3;
                  break;
                }
              for (int t : sTri.sTriangles)
                {
                  const int r = sPoly[t];
                  if (r < 0)        { rTris.push_back(crLoop[~r].lId); }
                  else if (r < lNew) { rTris.push_back(~(long)r); }
                  else              { rTris.push_back(crLoop[sPoleLoop[r - lNew]].lId); }
                  rUVs.push_back(SmPoint2d(sTri.sU[t], sTri.sV[t]));
                }
            }
        }
    };
  if (crIn.bParallelSafe) { sm_ParallelFor(tbb::blocked_range<int>(0, lChunks, 1), Triangulate); }
  else                    { Triangulate(tbb::blocked_range<int>(0, lChunks)); }
  auto Fail = [&]()
    {
      rOut.sNewUV.clear(); rOut.sNewPoints.clear(); rOut.sNewNormals.clear();
      return SM_ERR;
    };
  for (int c = 0; c < lChunks; c++)
    {
      if (sChunkFail[c] != 0)
        { return Fail(); }  // a leaf side walk failed, or no valid polygon triangulation
    }
  for (int i = 0; i < lVerts; i++)
    {
      const int lOnU = (sVerts[i].u == U0 || sVerts[i].u == U1) ? 1 : 0;
      const int lOnV = (sVerts[i].v == V0 || sVerts[i].v == V1) ? 1 : 0;
      const int lExpected = (lOnU && lOnV) ? 1 : ((lOnU || lOnV) ? 2 : 4);
      if (pTurns[i].load(std::memory_order_relaxed) != lExpected)
        { return Fail(); }  // not the T-junction structure of a subdivided rectangle
    }
  size_t lTotal = 0;
  for (int c = 0; c < lChunks; c++)
    { lTotal += sChunkTriangles[c].size(); }
  rOut.sTriangles.reserve(lTotal);
  rOut.sTriangleUVs.reserve(lTotal);
  for (int c = 0; c < lChunks; c++)
    {
      rOut.sTriangles.insert(rOut.sTriangles.end(), sChunkTriangles[c].begin(), sChunkTriangles[c].end());
      rOut.sTriangleUVs.insert(rOut.sTriangleUVs.end(), sChunkUVs[c].begin(), sChunkUVs[c].end());
    }

  rOut.eStatus = SM_SUCCESS;
  return SM_SUCCESS;

} // end SmTriangulateRectangularFace

/*******************************************************************//**
PURPOSE: The per-face dispatcher of SmTessellateFast().
***********************************************************************/
/*******************************************************************//**
   SmTriangulateTrimmedFace() (track G3) on this interface: the refinement
   tree's leaves as SmTrimLeaf rectangles, triangles and normals in the
   surface's orientation (the pipeline reverses reversed faces), the UV of
   each triangle corner for the pipeline's corner normals.
***********************************************************************/
SmStatus SmTriangulateTrimmedFastFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut)
{
  thread_local SmTrimFaceInput  tIn;
  thread_local SmTrimFaceOutput tOut;
  rOut.eStatus  = SM_ERR;
  rOut.eFailure = SM_FF_NONE;
  tIn.cpFace                      = crIn.cpFace;
  tIn.cpSurface                   = crIn.cpSurface;
  tIn.sParams.dChordHeightTol     = crIn.sParams.dChordHeightTol;
  tIn.sParams.dCurveAngleTolDeg   = crIn.sParams.dCurveAngleTolDeg;
  tIn.sParams.dSurfaceAngleTolDeg = crIn.sParams.dSurfaceAngleTolDeg;
  tIn.sParams.dMaxEdgeLength      = crIn.sParams.dMaxEdgeLength;
  tIn.sParams.dMaxAspectRatio     = crIn.sParams.dMaxAspectRatio;
  tIn.dTolerance                  = crIn.dTolerance;
  tIn.bSurfaceOrientation         = true;
  tIn.bTriangleUVs                = true;
  tIn.sLoops.resize(crIn.sLoops.size());
  for (size_t ii = 0; ii < crIn.sLoops.size(); ii++)
    {
      const std::vector<SmFastLoopVertex> & crLoop = crIn.sLoops[ii];
      std::vector<SmTrimLoopVertex>       & rLoop  = tIn.sLoops[ii];
      rLoop.resize(crLoop.size());
      for (size_t kk = 0; kk < crLoop.size(); kk++)
        {
          rLoop[kk].dU     = crLoop[kk].dU;
          rLoop[kk].dV     = crLoop[kk].dV;
          rLoop[kk].sPoint = crLoop[kk].sPoint;
          rLoop[kk].lId    = crLoop[kk].lId;
        }
    }
  tIn.sInteriorSamples = crIn.sInteriorSamples;
  tIn.sLeaves.clear();
  const SmTree     * cpTree = crIn.cpCache ? crIn.cpCache->GetBuiltTree() : NULL;
  const SmTreeNode * cpTop  = cpTree ? cpTree->GetTopNode() : NULL;
  if (cpTop != NULL)
    {
      thread_local std::vector<const SmTreeNode *> tStack;
      tStack.assign(1, cpTop);
      while (!tStack.empty())
        {
          const SmTreeNode * cpNode = tStack.back();
          tStack.pop_back();
          if (cpNode->m_pChild1 != NULL || cpNode->m_pChild2 != NULL)
            {
              if (cpNode->m_pChild2 != NULL) { tStack.push_back(cpNode->m_pChild2); }
              if (cpNode->m_pChild1 != NULL) { tStack.push_back(cpNode->m_pChild1); }
              continue;
            }
          if (cpNode->m_pData == NULL)
            { continue; }
          const SmExtent2d & crD = ((const SmBezierAux2d *)cpNode->m_pData)->m_sUVDomain;
          SmTrimLeaf sLeaf;
          sLeaf.dU0 = crD.GetMin().x;
          sLeaf.dV0 = crD.GetMin().y;
          sLeaf.dU1 = crD.GetMax().x;
          sLeaf.dV1 = crD.GetMax().y;
          tIn.sLeaves.push_back(sLeaf);
        }
    }
  if (SmTriangulateTrimmedFace(tIn, tOut) != SM_SUCCESS || tOut.eStatus != SM_SUCCESS)
    {
      const SmTrimFaceFailure eFailure = tOut.eFailure;
      rOut.eFailure = (   eFailure == SM_TF_LOOP_ORIENTATION || eFailure == SM_TF_LOOP_INTERSECTION
                       || eFailure == SM_TF_DUPLICATE_POINT  || eFailure == SM_TF_BAD_INPUT) ? SM_FF_LOOPS : SM_FF_OTHER;
      return SM_ERR;
    }
  // its own checks failed: refinement stopped at its budget (the unrefined mesh doesn't meet the
  // tolerances): the fallback; or triangles on loop segments stay flipped or degenerate until those edges
  // are split in both faces: the segments go back to the caller (SmTessellateWithFastBoundaries() splits
  // them; elsewhere the face falls back)
  if (!tOut.bTolerancesMet)
    {
      rOut.eFailure = SM_FF_OTHER;
      return SM_ERR;
    }
  if (!tOut.sSplitSegments.empty())
    {
      rOut.eFailure = SM_FF_SPLIT;
      rOut.sSplitSegments.swap(tOut.sSplitSegments);
      return SM_ERR;
    }
  // A periodic seam (a vertex at two places of the loops, not a pole run) where the surface creases: one
  // normal per vertex and face, as SMLib's output and the checks have, can't suit both sides; leave such
  // faces to SMLib, as SmTriangulateRectangularFace() does.
    {
      struct Occurrence { long lId; size_t lLoop, lPos; };
      thread_local std::vector<Occurrence> tOccurrences;
      tOccurrences.clear();
      for (size_t ii = 0; ii < tIn.sLoops.size(); ii++)
        {
          const std::vector<SmTrimLoopVertex> & crLoop = tIn.sLoops[ii];
          const size_t lCount = crLoop.size();
          for (size_t kk = 0; kk < lCount; kk++)
            {
              const long lId = crLoop[kk].lId;
              if (lCount > 1 && (crLoop[(kk + 1) % lCount].lId == lId || crLoop[(kk + lCount - 1) % lCount].lId == lId))
                { continue; }  // a pole run
              tOccurrences.push_back({lId, ii, kk});
            }
        }
      std::sort(tOccurrences.begin(), tOccurrences.end(), [](const Occurrence & a, const Occurrence & b)
        { return a.lId < b.lId || (a.lId == b.lId && (a.lLoop < b.lLoop || (a.lLoop == b.lLoop && a.lPos < b.lPos))); });
      for (size_t ii = 0; ii < tOccurrences.size(); ii++)
        {
          for (size_t jj = ii + 1; jj < tOccurrences.size() && tOccurrences[jj].lId == tOccurrences[ii].lId; jj++)
            {
              const SmVector3d & crA = tOut.sLoopNormals[tOccurrences[ii].lLoop][tOccurrences[ii].lPos];
              const SmVector3d & crB = tOut.sLoopNormals[tOccurrences[jj].lLoop][tOccurrences[jj].lPos];
              if (crA.Dot(crB) < 0.98480775301220802)  // cos(10 deg)
                {
                  rOut.eFailure = SM_FF_OTHER;
                  return SM_ERR;
                }
            }
        }
    }
  rOut.sNewUV.swap(tOut.sNewUV);
  rOut.sNewPoints.swap(tOut.sNewPoints);
  rOut.sNewNormals.swap(tOut.sNewNormals);
  rOut.sTriangles.swap(tOut.sTriangles);
  rOut.sTriangleUVs.swap(tOut.sTriangleUVs);
  rOut.eStatus = SM_SUCCESS;
  return SM_SUCCESS;
}

SmStatus SmTriangulateFastFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut)
{
  if (SmTriangulateRectangularFace(crIn, rOut) == SM_SUCCESS)
    { return SM_SUCCESS; }
  rOut = SmFastFaceOutput();
  if (SmTriangulateTrimmedFastFace(crIn, rOut) == SM_SUCCESS && rOut.eStatus == SM_SUCCESS)
    { return SM_SUCCESS; }
  const SmFastFaceFailure eFailure = (rOut.eFailure != SM_FF_NONE) ? rOut.eFailure : SM_FF_OTHER;
  std::vector<long> sSplitSegments;
  sSplitSegments.swap(rOut.sSplitSegments);
  rOut = SmFastFaceOutput();
  rOut.eFailure = eFailure;
  rOut.sSplitSegments.swap(sSplitSegments);
  return SM_ERR;
}

namespace {

/*******************************************************************//**
   Pipeline data of one face
***********************************************************************/
struct SmFastFaceWork
{
  SmFace                         * pFace     = NULL;
  SmTessSrfCache                 * pSC       = NULL;   // NULL: Phase 1 failed for the face
  SmBoolean                        bReversed = FALSE;
  SmBoolean                        bParallel = FALSE;  // its surface may be evaluated on worker threads
  std::vector<std::vector<std::pair<SmPolyVertex *, long>>> sLoops;  // Phase 1 loop vertices and their ids
  SmFastFaceOutput                 sOut;               // triangles (fast or fallback)
  std::vector<std::pair<long, SmVector3d>> sLoopNormals;  // surface normal (not reversed) per loop id, first use
  std::vector<SmVector3d>          sCornerNormals;     // when asked: surface normal (not reversed) per triangle corner
  SmBoolean                        bFast     = FALSE;
  SmBoolean                        bDone     = FALSE;  // sOut holds the face's triangles
};

// copies SMLib's Phase 2 triangles of a face into its output, mapping vertices as SmTess::OutputToPolyBrep():
// a PolyVertex with an original SmVertex is that vertex, any other is a new vertex of the face
SmStatus sm_CopyPhase2Triangles(SmFastFaceWork & rWork, const std::unordered_map<SmVertex *, long> & crVertexIds,
                                const std::vector<SmPoint3d> & crLoopPoints)
{
  SmFastFaceOutput & rOut = rWork.sOut;
  rOut = SmFastFaceOutput();
  SmPolyBrep * pPolyBrep = rWork.pSC ? rWork.pSC->GetTS_PolyBrep() : NULL;
  if (pPolyBrep == NULL)
    { return SM_ERR; }
  const SmSurface * cpSurface = rWork.pFace->GetSurface();
  const double      dTolerance = rWork.pFace->GetBrep()->GetTolerance();

  std::unordered_map<const SmPolyVertex *, long> sRefs;  // loop vertices: id, others: ~new index
  for (const auto & crLoop : rWork.sLoops)
    {
      for (const auto & crV : crLoop)
        { sRefs.emplace(crV.first, crV.second); }
    }

  SmTArray<SmPolyFace *> sPolyFaces;
  SmTArray<SmPolyEdge *> sEdges;
  pPolyBrep->GetPolyFaces(sPolyFaces);
  std::vector<long>      sPolygon;
  std::vector<SmPoint2d> sPolygonUV;
  for (ULONG ii = 0; ii < sPolyFaces.GetSize(); ii++)
    {
      SmPolyLoop * pLoop = sPolyFaces[ii]->GetOuterPolyLoop();
      if (pLoop == NULL)
        { continue; }
      pLoop->GetPolyEdges(sEdges);
      sPolygon.clear();
      sPolygonUV.clear();
      for (ULONG jj = 0; jj < sEdges.GetSize(); jj++)
        {
          const SmPolyVertex * cpVertex = sEdges[jj]->GetStartPolyVertex();
          long lRef;
          auto sFound = sRefs.find(cpVertex);
          if (sFound != sRefs.end())
            { lRef = sFound->second; }
          else
            {
              SmVertex * pOriginal = cpVertex->GetOriginalVertex();
              auto sId = pOriginal ? crVertexIds.find(pOriginal) : crVertexIds.end();
              if (sId != crVertexIds.end())
                { lRef = sId->second; }
              else
                {
                  const SmPoint3d sUV = cpVertex->GetPoint();
                  SmPoint3d  sPoint;
                  SmVector3d sNormal;
                  if (sm_PointAndNormal(cpSurface, sUV.x, sUV.y, sPoint, sNormal) != SM_SUCCESS)
                    { return SM_ERR; }
                  lRef = ~(long)rOut.sNewUV.size();
                  rOut.sNewUV.push_back(SmPoint2d(sUV.x, sUV.y));
                  rOut.sNewPoints.push_back(sPoint);
                  rOut.sNewNormals.push_back(sNormal);
                }
              sRefs.emplace(cpVertex, lRef);
            }
          if (sPolygon.empty() || sPolygon.back() != lRef)
            {
              sPolygon.push_back(lRef);
              const SmPoint3d sUV = cpVertex->GetPoint();
              sPolygonUV.push_back(SmPoint2d(sUV.x, sUV.y));
            }
        }
      while (sPolygon.size() > 1 && sPolygon.front() == sPolygon.back())
        { sPolygon.pop_back(); sPolygonUV.pop_back(); }
      // as SmTess::OutputToPolyBrep(): a vertex closer than tolerance/100 to the next is dropped from the
      // polygon (it merges the zero length edge), and polygons left with fewer than 3 vertices are skipped
      auto Point = [&](long r) -> const SmPoint3d & { return (r >= 0) ? crLoopPoints[r] : rOut.sNewPoints[~r]; };
      for (size_t kk = 0; kk < sPolygon.size() && sPolygon.size() >= 3; )
        {
          const size_t lNext = (kk + 1) % sPolygon.size();
          if (Point(sPolygon[kk]).DistanceBetween(Point(sPolygon[lNext])) < dTolerance / 100.0)
            {
              sPolygon.erase(sPolygon.begin() + kk);
              sPolygonUV.erase(sPolygonUV.begin() + kk);
            }
          else
            { kk++; }
        }
      if (sPolygon.size() < 3)
        { continue; }
      // polygons fan from their first vertex, as mesh consumers of SmPolyBrep::to_mesh_arrays() do
      for (size_t kk = 1; kk + 1 < sPolygon.size(); kk++)
        {
          rOut.sTriangles.insert(rOut.sTriangles.end(), {sPolygon[0], sPolygon[kk], sPolygon[kk + 1]});
          rOut.sTriangleUVs.insert(rOut.sTriangleUVs.end(), {sPolygonUV[0], sPolygonUV[kk], sPolygonUV[kk + 1]});
        }
    }
  rOut.eStatus = SM_SUCCESS;
  return SM_SUCCESS;
}

// SmTess::OutputToPolyBrep() drops a polygon vertex closer than tolerance/100 to the next, so SMLib's output has no
// edges that short (it skips the polygons left with fewer than 3 vertices, e.g. tree leaves thinner than that).
// The same for a fast face: the ends of a shorter triangle edge merge (a loop vertex absorbs a new vertex; two loop
// vertices stay, as the neighbor face has them too), and the triangles that become degenerate go.
void sm_CollapseShortEdges(SmFastFaceOutput & rOut, const std::vector<SmPoint3d> & crLoopPoints, double dMinLength,
                           SmBoolean bParallel)
{
  const size_t lNew = rOut.sNewPoints.size();
  if (lNew == 0 || !(dMinLength > 0.0))
    { return; }
  const double dMin2 = dMinLength * dMinLength;
  auto Point = [&](long r) -> const SmPoint3d & { return (r >= 0) ? crLoopPoints[r] : rOut.sNewPoints[~r]; };
  // most faces have no such edge: look for one first, in parallel for large faces
  auto HasShort = [&](size_t lFirst, size_t lEnd)
    {
      for (size_t kk = 3 * lFirst; kk < 3 * lEnd; kk += 3)
        {
          for (int e = 0; e < 3; e++)
            {
              const long a = rOut.sTriangles[kk + e], b = rOut.sTriangles[kk + (e + 1) % 3];
              if ((a < 0 || b < 0) && a != b && Point(a).DistanceBetweenSquared(Point(b)) < dMin2)
                { return true; }
            }
        }
      return false;
    };
  const size_t lTriangles = rOut.sTriangles.size() / 3;
  if (bParallel && lTriangles > 4096)
    {
      std::atomic<bool> sFound(false);
      sm_ParallelFor(tbb::blocked_range<size_t>(0, lTriangles, 1024), [&](const tbb::blocked_range<size_t> & crRange)
        {
          if (!sFound.load(std::memory_order_relaxed) && HasShort(crRange.begin(), crRange.end()))
            { sFound.store(true, std::memory_order_relaxed); }
        });
      if (!sFound.load())
        { return; }
    }
  else if (!HasShort(0, lTriangles))
    { return; }
  std::vector<long> sParent(lNew);  // union-find over the new vertices: ~self, a loop id or another new vertex
  for (size_t kk = 0; kk < lNew; kk++)
    { sParent[kk] = ~(long)kk; }
  auto Find = [&](long r)
    {
      while (r < 0 && sParent[~r] != r)
        {
          const long q = sParent[~r];
          if (q < 0 && sParent[~q] != q) { sParent[~r] = sParent[~q]; }  // path halving
          r = sParent[~r];
        }
      return r;
    };
  std::vector<long> & rT = rOut.sTriangles;
  SmBoolean bMerged = FALSE;
  for (size_t kk = 0; kk + 2 < rT.size(); kk += 3)
    {
      for (int e = 0; e < 3; e++)
        {
          const long a = rT[kk + e], b = rT[kk + (e + 1) % 3];
          if ((a >= 0 && b >= 0) || a == b || Point(a).DistanceBetweenSquared(Point(b)) >= dMin2)
            { continue; }
          const long ra = Find(a), rb = Find(b);
          if (ra == rb || (ra >= 0 && rb >= 0))
            { continue; }
          if      (ra >= 0)  { sParent[~rb] = ra; }  // the loop vertex represents the set
          else if (rb >= 0)  { sParent[~ra] = rb; }
          else if (~ra < ~rb) { sParent[~rb] = ra; }  // else the lowest new vertex
          else               { sParent[~ra] = rb; }
          bMerged = TRUE;
        }
    }
  if (!bMerged)
    { return; }
  const SmBoolean bUVs = (rOut.sTriangleUVs.size() == rT.size()) ? TRUE : FALSE;
  size_t lOut = 0;
  for (size_t kk = 0; kk + 2 < rT.size(); kk += 3)
    {
      const long r0 = Find(rT[kk]), r1 = Find(rT[kk + 1]), r2 = Find(rT[kk + 2]);
      if (r0 == r1 || r1 == r2 || r2 == r0)
        { continue; }
      rT[lOut] = r0; rT[lOut + 1] = r1; rT[lOut + 2] = r2;
      if (bUVs)
        {
          for (int c = 0; c < 3; c++)
            { rOut.sTriangleUVs[lOut + c] = rOut.sTriangleUVs[kk + c]; }
        }
      lOut += 3;
    }
  rT.resize(lOut);
  if (bUVs)
    { rOut.sTriangleUVs.resize(lOut); }
}

// The normals of a face's triangle corners (surface normals, not reversed): a new vertex its own; a loop
// vertex the normal at the corner's UV, which differs between the sides of a seam where the surface
// creases (one evaluation per vertex and UV), or, where the surface is singular there (Su x Sv vanishes at
// a pole or an apex, and the normal depends on the direction), the normal just inside the triangle, a
// thousandth of the way to its UV centroid. Also one normal per loop vertex, as SMLib's output has, at the
// vertex's first use in the triangles (at its first loop occurrence for output without corner UVs,
// crFirstLoopUV).
void sm_TriangleNormals(const SmSurface * cpSurface, const SmFastFaceOutput & crOut,
                        const std::unordered_map<long, SmPoint2d> & crFirstLoopUV, SmBoolean bCorners,
                        std::vector<std::pair<long, SmVector3d>> & rLoopNormals, std::vector<SmVector3d> & rCornerNormals)
{
  const size_t    lCorners = crOut.sTriangles.size();
  const SmBoolean bUVs     = (crOut.sTriangleUVs.size() == lCorners) ? TRUE : FALSE;
  struct Normal
  {
    SmPoint2d  sUV;
    SmVector3d sNormal;
    SmBoolean  bSingular;
  };
  struct Seen
  {
    size_t              lIndex;   // in rLoopNormals: the first use
    std::vector<Normal> sAt;      // normals at its UVs, the first use's first
  };
  std::unordered_map<long, Seen> sSeen;
  auto Evaluate = [&](const SmPoint2d & crUV, SmBoolean & rbSingular)
    {
      SmPoint3d  sPoint;
      SmVector3d sSu, sSv, sNormal(0, 0, 0);
      rbSingular = FALSE;
      if (cpSurface->Evaluate1stDerivatives(crUV, TRUE, TRUE, sPoint, sSu, sSv) == SM_SUCCESS)
        {
          sNormal = sSu * sSv;
          const double dLength = sNormal.Length(), dU = sSu.Length(), dV = sSv.Length();
          // near a singular point too (the derivatives nearly parallel, or one nearly collapsed), where the
          // normal turns fast with the direction it is approached from
          rbSingular = (   !(dLength > SmSurface::GetNormalLimit()) || dLength < 1.0e-3 * dU * dV
                        || std::min(dU, dV) < 1.0e-6 * std::max(dU, dV)) ? TRUE : FALSE;
          if (dLength > SmSurface::GetNormalLimit())
            { return sNormal / dLength; }
        }
      else
        { rbSingular = TRUE; }
      if (cpSurface->EvaluateNormal(crUV, TRUE, TRUE, sNormal) != SM_SUCCESS)
        { sNormal = SmVector3d(0, 0, 0); }
      return sNormal;
    };
  auto Same = [](const SmPoint2d & a, const SmPoint2d & b) { return a.x == b.x && a.y == b.y; };
  // normals that differ by rounding only (the sides of a smooth seam): the vertex keeps one normal per face
  auto Close = [](const SmVector3d & a, const SmVector3d & b) { return (a - b).Length() <= 1.0e-9; };
  rLoopNormals.clear();
  rCornerNormals.assign(bCorners ? lCorners : 0, SmVector3d(0, 0, 0));
  for (size_t kk = 0; kk + 2 < lCorners; kk += 3)
    {
      const long * aR = &crOut.sTriangles[kk];
      if (aR[0] == aR[1] || aR[1] == aR[2] || aR[2] == aR[0])
        { continue; }  // dropped by the assembly
      for (int c = 0; c < 3; c++)
        {
          const long r = aR[c];
          SmVector3d sNormal(0, 0, 0);
          if (r < 0)
            { sNormal = crOut.sNewNormals[~r]; }
          else
            {
              SmPoint2d sUV;
              if (bUVs)
                { sUV = crOut.sTriangleUVs[kk + c]; }
              else
                {
                  auto sFirst = crFirstLoopUV.find(r);
                  if (sFirst != crFirstLoopUV.end()) { sUV = sFirst->second; }
                }
              auto sFound = sSeen.find(r);
              const Normal * cpAt = NULL;
              if (sFound != sSeen.end())
                {
                  for (const Normal & crAt : sFound->second.sAt)
                    {
                      if (Same(crAt.sUV, sUV)) { cpAt = &crAt; break; }
                    }
                }
              Normal sAt;
              if (cpAt == NULL)
                {
                  sAt.sUV     = sUV;
                  sAt.sNormal = Evaluate(sUV, sAt.bSingular);
                  if (sFound == sSeen.end())
                    {
                      sFound = sSeen.emplace(r, Seen{rLoopNormals.size(), {}}).first;
                      rLoopNormals.push_back({r, sAt.sNormal});
                    }
                  else if (Close(sAt.sNormal, rLoopNormals[sFound->second.lIndex].second))
                    { sAt.sNormal = rLoopNormals[sFound->second.lIndex].second; }   // a smooth seam: one normal
                  sFound->second.sAt.push_back(sAt);
                  cpAt = &sFound->second.sAt.back();
                }
              sNormal = cpAt->sNormal;
              if (cpAt->bSingular && bUVs && bCorners)
                {
                  const SmPoint2d & a = crOut.sTriangleUVs[kk], & b = crOut.sTriangleUVs[kk + 1], & d = crOut.sTriangleUVs[kk + 2];
                  const SmPoint2d sInside(sUV.x + 1.0e-3 * ((a.x + b.x + d.x) / 3.0 - sUV.x),
                                          sUV.y + 1.0e-3 * ((a.y + b.y + d.y) / 3.0 - sUV.y));
                  SmBoolean bSingular = FALSE;
                  const SmVector3d sInsideNormal = Evaluate(sInside, bSingular);
                  if (!bSingular)
                    { sNormal = Close(sInsideNormal, sNormal) ? sNormal : sInsideNormal; }
                }
            }
          if (bCorners)
            { rCornerNormals[kk + c] = sNormal; }
        }
    }
}

// sm_TriangleNormals() of a face of the Phase 1 path
SmStatus sm_FaceNormals(SmFastFaceWork & rWork, SmBoolean bCorners)
{
  std::unordered_map<long, SmPoint2d> sFirstLoopUV;
  if (rWork.sOut.sTriangleUVs.size() != rWork.sOut.sTriangles.size())
    {
      for (const auto & crLoop : rWork.sLoops)
        {
          for (const auto & crV : crLoop)
            {
              const SmPoint3d sUV = crV.first->GetPoint();
              sFirstLoopUV.emplace(crV.second, SmPoint2d(sUV.x, sUV.y));
            }
        }
    }
  sm_TriangleNormals(rWork.pFace->GetSurface(), rWork.sOut, sFirstLoopUV, bCorners, rWork.sLoopNormals, rWork.sCornerNormals);
  return SM_SUCCESS;
}

// an edge between two vertex ids, either way round
inline uint64_t sm_EdgeKey(long a, long b)
{
  return (a < b) ? (((uint64_t)a << 32) | (uint64_t)b) : (((uint64_t)b << 32) | (uint64_t)a);
}

// the loop segments of a face, sorted
void sm_LoopSegments(const SmFastFaceWork & crWork, std::vector<uint64_t> & rSegments)
{
  rSegments.clear();
  for (const auto & crLoop : crWork.sLoops)
    {
      for (size_t kk = 0; kk < crLoop.size(); kk++)
        {
          const long a = crLoop[kk].second, b = crLoop[(kk + 1) % crLoop.size()].second;
          if (a != b)
            { rSegments.push_back(sm_EdgeKey(a, b)); }
        }
    }
  std::sort(rSegments.begin(), rSegments.end());
}

} // end anonymous namespace

/*******************************************************************//**
PURPOSE: Tessellate a Brep into one triangle mesh, fast faces in parallel,
   other faces through SMLib's Phase 2 (see SmTessFast.h).
***********************************************************************/
SmStatus SmTessellateFast
 (const SmBrep               * cpBrep,
  const SmTessellationParams & crParams,
  SmFastMesh                 & rMesh,
  const SmFastOptions        & crOptions)
{
  rMesh = SmFastMesh();
  NER(cpBrep);
  sm_ContextSerializer sSerial(cpBrep->GetContext());
  if (!sSerial.Acquired())
    { return SM_ERR; }   // called back from another context's call while another thread has this context

  // the boundaries of SmComputeFastBoundaries() first; a Brep with a face they can't do takes Phase 1
  if (crOptions.bFastBoundaries && crOptions.bFastFaces)
    {
      const SmStatus eBoundaries = SmTessellateWithFastBoundaries(cpBrep, crParams, rMesh, crOptions);
      if (eBoundaries == SM_SUCCESS)
        { return SM_SUCCESS; }
      const double dBoundariesMs = rMesh.dBoundariesMs;
      const long   lFallbackFace = rMesh.lFallbackFace;
      rMesh = SmFastMesh();
      rMesh.bWholeFallback   = TRUE;
      rMesh.dBoundariesMs    = dBoundariesMs;
      rMesh.lFallbackFace    = lFallbackFace;
    }

  const sm_Clock::time_point sStart = sm_Clock::now();
  SmBrep * pBrep = SM_CONST_CAST(SmBrep *, cpBrep);

  // parameters as SmApiTessellate() and SmBrep::ConvertToPolyBrep() use them
  const double dChordHeightTol     = crParams.dChordHeightTol;
  const double dCurveAngleTolDeg   = crParams.dCurveAngleTolDeg;
  const double dSurfaceAngleTolDeg = crParams.dSurfaceAngleTolDeg;
  double       dMaxEdgeLength      = crParams.dMaxEdgeLength;
  const double dMaxAspectRatio     = crParams.dMaxAspectRatio;
  if (dMaxEdgeLength < 0.0)
    {
      SmExtent3d sBox;
      pBrep->CalculateBoundingBox(sBox);
      double dSize = sBox.GetSize().Length();
      if (dSize >= 1.0e15 || dSize <= 0.0) dSize = 1.0;
      dMaxEdgeLength = -0.025 * dSize * dMaxEdgeLength;
    }
  SmTessellationParams sParams = crParams;
  sParams.dMaxEdgeLength = dMaxEdgeLength;

  const SmContext & crContext = *pBrep->GetContext();
  SmTemporaryChangeValue<SmBoolean> sDoingBool(SM_CONST_CAST(SmContext *, &crContext)->GetDoingBooleanRef(), TRUE);
  SmCurveTessDriver   sCrvTess(0, dChordHeightTol, dCurveAngleTolDeg, dMaxEdgeLength, 0.0001, FALSE);
  SmFastSurfaceTessDriver sFastSrfTess(dChordHeightTol, dSurfaceAngleTolDeg, dMaxEdgeLength, 0.0, 0.001, dMaxAspectRatio);
  SmSurfaceTessDriver     sPlainSrfTess(dChordHeightTol, dSurfaceAngleTolDeg, dMaxEdgeLength, 0.0, 0.001, dMaxAspectRatio);
  SmSurfaceTessDriver   & sSrfTess = crOptions.bFastRefinement ? (SmSurfaceTessDriver &)sFastSrfTess : sPlainSrfTess;
  SmStatus eResult = SM_SUCCESS;
  sm_Clock::time_point sCleanupStart;
    {
      SmTess sTess(crContext, sCrvTess, sSrfTess);
      sTess.SetAdvancingFront(FALSE);

      // Phase 1 on a copy, which sTess owns and deletes
      SmBrep * pCopy = new (crContext) SmBrep(*pBrep);
      const SmBoolean bThreads = sm_BrepThreadsSafe(pCopy);
      SmBoolean          bFailedFaces = FALSE;
      SmTArray<SmStatus> sStatuses;
      SER(sTess.Phase1SetupBrep(pCopy, bFailedFaces, NULL, &sStatuses));
      const sm_Clock::time_point sPhase1End = sm_Clock::now();
      rMesh.dPhase1Ms = sm_Ms(sStart, sPhase1End);

      SmTArray<SmFace *> sFaces;
      pCopy->GetFaces(sFaces);
      const ULONG lFaces = sFaces.GetSize();
      rMesh.lFaces = lFaces;

      // vertex ids of the Phase 1 loops, in face and loop order: one per SmVertex, and one per
      // PolyVertex without an original vertex
      std::vector<SmFastFaceWork>         sWork(lFaces);
      std::unordered_map<SmVertex *, long> sVertexIds;
      std::vector<SmPoint3d>              sLoopPoints;   // per id
      SmTArray<SmPolyLoop *> sPolyLoops;
      SmTArray<SmPolyEdge *> sPolyEdges;
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          SmFastFaceWork & rWork = sWork[ii];
          rWork.pFace = sFaces[ii];
          rWork.pSC   = sTess.GetTessSrfCacheOfFace(rWork.pFace);
          if (ii < sStatuses.GetSize() && sStatuses[ii] != SM_SUCCESS)
            { rWork.pSC = NULL; }
          SmPolyBrep * pPolyBrep = rWork.pSC ? rWork.pSC->GetTS_PolyBrep() : NULL;
          SmPolyFace * pPolyFace = pPolyBrep ? pPolyBrep->GetFirstPolyFace() : NULL;
          if (pPolyFace == NULL)
            {
              rWork.pSC = NULL;
              continue;
            }
          SmFaceuse * pFU = rWork.pFace->GetUpwardFaceuse();
          rWork.bReversed = (pFU->GetShell()->GetRegion() != rWork.pFace->GetBrep()->GetInfiniteRegion()) ? TRUE : FALSE;
          SmSurface * pSurface = rWork.pFace->GetSurface();
          SmBSplineSurface * pBSS = SM_CAST_PTR(SmBSplineSurface, pSurface);
          if (pBSS != NULL)
            {
              pBSS->GetOrCreateGwNurbPointer();  // built before sharing the surface with worker threads
              rWork.bParallel = bThreads;
            }
          pPolyFace->GetPolyLoops(sPolyLoops);
          for (ULONG jj = 0; jj < sPolyLoops.GetSize(); jj++)
            {
              sPolyLoops[jj]->GetPolyEdges(sPolyEdges);
              rWork.sLoops.emplace_back();
              auto & rLoop = rWork.sLoops.back();
              for (ULONG kk = 0; kk < sPolyEdges.GetSize(); kk++)
                {
                  SmPolyVertex * pPV       = sPolyEdges[kk]->GetStartPolyVertex();
                  SmVertex     * pOriginal = pPV->GetOriginalVertex();
                  long lId;
                  if (pOriginal != NULL)
                    {
                      auto sResult = sVertexIds.emplace(pOriginal, (long)sLoopPoints.size());
                      lId = sResult.first->second;
                      if (sResult.second)
                        { sLoopPoints.push_back(pOriginal->GetPoint()); }
                    }
                  else
                    {
                      lId = (long)sLoopPoints.size();
                      const SmPoint3d sUV = pPV->GetPoint();
                      SmPoint3d sPoint;
                      if (pSurface->EvaluatePoint(SmPoint2d(sUV.x, sUV.y), sPoint) != SM_SUCCESS)
                        { sPoint = SmPoint3d(0, 0, 0); }
                      sLoopPoints.push_back(sPoint);
                    }
                  rLoop.push_back({pPV, lId});
                }
            }
        }

      // fast triangulation, faces in parallel (surfaces other than BSpline ones serially). The trimmed-
      // face triangulator doesn't follow SMLib with a maximum edge length or aspect ratio (triangle
      // counts -65% to +94% on the benchmark models), so it only runs without them.
      const SmBoolean bTrimmedFaces = (crOptions.bTrimmedFaces && dMaxEdgeLength <= 0.0 && dMaxAspectRatio <= 0.0) ? TRUE : FALSE;
      const double    dTolerance    = pCopy->GetTolerance();  // as sm_CopyPhase2Triangles() and SmTess::OutputToPolyBrep()
      auto FastFace = [&](ULONG ii)
        {
          SmFastFaceWork & rWork = sWork[ii];
          if (rWork.pSC == NULL)
            { return; }
          if (!crOptions.bFastFaces)
            { return; }
          SmFastFaceInput sIn;
          sIn.cpFace     = rWork.pFace;
          sIn.cpSurface  = rWork.pFace->GetSurface();
          sIn.sParams    = sParams;
          sIn.cpCache    = rWork.pSC;
          sIn.bReversed  = rWork.bReversed;
          sIn.dTolerance = rWork.pFace->GetTolerance();
          sIn.bParallelSafe = rWork.bParallel;
          sIn.sLoops.resize(rWork.sLoops.size());
          for (size_t jj = 0; jj < rWork.sLoops.size(); jj++)
            {
              auto & rLoop = sIn.sLoops[jj];
              rLoop.resize(rWork.sLoops[jj].size());
              for (size_t kk = 0; kk < rLoop.size(); kk++)
                {
                  const auto      & crV = rWork.sLoops[jj][kk];
                  const SmPoint3d   sUV = crV.first->GetPoint();
                  rLoop[kk].dU     = sUV.x;
                  rLoop[kk].dV     = sUV.y;
                  rLoop[kk].sPoint = sLoopPoints[crV.second];
                  rLoop[kk].lId    = crV.second;
                }
            }
          const SmStatus eFast = bTrimmedFaces ? SmTriangulateFastFace(sIn, rWork.sOut)
                                               : SmTriangulateRectangularFace(sIn, rWork.sOut);
          if (eFast == SM_SUCCESS && rWork.sOut.eStatus == SM_SUCCESS)
            {
              sm_CollapseShortEdges(rWork.sOut, sLoopPoints, dTolerance / 100.0, rWork.bParallel);
              rWork.bFast = TRUE;
              rWork.bDone = TRUE;
            }
        };
      std::vector<ULONG> sParallelFaces, sSerialFaces;
      for (ULONG ii = 0; ii < lFaces; ii++)
        { (sWork[ii].bParallel ? sParallelFaces : sSerialFaces).push_back(ii); }
      sm_ParallelFor(tbb::blocked_range<size_t>(0, sParallelFaces.size(), 1), [&](const tbb::blocked_range<size_t> & crRange)
        {
          for (size_t kk = crRange.begin(); kk != crRange.end(); ++kk)
            { FastFace(sParallelFaces[kk]); }
        });
      for (ULONG ii : sSerialFaces)
        { FastFace(ii); }

      // neighbor fallback, first part: the fast faces that share a loop segment with a face rejected for
      // its loops go to SMLib too, so that SMLib's triangles of the broken face (which may cut across a
      // neighbor) meet SMLib's triangles of its neighbors, as in SMLib's output
      if (crOptions.bNeighborFallback)
        {
          std::vector<std::pair<uint64_t, ULONG>> sSegmentFaces;
          std::vector<uint64_t> sSegments;
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              sm_LoopSegments(sWork[ii], sSegments);
              for (uint64_t lKey : sSegments)
                { sSegmentFaces.push_back({lKey, ii}); }
            }
          std::sort(sSegmentFaces.begin(), sSegmentFaces.end());
          std::vector<char> sDemote(lFaces, 0);
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              const SmFastFaceWork & crWork = sWork[ii];
              if (crWork.pSC == NULL || crWork.bFast || crWork.sOut.eFailure != SM_FF_LOOPS)
                { continue; }
              sm_LoopSegments(crWork, sSegments);
              for (uint64_t lKey : sSegments)
                {
                  for (auto it = std::lower_bound(sSegmentFaces.begin(), sSegmentFaces.end(), std::make_pair(lKey, (ULONG)0));
                       it != sSegmentFaces.end() && it->first == lKey; ++it)
                    {
                      if (it->second != ii && sWork[it->second].bFast)
                        { sDemote[it->second] = 1; }
                    }
                }
            }
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              if (!sDemote[ii])
                { continue; }
              sWork[ii].bFast = FALSE;
              sWork[ii].bDone = FALSE;
              sWork[ii].sOut  = SmFastFaceOutput();
              rMesh.lDemotedFaces++;
            }
        }
      const sm_Clock::time_point sFacesEnd = sm_Clock::now();
      rMesh.dFacesMs = sm_Ms(sPhase1End, sFacesEnd);

      // fallback: SMLib's Phase 2 for the other faces, serially
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          SmFastFaceWork & rWork = sWork[ii];
          if (rWork.pSC == NULL || rWork.bDone)
            { continue; }
          if (   sTess.Phase2CreateFacePolygons(rWork.pFace) == SM_SUCCESS
              && sm_CopyPhase2Triangles(rWork, sVertexIds, sLoopPoints) == SM_SUCCESS)
            { rWork.bDone = TRUE; }
          // Phase2CreateFacePolygons() deletes the face's cache when it fails
          rWork.pSC = sTess.GetTessSrfCacheOfFace(rWork.pFace);
        }
      // normals of the loop vertices, and of the triangle corners when asked
      sm_ParallelFor(tbb::blocked_range<size_t>(0, sParallelFaces.size(), 1), [&](const tbb::blocked_range<size_t> & crRange)
        {
          for (size_t kk = crRange.begin(); kk != crRange.end(); ++kk)
            { if (sWork[sParallelFaces[kk]].bDone) sm_FaceNormals(sWork[sParallelFaces[kk]], crOptions.bCornerNormals); }
        });
      for (ULONG ii : sSerialFaces)
        { if (sWork[ii].bDone) sm_FaceNormals(sWork[ii], crOptions.bCornerNormals); }

      // neighbor fallback, second part: a fast face with a chord (an edge between two loop vertices that
      // isn't one of its loop segments) that more than two triangles of the mesh use goes to SMLib, where
      // SMLib's triangles of a face next to it cut the same chord; repeated while that demotes faces
      for (int iPass = 0; crOptions.bNeighborFallback && iPass < 4; iPass++)
        {
          auto Degenerate = [](const long * aR) { return aR[0] == aR[1] || aR[1] == aR[2] || aR[2] == aR[0]; };
          std::vector<uint64_t> sEdges;  // new vertices are the face's own: their edges can't be shared
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              if (!sWork[ii].bDone)
                { continue; }
              const std::vector<long> & crT = sWork[ii].sOut.sTriangles;
              for (size_t kk = 0; kk + 2 < crT.size(); kk += 3)
                {
                  if (Degenerate(&crT[kk]))
                    { continue; }
                  for (int e = 0; e < 3; e++)
                    {
                      const long a = crT[kk + e], b = crT[kk + (e + 1) % 3];
                      if (a >= 0 && b >= 0)
                        { sEdges.push_back(sm_EdgeKey(a, b)); }
                    }
                }
            }
          std::sort(sEdges.begin(), sEdges.end());
          std::vector<ULONG>    sDemoted;
          std::vector<uint64_t> sSegments;
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              if (!sWork[ii].bFast)
                { continue; }
              sm_LoopSegments(sWork[ii], sSegments);
              const std::vector<long> & crT = sWork[ii].sOut.sTriangles;
              bool bConflict = false;
              for (size_t kk = 0; kk + 2 < crT.size() && !bConflict; kk += 3)
                {
                  if (Degenerate(&crT[kk]))
                    { continue; }
                  for (int e = 0; e < 3 && !bConflict; e++)
                    {
                      const long a = crT[kk + e], b = crT[kk + (e + 1) % 3];
                      if (a < 0 || b < 0)
                        { continue; }
                      const uint64_t lKey = sm_EdgeKey(a, b);
                      if (std::binary_search(sSegments.begin(), sSegments.end(), lKey))
                        { continue; }
                      const auto sRange = std::equal_range(sEdges.begin(), sEdges.end(), lKey);
                      bConflict = (sRange.second - sRange.first) > 2;
                    }
                }
              if (bConflict)
                { sDemoted.push_back(ii); }
            }
          if (sDemoted.empty())
            { break; }
          for (ULONG ii : sDemoted)
            {
              SmFastFaceWork & rWork = sWork[ii];
              rWork.bFast = FALSE;
              rWork.bDone = FALSE;
              rWork.sOut  = SmFastFaceOutput();
              rMesh.lDemotedFaces++;
              if (   sTess.Phase2CreateFacePolygons(rWork.pFace) == SM_SUCCESS
                  && sm_CopyPhase2Triangles(rWork, sVertexIds, sLoopPoints) == SM_SUCCESS)
                {
                  rWork.bDone = TRUE;
                  sm_FaceNormals(rWork, crOptions.bCornerNormals);
                }
              rWork.pSC = sTess.GetTessSrfCacheOfFace(rWork.pFace);
            }
        }
      const sm_Clock::time_point sFallbackEnd = sm_Clock::now();
      rMesh.dFallbackMs = sm_Ms(sFacesEnd, sFallbackEnd);

      // assembly: loop vertices first (by id), then each face's new vertices
      const long lLoopVerts = (long)sLoopPoints.size();
      std::vector<long> sFaceBase(lFaces + 1, lLoopVerts);
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          const SmFastFaceWork & crWork = sWork[ii];
          sFaceBase[ii + 1] = sFaceBase[ii] + (crWork.bDone ? (long)crWork.sOut.sNewPoints.size() : 0);
          if (crWork.bDone) { (crWork.bFast ? rMesh.lFastFaces : rMesh.lFallbackFaces)++; }
          else              { rMesh.lFailedFaces++; }
          rMesh.sFaceKinds.push_back((int8_t)(!crWork.bDone ? 0 : (crWork.bFast ? 1 : 2)));
        }
      if (rMesh.lFailedFaces > 0 && !crOptions.bAllowPartial)
        { eResult = SM_ERR; }

      const long lVerts = sFaceBase[lFaces];
      if (lVerts < 0 || lVerts > (long)INT32_MAX)   // the arrays' indices are 32-bit
        { SER_MSG(SM_ERR, _T("SmTessellateFast: more vertices than 32-bit indices hold")); }
      rMesh.sPoints.resize((size_t)lVerts * 3);
      std::vector<SmVector3d> sNormalSums((size_t)lVerts, SmVector3d(0, 0, 0));
      for (long ii = 0; ii < lLoopVerts; ii++)
        {
          rMesh.sPoints[3 * (size_t)ii]     = sLoopPoints[ii].x;
          rMesh.sPoints[3 * (size_t)ii + 1] = sLoopPoints[ii].y;
          rMesh.sPoints[3 * (size_t)ii + 2] = sLoopPoints[ii].z;
        }
      size_t lTriangles = 0;
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          if (sWork[ii].bDone)
            { lTriangles += sWork[ii].sOut.sTriangles.size() / 3; }
        }
      rMesh.sTriangles.reserve(lTriangles * 3);
      rMesh.sTriangleFaces.reserve(lTriangles);
      if (crOptions.bCornerNormals)
        { rMesh.sCornerNormals.reserve(lTriangles * 9); }
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          const SmFastFaceWork & crWork = sWork[ii];
          if (!crWork.bDone)
            { continue; }
          const SmFastFaceOutput & crOut = crWork.sOut;
          const double dSign = crWork.bReversed ? -1.0 : 1.0;
          const long   lBase = sFaceBase[ii];
          for (size_t kk = 0; kk < crOut.sNewPoints.size(); kk++)
            {
              const size_t lV = (size_t)lBase + kk;
              rMesh.sPoints[3 * lV]     = crOut.sNewPoints[kk].x;
              rMesh.sPoints[3 * lV + 1] = crOut.sNewPoints[kk].y;
              rMesh.sPoints[3 * lV + 2] = crOut.sNewPoints[kk].z;
              sNormalSums[lV] = crOut.sNewNormals[kk] * dSign;
            }
          for (const auto & crN : crWork.sLoopNormals)
            { sNormalSums[crN.first] += crN.second * dSign; }
          const std::vector<long> & crTris = crOut.sTriangles;
          for (size_t kk = 0; kk + 2 < crTris.size(); kk += 3)
            {
              long aV[3];
              for (int c = 0; c < 3; c++)
                {
                  const long r = crTris[kk + c];
                  aV[c] = (r >= 0) ? r : lBase + ~r;
                }
              if (aV[0] == aV[1] || aV[1] == aV[2] || aV[2] == aV[0])
                {
                  rMesh.lDroppedTriangles++;
                  continue;
                }
              if (crWork.bReversed)
                { std::swap(aV[1], aV[2]); }
              for (int c = 0; c < 3; c++)
                { rMesh.sTriangles.push_back((int32_t)aV[c]); }
              rMesh.sTriangleFaces.push_back((int32_t)ii);
              if (crOptions.bCornerNormals)
                {
                  int aC[3] = {0, 1, 2};
                  if (crWork.bReversed)
                    { std::swap(aC[1], aC[2]); }
                  for (int c = 0; c < 3; c++)
                    {
                      const SmVector3d sN = crWork.sCornerNormals[kk + aC[c]] * dSign;
                      rMesh.sCornerNormals.push_back((float)sN.x);
                      rMesh.sCornerNormals.push_back((float)sN.y);
                      rMesh.sCornerNormals.push_back((float)sN.z);
                    }
                }
            }
        }
      // drop vertices no triangle uses (as SMLib deletes standalone PolyVertices), keeping the order
      std::vector<int32_t> sNewIndex((size_t)lVerts, -1);
      for (int32_t lV : rMesh.sTriangles)
        { sNewIndex[lV] = 0; }
      size_t lUsed = 0;
      for (size_t ii = 0; ii < (size_t)lVerts; ii++)
        {
          if (sNewIndex[ii] < 0)
            { continue; }
          sNewIndex[ii] = (int32_t)lUsed;   // lUsed <= ii < lVerts <= INT32_MAX
          if (lUsed != ii)
            {
              for (size_t a = 0; a < 3; a++) { rMesh.sPoints[3 * lUsed + a] = rMesh.sPoints[3 * ii + a]; }
              sNormalSums[lUsed] = sNormalSums[ii];
            }
          lUsed++;
        }
      for (int32_t & rV : rMesh.sTriangles)
        { rV = sNewIndex[rV]; }
      rMesh.sPoints.resize(lUsed * 3);
      rMesh.sNormals.resize(lUsed * 3);
      for (size_t ii = 0; ii < lUsed; ii++)
        {
          SmVector3d sN = sNormalSums[ii];
          const double dLength = sN.Length();
          if (std::isfinite(dLength) && dLength > 1e-30) { sN /= dLength; }
          else                                           { sN = SmVector3d(0, 0, 1); }
          rMesh.sNormals[3 * ii]     = (float)sN.x;
          rMesh.sNormals[3 * ii + 1] = (float)sN.y;
          rMesh.sNormals[3 * ii + 2] = (float)sN.z;
        }
      sCleanupStart = sm_Clock::now();
      rMesh.dAssemblyMs = sm_Ms(sFallbackEnd, sCleanupStart);
    } // sTess deletes the Phase 1 data and the copy
  rMesh.dCleanupMs = sm_Ms(sCleanupStart, sm_Clock::now());
  return eResult;

} // end SmTessellateFast

/*******************************************************************//**
   Fast refinement: SmTessSrfCache::BuildTreeFast()
***********************************************************************/
namespace {

// SmSurfaceCache.cpp sm_CullArray(): halve an array, keeping its first and last element
void sm_FastCullArray(SmTArray<double> & rArr)
{
  ULONG ii, jj, lSize = rArr.GetSize();
  for (ii = 2, jj = 1; ii < lSize; ii += 2, jj += 1)
    { rArr[jj] = rArr[ii]; }
  if (ii == lSize)
    { rArr[jj] = rArr[lSize - 1]; jj++; }
  rArr.SetSize(jj);
}

// SmSurfaceCache.cpp sm_ComputeBlockSplits(): the U and V splits of a SubdivideByBlock() block
SmStatus sm_FastBlockSplits(const SmSurface  * pSurface,
                            double             dChordHeightTolerance,
                            double             dAngTolRad,
                            double             dMaximumSideLength3D,
                            SmTArray<double> & rUSplits,
                            SmTArray<double> & rVSplits,
                            SmBoolean        & rbNoSplits)
{
  rbNoSplits = FALSE;
  SmExtent2d sUVDomain = pSurface->GetNaturalUVDomain();
  SmBoolean bCheckDiagonals = FALSE;
  SM_TYPE eSurfaceType = pSurface->GetType();
  if (eSurfaceType == SmBSplineSurface_TYPE || eSurfaceType == SmSurfOfExtrusion_TYPE)
    {
      double dTol = dChordHeightTolerance > 0.0 ? dChordHeightTolerance / 2.0 : SM_EFF_ZERO_SQRT;
      SmIsoCurve sU0(*pSurface, SM_SP_U, sUVDomain.GetMin().x, TRUE);
      if (sU0.IsLinear(dTol))
        {
          SmIsoCurve sV0(*pSurface, SM_SP_V, sUVDomain.GetMin().y, TRUE);
          if (sV0.IsLinear(dTol))
            {
              SmIsoCurve sU1(*pSurface, SM_SP_U, sUVDomain.GetMax().x, TRUE);
              if (sU1.IsLinear(dTol))
                {
                  SmIsoCurve sV1(*pSurface, SM_SP_V, sUVDomain.GetMax().y, TRUE);
                  if (sV1.IsLinear(dTol))
                    { bCheckDiagonals = TRUE; }
                }
            }
        }
    }
  SER(pSurface->FindTessellationSplits(SM_SP_U, sUVDomain, dChordHeightTolerance, dAngTolRad * 180.0 / SM_PI,
                                       dMaximumSideLength3D, rVSplits, bCheckDiagonals));
  SER(pSurface->FindTessellationSplits(SM_SP_V, sUVDomain, dChordHeightTolerance, dAngTolRad * 180 / SM_PI,
                                       dMaximumSideLength3D, rUSplits, bCheckDiagonals));
  static constexpr ULONG slMaxTotalFacets = 100000;
  ULONG lNumSplitsU = rUSplits.GetSize();
  ULONG lNumSplitsV = rVSplits.GetSize();
  ULONG lTotal = lNumSplitsU * lNumSplitsV;
  if (lNumSplitsU <= 2 && lNumSplitsV <= 2)
    {
      rbNoSplits = TRUE;
      return SM_SUCCESS;
    }
  while (lTotal > slMaxTotalFacets)
    {
      if (lNumSplitsU > 20)
        {
          sm_FastCullArray(rUSplits);
          lNumSplitsU = rUSplits.GetSize();
        }
      if (lNumSplitsV > 20)
        {
          sm_FastCullArray(rVSplits);
          lNumSplitsV = rVSplits.GetSize();
        }
      lTotal = lNumSplitsU * lNumSplitsV;
    }
  return SM_SUCCESS;
}

// sm_ParallelInvoke() inside an sm_RunIsolated() region, without isolating the two tasks again, so that
// threads waiting for a task keep running the region's other tasks: they run with the given tolerance
// defaults, and their diagnostics are reported on the calling thread
template <class F1, class F2>
void sm_ForkInRegion(const SmTol::ThreadDefaults & crDefaults, const F1 & crF1, const F2 & crF2)
{
  SmTaskDiagnostics sDiagnostics;
  tbb::parallel_invoke([&]
                         {
                           SmThreadDefaultsScope sScope(crDefaults);
                           SmDiagnosticCapture   sCapture(sDiagnostics);
                           crF1();
                           sDiagnostics.Add(0, sCapture);
                         },
                       [&]
                         {
                           SmThreadDefaultsScope sScope(crDefaults);
                           SmDiagnosticCapture   sCapture(sDiagnostics);
                           crF2();
                           sDiagnostics.Add(1, sCapture);
                         });
  sDiagnostics.Report();
}

} // end anonymous namespace

/*******************************************************************//**
PURPOSE: BuildTreeWithSubdivision() for SmTessellateFast(): the same tree,
   built faster.

NOTES: Makes the tests (SmSurfaceCache::TestAgainstTolerances() for
   SM_AD_AUX_DATA nodes), node splits (SubdivideNode()) and blocks
   (SubdivideByBlock()) of the serial SubdivideToTolerances() recursion,
   in its depth first order and with its m_bSwitchToFastSubdivision
   value, so the tree is the same. What tessellation doesn't use isn't
   computed: node bounding boxes, and the pseudo boxes when no chord
   height tolerance is set. Tests and splits run on worker threads first
   (as in SubdivideToTolerancesParallel(), without exploring under a
   node whose first child ends the recursion with the switch set), the
   serial pass only decides, then blocks run on worker threads, and the
   tree is built from the results. Subdivision surfaces are deleted at
   the end; node surfaces point at the cache surface, as after the
   clean-up in BuildTree().
***********************************************************************/
SmStatus SmTessSrfCache::BuildTreeFast(SmBSplineSurface * pBSS)
{
  NER(pBSS);

  // results of the recursion per node
  enum { SM_FR_LEAF = 0, SM_FR_SPLIT = 1, SM_FR_BLOCK = 2 };
  struct Node
  {
    SmExtent2d                sDomain;
    gw_SURFACE              * pSur      = NULL;   // subdivision surface (NLib data), owned but at the root
    SmBoolean                 bChordIn  = FALSE;  // tolerance states inherited from the parent
    SmBoolean                 bAngleIn  = FALSE;
    SmStatus                  eTest     = SM_ERR; // TestAgainstTolerances() results
    SmBoolean                 bNeedsSub = FALSE;
    SmBoolean                 bSetFast  = FALSE;
    SmBoolean                 bChordOut = FALSE;
    SmBoolean                 bAngleOut = FALSE;
    SmSurfParamType           eDir      = SM_SP_U;
    SmBoolean                 bSplit    = FALSE;  // SubdivideNode() results
    Node                    * pChild[2] = {NULL, NULL};  // in the node pool
    int                       eKind     = SM_FR_LEAF;  // decided by the serial pass
    SmStatus                  eBlock    = SM_ERR;      // SubdivideByBlock() results
    SmBoolean                 bNoSplits = FALSE;
    SmTArray<double>          sUSplits, sVSplits;
    SmBoolean                 bOwned    = FALSE;
    void FreeSurface() { if (bOwned && pSur != NULL) { smos_Free(pSur); pSur = NULL; } }
    ~Node() { FreeSurface(); }
  };

  // nodes in chunks, one pool per thread that splits
  struct Pool
  {
    std::vector<std::unique_ptr<Node[]>> sChunks;
    size_t                               lUsed = 0;  // in the last chunk
    Node * New()
    {
      static constexpr size_t slChunk = 256;
      if (sChunks.empty() || lUsed == slChunk)
        {
          sChunks.emplace_back(new Node[slChunk]);
          lUsed = 0;
        }
      return &sChunks.back()[lUsed++];
    }
  };

  struct Worker
  {
    SmTessSrfCache      * pCache;
    SmBoolean             bBlocksAllowed;
    SmBoolean             bParallel;   // worker tasks: their temporary surfaces notify the context's callbacks
    SmZoneTol3d           sZoneTol3d;
    SmTol::ThreadDefaults sDefaults;  // of the calling thread, for the tasks
    tbb::enumerable_thread_specific<Pool> * pPool;  // nodes, per thread that splits
    const SmFastTessSrfCache * cpFast;  // the cache of SmComputeFastBoundaries(), with its region, or NULL


    // SmSurfaceCache::TestAgainstTolerances() of an SM_AD_AUX_DATA node
    void Test(Node & rN) const
    {
      SmTessSrfCache & rC = *pCache;
      rN.bNeedsSub = FALSE;
      rN.bSetFast  = FALSE;
      rN.eDir      = SM_SP_U;
      // a node away from the face (SmFastTessSrfCache::m_sFastRegion) stays a leaf
      if (cpFast != NULL && cpFast->m_bFastRegion)
        {
          SmBoolean bMeets = TRUE;
          for (int d = 0; d < 2 && bMeets; d++)
            {
              const double dLo = (d == 0) ? rN.sDomain.GetMin().x : rN.sDomain.GetMin().y;
              const double dHi = (d == 0) ? rN.sDomain.GetMax().x : rN.sDomain.GetMax().y;
              const double dRLo = (d == 0) ? cpFast->m_sFastRegion.GetMin().x : cpFast->m_sFastRegion.GetMin().y;
              const double dRHi = (d == 0) ? cpFast->m_sFastRegion.GetMax().x : cpFast->m_sFastRegion.GetMax().y;
              const double dP   = cpFast->m_adFastPeriod[d];
              bMeets = (dHi >= dRLo && dLo <= dRHi) ? TRUE : FALSE;
              if (!bMeets && dP > 0.0)
                { bMeets = ((dHi >= dRLo - dP && dLo <= dRHi - dP) || (dHi >= dRLo + dP && dLo <= dRHi + dP)) ? TRUE : FALSE; }
            }
          if (!bMeets)
            {
              rN.eTest     = SM_SUCCESS;
              rN.bChordOut = rN.bChordIn;
              rN.bAngleOut = rN.bAngleIn;
              return;
            }
        }
      SmBoolean bChordSat = rN.bChordIn, bAngleSat = rN.bAngleIn;
      const double dChordHeightSquared = rC.m_dChordHeightTolerance * rC.m_dChordHeightTolerance;
      const double dAngTolDeg          = rC.m_dAngTolRad * 180 / SM_PI;
      if (dChordHeightSquared < SM_EFF_ZERO) { bChordSat = TRUE; }
      if (dAngTolDeg          < SM_EFF_ZERO) { bAngleSat = TRUE; }
      const SmBoolean bBothSatisfied = bChordSat && bAngleSat;
      gw_SURFACE * pSur = rN.pSur;
      SmZoneTol3d  sZone = sZoneTol3d;
      SmBoolean       bSubdivide = FALSE;
      SmSurfParamType eDir       = SM_SP_U;
      rN.eTest = SM_SUCCESS;

      // chord height, on the pseudo box (only computed when a chord height tolerance is set)
      if (dChordHeightSquared >= SM_EFF_ZERO)
        {
          SmPseudoBox sPSBox;
          if (!bBothSatisfied)
            {
              SmBSplineSurface sSurface(pSur, TRUE, rC.GetContext());  // borrows the NLib data
              SmExtent3d       sBox;
              if ((rN.eTest = sSurface.CalculateBoundingBox(rN.sDomain, &sBox, &sPSBox)) != SM_SUCCESS)
                { return; }
            }
          SmExtent1d sIvl1, sIvl2, sIvl3;
          sPSBox.GetIntervals(sIvl1, sIvl2, sIvl3);
          double dHeight = sIvl3.GetLength();
          if (dHeight * dHeight <= dChordHeightSquared)
            { bChordSat = TRUE; }
          else
            {
              bSubdivide = TRUE;
              double dUChordHeight, dVChordHeight, dUVChordHeight;
              if ((rN.eTest = sm_ComputeNetConstants(NULL, NULL, pSur, sZone, &dUChordHeight, &dVChordHeight,
                                                     NULL, NULL, &dUVChordHeight)) != SM_SUCCESS)
                { return; }
              if (   dUChordHeight  <= rC.m_dChordHeightTolerance
                  && dVChordHeight  <= rC.m_dChordHeightTolerance
                  && dUVChordHeight >  rC.m_dChordHeightTolerance)
                {
                  double us, ue, vs, ve;
                  SmVector3d aDerivs[4];
                  NL_POINT * SD[2];
                  SD[0] = (NL_POINT *)&aDerivs[0];
                  SD[1] = (NL_POINT *)&aDerivs[2];
                  N_SrfGetParameterBounds(pSur, &us, &ue, &vs, &ve);
                  N_SrfDerivs(pSur, (us + ue) / 2.0, (vs + ve) / 2.0, NL_LEFT, NL_LEFT, TRUE, 1, 1, SD);
                  if (aDerivs[2].LengthSquared() * (ue - us) < aDerivs[1].LengthSquared() * (ve - vs))
                    { eDir = SM_SP_V; }
                }
              else if (dVChordHeight > dUChordHeight)
                { eDir = SM_SP_V; }
            }
        }

      // turning angles (always tested for SM_AD_AUX_DATA nodes)
      double dUAngDeg = -1.0, dVAngDeg = -1.0;
      if ((rN.eTest = sm_ComputeNetConstants(NULL, NULL, pSur, sZone, NULL, NULL, &dUAngDeg, &dVAngDeg, NULL)) != SM_SUCCESS)
        { return; }
      if (dUAngDeg > 540 || dVAngDeg > 540)
        {
          if ((rN.eTest = sm_ComputeSurfConstants(rC.m_cpSurface, rN.sDomain, NULL, NULL, &dUAngDeg, &dVAngDeg, NULL)) != SM_SUCCESS)
            { return; }
        }
      if (dUAngDeg < rC.m_dStartFastSubdivisionAngleDeg && dVAngDeg < rC.m_dStartFastSubdivisionAngleDeg)
        { rN.bSetFast = TRUE; }
      if (dAngTolDeg < SM_EFF_ZERO || (dUAngDeg <= dAngTolDeg && dVAngDeg <= dAngTolDeg))
        { bAngleSat = TRUE; }
      else
        {
          if (!bSubdivide && dVAngDeg > dUAngDeg)
            { eDir = SM_SP_V; }
          bSubdivide = TRUE;
        }

      // size tolerances
      SmPoint2d sAspectRatio, sMaxSideLength3D, sMinSideLength3D;
      if (   (!bSubdivide && (rC.m_dAspectRatio3D > 0.0 || rC.m_dMaximumSideLength3D > 0.0))
          || (bSubdivide && rC.m_dMinimumSideLength3D > 0.0))
        {
          if (dUAngDeg <= dAngTolDeg && dVAngDeg <= dAngTolDeg)
            {
              SmExtent2d sDomain;
              SmPoint3d  sU0V0, sU1V0, sU1V1, sU0V1;
              if (   (rN.eTest = rC.GetCorners(pSur, rN.sDomain, sU0V0, sU1V0, sU1V1, sU0V1, sDomain)) != SM_SUCCESS
                  || (rN.eTest = rC.ComputeSizeTolerances(sU0V0, sU1V0, sU1V1, sU0V1, sAspectRatio, sMaxSideLength3D,
                                                          sMinSideLength3D)) != SM_SUCCESS)
                { return; }
            }
          else if ((rN.eTest = rC.ComputeSizeTolerances(rC.m_cpSurface, rN.sDomain, sAspectRatio, sMaxSideLength3D,
                                                        sMinSideLength3D)) != SM_SUCCESS)
            { return; }
        }
      if (   !bSubdivide
          && rC.m_dMaximumSideLength3D > 0.0
          && sMaxSideLength3D.Length() > rC.m_dMaximumSideLength3D)
        {
          bSubdivide = TRUE;
          eDir = (sMaxSideLength3D.x > sMaxSideLength3D.y) ? SM_SP_U : SM_SP_V;
        }
      if (   !bSubdivide
          && rC.m_dAspectRatio3D > 0.0
          && (sAspectRatio.x > rC.m_dAspectRatio3D || sAspectRatio.y > rC.m_dAspectRatio3D))
        {
          bSubdivide = TRUE;
          eDir = (sMaxSideLength3D.x > sMaxSideLength3D.y) ? SM_SP_U : SM_SP_V;
        }
      if (   bSubdivide
          && rC.m_dMinimumSideLength3D > 0.0
          && sMinSideLength3D.x <= rC.m_dMinimumSideLength3D
          && sMinSideLength3D.y <= rC.m_dMinimumSideLength3D)
        { bSubdivide = FALSE; }
      rN.bChordOut = bChordSat;
      rN.bAngleOut = bAngleSat;
      if (!bSubdivide)
        { return; }

      // stop at 1/1000 of the cache domain
      const SmVector2d sUVSize   = rC.m_sUVDomain.GetSize();
      const SmVector2d sNodeSize = rN.sDomain.GetSize();
      const double     dMinSLR   = (rC.m_dMinimumSideLengthRatioUV != 0.0) ? rC.m_dMinimumSideLengthRatioUV : 0.001;
      if (   (eDir == SM_SP_U && sNodeSize.x < sUVSize.x * dMinSLR)
          || (eDir == SM_SP_V && sNodeSize.y < sUVSize.y * dMinSLR))
        {
          if      (eDir == SM_SP_U && sNodeSize.y > sUVSize.y / 100.0) { eDir = SM_SP_V; }
          else if (eDir == SM_SP_V && sNodeSize.x > sUVSize.x / 100.0) { eDir = SM_SP_U; }
          else                                                       { return; }
        }
      rN.bNeedsSub = TRUE;
      rN.eDir      = eDir;
    }

    // SmSurfaceCache::SubdivideNode() of an SM_AD_AUX_DATA node with a subdivision surface, without the
    // child bounding boxes: SmSurface::SnapToKnots() and SmBSplineSurface::SplitAt() on the NLib data,
    // with the knot fix-up that SplitAt()'s result surfaces make; on failure bSplit stays FALSE
    void Split(Node & rN) const
    {
      gw_SURFACE      * pSur   = rN.pSur;
      const SmPoint2d   sUVMin = rN.sDomain.GetMin();
      const SmPoint2d   sUVMax = rN.sDomain.GetMax();
      const SmBoolean   bU     = (rN.eDir == SM_SP_U) ? TRUE : FALSE;
      const double      dLo    = bU ? sUVMin.x : sUVMin.y;
      const double      dHi    = bU ? sUVMax.x : sUVMax.y;
      double            adKnotData[256];
      SmTArray<double>  sKnots(256, adKnotData);
      if (sm_GetKnots(bU ? pSur->knu : pSur->knv, sKnots, NULL, NULL) != SM_SUCCESS)
        { return; }
      // SnapToKnots(): the nearest knot closer than a quarter of the node
      double dSplitParam = (dHi + dLo) / 2.0;
        {
          const double dSnapTolerance = (dHi - dLo) / 4.0;
          double dMinSnapDist = SM_BIG_DOUBLE, dSnapped = dSplitParam;
          for (ULONG ii = 0; ii < sKnots.GetSize(); ii++)
            {
              const double dSnapDist = smos_Fabs(sKnots[ii] - dSplitParam);
              if (dSnapDist < dMinSnapDist && dSnapDist < dSnapTolerance)
                {
                  dSnapped     = sKnots[ii];
                  dMinSnapDist = dSnapDist;
                }
            }
          dSplitParam = dSnapped;
        }
      // SplitAt(): a knot within numerical noise is the split parameter
      double dParam = dSplitParam;
        {
          const double dTol = SM_EFF_ZERO * 100.0 * (1.0 + smos_Fabs(dParam));
          for (ULONG ii = 0; ii < sKnots.GetSize(); ii++)
            {
              if (smos_Fabs(sKnots[ii] - dParam) < dTol)
                { dParam = sKnots[ii]; }
            }
        }
      gw_SURFACE * pSurL = NULL, * pSurR = NULL;
      if (   sm_SplitSrf(pSur, dParam, bU ? NL_UDIR : NL_VDIR, pSurL, pSurR) == NL_YES
          || pSurL == NULL || pSurR == NULL)
        {
          if (pSurL != NULL) { smos_Free(pSurL); }
          if (pSurR != NULL) { smos_Free(pSurR); }
          return;
        }
      gw_SURFACE * apChild[2] = {pSurL, pSurR};
      for (int ii = 0; ii < 2; ii++)
        {
          sm_FixupKnotVector(apChild[ii]->knu, SM_EFF_ZERO_PARAM);
          sm_FixupKnotVector(apChild[ii]->knv, SM_EFF_ZERO_PARAM);
          Node * pChild = pPool->local().New();
          rN.pChild[ii] = pChild;
          pChild->pSur     = apChild[ii];
          pChild->bOwned   = TRUE;
          pChild->bChordIn = rN.bChordOut;
          pChild->bAngleIn = rN.bAngleOut;
        }
      if (bU)
        {
          rN.pChild[0]->sDomain.SetMinMax(sUVMin, SmPoint2d(dSplitParam, sUVMax.y));
          rN.pChild[1]->sDomain.SetMinMax(SmPoint2d(dSplitParam, sUVMin.y), sUVMax);
        }
      else
        {
          rN.pChild[0]->sDomain.SetMinMax(sUVMin, SmPoint2d(sUVMax.x, dSplitParam));
          rN.pChild[1]->sDomain.SetMinMax(SmPoint2d(sUVMin.x, dSplitParam), sUVMax);
        }
      rN.bSplit = TRUE;
    }

    // large nodes fork worker tasks; small ones recurse on the calling thread
    static SmBoolean IsLarge(const Node & crN)
    {
      return ((long long)(crN.pSur->net->n + 1) * (crN.pSur->net->m + 1) >= 256) ? TRUE : FALSE;
    }

    // SubdivideByBlock() splits
    void Block(Node & rN) const
    {
      SmBSplineSurface sSurface(rN.pSur, TRUE, pCache->GetContext());  // borrows the NLib data
      rN.eBlock = sm_FastBlockSplits(&sSurface, pCache->m_dChordHeightTolerance, pCache->m_dAngTolRad,
                                     pCache->m_dMaximumSideLength3D, rN.sUSplits, rN.sVSplits, rN.bNoSplits);
    }

    // speculative explorations: a second child explored while the first one is, which is in vain when
    // the first one ends with the fast subdivision switch on
    struct Speculation
    {
      std::atomic<bool>   bInVain{false};
      const Speculation * pOuter = NULL;
    };
    static bool InVain(const Speculation * pSpec)
    {
      for ( ; pSpec != NULL; pSpec = pSpec->pOuter)
        {
          if (pSpec->bInVain.load(std::memory_order_relaxed))
            { return true; }
        }
      return false;
    }

    // tests and splits under a tested node, wherever the serial recursion may split; returns the fast
    // subdivision switch after the node, for a recursion that reaches it with the switch off (leaves
    // keep it or set it, blocks clear it, splits pass on the second child's result); leaves free their
    // surfaces, which nothing needs any more. An exploration in vain stops early, with any result.
    SmBoolean Explore(Node & rN, const Speculation * pSpec = NULL) const
    {
      if (rN.eTest != SM_SUCCESS)
        { return FALSE; }
      if (!rN.bNeedsSub)
        {
          rN.FreeSurface();
          return rN.bSetFast;
        }
      if (bBlocksAllowed && rN.bSetFast)
        { return FALSE; }  // a block wherever the serial recursion reaches it
      if (InVain(pSpec))
        { return FALSE; }
      Split(rN);
      if (!rN.bSplit)
        { return FALSE; }
      Node & r1 = *rN.pChild[0];
      Node & r2 = *rN.pChild[1];
      const SmBoolean bFork = (bParallel && IsLarge(rN)) ? TRUE : FALSE;
      if (bFork) { sm_ForkInRegion(sDefaults, [&] { Test(r1); }, [&] { Test(r2); }); }
      else       { Test(r1); Test(r2); }
      // the second child, reached with the switch on, is a leaf that keeps it or a block that clears it
      auto Second = [&]() -> SmBoolean
        {
          if (r2.eTest != SM_SUCCESS || r2.bNeedsSub)
            { return FALSE; }
          r2.FreeSurface();
          return TRUE;
        };
      // large nodes explore their children in parallel, unless the first one is a leaf that sets the
      // switch; the second one speculatively, until the first one's switch tells it's in vain
      const SmBoolean bSets1 = (bBlocksAllowed && r1.eTest == SM_SUCCESS && !r1.bNeedsSub && r1.bSetFast) ? TRUE : FALSE;
      if (bFork && !bSets1)
        {
          SmBoolean   bSwitch1 = FALSE, bSwitch2 = FALSE;
          Speculation sSpec2;
          sSpec2.pOuter = pSpec;
          sm_ForkInRegion(sDefaults,
                          [&]
                            {
                              bSwitch1 = Explore(r1, pSpec);
                              if (bBlocksAllowed && bSwitch1)
                                { sSpec2.bInVain.store(true, std::memory_order_relaxed); }
                            },
                          [&] { bSwitch2 = Explore(r2, bBlocksAllowed ? &sSpec2 : pSpec); });
          return (bBlocksAllowed && bSwitch1) ? Second() : bSwitch2;
        }
      const SmBoolean bSwitch1 = Explore(r1, pSpec);
      if (bBlocksAllowed && bSwitch1)
        { return Second(); }
      return Explore(r2, pSpec);
    }
  } ; // end struct Worker

  // tree and top node as in BuildTreeWithSubdivision()
  m_pTree = new (*GetContext()) SmTree(this);
  ULONG lPatchCount  = 64;
  ULONG lNumPerBlock = smos_Min(50, (3 * lPatchCount) + 16);
  m_pTree->m_sNodeMgr.Initialize(ALIGN_SIZE(sizeof(SmTreeNode)), lNumPerBlock);
  m_sAuxMgr.Initialize(ALIGN_SIZE(sizeof(SmBezierAux2d)), lNumPerBlock);
  SmTemporaryChangeValue<SmBoolean> sTCV(m_bPointTestEnabled, FALSE);

  SmTreeNode    * pTopNode = (SmTreeNode *)m_pTree->m_sNodeMgr.GetNewElement();
  SmBezierAux2d * pTopAux  = (SmBezierAux2d *)m_sAuxMgr.GetNewElement();
  NER(pTopNode);
  NER(pTopAux);
  m_pTree->m_pTopNode      = pTopNode;
  pTopNode->m_pTree        = m_pTree;
  pTopNode->m_pParent      = NULL;
  pTopNode->m_pChild1      = NULL;
  pTopNode->m_pChild2      = NULL;
  pTopNode->m_pData        = pTopAux;
  pTopNode->m_eAuxDataType = SM_AD_AUX_DATA;
  pTopNode->m_eGeomType    = SM_NG_DEFAULT;
  pTopAux->m_pOwningTreeNode       = pTopNode;
  pTopAux->m_eSplitDir             = SM_SP_U;
  pTopAux->m_bIsMarked             = FALSE;
  pTopAux->m_bIsTessOnly           = FALSE;
  pTopAux->m_eNodeClass            = SM_NC_UNKNOWN;
  pTopAux->m_pFace                 = NULL;
  pTopAux->m_sUVDomain             = pBSS->GetNaturalUVDomain();
  pTopAux->m_vVertexList.Init();
  pTopAux->m_sEdgeuseList.Init();
  pTopAux->m_sPolyEdgeList.Init();
  pTopAux->mBA_pSurface            = m_cpSurface;
  pTopAux->m_bChordHeightSatisfied = FALSE;
  pTopAux->m_bAngleTolSatisfied    = FALSE;

  // the root works on a copy of the surface, as BuildTreeWithSubdivision() does
  std::unique_ptr<SmBSplineSurface> pRootSurface(new (*GetContext()) SmBSplineSurface(*pBSS));
  tbb::enumerable_thread_specific<Pool> sPool;
  std::unique_ptr<Node> pRoot(new Node);
  pRoot->sDomain = pTopAux->m_sUVDomain;
  pRoot->pSur    = pRootSurface->GetOrCreateGwNurbPointer();
  NER(pRoot->pSur);

  // worker tasks as in BuildTreeWithSubdivision(): not with notification callbacks, which the tasks'
  // temporary surfaces would call concurrently, nor with SMLIB_TESS_SERIAL_REFINEMENT; the tree is the same
  static const SmBoolean sbParallelRefinement = (getenv("SMLIB_TESS_SERIAL_REFINEMENT") == NULL) ? TRUE : FALSE;
  const SmBoolean bParallel = (sbParallelRefinement && sm_NoNotifyCallbacks(GetContext())) ? TRUE : FALSE;
  const Worker sWorker = {this, (m_eTessellationAlgorithm == SM_TA_FASTER_TESSELLATION) ? TRUE : FALSE, bParallel,
                          SmTol::GetZoneTol3d(m_cpSurface), SmTol::GetThreadDefaults(), &sPool,
                          (GetType() == SmFastTessSrfCache_TYPE) ? static_cast<const SmFastTessSrfCache *>(this) : NULL};
  sm_RunIsolated([&]
    {
      sWorker.Test(*pRoot);
      sWorker.Explore(*pRoot);
    });

  // the serial recursion's decisions, with its fast subdivision switch
  std::vector<Node *> sBlocks;
    {
      SmBoolean bSwitch = m_bSwitchToFastSubdivision;
      std::vector<Node *> sStack(1, pRoot.get());
      while (!sStack.empty())
        {
          Node * pN = sStack.back();
          sStack.pop_back();
          if (pN->eTest != SM_SUCCESS)
            { SER(pN->eTest); }
          if (pN->bSetFast)
            { bSwitch = TRUE; }
          if (!pN->bNeedsSub)
            { pN->eKind = SM_FR_LEAF; continue; }
          if (bSwitch && sWorker.bBlocksAllowed)
            {
              pN->eKind = SM_FR_BLOCK;
              sBlocks.push_back(pN);
              bSwitch = FALSE;
              continue;
            }
          if (!pN->bSplit && pN->pSur != NULL)
            { sm_RunIsolated([&] { sWorker.Explore(*pN); }); }
          if (!pN->bSplit)
            { SER(SM_ERR); }
          pN->eKind = SM_FR_SPLIT;
          sStack.push_back(pN->pChild[1]);
          sStack.push_back(pN->pChild[0]);
        }
      m_bSwitchToFastSubdivision = bSwitch;
    }
  // the blocks' splits, in parallel, with freeing the surfaces that nothing needs any more: of blocks
  // when done, of split nodes and of nodes that the recursion didn't reach
    {
      std::vector<Node *> sTasks;
      for (Node * pN : sBlocks)
        {
          if (pN->pSur == NULL)
            { SER(SM_ERR); }
          sTasks.push_back(pN);
        }
      const size_t lBlocks = sTasks.size();
        {
          std::vector<Node *> sStack(1, pRoot.get());
          while (!sStack.empty())
            {
              Node * pN = sStack.back();
              sStack.pop_back();
              if (pN->eKind != SM_FR_BLOCK && pN->bOwned && pN->pSur != NULL)
                { sTasks.push_back(pN); }
              if (pN->pChild[0] != NULL)
                {
                  sStack.push_back(pN->pChild[0]);
                  sStack.push_back(pN->pChild[1]);
                }
            }
        }
      auto Task = [&](size_t ii)
        {
          if (ii < lBlocks)
            { sWorker.Block(*sTasks[ii]); }
          sTasks[ii]->FreeSurface();
        };
      if (bParallel)
        { sm_ParallelFor(size_t(0), sTasks.size(), Task); }
      else
        {
          for (size_t ii = 0; ii < sTasks.size(); ii++)
            { Task(ii); }
        }
    }
  for (Node * pN : sBlocks)
    { SER(pN->eBlock); }

  // build the tree as SubdivideNode() and SubdivideByBlock() do
  SmTree * pTree = m_pTree;
  auto NewAux = [&](SmTreeNode * pNode, const SmExtent2d & crDomain, SmNodeClassType eClass, SmBoolean bChord, SmBoolean bAngle) -> SmBezierAux2d *
    {
      SmBezierAux2d * pAux = (SmBezierAux2d *)m_sAuxMgr.GetNewElement();
      if (pAux == NULL)
        { return NULL; }
      pNode->m_eAuxDataType        = SM_AD_AUX_DATA;
      pNode->m_pData               = pAux;
      pNode->m_eGeomType           = SM_NG_DEFAULT;
      pAux->m_vVertexList.Init();
      pAux->m_sEdgeuseList.Init();
      pAux->m_sPolyEdgeList.Init();
      pAux->m_pOwningTreeNode       = pNode;
      pAux->m_eSplitDir             = SM_SP_U;
      pAux->m_bIsMarked             = FALSE;
      pAux->m_bIsTessOnly           = FALSE;
      pAux->m_eNodeClass            = eClass;
      pAux->mBA_pSurface            = m_cpSurface;
      pAux->m_pFace                 = NULL;
      pAux->m_bChordHeightSatisfied = bChord;
      pAux->m_bAngleTolSatisfied    = bAngle;
      pAux->m_sUVDomain             = crDomain;
      return pAux;
    };
  auto NewNode = [&](SmTreeNode * pParent) -> SmTreeNode *
    {
      SmTreeNode * pNode = (SmTreeNode *)pTree->m_sNodeMgr.GetNewElement();
      if (pNode == NULL)
        { return NULL; }
      pNode->m_pTree   = pTree;
      pNode->m_pParent = pParent;
      pNode->m_pChild1 = NULL;
      pNode->m_pChild2 = NULL;
      return pNode;
    };
  // SubdivideByBlock(): base nodes, assembled into a binary tree under pNode
  auto BuildBlock = [&](SmTreeNode * pNode, Node & rN) -> SmStatus
    {
      if (rN.bNoSplits)
        { return SM_SUCCESS; }
      const ULONG lNumSplitsU = rN.sUSplits.GetSize();
      const ULONG lNumSplitsV = rN.sVSplits.GetSize();
      std::vector<SmTreeNode *> sCurr, sNext;
      sCurr.reserve((size_t)(lNumSplitsU - 1) * (lNumSplitsV - 1));
      for (ULONG i = 1; i < lNumSplitsU; i++)
        {
          for (ULONG j = 1; j < lNumSplitsV; j++)
            {
              SmTreeNode * pBase = NewNode(NULL);
              NER(pBase);
              SmExtent2d sDomain(rN.sUSplits[i-1], rN.sVSplits[j-1], rN.sUSplits[i], rN.sVSplits[j]);
              NER(NewAux(pBase, sDomain, SM_NC_UNKNOWN, FALSE, FALSE));
              sCurr.push_back(pBase);
            }
        }
      ULONG lNumU = lNumSplitsU - 1;
      ULONG lNumV = lNumSplitsV - 1;
      while (lNumU > 1 || lNumV > 1)
        {
          const SmSurfParamType eCombine  = (lNumV > lNumU) ? SM_SP_V : SM_SP_U;
          const SmBoolean       bLastTime = (lNumU + lNumV == 3) ? TRUE : FALSE;
          sNext.clear();
          auto Combine = [&](SmTreeNode * pNode1, SmTreeNode * pNode2, SmSurfParamType eSplitDir) -> SmStatus
            {
              SmTreeNode * pNewNode = pNode;
              if (!bLastTime)
                {
                  pNewNode = NewNode(NULL);
                  NER(pNewNode);
                  SmExtent2d sUnion;
                  ((SmBezierAux2d *)pNode1->m_pData)->m_sUVDomain.Union(((SmBezierAux2d *)pNode2->m_pData)->m_sUVDomain, sUnion);
                  NER(NewAux(pNewNode, sUnion, SM_NC_UNKNOWN, FALSE, FALSE));
                }
              ((SmBezierAux2d *)pNewNode->m_pData)->m_eSplitDir = eSplitDir;
              sNext.push_back(pNewNode);
              pNewNode->m_pChild1 = pNode1;
              pNewNode->m_pChild2 = pNode2;
              pNode1->m_pParent   = pNewNode;
              pNode2->m_pParent   = pNewNode;
              return SM_SUCCESS;
            };
          if (eCombine == SM_SP_V)
            {
              for (ULONG ii = 0; ii < lNumU; ii++)
                {
                  for (ULONG kk = 1; kk < lNumV; kk += 2)
                    { SER(Combine(sCurr[ii * lNumV + kk - 1], sCurr[ii * lNumV + kk], SM_SP_V)); }
                  if (lNumV % 2 == 1)
                    { sNext.push_back(sCurr[ii * lNumV + lNumV - 1]); }
                }
              lNumV = (lNumV + 1) / 2;
            }
          else
            {
              for (ULONG ii = 1; ii < lNumU; ii += 2)
                {
                  for (ULONG kk = 0; kk < lNumV; kk++)
                    { SER(Combine(sCurr[(ii - 1) * lNumV + kk], sCurr[ii * lNumV + kk], SM_SP_U)); }
                }
              if (lNumU % 2 == 1)
                {
                  for (ULONG kk = 0; kk < lNumV; kk++)
                    { sNext.push_back(sCurr[(lNumU - 1) * lNumV + kk]); }
                }
              lNumU = (lNumU + 1) / 2;
            }
          sCurr.swap(sNext);
        }
      return SM_SUCCESS;
    };
    {
      std::vector<std::pair<SmTreeNode *, Node *>> sStack(1, {pTopNode, pRoot.get()});
      while (!sStack.empty())
        {
          SmTreeNode * pNode = sStack.back().first;
          Node       * pN    = sStack.back().second;
          sStack.pop_back();
          SmBezierAux2d * pAux = (SmBezierAux2d *)pNode->m_pData;
          pAux->m_bChordHeightSatisfied = pN->bChordOut;
          pAux->m_bAngleTolSatisfied    = pN->bAngleOut;
          if (pN->eKind == SM_FR_BLOCK)
            { SER(BuildBlock(pNode, *pN)); continue; }
          if (pN->eKind != SM_FR_SPLIT)
            { continue; }
          SmTreeNode * pNode1 = NewNode(pNode);
          SmTreeNode * pNode2 = NewNode(pNode);
          NER(pNode1);
          NER(pNode2);
          pNode->m_pChild1 = pNode1;
          pNode->m_pChild2 = pNode2;
          NER(NewAux(pNode1, pN->pChild[0]->sDomain, pAux->m_eNodeClass, pN->bChordOut, pN->bAngleOut));
          NER(NewAux(pNode2, pN->pChild[1]->sDomain, pAux->m_eNodeClass, pN->bChordOut, pN->bAngleOut));
          pAux->m_eSplitDir = pN->eDir;
          sStack.push_back({pNode2, pN->pChild[1]});
          sStack.push_back({pNode1, pN->pChild[0]});
        }
    }

  // the nodes and their remaining subdivision surfaces (of split nodes and nodes the recursion didn't
  // reach) go in parallel, each pool's on one thread, then pRootSurface the copy
    {
      std::vector<Pool *> sPools;
      for (Pool & rPool : sPool)
        { sPools.push_back(&rPool); }
      sm_ParallelFor(size_t(0), sPools.size(), [&](size_t ii) { sPools[ii]->sChunks.clear(); });
    }
  pRoot.reset();
  return SM_SUCCESS;

} // end SmTessSrfCache::BuildTreeFast

/*******************************************************************//**
   SmTessellateFast() on fast boundaries: SmTessellateWithFastBoundaries()
***********************************************************************/

namespace {

// pipeline data of one face on fast boundaries
struct SmBoundaryFaceWork
{
  const SmFastFaceBoundary * cpB       = NULL;
  SmBoolean                  bParallel = FALSE;  // its surface may be evaluated on worker threads
  SmBoolean                  bDone     = FALSE;  // sOut holds its triangles
  SmBoolean                  bFan      = FALSE;  // ... a degenerate face's loop, triangulated in 3D
  SmFastFaceOutput           sOut;
  std::vector<std::pair<long, SmVector3d>> sLoopNormals;  // surface normal (not reversed) per loop id, first use
  std::vector<SmVector3d>    sCornerNormals;              // when asked: surface normal (not reversed) per corner
};

// The UV on cpSurface of crPoint, a point of a loop segment between UVs crA and crB at fraction dF of the
// way: Gauss-Newton from that fraction of the UV segment; FALSE unless it comes within dTol of the point.
bool sm_SegmentUV(const SmSurface * cpSurface, const SmPoint2d & crA, const SmPoint2d & crB, double dF, const SmPoint3d & crPoint,
                  double dTol, SmPoint2d & rUV)
{
  SmPoint2d sUV(crA.x + dF * (crB.x - crA.x), crA.y + dF * (crB.y - crA.y));
  const double dSpan = std::max(std::fabs(crB.x - crA.x), std::fabs(crB.y - crA.y));
  for (int iIter = 0; iIter < 20; iIter++)
    {
      SmPoint3d  sP;
      SmVector3d sSu, sSv;
      if (cpSurface->Evaluate1stDerivatives(sUV, TRUE, TRUE, sP, sSu, sSv) != SM_SUCCESS)
        { return false; }
      const SmVector3d sR(crPoint.x - sP.x, crPoint.y - sP.y, crPoint.z - sP.z);
      const double a = sSu.Dot(sSu), b = sSu.Dot(sSv), c = sSv.Dot(sSv), d = sSu.Dot(sR), e = sSv.Dot(sR);
      const double dDet = a * c - b * b;
      if (!(std::fabs(dDet) > 0.0))
        { break; }
      double du = (c * d - b * e) / dDet, dv = (a * e - b * d) / dDet;
      // steps no longer than the segment's UV span (it starts close)
      const double dStep = std::max(std::fabs(du), std::fabs(dv));
      if (dSpan > 0.0 && dStep > dSpan) { du *= dSpan / dStep; dv *= dSpan / dStep; }
      sUV = SmPoint2d(sUV.x + du, sUV.y + dv);
      if (dStep <= 1.0e-15 * (1.0 + std::fabs(sUV.x) + std::fabs(sUV.y)))
        { break; }
    }
  SmPoint3d sAt;
  if (cpSurface->EvaluatePoint(sUV, sAt) != SM_SUCCESS || !(sAt.DistanceBetween(crPoint) <= dTol))
    { return false; }
  rUV = sUV;
  return true;
}

// The triangulation of a closed polygon in 3D that maximizes its worst triangle's shape quality
// (2 sqrt(3) |2 area| / sum of squared sides, 1 for an equilateral triangle), by dynamic programming over
// sub-polygons; triangles (i, k, j), i < k < j, follow the polygon's order. FALSE for more than
// kMaxLoop3d vertices, a degenerate triangle (twice its area at most 2e-12 times its longest side squared,
// the mesh checker's measure), or two triangles on one diagonal turned more than 90 degrees from each
// other: the polygon folds over itself there, and no triangulation of it alone covers the region.
static constexpr int kMaxLoop3d = 256;
SmBoolean sm_TriangulateLoop3d(const std::vector<SmPoint3d> & crP, std::vector<int> & rTriangles)
{
  rTriangles.clear();
  const int n = (int)crP.size();
  if (n < 3 || n > kMaxLoop3d)
    { return FALSE; }
  auto Normal = [&](int a, int b, int c) { return (crP[b] - crP[a]) * (crP[c] - crP[a]); };
  auto Quality = [&](int a, int b, int c)
    {
      const double dSides = crP[a].DistanceBetweenSquared(crP[b]) + crP[b].DistanceBetweenSquared(crP[c]) + crP[c].DistanceBetweenSquared(crP[a]);
      return (dSides > 0.0) ? 3.4641016151377546 * Normal(a, b, c).Length() / dSides : 0.0;
    };
  std::vector<double> sBest((size_t)n * n, 1.0e300);   // best[i][j], j > i + 1: the sub-polygon i..j
  std::vector<int>    sSplit((size_t)n * n, -1);
  for (int lLen = 2; lLen < n; lLen++)
    {
      for (int i = 0; i + lLen < n; i++)
        {
          const int j = i + lLen;
          double dBest = -1.0;
          for (int k = i + 1; k < j; k++)
            {
              const double dQ = std::min(Quality(i, k, j), std::min(sBest[(size_t)i * n + k], sBest[(size_t)k * n + j]));
              if (dQ > dBest) { dBest = dQ; sSplit[(size_t)i * n + j] = k; }
            }
          sBest[(size_t)i * n + j] = dBest;
        }
    }
  std::vector<std::pair<int, int>> sStack(1, {0, n - 1});
  while (!sStack.empty())
    {
      const std::pair<int, int> sIJ = sStack.back();
      sStack.pop_back();
      if (sIJ.second - sIJ.first < 2)
        { continue; }
      const int k = sSplit[(size_t)sIJ.first * n + sIJ.second];
      rTriangles.insert(rTriangles.end(), {sIJ.first, k, sIJ.second});
      sStack.push_back({sIJ.first, k});
      sStack.push_back({k, sIJ.second});
    }
  // degenerate triangles, and folds across diagonals (sides of the polygon are its pairs i, i + 1 and 0, n - 1)
  std::unordered_map<uint64_t, SmVector3d> sDiagonals;
  for (size_t tt = 0; tt < rTriangles.size(); tt += 3)
    {
      const int * aT = &rTriangles[tt];
      const SmVector3d sN = Normal(aT[0], aT[1], aT[2]);
      double dLongest = 0.0;
      for (int e = 0; e < 3; e++)
        { dLongest = std::max(dLongest, crP[aT[e]].DistanceBetweenSquared(crP[aT[(e + 1) % 3]])); }
      if (!(sN.Length() > 2.0e-12 * dLongest))
        { rTriangles.clear(); return FALSE; }
      for (int e = 0; e < 3; e++)
        {
          const int a = std::min(aT[e], aT[(e + 1) % 3]), b = std::max(aT[e], aT[(e + 1) % 3]);
          if (b - a == 1 || (a == 0 && b == n - 1))
            { continue; }
          const uint64_t lKey = ((uint64_t)a << 32) | (uint64_t)b;
          auto sFound = sDiagonals.find(lKey);
          if (sFound == sDiagonals.end())
            { sDiagonals.emplace(lKey, sN); }
          else if (sFound->second.Dot(sN) < 0.0)
            { rTriangles.clear(); return FALSE; }
        }
    }
  return TRUE;
}

// Triangulation of a polygon with holes in a plane: crRings[0] the outer ring, counterclockwise, the others
// holes inside it, clockwise, as indices into crXY. Holes are bridged to the outer ring (from a hole's
// rightmost vertex to a vertex it sees, as Eberly's ear clipping does), then ears are clipped, the best
// shaped (largest smallest angle) first. Triangles come counterclockwise, as index triples. FALSE when the
// rings aren't simple, disjoint, oriented and nested so, or with more than kMaxRings2d vertices.
static constexpr size_t kMaxRings2d = 512;
SmBoolean sm_TriangulateRings2d(const std::vector<SmPoint2d> & crXY, std::vector<std::vector<int>> sRings,
                                std::vector<int> & rTriangles)
{
  rTriangles.clear();
  size_t lTotal = 0;
  for (const std::vector<int> & crR : sRings)
    {
      if (crR.size() < 3)
        { return FALSE; }
      lTotal += crR.size();
    }
  if (sRings.empty() || lTotal > kMaxRings2d)
    { return FALSE; }
  SmExtent2d sBox;
  for (const SmPoint2d & crP : crXY)
    { sBox.AddPoint2d(crP); }
  const double dScale = std::sqrt(sBox.XLength() * sBox.XLength() + sBox.YLength() * sBox.YLength());
  if (!(dScale > 0.0))
    { return FALSE; }
  const double dEps = 1.0e-12 * dScale * dScale;   // on doubled triangle areas
  auto Orient = [&](int a, int b, int c)
    {
      const SmPoint2d & A = crXY[a], & B = crXY[b], & C = crXY[c];
      return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
    };
  auto Area2 = [&](const std::vector<int> & crR)
    {
      double d = 0.0;
      for (size_t ii = 0; ii < crR.size(); ii++)
        {
          const SmPoint2d & a = crXY[crR[ii]], & b = crXY[crR[(ii + 1) % crR.size()]];
          d += a.x * b.y - b.x * a.y;
        }
      return d;
    };
  auto Inside = [&](const std::vector<int> & crR, const SmPoint2d & crQ)   // even-odd rule
    {
      bool bIn = false;
      for (size_t ii = 0, jj = crR.size() - 1; ii < crR.size(); jj = ii++)
        {
          const SmPoint2d & a = crXY[crR[ii]], & b = crXY[crR[jj]];
          if ((a.y > crQ.y) != (b.y > crQ.y) && crQ.x < (b.x - a.x) * (crQ.y - a.y) / (b.y - a.y) + a.x)
            { bIn = !bIn; }
        }
      return bIn;
    };
  // orientations, and no two sides that cross or touch (but neighbors at their common vertex)
  if (!(Area2(sRings[0]) > dEps))
    { return FALSE; }
  for (size_t rr = 1; rr < sRings.size(); rr++)
    {
      if (!(Area2(sRings[rr]) < -dEps) || !Inside(sRings[0], crXY[sRings[rr][0]]))
        { return FALSE; }
      for (size_t qq = 1; qq < sRings.size(); qq++)
        {
          if (qq != rr && Inside(sRings[qq], crXY[sRings[rr][0]]))
            { return FALSE; }
        }
    }
  struct Side { int a, b; size_t lRing, lPos; };
  std::vector<Side> sSides;
  for (size_t rr = 0; rr < sRings.size(); rr++)
    {
      for (size_t ii = 0; ii < sRings[rr].size(); ii++)
        { sSides.push_back({sRings[rr][ii], sRings[rr][(ii + 1) % sRings[rr].size()], rr, ii}); }
    }
  for (size_t ii = 0; ii < sSides.size(); ii++)
    {
      for (size_t jj = ii + 1; jj < sSides.size(); jj++)
        {
          const Side & s1 = sSides[ii], & s2 = sSides[jj];
          const size_t lN = sRings[s1.lRing].size();
          const bool bNeighbors = s1.lRing == s2.lRing && ((s1.lPos + 1) % lN == s2.lPos || (s2.lPos + 1) % lN == s1.lPos);
          if (bNeighbors)
            {
              // only their common vertex: not folded back onto each other
              const int c = (s1.b == s2.a) ? s1.b : s2.b;
              const int x = (s1.b == s2.a) ? s1.a : s2.a, y = (s1.b == s2.a) ? s2.b : s1.b;
              if (std::fabs(Orient(x, c, y)) <= dEps)
                {
                  const SmPoint2d & X = crXY[x], & C = crXY[c], & Y = crXY[y];
                  if ((X.x - C.x) * (Y.x - C.x) + (X.y - C.y) * (Y.y - C.y) > 0.0)
                    { return FALSE; }
                }
              continue;
            }
          const double d1 = Orient(s1.a, s1.b, s2.a), d2 = Orient(s1.a, s1.b, s2.b);
          const double d3 = Orient(s2.a, s2.b, s1.a), d4 = Orient(s2.a, s2.b, s1.b);
          if (d1 * d2 <= dEps * dEps && d3 * d4 <= dEps * dEps)
            {
              // collinear sides apart from each other don't meet
              const bool bCollinear = std::fabs(d1) <= dEps && std::fabs(d2) <= dEps;
              if (!bCollinear)
                { return FALSE; }
              const SmPoint2d & A = crXY[s1.a], & B = crXY[s1.b];
              const double dx = B.x - A.x, dy = B.y - A.y, dLen2 = dx * dx + dy * dy;
              auto T = [&](int q) { return ((crXY[q].x - A.x) * dx + (crXY[q].y - A.y) * dy) / dLen2; };
              const double t1 = T(s2.a), t2 = T(s2.b);
              if (std::max(t1, t2) >= 0.0 && std::min(t1, t2) <= 1.0)
                { return FALSE; }
            }
        }
    }
  // bridges, holes with the rightmost vertices first
  std::vector<int> sPoly = sRings[0];
  std::vector<size_t> sOrder;
  for (size_t rr = 1; rr < sRings.size(); rr++)
    { sOrder.push_back(rr); }
  auto Rightmost = [&](const std::vector<int> & crR)
    {
      size_t lBest = 0;
      for (size_t ii = 1; ii < crR.size(); ii++)
        {
          const SmPoint2d & a = crXY[crR[ii]], & b = crXY[crR[lBest]];
          if (a.x > b.x || (a.x == b.x && a.y < b.y)) { lBest = ii; }
        }
      return lBest;
    };
  std::sort(sOrder.begin(), sOrder.end(), [&](size_t a, size_t b)
    {
      const SmPoint2d & pa = crXY[sRings[a][Rightmost(sRings[a])]], & pb = crXY[sRings[b][Rightmost(sRings[b])]];
      return pa.x > pb.x || (pa.x == pb.x && a < b);
    });
  for (size_t rr : sOrder)
    {
      const std::vector<int> & crHole = sRings[rr];
      const size_t lM = Rightmost(crHole);
      const SmPoint2d M = crXY[crHole[lM]];
      // the nearest side of the polygon the ray from M to +x hits
      double dBestX = 1.0e300;
      size_t lHit = (size_t)-1;
      for (size_t ii = 0; ii < sPoly.size(); ii++)
        {
          const SmPoint2d & a = crXY[sPoly[ii]], & b = crXY[sPoly[(ii + 1) % sPoly.size()]];
          if ((a.y > M.y) == (b.y > M.y))
            { continue; }
          const double x = a.x + (M.y - a.y) * (b.x - a.x) / (b.y - a.y);
          if (x >= M.x && x < dBestX) { dBestX = x; lHit = ii; }
        }
      if (lHit == (size_t)-1)
        { return FALSE; }
      // the side's end farther right, unless a reflex vertex lies in the triangle M, hit, end: then of
      // those, the one at the smallest angle to the ray (the nearest of equal ones)
      const size_t lA = lHit, lB = (lHit + 1) % sPoly.size();
      size_t lP = (crXY[sPoly[lA]].x > crXY[sPoly[lB]].x) ? lA : lB;
      const SmPoint2d I(dBestX, M.y), P = crXY[sPoly[lP]];
      double dBestAngle = 1.0e300, dBestDist = 1.0e300;
      size_t lReflex = (size_t)-1;
      const double dTri = (I.x - M.x) * (P.y - M.y) - (I.y - M.y) * (P.x - M.x);
      for (size_t ii = 0; ii < sPoly.size(); ii++)
        {
          if (ii == lP)
            { continue; }
          const int u = sPoly[(ii + sPoly.size() - 1) % sPoly.size()], v = sPoly[ii], w = sPoly[(ii + 1) % sPoly.size()];
          if (Orient(u, v, w) > 0.0)
            { continue; }   // convex
          const SmPoint2d & Q = crXY[v];
          auto Side2 = [&](const SmPoint2d & a, const SmPoint2d & b) { return (b.x - a.x) * (Q.y - a.y) - (b.y - a.y) * (Q.x - a.x); };
          const double s1 = Side2(M, I), s2 = Side2(I, P), s3 = Side2(P, M);
          const bool bIn = (dTri > 0.0) ? (s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0) : (s1 <= 0.0 && s2 <= 0.0 && s3 <= 0.0);
          if (!bIn || Q.x < M.x)
            { continue; }
          const double dAngle = std::atan2(std::fabs(Q.y - M.y), Q.x - M.x), dDist = M.DistanceBetween(Q);
          if (dAngle < dBestAngle || (dAngle == dBestAngle && dDist < dBestDist))
            { dBestAngle = dAngle; dBestDist = dDist; lReflex = ii; }
        }
      if (lReflex != (size_t)-1)
        { lP = lReflex; }
      // the polygon goes P, then the hole from M round to M, then P again
      std::vector<int> sMerged(sPoly.begin(), sPoly.begin() + lP + 1);
      for (size_t kk = 0; kk <= crHole.size(); kk++)
        { sMerged.push_back(crHole[(lM + kk) % crHole.size()]); }
      sMerged.insert(sMerged.end(), sPoly.begin() + lP, sPoly.end());
      sPoly.swap(sMerged);
    }
  // ears: convex corners whose triangle holds no other vertex (copies of its corners at the bridges aside)
  const size_t n = sPoly.size();
  std::vector<int> sPrev(n), sNext(n);
  for (size_t ii = 0; ii < n; ii++)
    {
      sPrev[ii] = (int)((ii + n - 1) % n);
      sNext[ii] = (int)((ii + 1) % n);
    }
  auto SamePoint = [&](int a, int b) { return crXY[a].x == crXY[b].x && crXY[a].y == crXY[b].y; };
  auto Quality = [&](int a, int b, int c)
    {
      const double dSides = crXY[a].DistanceBetween(crXY[b]) * crXY[a].DistanceBetween(crXY[b])
                          + crXY[b].DistanceBetween(crXY[c]) * crXY[b].DistanceBetween(crXY[c])
                          + crXY[c].DistanceBetween(crXY[a]) * crXY[c].DistanceBetween(crXY[a]);
      return dSides > 0.0 ? Orient(a, b, c) / dSides : 0.0;
    };
  auto IsEar = [&](int v)
    {
      const int a = sPoly[sPrev[v]], b = sPoly[v], c = sPoly[sNext[v]];
      if (!(Orient(a, b, c) > dEps))
        { return false; }
      for (int w = sNext[sNext[v]]; w != sPrev[v]; w = sNext[w])
        {
          const int q = sPoly[w];
          if (SamePoint(q, a) || SamePoint(q, b) || SamePoint(q, c))
            { continue; }
          if (Orient(a, b, q) >= -dEps && Orient(b, c, q) >= -dEps && Orient(c, a, q) >= -dEps)
            { return false; }
        }
      return true;
    };
  size_t lLeft = n;
  int lStart = 0;
  while (lLeft > 3)
    {
      int lBest = -1;
      double dBest = -1.0;
      int v = lStart;
      for (size_t kk = 0; kk < lLeft; kk++, v = sNext[v])
        {
          if (!IsEar(v))
            { continue; }
          const double dQ = Quality(sPoly[sPrev[v]], sPoly[v], sPoly[sNext[v]]);
          if (dQ > dBest) { dBest = dQ; lBest = v; }
        }
      if (lBest < 0)
        { rTriangles.clear(); return FALSE; }
      rTriangles.insert(rTriangles.end(), {sPoly[sPrev[lBest]], sPoly[lBest], sPoly[sNext[lBest]]});
      sNext[sPrev[lBest]] = sNext[lBest];
      sPrev[sNext[lBest]] = sPrev[lBest];
      lStart = sNext[lBest];
      lLeft--;
    }
  const int a = sPoly[sPrev[lStart]], b = sPoly[lStart], c = sPoly[sNext[lStart]];
  if (!(Orient(a, b, c) > dEps))
    { rTriangles.clear(); return FALSE; }
  rTriangles.insert(rTriangles.end(), {a, b, c});
  return TRUE;
}

// A face whose loops enclose no area in UV, whose surface can't place points inside it, from its loops'
// points alone (the edges', shared with the neighbors), after dropping a vertex closer than tolerance/100
// to the next as SmTess::OutputToPolyBrep() does: a single loop triangulated in 3D (sm_TriangulateLoop3d()),
// several (holes) in the plane of the first one's Newell normal (sm_TriangulateRings2d()). FALSE (the Brep
// takes the Phase 1 path) when that doesn't give a valid triangulation of the region they bound.
SmBoolean sm_TriangulateDegenerateFace(const SmFastFaceBoundary & crB, const std::vector<SmPoint3d> & crPoints,
                                       double dTolerance, SmFastFaceOutput & rOut)
{
  rOut = SmFastFaceOutput();
  if (crB.sLoops.empty())
    { return FALSE; }
  std::vector<long>      sIds;     // all loops' vertices, loop by loop
  std::vector<SmPoint2d> sUVs;
  std::vector<std::vector<int>> sRings;
  for (const std::vector<SmFastBoundaryVertex> & crLoop : crB.sLoops)
    {
      std::vector<long>      sL;
      std::vector<SmPoint2d> sU;
      for (const SmFastBoundaryVertex & crV : crLoop)
        {
          if (sL.empty() || sL.back() != crV.lId)
            {
              sL.push_back(crV.lId);
              sU.push_back(SmPoint2d(crV.dU, crV.dV));
            }
        }
      while (sL.size() > 1 && sL.front() == sL.back())
        { sL.pop_back(); sU.pop_back(); }
      for (size_t kk = 0; kk < sL.size() && sL.size() >= 3; )
        {
          const size_t lNext = (kk + 1) % sL.size();
          if (crPoints[sL[kk]].DistanceBetween(crPoints[sL[lNext]]) < dTolerance / 100.0)
            {
              sL.erase(sL.begin() + kk);
              sU.erase(sU.begin() + kk);
            }
          else
            { kk++; }
        }
      if (sL.size() < 3)
        { return FALSE; }
      sRings.emplace_back();
      for (size_t kk = 0; kk < sL.size(); kk++)
        {
          sRings.back().push_back((int)sIds.size());
          sIds.push_back(sL[kk]);
          sUVs.push_back(sU[kk]);
        }
    }
  std::vector<int> sTriangles;
  if (sRings.size() == 1)
    {
      std::vector<SmPoint3d> sP;
      for (long lId : sIds)
        { sP.push_back(crPoints[lId]); }
      if (!sm_TriangulateLoop3d(sP, sTriangles))
        { return FALSE; }
    }
  else
    {
      // the plane: the first loop's Newell normal, which it runs counterclockwise around
      SmVector3d sN(0, 0, 0);
      const std::vector<int> & crOuter = sRings[0];
      for (size_t kk = 0; kk < crOuter.size(); kk++)
        {
          const SmPoint3d & a = crPoints[sIds[crOuter[kk]]], & b = crPoints[sIds[crOuter[(kk + 1) % crOuter.size()]]];
          sN.x += (a.y - b.y) * (a.z + b.z);
          sN.y += (a.z - b.z) * (a.x + b.x);
          sN.z += (a.x - b.x) * (a.y + b.y);
        }
      if (sN.Unitize() != SM_SUCCESS)
        { return FALSE; }
      SmVector3d sX = (std::fabs(sN.x) < 0.6) ? SmVector3d(1, 0, 0) : SmVector3d(0, 1, 0);
      sX = sX - sN * sX.Dot(sN);
      if (sX.Unitize() != SM_SUCCESS)
        { return FALSE; }
      const SmVector3d sY = sN * sX;
      std::vector<SmPoint2d> sXY;
      for (long lId : sIds)
        {
          const SmVector3d sV(crPoints[lId].x, crPoints[lId].y, crPoints[lId].z);
          sXY.push_back(SmPoint2d(sV.Dot(sX), sV.Dot(sY)));
        }
      if (!sm_TriangulateRings2d(sXY, sRings, sTriangles))
        { return FALSE; }
    }
  for (int k : sTriangles)
    {
      rOut.sTriangles.push_back(sIds[k]);
      rOut.sTriangleUVs.push_back(sUVs[k]);
    }
  rOut.eStatus = SM_SUCCESS;
  return TRUE;
}

// sm_TriangleNormals() of a face on fast boundaries
void sm_BoundaryFaceNormals(SmBoundaryFaceWork & rWork, SmBoolean bCorners)
{
  std::unordered_map<long, SmPoint2d> sFirstLoopUV;
  if (rWork.sOut.sTriangleUVs.size() != rWork.sOut.sTriangles.size())
    {
      for (const auto & crLoop : rWork.cpB->sLoops)
        {
          for (const SmFastBoundaryVertex & crV : crLoop)
            { sFirstLoopUV.emplace(crV.lId, SmPoint2d(crV.dU, crV.dV)); }
        }
    }
  sm_TriangleNormals(rWork.cpB->cpFace->GetSurface(), rWork.sOut, sFirstLoopUV, bCorners, rWork.sLoopNormals,
                     rWork.sCornerNormals);
}

} // end anonymous namespace

/*******************************************************************//**
PURPOSE: SmTessellateFast() on the boundaries of SmComputeFastBoundaries()
   (see SmTessFast.h).
***********************************************************************/
SmStatus SmTessellateWithFastBoundaries
 (const SmBrep               * cpBrep,
  const SmTessellationParams & crParams,
  SmFastMesh                 & rMesh,
  const SmFastOptions        & crOptions)
{
  rMesh = SmFastMesh();
  NER(cpBrep);
  sm_ContextSerializer sSerial(cpBrep->GetContext());
  if (!sSerial.Acquired())
    { return SM_ERR; }   // called back from another context's call while another thread has this context
  SmBrep * pBrep = SM_CONST_CAST(SmBrep *, cpBrep);

  // parameters as SmTessellateFast() uses them
  const double dChordHeightTol     = crParams.dChordHeightTol;
  const double dSurfaceAngleTolDeg = crParams.dSurfaceAngleTolDeg;
  double       dMaxEdgeLength      = crParams.dMaxEdgeLength;
  const double dMaxAspectRatio     = crParams.dMaxAspectRatio;
  if (dMaxEdgeLength < 0.0)
    {
      SmExtent3d sBox;
      pBrep->CalculateBoundingBox(sBox);
      double dSize = sBox.GetSize().Length();
      if (dSize >= 1.0e15 || dSize <= 0.0) dSize = 1.0;
      dMaxEdgeLength = -0.025 * dSize * dMaxEdgeLength;
    }
  SmTessellationParams sParams = crParams;
  sParams.dMaxEdgeLength = dMaxEdgeLength;

  const SmContext & crContext = *pBrep->GetContext();
  SmTemporaryChangeValue<SmBoolean> sDoingBool(SM_CONST_CAST(SmContext *, &crContext)->GetDoingBooleanRef(), TRUE);
  SmFastSurfaceTessDriver sFastSrfTess(dChordHeightTol, dSurfaceAngleTolDeg, dMaxEdgeLength, 0.0, 0.001, dMaxAspectRatio);
  SmSurfaceTessDriver     sPlainSrfTess(dChordHeightTol, dSurfaceAngleTolDeg, dMaxEdgeLength, 0.0, 0.001, dMaxAspectRatio);
  SmSurfaceTessDriver   & sSrfTess = crOptions.bFastRefinement ? (SmSurfaceTessDriver &)sFastSrfTess : sPlainSrfTess;
  sm_Clock::time_point sCleanupStart;
    {
      // The boundaries, then the faces in rounds: a trimmed face whose triangulation needs loop segments split
      // (SM_FF_SPLIT) gets them split in quarters in every face using them (points on the edge's curve, their
      // UVs inverted on each face's surface from the segment's), as SMLib's Phase 2 splits such edges, and
      // those faces are triangulated again, up to kSplitRounds times; any other failure sends the Brep to the
      // Phase 1 path.
      static constexpr int kSplitRounds = 6;
      std::unique_ptr<SmFastBoundaries> pB(new SmFastBoundaries);   // owns the refinement trees
      SmFastBoundaries & sB = *pB;
      std::vector<SmBoundaryFaceWork> sWork;
      const double    dBrepTolerance = pBrep->GetTolerance();
      const SmBoolean bThreads       = sm_BrepThreadsSafe(pBrep);
      const SmBoolean bTrimmedFaces  = (crOptions.bTrimmedFaces && dMaxEdgeLength <= 0.0 && dMaxAspectRatio <= 0.0) ? TRUE : FALSE;
      double dBoundariesMs = 0.0, dFacesMs = 0.0;
      auto Fallback = [&](long lFace, SmStatus eStatus)
        {
          rMesh = SmFastMesh();
          rMesh.dBoundariesMs = dBoundariesMs;
          rMesh.lFallbackFace = lFace;
          return eStatus;
        };
      const sm_Clock::time_point sBoundariesStart = sm_Clock::now();
      SmFastBoundaryOptions sBoundaryOptions;
      sBoundaryOptions.pDriver             = &sSrfTess;
      sBoundaryOptions.bStopAtFirstFailure = TRUE;   // such a Brep takes the Phase 1 path: don't finish
      const SmStatus eBoundaries = SmComputeFastBoundaries(cpBrep, sParams, sB, sBoundaryOptions);
      sm_Clock::time_point sFacesStart = sm_Clock::now();
      dBoundariesMs = sm_Ms(sBoundariesStart, sFacesStart);
      if (eBoundaries != SM_SUCCESS)
        {
          long lFace = -1;
          for (size_t ii = 0; ii < sB.sFaces.size() && lFace < 0; ii++)
            {
              if (sB.sFaces[ii].eFailure != SM_FB_NONE) { lFace = (long)ii; }
            }
          return Fallback(lFace, eBoundaries);
        }

      // faces without a boundary: those whose single loop encloses no area in UV, triangulated in 3D; any
      // other, or a face whose faceuse contradicts it (SMLib's mesh of it folds, with mixed orientation, which
      // a clean mesh doesn't match), sends the Brep to the Phase 1 path
      const ULONG lFaces = (ULONG)sB.sFaces.size();
      sWork.assign(lFaces, SmBoundaryFaceWork());
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          SmBoundaryFaceWork & rWork = sWork[ii];
          rWork.cpB = &sB.sFaces[ii];
          if (rWork.cpB->eStatus == SM_SUCCESS && !rWork.cpB->bLoopsReversed)
            {
              rWork.bParallel = (bThreads && SM_CAST_PTR(SmBSplineSurface, rWork.cpB->cpFace->GetSurface()) != NULL) ? TRUE : FALSE;
              continue;
            }
          if (   rWork.cpB->eFailure != SM_FB_DEGENERATE
              || !sm_TriangulateDegenerateFace(*rWork.cpB, sB.sPoints, dBrepTolerance, rWork.sOut))
            { return Fallback((long)ii, SM_ERR_TESS_FAIL_FACE); }
          rWork.bFan  = TRUE;
          rWork.bDone = TRUE;
        }

      // fast triangulation as SmTessellateFast()'s, faces in parallel (surfaces other than BSpline ones
      // serially)
      auto FastFace = [&](ULONG ii)
        {
          SmBoundaryFaceWork & rWork = sWork[ii];
          if (rWork.bDone)
            { return; }
          const SmFastFaceBoundary & crB = *rWork.cpB;
          SmFastFaceInput sIn;
          sIn.cpFace     = crB.cpFace;
          sIn.cpSurface  = crB.cpFace->GetSurface();
          sIn.sParams    = sParams;
          sIn.cpCache    = crB.pCache;
          sIn.bReversed  = crB.bReversed;
          sIn.dTolerance = crB.dTolerance;
          sIn.bParallelSafe = rWork.bParallel;
          sIn.sLoops.resize(crB.sLoops.size());
          for (size_t jj = 0; jj < crB.sLoops.size(); jj++)
            {
              auto & rLoop = sIn.sLoops[jj];
              rLoop.resize(crB.sLoops[jj].size());
              for (size_t kk = 0; kk < rLoop.size(); kk++)
                {
                  const SmFastBoundaryVertex & crV = crB.sLoops[jj][kk];
                  rLoop[kk].dU     = crV.dU;
                  rLoop[kk].dV     = crV.dV;
                  rLoop[kk].sPoint = crV.sPoint;
                  rLoop[kk].lId    = crV.lId;
                }
            }
          rWork.sOut = SmFastFaceOutput();
          const SmStatus eFast = bTrimmedFaces ? SmTriangulateFastFace(sIn, rWork.sOut)
                                               : SmTriangulateRectangularFace(sIn, rWork.sOut);
          if (eFast == SM_SUCCESS && rWork.sOut.eStatus == SM_SUCCESS)
            {
              sm_CollapseShortEdges(rWork.sOut, sB.sPoints, dBrepTolerance / 100.0, rWork.bParallel);
              rWork.bDone = TRUE;
              sm_BoundaryFaceNormals(rWork, crOptions.bCornerNormals);
            }
        };
      for (int iRound = 0; ; iRound++)
        {
          std::vector<ULONG> sParallelFaces, sSerialFaces;
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              if (!sWork[ii].bDone)
                { (sWork[ii].bParallel ? sParallelFaces : sSerialFaces).push_back(ii); }
            }
          sm_ParallelFor(tbb::blocked_range<size_t>(0, sParallelFaces.size(), 1), [&](const tbb::blocked_range<size_t> & crRange)
            {
              for (size_t kk = crRange.begin(); kk != crRange.end(); ++kk)
                { FastFace(sParallelFaces[kk]); }
            });
          for (ULONG ii : sSerialFaces)
            { FastFace(ii); }

          // done, or a round with the segments split, or the Phase 1 path
          long lFirstLeft = -1;
          SmBoolean bSplitsOnly = TRUE;
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              if (sWork[ii].bDone)
                { continue; }
              if (lFirstLeft < 0) { lFirstLeft = (long)ii; }
              if (sWork[ii].sOut.eFailure != SM_FF_SPLIT || sWork[ii].sOut.sSplitSegments.empty())
                { bSplitsOnly = FALSE; }
            }
          if (lFirstLeft < 0)
            { break; }
          if (!bSplitsOnly || iRound >= kSplitRounds)
            {
              dFacesMs += sm_Ms(sFacesStart, sm_Clock::now());
              return Fallback(lFirstLeft, SM_ERR_TESS_FAIL_FACE);
            }
          // the segments' new points on their edges, ids after all others
          std::unordered_map<uint64_t, std::pair<int, int>> sSegmentEdge;   // segment -> edge, point index
          for (size_t e = 0; e < sB.sEdges.size(); e++)
            {
              const std::vector<long> & crIds = sB.sEdges[e].sIds;
              for (size_t k = 0; k + 1 < crIds.size(); k++)
                {
                  if (crIds[k] != crIds[k + 1])
                    { sSegmentEdge.emplace(sm_EdgeKey(crIds[k], crIds[k + 1]), std::make_pair((int)e, (int)k)); }
                }
            }
          struct Split
          {
            long lFrom, lTo;            // the segment's ids, in the edge's direction
            long aIds[3];               // the new points', from lFrom to lTo
            double dTol;
          };
          std::unordered_map<uint64_t, Split> sSplits;
          std::vector<std::pair<int, int>> sEdgeSplits;   // (edge, point index), in order of the requests
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              const std::vector<long> & crSegments = sWork[ii].sOut.sSplitSegments;
              for (size_t kk = 0; !sWork[ii].bDone && kk + 1 < crSegments.size(); kk += 2)
                {
                  const uint64_t lKey = sm_EdgeKey(crSegments[kk], crSegments[kk + 1]);
                  if (sSplits.count(lKey))
                    { continue; }
                  auto sFound = sSegmentEdge.find(lKey);
                  if (sFound == sSegmentEdge.end())
                    { return Fallback((long)ii, SM_ERR_TESS_FAIL_FACE); }
                  const SmFastEdgeBoundary & crE = sB.sEdges[sFound->second.first];
                  const int k = sFound->second.second;
                  const SmCurve * cpCurve = crE.cpEdge ? crE.cpEdge->GetCurve() : NULL;
                  if (cpCurve == NULL)
                    { return Fallback((long)ii, SM_ERR_TESS_FAIL_FACE); }
                  Split sSplit;
                  sSplit.lFrom = crE.sIds[k];
                  sSplit.lTo   = crE.sIds[k + 1];
                  sSplit.dTol  = 10.0 * std::max(dBrepTolerance, (double)crE.cpEdge->GetTolerance());
                  for (int q = 0; q < 3; q++)
                    {
                      SmPoint3d sPoint;
                      if (cpCurve->EvaluatePoint(crE.sParams[k] + 0.25 * (q + 1) * (crE.sParams[k + 1] - crE.sParams[k]), sPoint) != SM_SUCCESS)
                        { return Fallback((long)ii, SM_ERR_TESS_FAIL_FACE); }
                      sSplit.aIds[q] = (long)sB.sPoints.size();
                      sB.sPoints.push_back(sPoint);
                    }
                  sSplits.emplace(lKey, sSplit);
                  sEdgeSplits.push_back(sFound->second);
                }
            }
          // the edges' discretizations (later segments first, so that earlier indices hold)
          std::sort(sEdgeSplits.begin(), sEdgeSplits.end(), [](const std::pair<int, int> & a, const std::pair<int, int> & b)
            { return a.first < b.first || (a.first == b.first && a.second > b.second); });
          for (const std::pair<int, int> & crES : sEdgeSplits)
            {
              SmFastEdgeBoundary & rE = sB.sEdges[crES.first];
              const int k = crES.second;
              const Split & crS = sSplits[sm_EdgeKey(rE.sIds[k], rE.sIds[k + 1])];
              const double t0 = rE.sParams[k], t1 = rE.sParams[k + 1];
              rE.sParams.insert(rE.sParams.begin() + k + 1, {t0 + 0.25 * (t1 - t0), t0 + 0.5 * (t1 - t0), t0 + 0.75 * (t1 - t0)});
              rE.sIds.insert(rE.sIds.begin() + k + 1, {crS.aIds[0], crS.aIds[1], crS.aIds[2]});
            }
          // the points into every loop with the segment, UVs on that face's surface; those faces again
          for (ULONG ii = 0; ii < lFaces; ii++)
            {
              SmFastFaceBoundary & rB = sB.sFaces[ii];
              const SmSurface * cpSurface = rB.cpFace->GetSurface();
              bool bChanged = false;
              for (std::vector<SmFastBoundaryVertex> & rLoop : rB.sLoops)
                {
                  std::vector<SmFastBoundaryVertex> sNew;
                  sNew.reserve(rLoop.size() + 6);
                  for (size_t kk = 0; kk < rLoop.size(); kk++)
                    {
                      const SmFastBoundaryVertex & crA = rLoop[kk], & crB = rLoop[(kk + 1) % rLoop.size()];
                      sNew.push_back(crA);
                      auto sFound = (crA.lId != crB.lId) ? sSplits.find(sm_EdgeKey(crA.lId, crB.lId)) : sSplits.end();
                      if (sFound == sSplits.end())
                        { continue; }
                      const Split & crS = sFound->second;
                      const bool bForward = (crA.lId == crS.lFrom);
                      for (int q = 0; q < 3; q++)
                        {
                          const long lId = crS.aIds[bForward ? q : 2 - q];
                          SmFastBoundaryVertex sV;
                          sV.lId    = lId;
                          sV.sPoint = sB.sPoints[lId];
                          SmPoint2d sUV;
                          if (!sm_SegmentUV(cpSurface, SmPoint2d(crA.dU, crA.dV), SmPoint2d(crB.dU, crB.dV), 0.25 * (q + 1),
                                            sV.sPoint, crS.dTol, sUV))
                            { return Fallback((long)ii, SM_ERR_TESS_FAIL_FACE); }
                          sV.dU = sUV.x;
                          sV.dV = sUV.y;
                          sNew.push_back(sV);
                        }
                      bChanged = true;
                    }
                  rLoop.swap(sNew);
                }
              if (sWork[ii].bFan)
                {
                  // a face without area in UV next to the segment: its loop triangulated again
                  if (bChanged && !sm_TriangulateDegenerateFace(rB, sB.sPoints, dBrepTolerance, sWork[ii].sOut))
                    { return Fallback((long)ii, SM_ERR_TESS_FAIL_FACE); }
                  continue;
                }
              if (bChanged || !sWork[ii].bDone)
                {
                  sWork[ii].bDone = FALSE;
                  sWork[ii].sOut  = SmFastFaceOutput();
                  sWork[ii].sLoopNormals.clear();
                  sWork[ii].sCornerNormals.clear();
                }
            }
          rMesh.lSplitRounds++;
        }
      dFacesMs += sm_Ms(sFacesStart, sm_Clock::now());
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          if (sWork[ii].bFan)
            { sm_BoundaryFaceNormals(sWork[ii], crOptions.bCornerNormals); }
        }
      rMesh.lFaces        = lFaces;
      rMesh.dBoundariesMs = dBoundariesMs;
      const sm_Clock::time_point sFacesEnd = sm_Clock::now();
      rMesh.dFacesMs = dFacesMs;

      // assembly as SmTessellateFast()'s: loop vertices first (by id), then each face's new vertices
      const long lLoopVerts = (long)sB.sPoints.size();
      std::vector<long> sFaceBase(lFaces + 1, lLoopVerts);
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          const SmBoundaryFaceWork & crWork = sWork[ii];
          sFaceBase[ii + 1] = sFaceBase[ii] + (long)crWork.sOut.sNewPoints.size();
          (crWork.bFan ? rMesh.lFallbackFaces : rMesh.lFastFaces)++;
          if (crWork.bFan) { rMesh.lFanFaces++; }
          rMesh.sFaceKinds.push_back((int8_t)(crWork.bFan ? 2 : 1));
        }
      const long lVerts = sFaceBase[lFaces];
      if (lVerts < 0 || lVerts > (long)INT32_MAX)   // the arrays' indices are 32-bit
        {
          rMesh = SmFastMesh();
          SER_MSG(SM_ERR, _T("SmTessellateFast: more vertices than 32-bit indices hold"));
        }
      rMesh.sPoints.resize((size_t)lVerts * 3);
      std::vector<SmVector3d> sNormalSums((size_t)lVerts, SmVector3d(0, 0, 0));
      for (long ii = 0; ii < lLoopVerts; ii++)
        {
          rMesh.sPoints[3 * (size_t)ii]     = sB.sPoints[ii].x;
          rMesh.sPoints[3 * (size_t)ii + 1] = sB.sPoints[ii].y;
          rMesh.sPoints[3 * (size_t)ii + 2] = sB.sPoints[ii].z;
        }
      size_t lTriangles = 0;
      for (ULONG ii = 0; ii < lFaces; ii++)
        { lTriangles += sWork[ii].sOut.sTriangles.size() / 3; }
      rMesh.sTriangles.reserve(lTriangles * 3);
      rMesh.sTriangleFaces.reserve(lTriangles);
      if (crOptions.bCornerNormals)
        { rMesh.sCornerNormals.reserve(lTriangles * 9); }
      for (ULONG ii = 0; ii < lFaces; ii++)
        {
          const SmBoundaryFaceWork & crWork = sWork[ii];
          const SmFastFaceOutput   & crOut  = crWork.sOut;
          const SmBoolean bReversed = crWork.cpB->bReversed;
          const SmBoolean bSwap     = bReversed;
          const double    dSign     = bReversed ? -1.0 : 1.0;
          const long      lBase     = sFaceBase[ii];
          for (size_t kk = 0; kk < crOut.sNewPoints.size(); kk++)
            {
              const size_t lV = (size_t)lBase + kk;
              rMesh.sPoints[3 * lV]     = crOut.sNewPoints[kk].x;
              rMesh.sPoints[3 * lV + 1] = crOut.sNewPoints[kk].y;
              rMesh.sPoints[3 * lV + 2] = crOut.sNewPoints[kk].z;
              sNormalSums[lV] = crOut.sNewNormals[kk] * dSign;
            }
          for (const auto & crN : crWork.sLoopNormals)
            { sNormalSums[crN.first] += crN.second * dSign; }
          const std::vector<long> & crTris = crOut.sTriangles;
          for (size_t kk = 0; kk + 2 < crTris.size(); kk += 3)
            {
              long aV[3];
              for (int c = 0; c < 3; c++)
                {
                  const long r = crTris[kk + c];
                  aV[c] = (r >= 0) ? r : lBase + ~r;
                }
              if (aV[0] == aV[1] || aV[1] == aV[2] || aV[2] == aV[0])
                {
                  rMesh.lDroppedTriangles++;
                  continue;
                }
              if (bSwap)
                { std::swap(aV[1], aV[2]); }
              for (int c = 0; c < 3; c++)
                { rMesh.sTriangles.push_back((int32_t)aV[c]); }
              rMesh.sTriangleFaces.push_back((int32_t)ii);
              if (crOptions.bCornerNormals)
                {
                  int aC[3] = {0, 1, 2};
                  if (bSwap)
                    { std::swap(aC[1], aC[2]); }
                  for (int c = 0; c < 3; c++)
                    {
                      const SmVector3d sN = crWork.sCornerNormals[kk + aC[c]] * dSign;
                      rMesh.sCornerNormals.push_back((float)sN.x);
                      rMesh.sCornerNormals.push_back((float)sN.y);
                      rMesh.sCornerNormals.push_back((float)sN.z);
                    }
                }
            }
        }
      // drop vertices no triangle uses (as SMLib deletes standalone PolyVertices), keeping the order
      std::vector<int32_t> sNewIndex((size_t)lVerts, -1);
      for (int32_t lV : rMesh.sTriangles)
        { sNewIndex[lV] = 0; }
      size_t lUsed = 0;
      for (size_t ii = 0; ii < (size_t)lVerts; ii++)
        {
          if (sNewIndex[ii] < 0)
            { continue; }
          sNewIndex[ii] = (int32_t)lUsed;   // lUsed <= ii < lVerts <= INT32_MAX
          if (lUsed != ii)
            {
              for (size_t a = 0; a < 3; a++) { rMesh.sPoints[3 * lUsed + a] = rMesh.sPoints[3 * ii + a]; }
              sNormalSums[lUsed] = sNormalSums[ii];
            }
          lUsed++;
        }
      for (int32_t & rV : rMesh.sTriangles)
        { rV = sNewIndex[rV]; }
      rMesh.sPoints.resize(lUsed * 3);
      rMesh.sNormals.resize(lUsed * 3);
      for (size_t ii = 0; ii < lUsed; ii++)
        {
          SmVector3d sN = sNormalSums[ii];
          const double dLength = sN.Length();
          if (std::isfinite(dLength) && dLength > 1e-30) { sN /= dLength; }
          else                                           { sN = SmVector3d(0, 0, 1); }
          rMesh.sNormals[3 * ii]     = (float)sN.x;
          rMesh.sNormals[3 * ii + 1] = (float)sN.y;
          rMesh.sNormals[3 * ii + 2] = (float)sN.z;
        }
      sCleanupStart = sm_Clock::now();
      rMesh.dAssemblyMs = sm_Ms(sFacesEnd, sCleanupStart);
    } // sB deletes the refinement trees
  rMesh.dCleanupMs = sm_Ms(sCleanupStart, sm_Clock::now());
  return SM_SUCCESS;

} // end SmTessellateWithFastBoundaries
