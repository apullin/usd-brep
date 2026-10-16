// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmTessFast.cpp
* PURPOSE: Opt-in fast tessellation of a Brep into flat triangle arrays
*          (see SmTessFast.h).
**********************************************************************/

#include "StdAfx.h"

#include <SmTessFast.h>
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

SmFastFaceTriangulator gs_pfnTrimmedTriangulator = NULL;

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
  // can disagree in the last bits about a shared split. Use one value per split: values closer than
  // 1e-12 of the domain size become the smallest of them (the domain bounds win), and leaves that
  // collapse to zero width are dropped.
    {
      const bool bPar = crIn.bParallelSafe ? true : false;
      auto Canonical = [&](double SmLeafRect::*pLo, double SmLeafRect::*pHi, double dMin, double dMax) -> SmBoolean
        {
          const double dEps = 1e-12 * (dMax - dMin);
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
  std::vector<int>     sLeafCorner((size_t)lLeaves * 4);   // BL, BR, TR, TL vertex of each leaf
    {
      struct SmCornerKey { double u, v; int lc; };
      std::vector<SmCornerKey> sAll((size_t)lLeaves * 4);
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
      for (size_t i = 0; i < sAll.size(); i++)
        {
          if (sVerts.empty() || sAll[i].u != sVerts.back().u || sAll[i].v != sVerts.back().v)
            {
              sVerts.push_back({sAll[i].u, sAll[i].v});
              sCornerCount.push_back(0);
            }
          sCornerCount.back()++;
          sLeafCorner[sAll[i].lc] = (int)sVerts.size() - 1;
        }
    }
  const int lVerts = (int)sVerts.size();
  std::vector<int> sNextH(lVerts, -1), sNextV(lVerts, -1);
    {
      std::vector<int> sByV(lVerts);
      std::iota(sByV.begin(), sByV.end(), 0);
      auto LessVU = [&](int a, int b) { return sm_LessVU(sVerts[a], sVerts[b]); };
      if (crIn.bParallelSafe) { sm_RunIsolated([&] { tbb::parallel_sort(sByV.begin(), sByV.end(), LessVU); }); }
      else                    { std::sort(sByV.begin(), sByV.end(), LessVU); }
      for (int k = 0; k + 1 < lVerts; k++)
        {
          if (sVerts[k + 1].u == sVerts[k].u) { sNextV[k] = k + 1; }
          if (sVerts[sByV[k + 1]].v == sVerts[sByV[k]].v) { sNextH[sByV[k]] = sByV[k + 1]; }
        }
    }

  // 5. polygon references of the vertices: r >= 0 new vertex r (interior), r >= lNew pole point r - lNew,
  //    r < 0 loop vertex ~r (corners and snapped boundary points)
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
          sRef[i] = (int)sNewVert.size();
          sNewVert.push_back(i);
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
SmStatus SmTriangulateFastFace(const SmFastFaceInput & crIn, SmFastFaceOutput & rOut)
{
  if (SmTriangulateRectangularFace(crIn, rOut) == SM_SUCCESS)
    { return SM_SUCCESS; }
  if (gs_pfnTrimmedTriangulator != NULL)
    {
      rOut = SmFastFaceOutput();
      if (gs_pfnTrimmedTriangulator(crIn, rOut) == SM_SUCCESS && rOut.eStatus == SM_SUCCESS)
        { return SM_SUCCESS; }
    }
  rOut = SmFastFaceOutput();
  return SM_ERR;
}

void SmSetTrimmedFaceTriangulator(SmFastFaceTriangulator pfnTriangulator)
{
  gs_pfnTrimmedTriangulator = pfnTriangulator;
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

// one surface normal (not reversed) per loop vertex of the face, as SMLib's output has: at the UV of the
// vertex's first use in the face's triangles (seams, poles and creases have several), or at its first
// loop occurrence when the output has no corner UVs; and the normal at each triangle corner
SmStatus sm_FaceNormals(SmFastFaceWork & rWork, SmBoolean bCorners)
{
  const SmFastFaceOutput & crOut = rWork.sOut;
  const size_t    lCorners  = crOut.sTriangles.size();
  const SmBoolean bUVs      = (crOut.sTriangleUVs.size() == lCorners) ? TRUE : FALSE;
  const SmSurface * cpSurface = rWork.pFace->GetSurface();
  std::unordered_map<long, SmPoint2d> sFirstLoopUV;
  if (!bUVs)
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
  std::unordered_map<long, size_t> sIndex;  // loop id -> rWork.sLoopNormals index
  rWork.sLoopNormals.clear();
  rWork.sCornerNormals.assign(bCorners ? lCorners : 0, SmVector3d(0, 0, 0));
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
              auto sFound = sIndex.find(r);
              if (sFound == sIndex.end())
                {
                  SmPoint2d sUV = bUVs ? crOut.sTriangleUVs[kk + c] : sFirstLoopUV[r];
                  if (cpSurface->EvaluateNormal(sUV, TRUE, TRUE, sNormal) != SM_SUCCESS)
                    { sNormal = SmVector3d(0, 0, 0); }
                  sFound = sIndex.emplace(r, rWork.sLoopNormals.size()).first;
                  rWork.sLoopNormals.push_back({r, sNormal});
                }
              sNormal = rWork.sLoopNormals[sFound->second].second;
            }
          if (bCorners)
            { rWork.sCornerNormals[kk + c] = sNormal; }
        }
    }
  return SM_SUCCESS;
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
              rWork.bParallel = TRUE;
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

      // fast triangulation, faces in parallel (surfaces other than BSpline ones serially)
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
          if (SmTriangulateFastFace(sIn, rWork.sOut) == SM_SUCCESS && rWork.sOut.eStatus == SM_SUCCESS)
            {
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
      rMesh.sPoints.resize((size_t)lVerts * 3);
      std::vector<SmVector3d> sNormalSums((size_t)lVerts, SmVector3d(0, 0, 0));
      for (long ii = 0; ii < lLoopVerts; ii++)
        {
          rMesh.sPoints[3 * ii]     = sLoopPoints[ii].x;
          rMesh.sPoints[3 * ii + 1] = sLoopPoints[ii].y;
          rMesh.sPoints[3 * ii + 2] = sLoopPoints[ii].z;
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
              const long lV = lBase + (long)kk;
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
      int32_t lUsed = 0;
      for (long ii = 0; ii < lVerts; ii++)
        {
          if (sNewIndex[ii] < 0)
            { continue; }
          sNewIndex[ii] = lUsed;
          if (lUsed != ii)
            {
              for (int a = 0; a < 3; a++) { rMesh.sPoints[3 * lUsed + a] = rMesh.sPoints[3 * ii + a]; }
              sNormalSums[lUsed] = sNormalSums[ii];
            }
          lUsed++;
        }
      for (int32_t & rV : rMesh.sTriangles)
        { rV = sNewIndex[rV]; }
      rMesh.sPoints.resize((size_t)lUsed * 3);
      rMesh.sNormals.resize((size_t)lUsed * 3);
      for (long ii = 0; ii < lUsed; ii++)
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
    SmZoneTol3d           sZoneTol3d;
    SmTol::ThreadDefaults sDefaults;  // of the calling thread, for the tasks
    tbb::enumerable_thread_specific<Pool> * pPool;  // nodes, per thread that splits


    // SmSurfaceCache::TestAgainstTolerances() of an SM_AD_AUX_DATA node
    void Test(Node & rN) const
    {
      SmTessSrfCache & rC = *pCache;
      rN.bNeedsSub = FALSE;
      rN.bSetFast  = FALSE;
      rN.eDir      = SM_SP_U;
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
      const SmBoolean bFork = IsLarge(rN);
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

  const Worker sWorker = {this, (m_eTessellationAlgorithm == SM_TA_FASTER_TESSELLATION) ? TRUE : FALSE,
                          SmTol::GetZoneTol3d(m_cpSurface), SmTol::GetThreadDefaults(), &sPool};
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
      sm_ParallelFor(size_t(0), sTasks.size(), [&](size_t ii)
        {
          if (ii < lBlocks)
            { sWorker.Block(*sTasks[ii]); }
          sTasks[ii]->FreeSurface();
        });
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
