// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmFastFace.cpp
* PURPOSE: SmTriangulateTrimmedFace(): display triangulation of one trimmed
*    face from its boundary loops.
*
* METHOD:
*   1. Map UV affinely to a working plane whose metric averages, over
*      the loops' UV box, the first fundamental form (over a length
*      scale) plus the third fundamental form (normal turning over the
*      angle tolerance): working-plane lengths measure both 3D size and
*      normal turning, so thin cylinder strips, cones and stretched
*      parameterizations triangulate like their 3D shape.
*   2. Delaunay-triangulate the loop vertices inside a large enclosing
*      triangle, insert the loop segments as constraints (edge flips),
*      and classify the triangles: the face lies to the left of every
*      directed loop segment (faces whose loops don't bound a region
*      that way are rejected).
*   3. Interior vertices: the corners of the refinement tree leaves that
*      lie inside the face without touching its loops (SMLib's interior
*      vertex set; the leaf sides become constraints, so each leaf is
*      triangulated on its own as SMLib does with quadrilaterals), plus
*      any given samples clear of the boundary.
*   4. Flip interior edges where that brings the facet normals closer to
*      the surface normals at the corners (diagonals across curvature
*      that the working-plane criterion can't see), then where the other
*      diagonal bulges clearly less from the surface (leaf diagonals by
*      the turning of the corner normals, other edges by the surface's
*      offset at their UV midpoint). Then refine, best effort: triangles
*      whose facet normal strays from the surface normals beyond the
*      angle tolerance, that deviate from the surface, exceed the maximum
*      edge length, are flipped or degenerate, get a new vertex at the
*      working-plane circumcenter (or the centroid when that lies outside
*      the face or crowds its boundary in 3D). With tree leaves, which
*      meet the tolerances already, only the triangles between the leaves
*      and the loops are checked, against the deviation the angle
*      tolerance allows for their size. Loop segments are never split.
*      When refinement doesn't converge within its budget or leaves
*      invalid triangles away from the loops, the unrefined triangulation
*      is used.
*
*   Orientation tests are exact (adaptive filter, then expansion
*   arithmetic); in-circle tests only flip edges when the filtered
*   determinant is certain, which keeps flips valid without exact
*   in-circle arithmetic (ties are left as they are). Everything is
*   deterministic: nothing depends on hashing order, and the
*   point-location walk uses a fixed-seed generator.
**********************************************************************/

#include "StdAfx.h"

#include <SmFastFace.h>

#include <SmBrep.h>
#include <SmFace.h>
#include <SmFaceuse.h>
#include <SmShell.h>
#include <SmSurface.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef SM_FASTFACE_DEBUG
SmTrimFaceDebug & SmTrimFaceDebugState()
{
  static thread_local SmTrimFaceDebug tState;
  return tState;
}
static thread_local long t_aReasons[8];
#define FF_REASON(k) (++t_aReasons[k])
#else
#define FF_REASON(k) ((void)0)
#endif

namespace
{

// ---------------------------------------------------------------------------------------------
// Robust predicates (after J. R. Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast
// Robust Geometric Predicates", 1997)

const double kEpsilon     = std::numeric_limits<double>::epsilon() * 0.5;  // 2^-53
const double kCcwErrBound = (3.0 + 16.0 * kEpsilon) * kEpsilon;

inline void TwoSum(double a, double b, double & x, double & y)
{
  x = a + b;
  const double bv = x - a;
  const double av = x - bv;
  y = (a - av) + (b - bv);
}

// x + y == a * b exactly (fma is exact whatever the compiler's contraction settings)
inline void TwoProduct(double a, double b, double & x, double & y)
{
  x = a * b;
  y = std::fma(a, b, -x);
}

// e += b for a nonoverlapping expansion e of length elen (in place, zero components removed)
int GrowExpansion(int elen, double * e, double b)
{
  double q = b;
  int h = 0;
  for (int i = 0; i < elen; ++i)
    {
      double qnew, hh;
      TwoSum(q, e[i], qnew, hh);
      q = qnew;
      if (hh != 0.0) e[h++] = hh;
    }
  if (q != 0.0 || h == 0) e[h++] = q;
  return h;
}

double Orient2dExact(double ax, double ay, double bx, double by, double cx, double cy)
{
  double e[16];
  int n = 0;
  auto add = [&](double p, double q)
    {
      double x, y;
      TwoProduct(p, q, x, y);
      n = GrowExpansion(n, e, y);
      n = GrowExpansion(n, e, x);
    };
  add(ax, by);
  add(-ay, bx);
  add(bx, cy);
  add(-by, cx);
  add(cx, ay);
  add(-cy, ax);
  return e[n - 1];
}

// > 0 when a, b, c are counterclockwise, < 0 clockwise, 0 collinear (exact sign)
inline double Orient2d(double ax, double ay, double bx, double by, double cx, double cy)
{
  const double detleft  = (ax - cx) * (by - cy);
  const double detright = (ay - cy) * (bx - cx);
  const double det      = detleft - detright;
  double detsum;
  if (detleft > 0.0)
    {
      if (detright <= 0.0) return det;
      detsum = detleft + detright;
    }
  else if (detleft < 0.0)
    {
      if (detright >= 0.0) return det;
      detsum = -detleft - detright;
    }
  else
    return det;
  const double errbound = kCcwErrBound * detsum;
  if (det >= errbound || -det >= errbound) return det;
  return Orient2dExact(ax, ay, bx, by, cx, cy);
}

// +1 when d is certainly inside the circle through the counterclockwise a, b, c; -1 when certainly
// outside; 0 when the floating-point filter can't tell (nearly cocircular)
inline int InCircle(double ax, double ay, double bx, double by, double cx, double cy, double dx, double dy)
{
  const double adx = ax - dx, bdx = bx - dx, cdx = cx - dx;
  const double ady = ay - dy, bdy = by - dy, cdy = cy - dy;
  const double bdxcdy = bdx * cdy, cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady, adxcdy = adx * cdy;
  const double adxbdy = adx * bdy, bdxady = bdx * ady;
  const double alift = adx * adx + ady * ady;
  const double blift = bdx * bdx + bdy * bdy;
  const double clift = cdx * cdx + cdy * cdy;
  const double det = alift * (bdxcdy - cdxbdy) + blift * (cdxady - adxcdy) + clift * (adxbdy - bdxady);
  const double permanent = (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift
                         + (std::fabs(cdxady) + std::fabs(adxcdy)) * blift
                         + (std::fabs(adxbdy) + std::fabs(bdxady)) * clift;
  const double errbound = (10.0 + 96.0 * kEpsilon) * kEpsilon * permanent;
  if (det > errbound) return 1;
  if (-det > errbound) return -1;
  return 0;
}

inline int Sign(double d) { return (d > 0.0) - (d < 0.0); }

// ---------------------------------------------------------------------------------------------
// Constrained Delaunay triangulation

// Triangle with counterclockwise vertices; edge i is opposite v[i], from v[i+1] to v[i+2], and
// n[i] is the triangle across it (-1 for none).
struct Tri
{
  int           v[3];
  int           n[3];
  unsigned char c;      // bit i: edge i is a loop segment; bit i + 3: edge i is a leaf side
  signed char   label;  // +1 inside the face, -1 outside, 0 unknown
  unsigned char state;  // refinement: kUnchecked, kGood, kDropped
};

enum { kUnchecked = 0, kGood = 1, kDropped = 2 };
enum { kLocIn, kLocEdge, kLocVertex, kLocFail, kLocBlocked };

inline int Next3(int i) { return (i == 2) ? 0 : i + 1; }
inline int Prev3(int i) { return (i == 0) ? 2 : i - 1; }

// Constraint bits of edge e: 1 = loop segment, 2 = leaf side (both may be set)
inline int EdgeBits(unsigned char c, int e) { return ((c >> e) & 1) | (((c >> (e + 3)) & 1) << 1); }
inline unsigned char PutBits(int e, int bits) { return (unsigned char)(((bits & 1) << e) | (((bits >> 1) & 1) << (e + 3))); }
const int kLoopEdge = 1, kLeafEdge = 2;

class Cdt
{
public:
  std::vector<double>              mX, mY;    // working-plane coordinates; vertices 0-2 enclose the rest
  std::vector<int>                 mVT;       // a triangle that has each vertex
  std::vector<Tri>                 mT;
  std::vector<std::pair<int, int>> mStack;    // edges to legalize
  std::vector<int>                 mTouched;  // triangles changed since the last ClearTouched()
  int                              mLast = 0; // where the next walk starts
  uint32_t                         mRand = 0;

  void Clear()
  {
    mX.clear(); mY.clear(); mVT.clear(); mT.clear(); mStack.clear(); mTouched.clear();
    mLast = 0;
    mRand = 2463534242u;
  }

  int NumVertices() const { return (int)mX.size(); }

  int AddVertex(double x, double y)
  {
    mX.push_back(x);
    mY.push_back(y);
    mVT.push_back(-1);
    return (int)mX.size() - 1;
  }

  double Orient(int a, int b, int c) const { return Orient2d(mX[a], mY[a], mX[b], mY[b], mX[c], mY[c]); }
  double OrientP(int a, int b, double px, double py) const { return Orient2d(mX[a], mY[a], mX[b], mY[b], px, py); }

  uint32_t NextRand()
  {
    mRand ^= mRand << 13;
    mRand ^= mRand >> 17;
    mRand ^= mRand << 5;
    return mRand;
  }

  void Touch(int t)
  {
    mT[t].state = kUnchecked;
    mTouched.push_back(t);
  }

  int NewTri()
  {
    mT.emplace_back();
    Tri & rT = mT.back();
    rT.c = 0;
    rT.label = 0;
    rT.state = kUnchecked;
    return (int)mT.size() - 1;
  }

  int NeighborIndex(int t, int nb) const
  {
    const Tri & rT = mT[t];
    if (rT.n[0] == nb) return 0;
    if (rT.n[1] == nb) return 1;
    return 2;
  }

  void ReplaceNeighbor(int t, int oldNb, int newNb)
  {
    if (t < 0) return;
    Tri & rT = mT[t];
    for (int k = 0; k < 3; ++k)
      if (rT.n[k] == oldNb) { rT.n[k] = newNb; return; }
  }

  // Enclosing triangle for points in [xmin,xmax] x [ymin,ymax]
  void Init(double xmin, double ymin, double xmax, double ymax)
  {
    const double cx = 0.5 * (xmin + xmax), cy = 0.5 * (ymin + ymax);
    double s = std::max(xmax - xmin, ymax - ymin);
    if (!(s > 0.0)) s = 1.0;
    const double m = 64.0 * s;
    AddVertex(cx - 2.0 * m, cy - m);
    AddVertex(cx + 2.0 * m, cy - m);
    AddVertex(cx, cy + 2.0 * m);
    const int t = NewTri();
    Tri & rT = mT[t];
    rT.v[0] = 0; rT.v[1] = 1; rT.v[2] = 2;
    rT.n[0] = rT.n[1] = rT.n[2] = -1;
    mVT[0] = mVT[1] = mVT[2] = t;
    mLast = t;
  }

  // Finds the triangle containing (px, py), walking from triangle t. kLocEdge: on edge rIdx of
  // rTri; kLocVertex: on vertex rTri.v[rIdx]. With bStopAtConstraints the walk doesn't cross
  // constraints, loop segments or leaf sides (kLocBlocked).
  int Locate(double px, double py, int t, int & rTri, int & rIdx, bool bStopAtConstraints)
  {
    if (t < 0 || t >= (int)mT.size()) t = mLast;
    int prev = -1;
    const int lMaxSteps = 4 * (int)mT.size() + 64;
    for (int step = 0; step < lMaxSteps; ++step)
      {
        const Tri & rT = mT[t];
        const int r = (int)(NextRand() % 3);
        int next = -2;
        for (int k = 0; k < 3; ++k)
          {
            const int e = (r + k) % 3;
            const int nb = rT.n[e];
            if (prev >= 0 && nb == prev) continue;
            if (OrientP(rT.v[Next3(e)], rT.v[Prev3(e)], px, py) < 0.0)
              {
                if (nb < 0) return kLocFail;
                if (bStopAtConstraints && EdgeBits(rT.c, e)) return kLocBlocked;
                next = nb;
                break;
              }
          }
        if (next == -2)
          {
            double o[3];
            int zeros = 0, zeroEdge = -1;
            for (int e = 0; e < 3; ++e)
              {
                o[e] = OrientP(rT.v[Next3(e)], rT.v[Prev3(e)], px, py);
                if (o[e] == 0.0) { ++zeros; zeroEdge = (zeroEdge < 0) ? e : zeroEdge; }
              }
            rTri = t;
            mLast = t;
            if (zeros == 0) return kLocIn;
            if (zeros == 1) { rIdx = zeroEdge; return kLocEdge; }
            // on the vertex shared by two edges with zero orientation
            for (int k = 0; k < 3; ++k)
              if (o[k] != 0.0) { rIdx = k; return kLocVertex; }
            return kLocFail;  // degenerate triangle
          }
        prev = t;
        t = next;
      }
    return kLocFail;
  }

  // Flips edge i of triangle t: (o, u, w) + (o2, w, u) become t = (o, u, o2), nb = (o2, w, o).
  void Flip(int t, int i)
  {
    const int nb = mT[t].n[i];
    const int j = NeighborIndex(nb, t);
    Tri & rT = mT[t];
    Tri & rN = mT[nb];
    const int o = rT.v[i], u = rT.v[Next3(i)], w = rT.v[Prev3(i)];
    const int o2 = rN.v[j];
    const int nW = rT.n[Next3(i)], nU = rT.n[Prev3(i)];
    const int cW = EdgeBits(rT.c, Next3(i)), cU = EdgeBits(rT.c, Prev3(i));
    const int mU = rN.n[Next3(j)], mW = rN.n[Prev3(j)];
    const int dU = EdgeBits(rN.c, Next3(j)), dW = EdgeBits(rN.c, Prev3(j));
    rT.v[0] = o;  rT.v[1] = u; rT.v[2] = o2;
    rT.n[0] = mU; rT.n[1] = nb; rT.n[2] = nU;
    rT.c = PutBits(0, dU) | PutBits(2, cU);
    rN.v[0] = o2; rN.v[1] = w; rN.v[2] = o;
    rN.n[0] = nW; rN.n[1] = t;  rN.n[2] = mW;
    rN.c = PutBits(0, cW) | PutBits(2, dW);
    ReplaceNeighbor(mU, nb, t);
    ReplaceNeighbor(nW, t, nb);
    mVT[o] = t; mVT[u] = t; mVT[o2] = t; mVT[w] = nb;
    Touch(t);
    Touch(nb);
  }

  // Whether edge i of triangle t isn't locally Delaunay (certainly, see InCircle()), which implies
  // that the quadrilateral is strictly convex, so flipping it is valid
  bool ShouldFlip(int t, int i) const
  {
    const Tri & rT = mT[t];
    if (EdgeBits(rT.c, i)) return false;
    const int nb = rT.n[i];
    if (nb < 0) return false;
    const int o = rT.v[i], u = rT.v[Next3(i)], w = rT.v[Prev3(i)];
    const int o2 = mT[nb].v[NeighborIndex(nb, t)];
    return InCircle(mX[o], mY[o], mX[u], mY[u], mX[w], mY[w], mX[o2], mY[o2]) > 0;
  }

  // Restores the Delaunay property from the edges on mStack.
  void Legalize()
  {
    while (!mStack.empty())
      {
        const int t = mStack.back().first, i = mStack.back().second;
        mStack.pop_back();
        if (!ShouldFlip(t, i)) continue;
        const int nb = mT[t].n[i];
        Flip(t, i);
        mStack.emplace_back(t, 0);
        mStack.emplace_back(t, 2);
        mStack.emplace_back(nb, 0);
        mStack.emplace_back(nb, 2);
      }
  }

  // Inserts a new vertex strictly inside triangle t.
  int InsertInTriangle(int t, double px, double py)
  {
    const int p = AddVertex(px, py);
    const Tri old = mT[t];
    const int a = old.v[0], b = old.v[1], c = old.v[2];
    const int tA = t, tB = NewTri(), tC = NewTri();
    Tri & rA = mT[tA];
    rA.v[0] = p; rA.v[1] = b; rA.v[2] = c;
    rA.n[0] = old.n[0]; rA.n[1] = tB; rA.n[2] = tC;
    rA.c = PutBits(0, EdgeBits(old.c, 0));
    Tri & rB = mT[tB];
    rB.v[0] = p; rB.v[1] = c; rB.v[2] = a;
    rB.n[0] = old.n[1]; rB.n[1] = tC; rB.n[2] = tA;
    rB.c = PutBits(0, EdgeBits(old.c, 1));
    rB.label = old.label;
    Tri & rC = mT[tC];
    rC.v[0] = p; rC.v[1] = a; rC.v[2] = b;
    rC.n[0] = old.n[2]; rC.n[1] = tA; rC.n[2] = tB;
    rC.c = PutBits(0, EdgeBits(old.c, 2));
    rC.label = old.label;
    ReplaceNeighbor(old.n[1], t, tB);
    ReplaceNeighbor(old.n[2], t, tC);
    mVT[p] = tA; mVT[b] = tA; mVT[c] = tA; mVT[a] = tB;
    Touch(tA); Touch(tB); Touch(tC);
    mStack.emplace_back(tA, 0);
    mStack.emplace_back(tB, 0);
    mStack.emplace_back(tC, 0);
    Legalize();
    mLast = mVT[p];
    return p;
  }

  // Inserts a new vertex on edge i of triangle t (which has a neighbor across it).
  int InsertOnEdge(int t, int i, double px, double py)
  {
    const int t2 = mT[t].n[i];
    if (t2 < 0) return -1;
    const int p = AddVertex(px, py);
    const Tri oT = mT[t];
    const int j = NeighborIndex(t2, t);
    const Tri oN = mT[t2];
    const int o = oT.v[i], u = oT.v[Next3(i)], w = oT.v[Prev3(i)];
    const int o2 = oN.v[j];
    const int nW = oT.n[Next3(i)], nU = oT.n[Prev3(i)];
    const int cW = EdgeBits(oT.c, Next3(i)), cU = EdgeBits(oT.c, Prev3(i));
    const int mU = oN.n[Next3(j)], mW = oN.n[Prev3(j)];
    const int dU = EdgeBits(oN.c, Next3(j)), dW = EdgeBits(oN.c, Prev3(j));
    const int cs = EdgeBits(oT.c, i);
    const int A = t, B = NewTri(), C = t2, D = NewTri();
    Tri & rA = mT[A];
    rA.v[0] = p; rA.v[1] = w; rA.v[2] = o;
    rA.n[0] = nW; rA.n[1] = B; rA.n[2] = D;
    rA.c = PutBits(0, cW) | PutBits(2, cs);
    Tri & rB = mT[B];
    rB.v[0] = p; rB.v[1] = o; rB.v[2] = u;
    rB.n[0] = nU; rB.n[1] = C; rB.n[2] = A;
    rB.c = PutBits(0, cU) | PutBits(1, cs);
    rB.label = oT.label;
    Tri & rC = mT[C];
    rC.v[0] = p; rC.v[1] = u; rC.v[2] = o2;
    rC.n[0] = mU; rC.n[1] = D; rC.n[2] = B;
    rC.c = PutBits(0, dU) | PutBits(2, cs);
    Tri & rD = mT[D];
    rD.v[0] = p; rD.v[1] = o2; rD.v[2] = w;
    rD.n[0] = mW; rD.n[1] = A; rD.n[2] = C;
    rD.c = PutBits(0, dW) | PutBits(1, cs);
    rD.label = oN.label;
    ReplaceNeighbor(nU, t, B);
    ReplaceNeighbor(mW, t2, D);
    mVT[p] = A; mVT[w] = A; mVT[o] = A; mVT[u] = B; mVT[o2] = C;
    Touch(A); Touch(B); Touch(C); Touch(D);
    mStack.emplace_back(A, 0);
    mStack.emplace_back(B, 0);
    mStack.emplace_back(C, 0);
    mStack.emplace_back(D, 0);
    Legalize();
    mLast = mVT[p];
    return p;
  }

  // Finds the triangle with the directed edge a->b: rT.v[rI+1] == a, rT.v[rI+2] == b.
  bool FindEdge(int a, int b, int & rT, int & rI) const
  {
    const int t0 = mVT[a];
    int t = t0;
    for (int guard = 0; guard < 4096 && t >= 0; ++guard)
      {
        const Tri & rTri = mT[t];
        const int k = (rTri.v[0] == a) ? 0 : (rTri.v[1] == a) ? 1 : 2;
        if (rTri.v[Next3(k)] == b) { rT = t; rI = Prev3(k); return true; }
        t = rTri.n[Next3(k)];  // counterclockwise around a
        if (t == t0) break;
      }
    // slow path (very high vertex degree)
    for (int tt = 0; tt < (int)mT.size(); ++tt)
      for (int k = 0; k < 3; ++k)
        if (mT[tt].v[Next3(k)] == a && mT[tt].v[Prev3(k)] == b) { rT = tt; rI = k; return true; }
    return false;
  }

  void SetConstraint(int t, int i, int bits)
  {
    mT[t].c |= PutBits(i, bits);
    const int nb = mT[t].n[i];
    if (nb >= 0) mT[nb].c |= PutBits(NeighborIndex(nb, t), bits);
  }

  // Inserts the constraint segment a-b. Returns false when a vertex lies on it or it crosses
  // another constraint.
  bool InsertSegment(int a, int b, int bits, bool & rbInternal)
  {
    rbInternal = false;
    if (a == b) return true;
    int t, i;
    if (FindEdge(a, b, t, i)) { SetConstraint(t, i, bits); return true; }
    const double ax = mX[a], ay = mY[a], bx = mX[b], by = mY[b];
    auto between = [&](int v)  // v (collinear with a-b) strictly between a and b
      {
        const double dx = bx - ax, dy = by - ay;
        const double s = (mX[v] - ax) * dx + (mY[v] - ay) * dy;
        return s > 0.0 && s < dx * dx + dy * dy;
      };
    // the triangle around a that the segment leaves through its edge opposite a
    int start = -1, startEdge = -1;
      {
        const int t0 = mVT[a];
        int tt = t0;
        for (int guard = 0; guard < 4096 && tt >= 0; ++guard)
          {
            const Tri & rTri = mT[tt];
            const int k = (rTri.v[0] == a) ? 0 : (rTri.v[1] == a) ? 1 : 2;
            const int x = rTri.v[Next3(k)], y = rTri.v[Prev3(k)];
            const double ox = OrientP(a, b, mX[x], mY[x]);
            const double oy = OrientP(a, b, mX[y], mY[y]);
            if (ox == 0.0 && between(x)) return false;
            if (oy == 0.0 && between(y)) return false;
            if (ox < 0.0 && oy > 0.0) { start = tt; startEdge = k; break; }
            tt = rTri.n[Next3(k)];
            if (tt == t0) break;
          }
      }
    if (start < 0) { rbInternal = true; return false; }
    // walk to b, collecting the crossed edges as (right, left) vertex pairs
    std::vector<std::pair<int, int>> & rCross = mCross;
    rCross.clear();
    t = start;
    i = startEdge;
    for (int guard = 0; guard < 1000000; ++guard)
      {
        const Tri & rTri = mT[t];
        if (EdgeBits(rTri.c, i)) return false;  // crosses a constraint
        const int right = rTri.v[Next3(i)], left = rTri.v[Prev3(i)];
        rCross.emplace_back(right, left);
        const int t2 = rTri.n[i];
        if (t2 < 0) { rbInternal = true; return false; }
        const int j = NeighborIndex(t2, t);
        const int z = mT[t2].v[j];
        if (z == b) break;
        const double oz = OrientP(a, b, mX[z], mY[z]);
        if (oz == 0.0) return false;  // z lies on the segment
        t = t2;
        i = (oz < 0.0) ? Prev3(j) : Next3(j);  // leave through z-left or right-z
      }
    // flip the crossed edges away (Sloan 1993)
    std::vector<std::pair<int, int>> & rNew = mNewEdges;
    rNew.clear();
    size_t head = 0;
    size_t lGuard = 0;
    const size_t lMaxGuard = 64 + 16 * rCross.size() * rCross.size();
    while (head < rCross.size())
      {
        if (++lGuard > lMaxGuard) { rbInternal = true; return false; }
        const std::pair<int, int> e = rCross[head++];
        int te, ie;
        if (!FindEdge(e.first, e.second, te, ie)) { rbInternal = true; return false; }
        const int nb = mT[te].n[ie];
        if (nb < 0) { rbInternal = true; return false; }
        const int o = mT[te].v[ie];
        const int o2 = mT[nb].v[NeighborIndex(nb, te)];
        const int u = mT[te].v[Next3(ie)], w = mT[te].v[Prev3(ie)];
        if (!(Sign(Orient(o, o2, u)) * Sign(Orient(o, o2, w)) < 0))
          {
            rCross.push_back(e);  // not convex yet: retry later
            continue;
          }
        Flip(te, ie);
        const int s1 = Sign(OrientP(a, b, mX[o], mY[o])), s2 = Sign(OrientP(a, b, mX[o2], mY[o2]));
        const bool bCross = s1 * s2 < 0
                         && Sign(Orient(o, o2, a)) * Sign(Orient(o, o2, b)) < 0;
        if (bCross) rCross.emplace_back(o, o2);
        else        rNew.emplace_back(o, o2);
      }
    if (!FindEdge(a, b, t, i)) { rbInternal = true; return false; }
    SetConstraint(t, i, bits);
    for (const std::pair<int, int> & e : rNew)
      {
        if ((e.first == a && e.second == b) || (e.first == b && e.second == a)) continue;
        int te, ie;
        if (FindEdge(e.first, e.second, te, ie)) mStack.emplace_back(te, ie);
      }
    Legalize();
    return true;
  }

  // Copies the triangulation (not the scratch buffers) from crOther.
  void CopyFrom(const Cdt & crOther)
  {
    mX = crOther.mX; mY = crOther.mY; mVT = crOther.mVT; mT = crOther.mT;
    mLast = crOther.mLast;
    mRand = crOther.mRand;
  }

private:
  std::vector<std::pair<int, int>> mCross, mNewEdges;
};


// ---------------------------------------------------------------------------------------------
// Face triangulation

// Tuning (adjustable through SmTrimFaceDebugTuning() in SM_FASTFACE_DEBUG builds)
struct Tuning
{
  double dAngleFactor     = 1.0;   // facet normal vs surface normals: this * surface angle tolerance ...
  double dLeafAngleFactor = 0.0;   // ... or this when tree leaves give the interior vertices (0: no angle
                                   // refinement then; SMLib's leaves already resolve the curvature)
  double dDeviationFactor = 0.45;  // centroid deviation: this * tan(angle / 2) * longest edge
  double dBoundaryMargin  = 2.0;   // degrees over half the turning of a loop segment's normals
  double dSampleClearance = 0.35;  // samples nearer a loop segment than this * its length are skipped ...
  double dLeafClearance   = 0.05;  // ... and leaf corners nearer than this (SMLib keeps them all)
  double dSplitClearance  = 0.25;  // refinement points nearer a loop segment than this * its length (3D) aren't used
  double dMinBudget       = 2000;  // new vertices allowed: at least this ...
  double dBudgetPerVertex = 16;    // ... or this per loop vertex and sample
  double dLeafBudgetPerVertex = 1.0;   // with tree leaves: this per loop vertex and leaf (at least 16)
  double dBulgeFlip       = 0.8;   // flip an interior edge when the other diagonal bulges less than this * its bulge (0: off)
  double dLeafDeviation   = 1.0;   // with tree leaves, triangles outside the leaves whose centroid deviates more than
                                   // this * (2/3) tan(angle sqrt(2) / 4) * longest edge are refined (0: off)
  double dChordFactor     = 1.5;   // triangles whose centroid deviates more than this * the chord height tolerance are
                                   // refined (SMLib's meshes exceed the tolerance by up to 2x)
  double dLeafSpan        = 1.5;   // ImproveNormals() makes no interior edge whose UV extent exceeds this many times the
                                   // size of the smallest leaf its UV box meets, unless it replaces a wider one (0: off)
};
#ifdef SM_FASTFACE_DEBUG
Tuning g_sTuning;
#else
const Tuning g_sTuning;
#endif

const double kDegToRad            = 3.14159265358979323846 / 180.0;
const double kDegenerateArea      = 2.0e-12;  // degenerate: twice the area <= this * longest edge^2
const double kSliverSin2          = 3.0462e-4; // sin^2(1 degree): smaller angles make a sliver
const double kEarSin2             = 6.6987e-2; // sin^2(15 degrees): thinner ears of the loop are avoided
const double kBulgeSuspect        = 0.01;      // ReduceBulge(): edges estimated to bulge more than this
                                               // times their length are measured

struct VertexData
{
  SmPoint2d  sUV;
  SmPoint3d  sPoint;
  SmVector3d sNormal;            // unit surface normal (surface orientation)
  long       lRef = 0;           // output reference: lId, or ~k for the k-th new vertex
  long       lPole = -1;         // pole run: loop vertices sharing an lId with a loop neighbor ...
  SmVector2d sPoleDir;           // ... and the direction of the pole line in UV
  bool       bSingular = false;  // normal not defined there (pole)
};

struct Workspace
{
  Cdt                              sCdt;
  Cdt                              sSaved;           // the triangulation before refinement
  std::vector<VertexData>          sSavedVerts;
  std::vector<VertexData>          sVerts;           // per CDT vertex (0-2 unused)
  std::vector<SmVector3d>          sLoopNormal;      // per flat loop vertex
  std::vector<double>              sLoopSu, sLoopSv; // per flat loop vertex: |Su|, |Sv| (non-planar faces with leaves)
  std::vector<double>              sLoopXY;          // per flat loop vertex: working-plane x, y
  std::vector<char>                sLoopPole;        // per flat loop vertex
  std::vector<SmVector2d>          sLoopPoleDir;     // per flat loop vertex (pole vertices)
  std::vector<int>                 sLoopCdt;         // per flat loop vertex: CDT vertex
  std::vector<std::pair<int, int>> sSegments;        // directed loop segments (CDT vertices) ...
  std::vector<int>                 sSegFlat, sSegFlatNext;  // ... and their ends' flat loop vertex indices
  std::vector<std::pair<int, int>> sSortedSegments;
  std::vector<int>                 sFlood;
  std::vector<int>                 sQueue;
  std::vector<int>                 sGridCell, sGridStart, sGridFill;
  std::vector<double>              sSampleXY;
  std::vector<int>                 sSampleIndex, sSampleCell, sSampleStart;
  std::vector<SmPoint2d>           sCandidates;      // interior vertex candidates (UV)
  std::vector<SmPoint2d>           sCornerKeys;      // the leaf corner candidates' own corners (before SnapLeafCorners())
  std::vector<std::array<double, 4>> sCornerSnap;    // leaf corner (u, v) -> the corner it merges into, sorted
  std::vector<int>                 sInsideLeaves;    // indices of the leaves inside the face
  std::vector<std::array<double, 3>> sCornerVerts, sCornerByV;  // leaf corners (u, v, vertex)
  std::vector<int>                 sUVCell, sUVStart;
  std::unordered_map<uint64_t, double> sEdgeDev;     // ReduceBulge(): evaluated edge deviations
  std::vector<int>                 sLeafCell, sLeafStart;  // leaves per cell of a grid over the loops' UV box
  std::unordered_map<uint64_t, double> sEdgeSpan;    // ImproveNormals(): evaluated EdgeSpan() values

  void Clear()
  {
    sCdt.Clear();
    sVerts.clear();
    sSavedVerts.clear();
    sLoopNormal.clear();
    sLoopSu.clear();
    sLoopSv.clear();
    sLoopXY.clear();
    sLoopPole.clear();
    sLoopPoleDir.clear();
    sLoopCdt.clear();
    sSegments.clear();
    sSegFlat.clear();
    sSegFlatNext.clear();
    sSortedSegments.clear();
    sFlood.clear();
    sQueue.clear();
    sGridCell.clear();
    sGridStart.clear();
    sGridFill.clear();
    sSampleXY.clear();
    sSampleIndex.clear();
    sSampleCell.clear();
    sSampleStart.clear();
    sCandidates.clear();
    sCornerKeys.clear();
    sCornerSnap.clear();
    sInsideLeaves.clear();
    sCornerVerts.clear();
    sCornerByV.clear();
    sUVCell.clear();
    sUVStart.clear();
    sLeafCell.clear();
    sLeafStart.clear();
  }
};

Workspace & GetWorkspace()
{
  static thread_local Workspace tWorkspace;
  return tWorkspace;
}

// Affine map from UV to the working plane: x = l11 du + l21 dv, y = l22 dv (du = u - u0, dv = v - v0),
// where L L^T is the working metric.
struct WorkMap
{
  double u0 = 0.0, v0 = 0.0, l11 = 1.0, l21 = 0.0, l22 = 1.0;
  void Forward(double u, double v, double & x, double & y) const
  {
    const double du = u - u0, dv = v - v0;
    x = l11 * du + l21 * dv;
    y = l22 * dv;
  }
  void Inverse(double x, double y, double & u, double & v) const
  {
    const double dv = y / l22;
    v = v0 + dv;
    u = u0 + (x - l21 * dv) / l11;
  }
  // Cholesky factor of the symmetric positive definite [a b; b c]
  bool SetMetric(double a, double b, double c)
  {
    if (!(a > 0.0) || !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) return false;
    l11 = std::sqrt(a);
    l21 = b / l11;
    const double r = c - l21 * l21;
    if (!(r > 1.0e-12 * c)) return false;
    l22 = std::sqrt(r);
    return std::isfinite(l11) && std::isfinite(l21) && std::isfinite(l22);
  }
};

inline bool IsFinite(double d) { return std::isfinite(d); }

SmBoolean FaceIsReversed(const SmFace * cpFace)
{
  if (!cpFace) return FALSE;
  SmFaceuse * pFU = cpFace->GetUpwardFaceuse();
  if (!pFU || !pFU->GetShell() || !cpFace->GetBrep()) return FALSE;
  return (pFU->GetShell()->GetRegion() != cpFace->GetBrep()->GetInfiniteRegion()) ? TRUE : FALSE;
}

class FaceTriangulator
{
public:
  FaceTriangulator(const SmTrimFaceInput & crIn, SmTrimFaceOutput & rOut, Workspace & rW)
    : mIn(crIn), mOut(rOut), mW(rW), mCdt(rW.sCdt), mpSurface(crIn.cpSurface) {}

  SmStatus Run()
  {
    SmTrimFaceFailure eFail = SM_TF_NONE;
    const bool bOk =    Setup(eFail)
                     && LoopNormals(eFail)
                     && Metric(eFail)
                     && InsertLoops(eFail)
                     && Classify(eFail)
                     && InsertSamples(eFail)
                     && Refine(eFail)
                     && Output(eFail);
#ifdef SM_FASTFACE_DEBUG
    DebugCapture();
#endif
    if (!bOk) return Fail(eFail);
    mOut.eStatus = SM_SUCCESS;
    mOut.eFailure = SM_TF_NONE;
    return SM_SUCCESS;
  }

#ifdef SM_FASTFACE_DEBUG
  void DebugCapture()
  {
    SmTrimFaceDebug & rD = SmTrimFaceDebugState();
    rD = SmTrimFaceDebug();
    for (int k = 0; k < 8; ++k) { rD.aReasons[k] = t_aReasons[k]; t_aReasons[k] = 0; }
    rD.dMap[0] = mMap.u0; rD.dMap[1] = mMap.v0; rD.dMap[2] = mMap.l11; rD.dMap[3] = mMap.l21; rD.dMap[4] = mMap.l22;
    for (size_t v = 0; v < mW.sVerts.size() && v < mCdt.mX.size(); ++v)
      {
        const VertexData & crV = mW.sVerts[v];
        rD.sVerts.push_back({crV.sUV.x, crV.sUV.y, mCdt.mX[v], mCdt.mY[v], crV.sPoint.x, crV.sPoint.y, crV.sPoint.z,
                             crV.sNormal.x, crV.sNormal.y, crV.sNormal.z, (double)crV.lRef, (double)crV.lPole,
                             crV.bSingular ? 1.0 : 0.0});
      }
    for (const Tri & crT : mCdt.mT)
      rD.sTris.push_back({crT.v[0], crT.v[1], crT.v[2], (int)crT.label, (int)crT.state, (int)crT.c});
  }
#endif

private:
  const SmTrimFaceInput & mIn;
  SmTrimFaceOutput      & mOut;
  Workspace             & mW;
  Cdt                   & mCdt;
  const SmSurface       * mpSurface;
  WorkMap                 mMap;
  bool                    mbPlanar = false;
  bool                    mbOwnNormal = false;  // EvaluateNormal() == unitized Su x Sv (no override)
  bool                    mbReversed = false;
  ULONG                   mlLoopVertices = 0;
  double                  mdAngle = 0.0;        // facet vs surface normal bound (radians); 0 disables
  double                  mdMetricAngle = 0.0;  // normal turning per unit length of the working metric
  double                  mdSu = 1.0, mdSv = 1.0;  // average |Su|, |Sv|
  double                  mdCosAngle = -2.0;
  int                     mlLeafGrid = 1;       // cells per side of the leaf grid (BuildLeafGrid())
  double                  mdDevFactor = 0.0;    // centroid deviation bound per longest edge; 0 disables
  double                  mdLeafDev = 0.0;      // the same for triangles outside the leaves (with leaves); 0 disables
  double                  mdMinEdge = 0.0;      // triangles smaller than this (3D) aren't refined
  double                  mdDiag = 0.0;         // 3D diagonal of the loop vertices
  double                  mdUVMin[2] = {0, 0}, mdUVMax[2] = {0, 0};
  double                  mdWorkDiag = 0.0;
  double                  mdWorkMin[2] = {0, 0}, mdWorkMax[2] = {0, 0};
  ULONG                   mlNew = 0;            // new vertices
  ULONG                   mlBudget = 0;
  mutable long            mlBlockA = -1, mlBlockB = -1;   // Clear3D(): the loop segment too near (loop vertex ids)
  std::vector<long>       msBlocking;           // RefineOnce(): those loop segments, as pairs of loop vertex ids
  std::vector<std::array<long, 3>> msBlockedTris;  // RefineOnce(): the triangles they kept unsplit (sorted ids)
  bool                    mbJudgeFlips = false; // ImproveNormals() after refinement: flips must keep Judge() happy

  SmStatus Fail(SmTrimFaceFailure eFail)
  {
    mOut.eStatus = SM_ERR;
    mOut.eFailure = (eFail == SM_TF_NONE) ? SM_TF_INTERNAL : eFail;
    mOut.sNewUV.clear();
    mOut.sNewPoints.clear();
    mOut.sNewNormals.clear();
    mOut.sTriangles.clear();
    mOut.sLoopNormals.clear();
    return SM_ERR;
  }

  // Normal at uv with SMLib's semantics (SmSurface::EvaluateNormal)
  bool SurfaceNormal(double u, double v, SmVector3d & rN) const
  {
    if (mpSurface->EvaluateNormal(SmPoint2d(u, v), TRUE, TRUE, rN) != SM_SUCCESS) return false;
    const double len = rN.Length();
    return len > 0.5 && IsFinite(len);
  }

  // Point and unit normal from first derivatives (a check, not output); rbSingular when the
  // derivatives are degenerate
  bool PointAndNormal(double u, double v, SmPoint3d & rP, SmVector3d & rN, bool & rbSingular) const
  {
    SmVector3d sSu, sSv;
    rbSingular = false;
    if (mpSurface->Evaluate1stDerivatives(SmPoint2d(u, v), TRUE, TRUE, rP, sSu, sSv) != SM_SUCCESS) return false;
    rN = sSu * sSv;
    const double len = rN.Length();
    const double scale = std::max(sSu.Length(), sSv.Length());
    if (!(len > 1.0e-8 * scale * scale) || !IsFinite(len))
      {
        rbSingular = true;
        return SurfaceNormal(u, v, rN);
      }
    rN /= len;
    return true;
  }

  bool Setup(SmTrimFaceFailure & rFail)
  {
    rFail = SM_TF_BAD_INPUT;
    if (!mpSurface || mIn.sLoops.empty()) return false;
    mbPlanar = mpSurface->IsKindOf(SmPlane_TYPE) ? true : false;
    // surfaces whose EvaluateNormal() is SmSurface's: unitized cross product of the first derivatives
    const SM_TYPE eType = mpSurface->GetType();
    mbOwnNormal = eType == SmBSplineSurface_TYPE || eType == SmCylinder_TYPE || eType == SmCone_TYPE
               || eType == SmTorus_TYPE || eType == SmSurfOfRevolution_TYPE || eType == SmSurfOfExtrusion_TYPE;
    mbReversed = !mIn.bSurfaceOrientation && FaceIsReversed(mIn.cpFace);
    double umin = 1e300, vmin = 1e300, umax = -1e300, vmax = -1e300;
    SmPoint3d sPmin(1e300, 1e300, 1e300), sPmax(-1e300, -1e300, -1e300);
    mlLoopVertices = 0;
    for (const std::vector<SmTrimLoopVertex> & crLoop : mIn.sLoops)
      {
        if (crLoop.size() < 2) return false;
        for (const SmTrimLoopVertex & crV : crLoop)
          {
            if (   !IsFinite(crV.dU) || !IsFinite(crV.dV) || !IsFinite(crV.sPoint.x) || !IsFinite(crV.sPoint.y)
                || !IsFinite(crV.sPoint.z) || crV.lId < 0)
              return false;
            umin = std::min(umin, crV.dU); umax = std::max(umax, crV.dU);
            vmin = std::min(vmin, crV.dV); vmax = std::max(vmax, crV.dV);
            sPmin.x = std::min(sPmin.x, crV.sPoint.x); sPmax.x = std::max(sPmax.x, crV.sPoint.x);
            sPmin.y = std::min(sPmin.y, crV.sPoint.y); sPmax.y = std::max(sPmax.y, crV.sPoint.y);
            sPmin.z = std::min(sPmin.z, crV.sPoint.z); sPmax.z = std::max(sPmax.z, crV.sPoint.z);
            ++mlLoopVertices;
          }
      }
    mdUVMin[0] = umin; mdUVMin[1] = vmin; mdUVMax[0] = umax; mdUVMax[1] = vmax;
    mdDiag = (sPmax - sPmin).Length();
    if (!(mdDiag > 0.0) || !(umax > umin) || !(vmax > vmin)) return false;

    const SmTrimFaceParams & crP = mIn.sParams;
    if (crP.dSurfaceAngleTolDeg > 0.0)
      {
        mdMetricAngle = g_sTuning.dAngleFactor * std::min(crP.dSurfaceAngleTolDeg, 90.0) * kDegToRad;
        const double dFactor = mIn.sLeaves.empty() ? g_sTuning.dAngleFactor : g_sTuning.dLeafAngleFactor;
        if (dFactor > 0.0)
          {
            mdAngle = std::min(dFactor * std::min(crP.dSurfaceAngleTolDeg, 90.0), 180.0) * kDegToRad;
            mdCosAngle = std::cos(mdAngle);
            mdDevFactor = (mdAngle < 3.0) ? g_sTuning.dDeviationFactor * std::tan(0.5 * mdAngle) : 0.0;
          }
        // with leaves, the triangles between the leaves and the loops against the bulge an angle
        // tolerance allows (see SmTrimFaceInput::sLeaves): an arc turning by angle * sqrt(2) over a
        // cell's diagonal, and 4/3 for the bulge inside a triangle
        if (!mIn.sLeaves.empty() && mdAngle == 0.0)
          mdLeafDev = g_sTuning.dLeafDeviation * (2.0 / 3.0)
                    * std::tan(std::min(crP.dSurfaceAngleTolDeg, 90.0) * kDegToRad * std::sqrt(2.0) / 4.0);
      }
    mdMinEdge = 1.0e-4 * mdDiag;
    mlBudget = mIn.sLeaves.empty()
             ? (ULONG)std::max(g_sTuning.dMinBudget,
                               g_sTuning.dBudgetPerVertex * (double)(mlLoopVertices + mIn.sInteriorSamples.size()))
             : (ULONG)std::max(16.0, g_sTuning.dLeafBudgetPerVertex
                               * (double)(mlLoopVertices + mIn.sInteriorSamples.size() + mIn.sLeaves.size()));
    rFail = SM_TF_NONE;
    return true;
  }

  // Normals at the loop vertices; pole runs (consecutive loop vertices with one lId at different
  // UV points)
  bool LoopNormals(SmTrimFaceFailure & rFail)
  {
    rFail = SM_TF_EVALUATION;
    mW.sLoopNormal.resize(mlLoopVertices);
    mW.sLoopSu.assign(mlLoopVertices, 0.0);
    mW.sLoopSv.assign(mlLoopVertices, 0.0);
    mW.sLoopPole.assign(mlLoopVertices, 0);
    mW.sLoopPoleDir.assign(mlLoopVertices, SmVector2d(0.0, 0.0));
    ULONG k = 0;
    for (const std::vector<SmTrimLoopVertex> & crLoop : mIn.sLoops)
      {
        const size_t n = crLoop.size();
        for (size_t i = 0; i < n; ++i, ++k)
          {
            const SmTrimLoopVertex & crV = crLoop[i];
            if (mbPlanar && k > 0) mW.sLoopNormal[k] = mW.sLoopNormal[0];
            else if (!mbPlanar && !mIn.sLeaves.empty())
              {
                // the first derivatives too, for the leaf margins (LeafCorners())
                SmPoint3d sP;
                SmVector3d sSu, sSv;
                if (mpSurface->Evaluate1stDerivatives(SmPoint2d(crV.dU, crV.dV), TRUE, TRUE, sP, sSu, sSv) != SM_SUCCESS)
                  return false;
                mW.sLoopSu[k] = sSu.Length();
                mW.sLoopSv[k] = sSv.Length();
                SmVector3d sN = sSu * sSv;
                const double len = sN.Length();
                if (mbOwnNormal && len > 1.0e-8 && IsFinite(len))  // exactly as SmSurface::EvaluateNormal()
                  {
                    sN /= len;
                    mW.sLoopNormal[k] = sN;
                  }
                else if (!SurfaceNormal(crV.dU, crV.dV, mW.sLoopNormal[k])) return false;
              }
            else if (!SurfaceNormal(crV.dU, crV.dV, mW.sLoopNormal[k])) return false;
            const SmTrimLoopVertex & crPrev = crLoop[(i + n - 1) % n];
            const SmTrimLoopVertex & crNext = crLoop[(i + 1) % n];
            const SmTrimLoopVertex * cpRun = nullptr;
            if (crPrev.lId == crV.lId && (crPrev.dU != crV.dU || crPrev.dV != crV.dV)) cpRun = &crPrev;
            if (crNext.lId == crV.lId && (crNext.dU != crV.dU || crNext.dV != crV.dV)) cpRun = &crNext;
            if (cpRun)
              {
                mW.sLoopPole[k] = 1;
                SmVector2d sDir(cpRun->dU - crV.dU, cpRun->dV - crV.dV);
                const double len = std::hypot(sDir.x, sDir.y);
                mW.sLoopPoleDir[k] = SmVector2d(sDir.x / len, sDir.y / len);
              }
          }
      }
    rFail = SM_TF_NONE;
    return true;
  }

  // Working metric: the first fundamental form over a length scale plus the third fundamental form
  // (normal turning) over the angle bound, averaged over a 3x3 grid of the loops' UV box, so that
  // working-plane lengths measure both size and normal turning.
  bool Metric(SmTrimFaceFailure & rFail)
  {
    rFail = SM_TF_EVALUATION;
    double dLength = mdDiag;
    if (mIn.sParams.dMaxEdgeLength > 0.0) dLength = std::min(dLength, mIn.sParams.dMaxEdgeLength);
    const double wl = 1.0 / (dLength * dLength);
    const double wa = (mdMetricAngle > 0.0 && !mbPlanar) ? 1.0 / (mdMetricAngle * mdMetricAngle) : 0.0;
    double m00 = 0.0, m01 = 0.0, m11 = 0.0;  // length part
    double t00 = 0.0, t01 = 0.0, t11 = 0.0;  // curvature part
    int lUsed = 0;
    const int n = mbPlanar ? 1 : 3;
    for (int iu = 0; iu < n; ++iu)
      for (int iv = 0; iv < n; ++iv)
        {
          const double fu = (n == 1) ? 0.5 : (1.0 + 2.0 * iu) / 6.0, fv = (n == 1) ? 0.5 : (1.0 + 2.0 * iv) / 6.0;
          const SmPoint2d sUV(mdUVMin[0] + fu * (mdUVMax[0] - mdUVMin[0]), mdUVMin[1] + fv * (mdUVMax[1] - mdUVMin[1]));
          SmVector3d aD[9];  // [D Dv Dvv; Du Duv -; Duu - -] (SM_S2 layout)
          if (wa > 0.0)
            {
              if (mpSurface->Evaluate(sUV, 2, 2, TRUE, TRUE, TRUE, aD, TRUE) != SM_SUCCESS) continue;
            }
          else
            {
              if (mpSurface->Evaluate1stDerivatives(sUV, TRUE, TRUE, aD[SM_S2(0, 0)], aD[SM_S2(1, 0)], aD[SM_S2(0, 1)])
                  != SM_SUCCESS)
                continue;
            }
          const SmVector3d & crSu = aD[SM_S2(1, 0)];
          const SmVector3d & crSv = aD[SM_S2(0, 1)];
          const double E = crSu.Dot(crSu), F = crSu.Dot(crSv), G = crSv.Dot(crSv);
          const double det = E * G - F * F;
          if (!(det > 1.0e-12 * E * G) || !IsFinite(det)) continue;  // singular here
          m00 += E; m01 += F; m11 += G;
          if (wa > 0.0)
            {
              SmVector3d sN = crSu * crSv;
              sN /= std::sqrt(det);
              const double L = aD[SM_S2(2, 0)].Dot(sN), M = aD[SM_S2(1, 1)].Dot(sN), N = aD[SM_S2(0, 2)].Dot(sN);
              // III = II I^-1 II
              const double i00 = G / det, i01 = -F / det, i11 = E / det;
              const double a00 = L * i00 + M * i01, a01 = L * i01 + M * i11;  // II I^-1, first row
              const double a10 = M * i00 + N * i01, a11 = M * i01 + N * i11;  // second row
              const double c00 = a00 * L + a01 * M, c01 = a00 * M + a01 * N, c11 = a10 * M + a11 * N;
              if (IsFinite(c00) && IsFinite(c01) && IsFinite(c11)) { t00 += c00; t01 += c01; t11 += c11; }
            }
          ++lUsed;
        }
    if (lUsed == 0)
      {
        // singular everywhere sampled: scale each parameter to the 3D size
        const double du = mdUVMax[0] - mdUVMin[0], dv = mdUVMax[1] - mdUVMin[1];
        m00 = mdDiag * mdDiag / (du * du);
        m11 = mdDiag * mdDiag / (dv * dv);
        m01 = 0.0;
        t00 = t01 = t11 = 0.0;
        lUsed = 1;
      }
    m00 /= lUsed; m01 /= lUsed; m11 /= lUsed;
    t00 /= lUsed; t01 /= lUsed; t11 /= lUsed;
    mdSu = std::sqrt(m00);
    mdSv = std::sqrt(m11);
    mMap.u0 = mdUVMin[0];
    mMap.v0 = mdUVMin[1];
    double a = wl * m00 + wa * t00, b = wl * m01 + wa * t01, c = wl * m11 + wa * t11;
    if (std::fabs(b) < 1.0e-3 * std::sqrt(std::fabs(a * c))) b = 0.0;  // keeps iso-parameter lines exactly straight
    if (!mMap.SetMetric(a, b, c) && !mMap.SetMetric(m00, m01, m11)) return false;
    rFail = SM_TF_NONE;
    return true;
  }

  bool InsertLoops(SmTrimFaceFailure & rFail)
  {
    double xmin = 1e300, ymin = 1e300, xmax = -1e300, ymax = -1e300;
    std::vector<double> & rXY = mW.sLoopXY;
    rXY.resize(2 * mlLoopVertices);
    ULONG k = 0;
    for (const std::vector<SmTrimLoopVertex> & crLoop : mIn.sLoops)
      for (const SmTrimLoopVertex & crV : crLoop)
        {
          double x, y;
          mMap.Forward(crV.dU, crV.dV, x, y);
          rXY[2 * k] = x;
          rXY[2 * k + 1] = y;
          xmin = std::min(xmin, x); xmax = std::max(xmax, x);
          ymin = std::min(ymin, y); ymax = std::max(ymax, y);
          ++k;
        }
    mdWorkMin[0] = xmin; mdWorkMin[1] = ymin; mdWorkMax[0] = xmax; mdWorkMax[1] = ymax;
    mdWorkDiag = std::hypot(xmax - xmin, ymax - ymin);
    mCdt.Init(xmin, ymin, xmax, ymax);
    mW.sVerts.resize(3);
    mW.sLoopCdt.resize(mlLoopVertices);

    // vertices
    k = 0;
    for (const std::vector<SmTrimLoopVertex> & crLoop : mIn.sLoops)
      for (const SmTrimLoopVertex & crV : crLoop)
        {
          const double x = rXY[2 * k], y = rXY[2 * k + 1];
          int t = -1, idx = 0;
          const int loc = mCdt.Locate(x, y, mCdt.mLast, t, idx, false);
          int v = -1;
          if (loc == kLocVertex)
            {
              v = mCdt.mT[t].v[idx];
              if (v < 3 || mW.sVerts[v].lRef != crV.lId) { rFail = SM_TF_DUPLICATE_POINT; return false; }
            }
          else if (loc == kLocIn)   v = mCdt.InsertInTriangle(t, x, y);
          else if (loc == kLocEdge) v = mCdt.InsertOnEdge(t, idx, x, y);
          if (v < 0) { rFail = SM_TF_INTERNAL; return false; }
          if (v == (int)mW.sVerts.size())
            {
              mW.sVerts.emplace_back();
              VertexData & rV = mW.sVerts.back();
              rV.sUV = SmPoint2d(crV.dU, crV.dV);
              rV.sPoint = crV.sPoint;
              rV.sNormal = mW.sLoopNormal[k];
              rV.lRef = crV.lId;
            }
          if (mW.sLoopPole[k])
            {
              mW.sVerts[v].lPole = crV.lId;
              mW.sVerts[v].sPoleDir = mW.sLoopPoleDir[k];
              mW.sVerts[v].bSingular = true;
            }
          mW.sLoopCdt[k] = v;
          ++k;
        }

    // constraints
    k = 0;
    for (const std::vector<SmTrimLoopVertex> & crLoop : mIn.sLoops)
      {
        const size_t n = crLoop.size();
        const ULONG first = k;
        for (size_t i = 0; i < n; ++i, ++k)
          {
            const int a = mW.sLoopCdt[k];
            const ULONG k2 = (i + 1 < n) ? k + 1 : first;
            const int b = mW.sLoopCdt[k2];
            if (a != b)
              {
                mW.sSegments.emplace_back(a, b);
                mW.sSegFlat.push_back((int)k);
                mW.sSegFlatNext.push_back((int)k2);
              }
          }
      }
    if (mW.sSegments.size() < 3) { rFail = SM_TF_BAD_INPUT; return false; }
    for (const std::pair<int, int> & crS : mW.sSegments)
      {
        bool bInternal = false;
        if (!mCdt.InsertSegment(crS.first, crS.second, kLoopEdge, bInternal))
          {
            rFail = bInternal ? SM_TF_INTERNAL : SM_TF_LOOP_INTERSECTION;
            return false;
          }
      }
    return true;
  }

  // Labels the triangles inside the face: those to the left of the directed loop segments and the
  // triangles reached from them without crossing a segment. False when the loops don't bound a
  // region that way.
  bool LabelRegion()
  {
    std::vector<Tri> & rT = mCdt.mT;
    for (Tri & rTri : rT) rTri.label = 0;
    std::vector<int> & rFlood = mW.sFlood;
    rFlood.clear();
    for (const std::pair<int, int> & crS : mW.sSegments)
      {
        int t, i;
        if (!mCdt.FindEdge(crS.first, crS.second, t, i)) return false;
        const int seed = t;
        if (rT[seed].label == 0) { rT[seed].label = 1; rFlood.push_back(seed); }
      }
    while (!rFlood.empty())
      {
        const int t = rFlood.back();
        rFlood.pop_back();
        for (int e = 0; e < 3; ++e)
          {
            if ((rT[t].c >> e) & 1) continue;
            const int nb = rT[t].n[e];
            if (nb >= 0 && rT[nb].label == 0) { rT[nb].label = 1; rFlood.push_back(nb); }
          }
      }
    for (Tri & rTri : rT)
      {
        if (rTri.label == 0) rTri.label = -1;
        if (rTri.label > 0 && (rTri.v[0] < 3 || rTri.v[1] < 3 || rTri.v[2] < 3)) return false;
      }
    // the other side of every segment is outside, unless the segment is used both ways (a slit)
    for (const std::pair<int, int> & crS : mW.sSegments)
      {
        int t, i;
        if (!mCdt.FindEdge(crS.first, crS.second, t, i)) return false;
        const int other = rT[t].n[i];
        if (other < 0 || rT[other].label < 0) continue;
        const std::pair<int, int> sRev(crS.second, crS.first);
        if (!std::binary_search(mW.sSortedSegments.begin(), mW.sSortedSegments.end(), sRev)) return false;
      }
    return true;
  }

  bool Classify(SmTrimFaceFailure & rFail)
  {
    mW.sSortedSegments = mW.sSegments;
    std::sort(mW.sSortedSegments.begin(), mW.sSortedSegments.end());
    if (LabelRegion()) return true;
    rFail = SM_TF_LOOP_ORIENTATION;
    return false;
  }

  // squared working-plane distance from (px, py) to segment a-b
  static double SegDist2(double px, double py, double ax, double ay, double bx, double by)
  {
    const double dx = bx - ax, dy = by - ay;
    const double len2 = dx * dx + dy * dy;
    double s = (len2 > 0.0) ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0.0;
    s = std::min(1.0, std::max(0.0, s));
    const double ex = ax + s * dx - px, ey = ay + s * dy - py;
    return ex * ex + ey * ey;
  }

  // SMLib's Phase 2 ends the split of a tree node on a vertex of the node's side, made by the neighbor
  // across it, within a quarter of the adjacent leaves (PrepareTreeNodeData()), so the near coincident
  // splits of neighbor blocks become one vertex instead of slivers. The same for the leaf corners (as
  // SmTriangulateRectangularFace() does it): a T-junction whose leaves lie right of or above its line
  // (made after the line's other side in SMLib's depth first order) merges into the next vertex on the
  // line that is a T-junction from the other side, when that is within a quarter of the leaf between
  // them. Fills mW.sCornerSnap (sorted by the merged corner).
  void SnapLeafCorners()
  {
    mW.sCornerSnap.clear();
    const std::vector<SmTrimLeaf> & crLeaves = mIn.sLeaves;
    const int lLeaves = (int)crLeaves.size();
    if (lLeaves < 3) return;
    struct Corner { double u, v; int lc; };
    std::vector<Corner> sAll((size_t)lLeaves * 4);
    for (int l = 0; l < lLeaves; ++l)
      {
        const SmTrimLeaf & crL = crLeaves[l];
        sAll[4 * l]     = {crL.dU0, crL.dV0, 4 * l};
        sAll[4 * l + 1] = {crL.dU1, crL.dV0, 4 * l + 1};
        sAll[4 * l + 2] = {crL.dU1, crL.dV1, 4 * l + 2};
        sAll[4 * l + 3] = {crL.dU0, crL.dV1, 4 * l + 3};
      }
    std::sort(sAll.begin(), sAll.end(), [](const Corner & a, const Corner & b)
      { return a.u < b.u || (a.u == b.u && (a.v < b.v || (a.v == b.v && a.lc < b.lc))); });
    // the corner positions (u, v order), the leaf corners at each (count, mask, the first two)
    std::vector<double> sU, sV;
    std::vector<int> sCount, sLC;
    std::vector<unsigned char> sMask;
    for (const Corner & crC : sAll)
      {
        if (sU.empty() || crC.u != sU.back() || crC.v != sV.back())
          {
            sU.push_back(crC.u); sV.push_back(crC.v);
            sCount.push_back(0); sMask.push_back(0); sLC.push_back(-1); sLC.push_back(-1);
          }
        const size_t k = sU.size() - 1;
        if (sCount[k] < 2) sLC[2 * k + sCount[k]] = crC.lc;
        ++sCount[k];
        sMask[k] |= (unsigned char)(1 << (crC.lc & 3));
      }
    const int n = (int)sU.size();
    std::vector<int> sNextV(n, -1), sPrevV(n, -1), sNextH(n, -1), sPrevH(n, -1);
    for (int k = 0; k + 1 < n; ++k)
      if (sU[k + 1] == sU[k]) { sNextV[k] = k + 1; sPrevV[k + 1] = k; }
    std::vector<int> sByV(n);
    for (int k = 0; k < n; ++k) sByV[k] = k;
    std::sort(sByV.begin(), sByV.end(), [&](int a, int b) { return sV[a] < sV[b] || (sV[a] == sV[b] && sU[a] < sU[b]); });
    for (int k = 0; k + 1 < n; ++k)
      if (sV[sByV[k + 1]] == sV[sByV[k]]) { sNextH[sByV[k]] = sByV[k + 1]; sPrevH[sByV[k + 1]] = sByV[k]; }
    auto TJunction = [&](int k, unsigned char cMask) { return k >= 0 && sCount[k] == 2 && sMask[k] == cMask; };
    for (int k = 0; k < n; ++k)
      {
        // leaves right of a vertical line (BL of the upper leaf, TL of the lower), merging along v into a
        // vertex whose leaves lie left of it (BR, TR); or leaves above a horizontal line (BL of the right
        // leaf, BR of the left one), merging along u into one with leaves below it (TL, TR)
        const bool bVertical = TJunction(k, 9);
        if (!bVertical && !TJunction(k, 3)) continue;
        const unsigned char cOther = bVertical ? 6 : 12;
        double dBest = std::numeric_limits<double>::infinity();
        int lKeep = -1;
        for (int c = 0; c < 2; ++c)
          {
            const int lc = sLC[2 * k + c];
            const SmTrimLeaf & crL = crLeaves[lc / 4];
            const bool bUp = (lc & 3) == 0;  // the leaf with its BL corner here lies up (right) of the vertex
            const int lNext = bVertical ? (bUp ? sNextV[k] : sPrevV[k]) : (bUp ? sNextH[k] : sPrevH[k]);
            if (!TJunction(lNext, cOther)) continue;
            const double dExtent = bVertical ? crL.dV1 - crL.dV0 : crL.dU1 - crL.dU0;
            const double dDist = bVertical ? std::fabs(sV[lNext] - sV[k]) : std::fabs(sU[lNext] - sU[k]);
            if (dDist <= 0.25 * dExtent && (dDist < dBest || (dDist == dBest && !bUp)))
              {
                dBest = dDist;
                lKeep = lNext;
              }
          }
        if (lKeep >= 0) mW.sCornerSnap.push_back({sU[k], sV[k], sU[lKeep], sV[lKeep]});
      }
    std::sort(mW.sCornerSnap.begin(), mW.sCornerSnap.end());
  }

  // The position of a leaf corner after SnapLeafCorners()
  SmPoint2d SnappedCorner(const SmPoint2d & crKey) const
  {
    const std::array<double, 4> sKey = {crKey.x, crKey.y, -1e300, -1e300};
    auto it = std::lower_bound(mW.sCornerSnap.begin(), mW.sCornerSnap.end(), sKey);
    if (it != mW.sCornerSnap.end() && (*it)[0] == crKey.x && (*it)[1] == crKey.y) return SmPoint2d((*it)[2], (*it)[3]);
    return crKey;
  }

  // Corners of the leaves inside the face, as SMLib's tessellation uses them: a leaf touched by a
  // loop segment (within a tiny margin) is on the boundary; otherwise it is inside when its center
  // is. Appends to rCorners (with duplicates where leaves share corners).
  void LeafCorners(std::vector<SmPoint2d> & rCorners)
  {
    const std::vector<SmTrimLeaf> & crLeaves = mIn.sLeaves;
    const double w = mdUVMax[0] - mdUVMin[0], h = mdUVMax[1] - mdUVMin[1];
    // SMLib's margin around each segment: 10 times the tolerance mapped to UV by the derivatives at
    // the segment (the mean of its ends' here, its midpoint in SMLib; the face average for planes),
    // at most a hundredth of the surface's domain (SmTessSrfCache::ImplantUVSegment())
    const SmExtent2d sDomain = mpSurface->GetNaturalUVDomain();
    const double dTol = std::max(mIn.dTolerance, 0.0);
    const double dCapU = sDomain.XLength() / 100.0, dCapV = sDomain.YLength() / 100.0;
    auto margins = [&](int sg, double & rmu, double & rmv)
      {
        double su = mdSu, sv = mdSv;
        if (!mW.sLoopSu.empty())
          {
            const int ka = mW.sSegFlat[sg], kb = mW.sSegFlatNext[sg];
            su = 0.5 * (mW.sLoopSu[ka] + mW.sLoopSu[kb]);
            sv = 0.5 * (mW.sLoopSv[ka] + mW.sLoopSv[kb]);
          }
        rmu = std::max(1.0e-9 * w, 10.0 * ((su > 0.0) ? std::min(dTol / su, dCapU) : dCapU));
        rmv = std::max(1.0e-9 * h, 10.0 * ((sv > 0.0) ? std::min(dTol / sv, dCapV) : dCapV));
      };
    double mu = 1.0e-9 * w, mv = 1.0e-9 * h;  // the largest margins, for the grid lookup
    for (int sg = 0; sg < (int)mW.sSegments.size(); ++sg)
      {
        double a, b;
        margins(sg, a, b);
        mu = std::max(mu, a);
        mv = std::max(mv, b);
      }
    const int nseg = (int)mW.sSegments.size();
    const int g = std::max(1, std::min(256, (int)std::sqrt((double)nseg)));
    const double cw = w / g, ch = h / g;
    std::vector<int> & rStart = mW.sUVStart;
    std::vector<int> & rCell = mW.sUVCell;
    std::vector<int> & rFill = mW.sGridFill;
    auto range = [&](double u0, double v0, double u1, double v1, int & x0, int & x1, int & y0, int & y1)
      {
        x0 = std::max(0, std::min(g - 1, (int)((u0 - mdUVMin[0]) / cw)));
        x1 = std::max(0, std::min(g - 1, (int)((u1 - mdUVMin[0]) / cw)));
        y0 = std::max(0, std::min(g - 1, (int)((v0 - mdUVMin[1]) / ch)));
        y1 = std::max(0, std::min(g - 1, (int)((v1 - mdUVMin[1]) / ch)));
      };
    auto segRange = [&](int s, int & x0, int & x1, int & y0, int & y1)
      {
        const SmPoint2d & a = mW.sVerts[mW.sSegments[s].first].sUV, & b = mW.sVerts[mW.sSegments[s].second].sUV;
        range(std::min(a.x, b.x), std::min(a.y, b.y), std::max(a.x, b.x), std::max(a.y, b.y), x0, x1, y0, y1);
      };
    rStart.assign(g * g + 1, 0);
    for (int s = 0; s < nseg; ++s)
      {
        int x0, x1, y0, y1;
        segRange(s, x0, x1, y0, y1);
        for (int yy = y0; yy <= y1; ++yy)
          for (int xx = x0; xx <= x1; ++xx) ++rStart[yy * g + xx + 1];
      }
    for (int c = 0; c < g * g; ++c) rStart[c + 1] += rStart[c];
    rCell.assign(rStart[g * g], 0);
    rFill.assign(rStart.begin(), rStart.end() - 1);
    for (int s = 0; s < nseg; ++s)
      {
        int x0, x1, y0, y1;
        segRange(s, x0, x1, y0, y1);
        for (int yy = y0; yy <= y1; ++yy)
          for (int xx = x0; xx <= x1; ++xx) rCell[rFill[yy * g + xx]++] = s;
      }
    // segment a-b meets the rectangle [u0,u1] x [v0,v1] (Liang-Barsky clipping)
    auto meets = [](const SmPoint2d & a, const SmPoint2d & b, double u0, double v0, double u1, double v1)
      {
        double t0 = 0.0, t1 = 1.0;
        const double d[2] = {b.x - a.x, b.y - a.y}, p[2] = {a.x, a.y}, lo[2] = {u0, v0}, hi[2] = {u1, v1};
        for (int k = 0; k < 2; ++k)
          {
            if (d[k] == 0.0)
              {
                if (p[k] < lo[k] || p[k] > hi[k]) return false;
                continue;
              }
            double ta = (lo[k] - p[k]) / d[k], tb = (hi[k] - p[k]) / d[k];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
            if (t0 > t1) return false;
          }
        return true;
      };
    int hint = mCdt.mLast;
    for (const SmTrimLeaf & crL : crLeaves)
      {
        const double u0 = crL.dU0 - mu, u1 = crL.dU1 + mu, v0 = crL.dV0 - mv, v1 = crL.dV1 + mv;
        if (!(u1 > u0 && v1 > v0) || u1 < mdUVMin[0] || u0 > mdUVMax[0] || v1 < mdUVMin[1] || v0 > mdUVMax[1]) continue;
        int x0, x1, y0, y1;
        range(u0, v0, u1, v1, x0, x1, y0, y1);
        bool bTouched = false;
        for (int yy = y0; yy <= y1 && !bTouched; ++yy)
          for (int xx = x0; xx <= x1 && !bTouched; ++xx)
            for (int q = rStart[yy * g + xx]; q < rStart[yy * g + xx + 1] && !bTouched; ++q)
              {
                const int sg = rCell[q];
                double smu, smv;
                margins(sg, smu, smv);
                bTouched = meets(mW.sVerts[mW.sSegments[sg].first].sUV, mW.sVerts[mW.sSegments[sg].second].sUV,
                                 crL.dU0 - smu, crL.dV0 - smv, crL.dU1 + smu, crL.dV1 + smv);
              }
        if (bTouched) continue;
        double x, y;
        mMap.Forward(0.5 * (crL.dU0 + crL.dU1), 0.5 * (crL.dV0 + crL.dV1), x, y);
        int t = -1, idx = 0;
        const int loc = mCdt.Locate(x, y, hint, t, idx, false);
        if (loc == kLocFail) continue;
        hint = t;
        if (mCdt.mT[t].label <= 0) continue;
        const SmPoint2d aKeys[4] = {SmPoint2d(crL.dU0, crL.dV0), SmPoint2d(crL.dU1, crL.dV0), SmPoint2d(crL.dU1, crL.dV1),
                                    SmPoint2d(crL.dU0, crL.dV1)};
        for (const SmPoint2d & crKey : aKeys)
          {
            mW.sCornerKeys.push_back(crKey);
            rCorners.push_back(SnappedCorner(crKey));
          }
        mW.sInsideLeaves.push_back((int)(&crL - crLeaves.data()));
      }
  }

  // Constrains the sides of the inside leaves between the corners in the triangulation (leaf
  // corners on a side, where smaller neighbors meet it, split it), so that each leaf is
  // triangulated on its own, as SMLib turns inside leaves into quadrilaterals: the leaves already
  // satisfy the tolerances. rCornerVerts holds (u, v, vertex) of the corners in the triangulation.
  void ConstrainLeafSides(std::vector<std::array<double, 3>> & rCornerVerts)
  {
    if (rCornerVerts.empty() || mW.sInsideLeaves.empty()) return;
    // sorted by (u, v) for vertical sides and by (v, u) for horizontal sides
    std::vector<std::array<double, 3>> & rByU = rCornerVerts;
    std::sort(rByU.begin(), rByU.end());
    rByU.erase(std::unique(rByU.begin(), rByU.end(),
                           [](const std::array<double, 3> & a, const std::array<double, 3> & b)
                             { return a[0] == b[0] && a[1] == b[1]; }), rByU.end());
    std::vector<std::array<double, 3>> & rByV = mW.sCornerByV;
    rByV.clear();
    for (const std::array<double, 3> & crC : rByU) rByV.push_back({crC[1], crC[0], crC[2]});
    std::sort(rByV.begin(), rByV.end());
    // the vertices on the line key == c (first coordinate) with lo <= second <= hi, in order
    auto side = [&](const std::vector<std::array<double, 3>> & crList, double c, double lo, double hi)
      {
        const std::array<double, 3> sKey = {c, lo, -1e300};
        auto it = std::lower_bound(crList.begin(), crList.end(), sKey);
        int prev = -1;
        for (; it != crList.end() && (*it)[0] == c && (*it)[1] <= hi; ++it)
          {
            const int v = (int)(*it)[2];
            if (prev >= 0 && prev != v)
              {
                bool bInternal = false;
                mCdt.InsertSegment(prev, v, kLeafEdge, bInternal);  // a failure just leaves the side open
              }
            prev = v;
          }
      };
    for (int l : mW.sInsideLeaves)
      {
        const SmTrimLeaf & crL = mIn.sLeaves[l];
        side(rByV, crL.dV0, crL.dU0, crL.dU1);
        side(rByV, crL.dV1, crL.dU0, crL.dU1);
        side(rByU, crL.dU0, crL.dV0, crL.dV1);
        side(rByU, crL.dU1, crL.dV0, crL.dV1);
      }
  }

  bool InsertSamples(SmTrimFaceFailure & rFail)
  {
    std::vector<SmPoint2d> & rCandidates = mW.sCandidates;
    rCandidates.clear();
    mW.sInsideLeaves.clear();
    if (!mIn.sLeaves.empty())
      {
        SnapLeafCorners();
        LeafCorners(rCandidates);
      }
    const size_t lLeafCorners = rCandidates.size();
    std::vector<std::array<double, 3>> & rCornerVerts = mW.sCornerVerts;
    rCornerVerts.clear();
    rCandidates.insert(rCandidates.end(), mIn.sInteriorSamples.begin(), mIn.sInteriorSamples.end());
    const std::vector<SmPoint2d> & crSamples = rCandidates;
    if (crSamples.empty()) return true;
    // uniform grid of the loop segments over the working-plane bounds, for clearance queries
    const double w = std::max(mdWorkMax[0] - mdWorkMin[0], 1e-300), h = std::max(mdWorkMax[1] - mdWorkMin[1], 1e-300);
    const int nseg = (int)mW.sSegments.size();
    const double cell = std::max(2.0 * std::sqrt(w * h / std::max(1, nseg)), 1e-300);
    const int gx = std::max(1, std::min(256, (int)(w / cell) + 1)), gy = std::max(1, std::min(256, (int)(h / cell) + 1));
    const double cw = w / gx, ch = h / gy;
    std::vector<int> & rStart = mW.sGridStart;
    std::vector<int> & rCell = mW.sGridCell;
    std::vector<int> & rFill = mW.sGridFill;
    rStart.assign(gx * gy + 1, 0);
    auto cellRange = [&](int s, int & x0, int & x1, int & y0, int & y1)
      {
        const int a = mW.sSegments[s].first, b = mW.sSegments[s].second;
        const double ax = mCdt.mX[a], ay = mCdt.mY[a], bx = mCdt.mX[b], by = mCdt.mY[b];
        x0 = std::max(0, std::min(gx - 1, (int)((std::min(ax, bx) - mdWorkMin[0]) / cw)));
        x1 = std::max(0, std::min(gx - 1, (int)((std::max(ax, bx) - mdWorkMin[0]) / cw)));
        y0 = std::max(0, std::min(gy - 1, (int)((std::min(ay, by) - mdWorkMin[1]) / ch)));
        y1 = std::max(0, std::min(gy - 1, (int)((std::max(ay, by) - mdWorkMin[1]) / ch)));
      };
    for (int s = 0; s < nseg; ++s)
      {
        int x0, x1, y0, y1;
        cellRange(s, x0, x1, y0, y1);
        for (int yy = y0; yy <= y1; ++yy)
          for (int xx = x0; xx <= x1; ++xx) ++rStart[yy * gx + xx + 1];
      }
    for (int c = 0; c < gx * gy; ++c) rStart[c + 1] += rStart[c];
    rCell.assign(rStart[gx * gy], 0);
    rFill.assign(rStart.begin(), rStart.end() - 1);
    for (int s = 0; s < nseg; ++s)
      {
        int x0, x1, y0, y1;
        cellRange(s, x0, x1, y0, y1);
        for (int yy = y0; yy <= y1; ++yy)
          for (int xx = x0; xx <= x1; ++xx) rCell[rFill[yy * gx + xx]++] = s;
      }
    // too close to a loop segment: within dSampleClearance of the segment's length or of the sample
    // spacing, whichever is smaller
    auto crowds = [&](double px, double py, double spacing, double clearance)
      {
        const int cx = (int)std::floor((px - mdWorkMin[0]) / cw), cy = (int)std::floor((py - mdWorkMin[1]) / ch);
        double best2 = 1e300;
        for (int ring = 0; ring <= std::max(gx, gy); ++ring)
          {
            // segments beyond this ring are at least ring * cell away
            const double reach = (ring - 1) * std::min(cw, ch);
            if (ring > 0 && reach > 0.0 && reach * reach > best2) break;
            for (int yy = cy - ring; yy <= cy + ring; ++yy)
              for (int xx = cx - ring; xx <= cx + ring; ++xx)
                {
                  if (std::max(std::abs(yy - cy), std::abs(xx - cx)) != ring) continue;
                  if (xx < 0 || yy < 0 || xx >= gx || yy >= gy) continue;
                  for (int q = rStart[yy * gx + xx]; q < rStart[yy * gx + xx + 1]; ++q)
                    {
                      const int s = rCell[q];
                      const int a = mW.sSegments[s].first, b = mW.sSegments[s].second;
                      const double ax = mCdt.mX[a], ay = mCdt.mY[a], bx = mCdt.mX[b], by = mCdt.mY[b];
                      const double d2 = SegDist2(px, py, ax, ay, bx, by);
                      best2 = std::min(best2, d2);
                      if (mW.sVerts[a].lRef == mW.sVerts[b].lRef) continue;  // a pole: one point in 3D
                      const double len2 = std::min((bx - ax) * (bx - ax) + (by - ay) * (by - ay), spacing * spacing);
                      if (d2 < clearance * clearance * len2) return true;
                    }
                }
          }
        return false;
      };

    // candidates inside the loops' box, and their spacing: the distance to the nearest other sample
    std::vector<double> & rSX = mW.sSampleXY;
    rSX.clear();
    std::vector<int> & rSIdx = mW.sSampleIndex;
    rSIdx.clear();
    for (size_t k = 0; k < crSamples.size(); ++k)
      {
        const SmPoint2d & crS = crSamples[k];
        if (!IsFinite(crS.x) || !IsFinite(crS.y)) continue;
        double x, y;
        mMap.Forward(crS.x, crS.y, x, y);
        if (x <= mdWorkMin[0] || x >= mdWorkMax[0] || y <= mdWorkMin[1] || y >= mdWorkMax[1]) continue;
        rSX.push_back(x);
        rSX.push_back(y);
        rSIdx.push_back((int)k);
      }
    const int ns = (int)rSIdx.size();
    if (ns == 0) return true;
    const double scell = std::max(std::sqrt(w * h / ns), 1e-300);
    const int sgx = std::max(1, std::min(512, (int)(w / scell) + 1)), sgy = std::max(1, std::min(512, (int)(h / scell) + 1));
    const double scw = w / sgx, sch = h / sgy;
    std::vector<int> & rSStart = mW.sSampleStart;
    std::vector<int> & rSCell = mW.sSampleCell;
    rSStart.assign(sgx * sgy + 1, 0);
    auto sampleCell = [&](int q)
      {
        const int cx = std::max(0, std::min(sgx - 1, (int)((rSX[2 * q] - mdWorkMin[0]) / scw)));
        const int cy = std::max(0, std::min(sgy - 1, (int)((rSX[2 * q + 1] - mdWorkMin[1]) / sch)));
        return cy * sgx + cx;
      };
    for (int q = 0; q < ns; ++q) ++rSStart[sampleCell(q) + 1];
    for (int c = 0; c < sgx * sgy; ++c) rSStart[c + 1] += rSStart[c];
    rSCell.assign(ns, 0);
    rFill.assign(rSStart.begin(), rSStart.end() - 1);
    for (int q = 0; q < ns; ++q) rSCell[rFill[sampleCell(q)]++] = q;
    const double dDup2 = 1.0e-18 * mdWorkDiag * mdWorkDiag;
    auto spacingOf = [&](int q)
      {
        const double px = rSX[2 * q], py = rSX[2 * q + 1];
        const int c = sampleCell(q), cx = c % sgx, cy = c / sgx;
        double best2 = 1e300;
        for (int ring = 0; ring <= std::max(sgx, sgy); ++ring)
          {
            const double reach = (ring - 1) * std::min(scw, sch);
            if (ring > 0 && reach > 0.0 && reach * reach > best2) break;
            for (int yy = cy - ring; yy <= cy + ring; ++yy)
              for (int xx = cx - ring; xx <= cx + ring; ++xx)
                {
                  if (std::max(std::abs(yy - cy), std::abs(xx - cx)) != ring) continue;
                  if (xx < 0 || yy < 0 || xx >= sgx || yy >= sgy) continue;
                  for (int r = rSStart[yy * sgx + xx]; r < rSStart[yy * sgx + xx + 1]; ++r)
                    {
                      const int o = rSCell[r];
                      if (o == q) continue;
                      const double dx = rSX[2 * o] - px, dy = rSX[2 * o + 1] - py, d2 = dx * dx + dy * dy;
                      if (d2 > dDup2) best2 = std::min(best2, d2);
                    }
                }
          }
        return (best2 < 1e300) ? std::sqrt(best2) : mdWorkDiag;
      };

    rFail = SM_TF_EVALUATION;
    int hint = mCdt.mLast;
    for (int q = 0; q < ns; ++q)
      {
        const SmPoint2d & crS = crSamples[rSIdx[q]];
        const double x = rSX[2 * q], y = rSX[2 * q + 1];
        const bool bLeafCorner = (size_t)rSIdx[q] < lLeafCorners;
        int t = -1, idx = 0;
        const int loc = mCdt.Locate(x, y, hint, t, idx, false);
        if (loc == kLocVertex && bLeafCorner)
          {
            const int v = mCdt.mT[t].v[idx];  // a corner shared with an inserted leaf corner
            if (v >= 3 && mW.sVerts[v].lRef < 0 && mW.sVerts[v].sUV.x == crS.x && mW.sVerts[v].sUV.y == crS.y)
              rCornerVerts.push_back({mW.sCornerKeys[rSIdx[q]].x, mW.sCornerKeys[rSIdx[q]].y, (double)v});
          }
        if (loc != kLocIn && loc != kLocEdge) continue;
        hint = t;
        const Tri & rTri = mCdt.mT[t];
        if (rTri.label <= 0) continue;
        if (loc == kLocEdge && (((rTri.c >> idx) & 1) || mCdt.mT[rTri.n[idx]].label <= 0)) continue;
        bool bNear = false;
        for (int k = 0; k < 3 && !bNear; ++k)
          {
            const int v = rTri.v[k];
            if (std::hypot(mCdt.mX[v] - x, mCdt.mY[v] - y) < 1.0e-6 * mdWorkDiag) bNear = true;
          }
        if (bNear || crowds(x, y, spacingOf(q), bLeafCorner ? g_sTuning.dLeafClearance : g_sTuning.dSampleClearance))
          continue;
        if (!AddVertexAt(t, loc, idx, x, y, crS.x, crS.y, nullptr)) return false;
        if (bLeafCorner)
          rCornerVerts.push_back({mW.sCornerKeys[rSIdx[q]].x, mW.sCornerKeys[rSIdx[q]].y, (double)(mW.sVerts.size() - 1)});
        ++mOut.lSamplesUsed;
        hint = mCdt.mLast;
      }
    ConstrainLeafSides(rCornerVerts);
    rFail = SM_TF_NONE;
    return true;
  }

  // Inserts a new vertex at working point (x, y), located in triangle t (on its edge idx for
  // kLocEdge), with UV (u, v); evaluates its point (unless given) and normal.
  bool AddVertexAt(int t, int loc, int idx, double x, double y, double u, double v, const SmPoint3d * cpP)
  {
    SmPoint3d sP;
    SmVector3d sN;
    if (cpP) sP = *cpP;
    else if (mpSurface->EvaluatePoint(SmPoint2d(u, v), sP) != SM_SUCCESS) return false;
    if (!SurfaceNormal(u, v, sN)) return false;
    if (!IsFinite(sP.x) || !IsFinite(sP.y) || !IsFinite(sP.z)) return false;
    const int p = (loc == kLocIn) ? mCdt.InsertInTriangle(t, x, y) : mCdt.InsertOnEdge(t, idx, x, y);
    if (p < 0 || p != (int)mW.sVerts.size()) return false;
    mW.sVerts.emplace_back();
    VertexData & rV = mW.sVerts.back();
    rV.sUV = SmPoint2d(u, v);
    rV.sPoint = sP;
    rV.sNormal = sN;
    rV.lRef = ~(long)mlNew;
    ++mlNew;
    return true;
  }

  // UV of corner P of a triangle with the other corners A, B, for interpolation. Along a pole line
  // every UV point is the same 3D point, so a pole corner's own UV is arbitrary: take its projection
  // onto the pole line from the midpoint of A-B instead (within the loops' UV box).
  SmPoint2d CornerUV(const VertexData & P, const VertexData & A, const VertexData & B) const
  {
    if (P.lPole < 0) return P.sUV;
    const double mx = 0.5 * (A.sUV.x + B.sUV.x) - P.sUV.x, my = 0.5 * (A.sUV.y + B.sUV.y) - P.sUV.y;
    const double s = mx * P.sPoleDir.x + my * P.sPoleDir.y;
    const double u = std::min(mdUVMax[0], std::max(mdUVMin[0], P.sUV.x + s * P.sPoleDir.x));
    const double v = std::min(mdUVMax[1], std::max(mdUVMin[1], P.sUV.y + s * P.sPoleDir.y));
    return SmPoint2d(u, v);
  }

  // UV centroid of a triangle (pole corners as in CornerUV())
  SmPoint2d CentroidUV(const VertexData & A, const VertexData & B, const VertexData & C) const
  {
    const SmPoint2d a = CornerUV(A, B, C), b = CornerUV(B, C, A), c = CornerUV(C, A, B);
    return SmPoint2d((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0);
  }

  // What a triangle needs
  enum Verdict { kOk, kRefine, kRefineCentroid, kDrop, kInvalid, kEvalError };

  Verdict Judge(int t, SmPoint3d & rCentroidPoint, bool & rbHaveCentroid)
  {
    rbHaveCentroid = false;
    const Tri & rTri = mCdt.mT[t];
    const VertexData & A = mW.sVerts[rTri.v[0]];
    const VertexData & B = mW.sVerts[rTri.v[1]];
    const VertexData & C = mW.sVerts[rTri.v[2]];
    // corners that are one 3D vertex: degenerate at a pole (dropped); elsewhere (both copies of
    // a seam vertex) the triangle spans the period and must be split
    if (A.lRef == B.lRef || B.lRef == C.lRef || C.lRef == A.lRef)
      {
        const VertexData * cpX = (A.lRef == B.lRef) ? &A : (B.lRef == C.lRef) ? &B : &C;
        const VertexData * cpY = (A.lRef == B.lRef) ? &B : (B.lRef == C.lRef) ? &C : &A;
        if (cpX->lPole >= 0 && cpX->lPole == cpY->lPole) return kDrop;
        FF_REASON(0);
        return kRefineCentroid;
      }
    const VertexData * caV[3] = {&A, &B, &C};
    const SmVector3d sAB = B.sPoint - A.sPoint, sBC = C.sPoint - B.sPoint, sCA = A.sPoint - C.sPoint;
    const double caLen[3] = {sBC.Length(), sCA.Length(), sAB.Length()};  // edge k is opposite corner k
    const double lMax = std::max(caLen[0], std::max(caLen[1], caLen[2]));
    SmVector3d sNT = sAB * (C.sPoint - A.sPoint);
    const double area2 = sNT.Length();
    const bool bSmall = lMax < mdMinEdge;
    if (!(area2 > kDegenerateArea * lMax * lMax)) { FF_REASON(1); return bSmall ? kInvalid : kRefine; }
    sNT /= area2;
    // facet normal against the corner normals; at the ends of a loop segment whose normals turn by
    // phi, the facet can't do better than phi / 2
    double caCos[3] = {mdCosAngle, mdCosAngle, mdCosAngle};
    bool bBoundary = false;
    if ((rTri.c & 7) != 0 && mdAngle > 0.0)
      {
        const double cm = std::cos(g_sTuning.dBoundaryMargin * kDegToRad), sm = std::sin(g_sTuning.dBoundaryMargin * kDegToRad);
        for (int e = 0; e < 3; ++e)
          {
            if (!((rTri.c >> e) & 1)) continue;
            bBoundary = true;
            const int p = Next3(e), q = Prev3(e);
            const double d = std::max(-1.0, std::min(1.0, caV[p]->sNormal.Dot(caV[q]->sNormal)));
            const double ch = std::sqrt(0.5 * (1.0 + d)), sh = std::sqrt(0.5 * (1.0 - d));
            const double c = ch * cm - sh * sm;  // cos(phi / 2 + margin)
            caCos[p] = std::min(caCos[p], c);
            caCos[q] = std::min(caCos[q], c);
          }
      }
    bool bBad = false;
    SmVector3d sSum(0.0, 0.0, 0.0);
    for (int k = 0; k < 3; ++k)
      {
        if (caV[k]->bSingular) continue;
        sSum += caV[k]->sNormal;
        if (sNT.Dot(caV[k]->sNormal) < caCos[k]) { bBad = true; FF_REASON(3); }
      }
    if (!(sNT.Dot(sSum) > 0.0)) { FF_REASON(2); return bSmall ? kInvalid : kRefine; }  // flipped
    // interior edges against the maximum edge length
    const double dMaxEdge = mIn.sParams.dMaxEdgeLength;
    if (dMaxEdge > 0.0)
      for (int k = 0; k < 3; ++k)
        if (!((rTri.c >> k) & 1) && caLen[k] > dMaxEdge) { bBad = true; FF_REASON(4); }
    // a triangle with two leaf sides lies in an inside leaf, which meets the tolerances already
    const bool bLeafDev = mdLeafDev > 0.0
                       && ((EdgeBits(rTri.c, 0) >> 1) + (EdgeBits(rTri.c, 1) >> 1) + (EdgeBits(rTri.c, 2) >> 1)) < 2;
    if (!bBad && !mbPlanar && (mdAngle > 0.0 || mIn.sParams.dChordHeightTol > 0.0 || bLeafDev))
      {
        // the surface at the centroid
        const SmPoint2d sG = CentroidUV(A, B, C);
        const double uc = sG.x, vc = sG.y;
        SmVector3d sN;
        bool bSingular = false;
        if (!PointAndNormal(uc, vc, rCentroidPoint, sN, bSingular)) return kEvalError;
        rbHaveCentroid = true;
        if (!bSingular && mdAngle > 0.0 && sNT.Dot(sN) < mdCosAngle) { bBad = true; FF_REASON(5); }
        const double dev = std::fabs((rCentroidPoint - A.sPoint).Dot(sNT));
        if (mIn.sParams.dChordHeightTol > 0.0 && dev > g_sTuning.dChordFactor * mIn.sParams.dChordHeightTol) { bBad = true; FF_REASON(6); }
        if (mdDevFactor > 0.0 && dev > (bBoundary ? 1.5 : 1.0) * mdDevFactor * lMax) { bBad = true; FF_REASON(7); }
        if (bLeafDev && dev > mdLeafDev * lMax) { bBad = true; FF_REASON(7); }
      }
    if (bBad && !bSmall) return kRefine;
    return kOk;
  }

  enum SplitResult { kSplitDone, kSplitCannot, kSplitError };

  // A triangle's corner ids, sorted
  std::array<long, 3> TriIds(const Tri & crT) const
  {
    std::array<long, 3> aIds = {mW.sVerts[crT.v[0]].lRef, mW.sVerts[crT.v[1]].lRef, mW.sVerts[crT.v[2]].lRef};
    std::sort(aIds.begin(), aIds.end());
    return aIds;
  }

  // Whether splitting loop segments can fix an invalid triangle (sSplitSegments reports them): it has
  // one, or one kept it from being split (msBlocking)
  bool SplitActionable(const Tri & crT) const
  {
    return (crT.c & 7) != 0
        || (!msBlockedTris.empty() && std::find(msBlockedTris.begin(), msBlockedTris.end(), TriIds(crT)) != msBlockedTris.end());
  }

  // Number of the triangle's edges that are loop segments (in either direction)
  int LoopEdges(int a, int b, int c) const
  {
    auto isLoop = [&](int p, int q)
      {
        return std::binary_search(mW.sSortedSegments.begin(), mW.sSortedSegments.end(), std::make_pair(p, q))
            || std::binary_search(mW.sSortedSegments.begin(), mW.sSortedSegments.end(), std::make_pair(q, p));
      };
    return (int)isLoop(a, b) + (int)isLoop(b, c) + (int)isLoop(c, a);
  }

  // How far a triangle's facet normal strays from the surface normals at its corners: max over
  // the corners of 1 - cos(angle) (singular corners skipped); 2 when degenerate in 3D or a sliver
  // (smallest angle under 1 degree), 0 for a pole triangle (left out of the output). A function of
  // the vertex set.
  double NormalDeviation(int a, int b, int c) const
  {
    const VertexData & A = mW.sVerts[a], & B = mW.sVerts[b], & C = mW.sVerts[c];
    if (A.lRef == B.lRef || B.lRef == C.lRef || C.lRef == A.lRef)
      {
        const VertexData * cpX = (A.lRef == B.lRef) ? &A : (B.lRef == C.lRef) ? &B : &C;
        const VertexData * cpY = (A.lRef == B.lRef) ? &B : (B.lRef == C.lRef) ? &C : &A;
        return (cpX->lPole >= 0 && cpX->lPole == cpY->lPole) ? 0.0 : 2.0;
      }
    SmVector3d sN = (B.sPoint - A.sPoint) * (C.sPoint - A.sPoint);
    const double len = sN.Length();
    const double l0 = (B.sPoint - A.sPoint).LengthSquared(), l1 = (C.sPoint - B.sPoint).LengthSquared();
    const double l2 = (A.sPoint - C.sPoint).LengthSquared();
    const double lMax2 = std::max(l0, std::max(l1, l2)), lMid2 = l0 + l1 + l2 - lMax2 - std::min(l0, std::min(l1, l2));
    if (!(len > kDegenerateArea * lMax2)) return 2.0;
    if (len * len < kSliverSin2 * lMid2 * lMax2) return 2.0;  // a sliver: its facet normal means nothing
    // an ear of the loop over a nearly straight run of three loop vertices: when the neighbor face
    // cuts the same ear, the two coincide (a non-manifold edge), so connect inward when possible
    if (len * len < kEarSin2 * lMid2 * lMax2 && LoopEdges(a, b, c) >= 2) return 2.0;
    sN /= len;
    double dWorst = 0.0;
    const VertexData * caV[3] = {&A, &B, &C};
    for (int k = 0; k < 3; ++k)
      if (!caV[k]->bSingular) dWorst = std::max(dWorst, 1.0 - sN.Dot(caV[k]->sNormal));
    return dWorst;
  }

  // A triangle that is degenerate, folded against a corner normal or an ear of the loops (see
  // NormalDeviation()), unlike a mere sliver
  bool Broken(int a, int b, int c) const
  {
    const VertexData & A = mW.sVerts[a], & B = mW.sVerts[b], & C = mW.sVerts[c];
    if (A.lRef == B.lRef || B.lRef == C.lRef || C.lRef == A.lRef) return true;
    const SmVector3d sN = (B.sPoint - A.sPoint) * (C.sPoint - A.sPoint);
    const double len = sN.Length();
    const double l0 = (B.sPoint - A.sPoint).LengthSquared(), l1 = (C.sPoint - B.sPoint).LengthSquared();
    const double l2 = (A.sPoint - C.sPoint).LengthSquared();
    const double lMax2 = std::max(l0, std::max(l1, l2)), lMid2 = l0 + l1 + l2 - lMax2 - std::min(l0, std::min(l1, l2));
    if (!(len > kDegenerateArea * lMax2)) return true;
    if (len * len < kEarSin2 * lMid2 * lMax2 && LoopEdges(a, b, c) >= 2) return true;
    const VertexData * caV[3] = {&A, &B, &C};
    for (int k = 0; k < 3; ++k)
      if (!caV[k]->bSingular && !(sN.Dot(caV[k]->sNormal) > 0.0)) return true;
    return false;
  }

  // A uniform grid of the tree leaves over the loops' UV box, for EdgeSpan()
  void BuildLeafGrid()
  {
    mW.sLeafCell.clear();
    mW.sLeafStart.clear();
    mW.sEdgeSpan.clear();
    const std::vector<SmTrimLeaf> & crLeaves = mIn.sLeaves;
    if (mbPlanar || crLeaves.size() < 2 || !(g_sTuning.dLeafSpan > 0.0)) return;
    const int g = mlLeafGrid = std::max(1, std::min(256, (int)std::sqrt((double)crLeaves.size())));
    const double cw = (mdUVMax[0] - mdUVMin[0]) / g, ch = (mdUVMax[1] - mdUVMin[1]) / g;
    auto range = [&](const SmTrimLeaf & crL, int & x0, int & x1, int & y0, int & y1)
      {
        x0 = std::max(0, std::min(g - 1, (int)((crL.dU0 - mdUVMin[0]) / cw)));
        x1 = std::max(0, std::min(g - 1, (int)((crL.dU1 - mdUVMin[0]) / cw)));
        y0 = std::max(0, std::min(g - 1, (int)((crL.dV0 - mdUVMin[1]) / ch)));
        y1 = std::max(0, std::min(g - 1, (int)((crL.dV1 - mdUVMin[1]) / ch)));
      };
    auto usable = [&](const SmTrimLeaf & crL)
      {
        return crL.dU1 > crL.dU0 && crL.dV1 > crL.dV0 && crL.dU1 >= mdUVMin[0] && crL.dU0 <= mdUVMax[0]
            && crL.dV1 >= mdUVMin[1] && crL.dV0 <= mdUVMax[1];
      };
    std::vector<int> & rStart = mW.sLeafStart;
    rStart.assign(g * g + 1, 0);
    for (const SmTrimLeaf & crL : crLeaves)
      {
        if (!usable(crL)) continue;
        int x0, x1, y0, y1;
        range(crL, x0, x1, y0, y1);
        for (int yy = y0; yy <= y1; ++yy)
          for (int xx = x0; xx <= x1; ++xx) ++rStart[yy * g + xx + 1];
      }
    for (int c = 0; c < g * g; ++c) rStart[c + 1] += rStart[c];
    mW.sLeafCell.assign(rStart[g * g], 0);
    std::vector<int> & rFill = mW.sGridFill;
    rFill.assign(rStart.begin(), rStart.end() - 1);
    for (int l = 0; l < (int)crLeaves.size(); ++l)
      {
        if (!usable(crLeaves[l])) continue;
        int x0, x1, y0, y1;
        range(crLeaves[l], x0, x1, y0, y1);
        for (int yy = y0; yy <= y1; ++yy)
          for (int xx = x0; xx <= x1; ++xx) mW.sLeafCell[rFill[yy * g + xx]++] = l;
      }
  }

  // How many times the UV extent of edge a-b exceeds the size of the smallest tree leaf its UV box
  // meets, in u or v: SmTess::CheckAgainstSurfaceCache()'s measure, by which SMLib splits the
  // interior edges it makes. 0 without a leaf grid and for edges at poles (cached).
  double EdgeSpan(int a, int b)
  {
    const VertexData & A = mW.sVerts[a], & B = mW.sVerts[b];
    if (mW.sLeafStart.empty() || A.lPole >= 0 || B.lPole >= 0) return 0.0;
    const uint64_t key = (a < b) ? (((uint64_t)(uint32_t)a << 32) | (uint32_t)b) : (((uint64_t)(uint32_t)b << 32) | (uint32_t)a);
    auto it = mW.sEdgeSpan.find(key);
    if (it != mW.sEdgeSpan.end()) return it->second;
    // the segment without its first and last hundredth, as SMLib measures it
    const double du = B.sUV.x - A.sUV.x, dv = B.sUV.y - A.sUV.y;
    const double u0 = std::min(A.sUV.x + 0.01 * du, A.sUV.x + 0.99 * du), u1 = std::max(A.sUV.x + 0.01 * du, A.sUV.x + 0.99 * du);
    const double v0 = std::min(A.sUV.y + 0.01 * dv, A.sUV.y + 0.99 * dv), v1 = std::max(A.sUV.y + 0.01 * dv, A.sUV.y + 0.99 * dv);
    const int g = mlLeafGrid;
    const double cw = (mdUVMax[0] - mdUVMin[0]) / g, ch = (mdUVMax[1] - mdUVMin[1]) / g;
    const int x0 = std::max(0, std::min(g - 1, (int)((u0 - mdUVMin[0]) / cw))), x1 = std::max(0, std::min(g - 1, (int)((u1 - mdUVMin[0]) / cw)));
    const int y0 = std::max(0, std::min(g - 1, (int)((v0 - mdUVMin[1]) / ch))), y1 = std::max(0, std::min(g - 1, (int)((v1 - mdUVMin[1]) / ch)));
    double dSpan = 0.0;
    for (int yy = y0; yy <= y1; ++yy)
      for (int xx = x0; xx <= x1; ++xx)
        for (int q = mW.sLeafStart[yy * g + xx]; q < mW.sLeafStart[yy * g + xx + 1]; ++q)
          {
            const SmTrimLeaf & crL = mIn.sLeaves[mW.sLeafCell[q]];
            if (crL.dU1 < u0 || crL.dU0 > u1 || crL.dV1 < v0 || crL.dV0 > v1) continue;
            dSpan = std::max(dSpan, std::max((u1 - u0) / (crL.dU1 - crL.dU0), (v1 - v0) / (crL.dV1 - crL.dV0)));
          }
    mW.sEdgeSpan.emplace(key, dSpan);
    return dSpan;
  }

  // Flips edge i of triangle t (ImproveNormals(), ReduceBulge()). After refinement (mbJudgeFlips) the
  // flip stays only if both new triangles pass Judge(); otherwise it is undone exactly, so the
  // triangles that met the tolerances keep meeting them. Returns whether the flip stays.
  bool TryFlip(int t, int i)
  {
    if (!mbJudgeFlips)
      {
        mCdt.Flip(t, i);
        return true;
      }
    const int nb = mCdt.mT[t].n[i];
    const int j = mCdt.NeighborIndex(nb, t);
    const Tri sOldT = mCdt.mT[t], sOldN = mCdt.mT[nb];
    const int nW = sOldT.n[Next3(i)], mU = sOldN.n[Next3(j)];   // the outer neighbors Flip() relinks
    const int caV[4] = {sOldT.v[0], sOldT.v[1], sOldT.v[2], sOldN.v[j]};
    int caVT[4];
    for (int k = 0; k < 4; ++k) caVT[k] = mCdt.mVT[caV[k]];
    mCdt.Flip(t, i);
    SmPoint3d sCentroid;
    bool bHaveCentroid = false;
    if (Judge(t, sCentroid, bHaveCentroid) == kOk && Judge(nb, sCentroid, bHaveCentroid) == kOk)
      {
        mCdt.mT[t].state = mCdt.mT[nb].state = kGood;
        return true;
      }
    mCdt.ReplaceNeighbor(mU, t, nb);
    mCdt.ReplaceNeighbor(nW, nb, t);
    mCdt.mT[t] = sOldT;
    mCdt.mT[nb] = sOldN;
    for (int k = 0; k < 4; ++k) mCdt.mVT[caV[k]] = caVT[k];
    return false;
  }

  // Flips interior edges (strictly convex in UV) wherever that lowers the larger NormalDeviation()
  // of the two triangles: picks the diagonals that follow the surface where the working-plane
  // Delaunay criterion doesn't (high curvature next to coarse leaves). Each flip lowers the sorted
  // vector of deviations, so this terminates. A flip that makes an interior edge wider than
  // dLeafSpan leaves (EdgeSpan()), and wider than the edge it replaces, is left out unless one of the
  // triangles is Broken(): NormalDeviation() rates slivers as bad as degenerate triangles, and would
  // turn the strips of slivers along a curved face that SMLib makes (and that follow its leaves)
  // into fans across its curvature, which deviate from it many times more.
  void ImproveNormals()
  {
    if (mbPlanar) return;
    std::vector<std::pair<int, int>> & rStack = mCdt.mStack;
    rStack.clear();
    for (int t = 0; t < (int)mCdt.mT.size(); ++t)
      {
        const Tri & rT = mCdt.mT[t];
        if (rT.label <= 0) continue;
        for (int e = 0; e < 3; ++e)
          if (rT.n[e] > t) rStack.emplace_back(t, e);
      }
    while (!rStack.empty())
      {
        const int t = rStack.back().first, i = rStack.back().second;
        rStack.pop_back();
        const Tri & rT = mCdt.mT[t];
        if (rT.label <= 0 || EdgeBits(rT.c, i)) continue;
        const int nb = rT.n[i];
        if (nb < 0 || mCdt.mT[nb].label <= 0) continue;
        const int o = rT.v[i], u = rT.v[Next3(i)], w = rT.v[Prev3(i)];
        const int o2 = mCdt.mT[nb].v[mCdt.NeighborIndex(nb, t)];
        const double dOld = std::max(NormalDeviation(o, u, w), NormalDeviation(o2, w, u));
        if (dOld < 1.0e-12) continue;
        const double dNew = std::max(NormalDeviation(o, u, o2), NormalDeviation(o2, w, o));
        if (!(dNew < dOld)) continue;
        if (!(Sign(mCdt.Orient(o, o2, u)) * Sign(mCdt.Orient(o, o2, w)) < 0)) continue;
        if (!mW.sLeafStart.empty())
          {
            const double dSpan = EdgeSpan(o, o2);
            if (dSpan > g_sTuning.dLeafSpan && dSpan > EdgeSpan(u, w) && !Broken(o, u, w) && !Broken(o2, w, u)) continue;
          }
        if (!TryFlip(t, i)) continue;
        rStack.emplace_back(t, 0);
        rStack.emplace_back(t, 2);
        rStack.emplace_back(nb, 0);
        rStack.emplace_back(nb, 2);
      }
    ReduceBulge();
  }

  // Chord height of the straight edge a-b over the surface, estimated from the turning of the
  // corner normals along it: L^2 kappa / 8 = |(Nb - Na) . (b - a)| / 8 (0 at singular corners)
  double EdgeBulge(int a, int b) const
  {
    const VertexData & A = mW.sVerts[a], & B = mW.sVerts[b];
    if (A.bSingular || B.bSingular) return 0.0;
    return 0.125 * std::fabs((B.sNormal - A.sNormal).Dot(B.sPoint - A.sPoint));
  }

  // Chord height of the edge a-b: the offset of the surface point at its UV midpoint from the
  // chord's midpoint along the mean corner normal; -1 at a pole or singular corner or when the
  // evaluation fails
  double EdgeDeviation(int a, int b) const
  {
    const VertexData & A = mW.sVerts[a], & B = mW.sVerts[b];
    if (A.lPole >= 0 || B.lPole >= 0 || A.bSingular || B.bSingular) return -1.0;
    SmVector3d sN = A.sNormal + B.sNormal;
    const double len = sN.Length();
    if (!(len > 1.0e-3)) return -1.0;
    SmPoint3d sP;
    if (mpSurface->EvaluatePoint(SmPoint2d(0.5 * (A.sUV.x + B.sUV.x), 0.5 * (A.sUV.y + B.sUV.y)), sP) != SM_SUCCESS)
      return -1.0;
    const double d = std::fabs((sP - (A.sPoint + B.sPoint) * 0.5).Dot(sN)) / len;
    return IsFinite(d) ? d : -1.0;
  }

  // Flips interior edges (strictly convex in UV) whose other diagonal bulges clearly less from the
  // surface, unless that makes a triangle flipped, degenerate or a sliver, so that the diagonals
  // follow the surface's flatter direction: the diagonals of the tree leaves (two triangles whose
  // other sides are leaf sides) by EdgeBulge(), other edges, where they visibly bulge, by
  // EdgeDeviation(). Each flip lowers the sum of the edges' bulges, so this terminates.
  void ReduceBulge()
  {
    const double dRatio = g_sTuning.dBulgeFlip;
    if (mbPlanar || !(dRatio > 0.0)) return;
    std::unordered_map<uint64_t, double> & rCache = mW.sEdgeDev;
    rCache.clear();
    auto measured = [&](int a, int b)
      {
        if (a > b) std::swap(a, b);
        const uint64_t key = ((uint64_t)(uint32_t)a << 32) | (uint64_t)(uint32_t)b;
        auto it = rCache.find(key);
        if (it == rCache.end()) it = rCache.emplace(key, EdgeDeviation(a, b)).first;
        return it->second;
      };
    std::vector<std::pair<int, int>> & rStack = mCdt.mStack;
    rStack.clear();
    for (int t = 0; t < (int)mCdt.mT.size(); ++t)
      {
        const Tri & rT = mCdt.mT[t];
        if (rT.label <= 0) continue;
        for (int e = 0; e < 3; ++e)
          if (rT.n[e] > t) rStack.emplace_back(t, e);
      }
    while (!rStack.empty())
      {
        const int t = rStack.back().first, i = rStack.back().second;
        rStack.pop_back();
        const Tri & rT = mCdt.mT[t];
        if (rT.label <= 0 || EdgeBits(rT.c, i)) continue;
        const int nb = rT.n[i];
        if (nb < 0 || mCdt.mT[nb].label <= 0) continue;
        const int o = rT.v[i], u = rT.v[Next3(i)], w = rT.v[Prev3(i)];
        const int j = mCdt.NeighborIndex(nb, t);
        const int o2 = mCdt.mT[nb].v[j];
        const int leaf = kLeafEdge;
        const bool bLeafQuad = EdgeBits(rT.c, Next3(i)) == leaf && EdgeBits(rT.c, Prev3(i)) == leaf
                            && EdgeBits(mCdt.mT[nb].c, Next3(j)) == leaf && EdgeBits(mCdt.mT[nb].c, Prev3(j)) == leaf;
        double dOld = EdgeBulge(u, w), dNew = EdgeBulge(o, o2);
        if (!bLeafQuad)
          {
            const double dLenOld = mW.sVerts[u].sPoint.DistanceBetween(mW.sVerts[w].sPoint);
            const double dLenNew = mW.sVerts[o].sPoint.DistanceBetween(mW.sVerts[o2].sPoint);
            if (!(dOld > kBulgeSuspect * dLenOld || dNew > kBulgeSuspect * dLenNew)) continue;
            dOld = measured(u, w);
            dNew = measured(o, o2);
            if (dOld < 0.0 || dNew < 0.0) continue;
          }
        if (!(dNew < dRatio * dOld)) continue;
        if (!(Sign(mCdt.Orient(o, o2, u)) * Sign(mCdt.Orient(o, o2, w)) < 0)) continue;
        if (!(std::max(NormalDeviation(o, u, o2), NormalDeviation(o2, w, o)) < 1.0)) continue;
        if (!TryFlip(t, i)) continue;
        rStack.emplace_back(t, 0);
        rStack.emplace_back(t, 2);
        rStack.emplace_back(nb, 0);
        rStack.emplace_back(nb, 2);
      }
  }

  // Invalid triangles (see TriangleValid()): rlInterior counts those not SplitActionable()
  void CountInvalid(ULONG & rlInterior, ULONG & rlBoundary) const
  {
    rlInterior = rlBoundary = 0;
    for (int t = 0; t < (int)mCdt.mT.size(); ++t)
      {
        const Tri & rTri = mCdt.mT[t];
        if (rTri.label <= 0 || rTri.state == kDropped || TriangleValid(t)) continue;
        if (SplitActionable(rTri)) ++rlBoundary; else ++rlInterior;
      }
  }

  // Refinement never makes the triangulation worse: when it doesn't meet the tolerances (budget
  // exhausted, or a triangle that needs refining can't take a new vertex, nor a loop segment be split
  // for it) or leaves invalid triangles without a loop segment, the unrefined triangulation is used
  // instead if it has none, with bTolerancesMet false (the caller's fallback).
  bool Refine(SmTrimFaceFailure & rFail)
  {
    BuildLeafGrid();
    ImproveNormals();
    mW.sSaved.CopyFrom(mCdt);
    mW.sSavedVerts = mW.sVerts;
    const ULONG lSavedNew = mlNew;
    if (!RefineOnce(rFail)) return false;
    if (mOut.lRefinePoints > 0)   // the flips may not undo what refinement met
      {
        mbJudgeFlips = true;
        ImproveNormals();
        mbJudgeFlips = false;
      }
    ULONG lInterior = 0, lBoundary = 0;
    CountInvalid(lInterior, lBoundary);
    if (lInterior > 0 || !mOut.bTolerancesMet)  // made invalid triangles, or didn't converge
      {
        mCdt.CopyFrom(mW.sSaved);
        mW.sVerts = mW.sSavedVerts;
        mlNew = lSavedNew;
        mOut.lRefinePoints = 0;
        mOut.bTolerancesMet = false;
        for (Tri & rTri : mCdt.mT)  // drop pole triangles again
          if (rTri.label > 0 && rTri.state != kDropped)
            {
              const long r0 = mW.sVerts[rTri.v[0]].lRef, r1 = mW.sVerts[rTri.v[1]].lRef, r2 = mW.sVerts[rTri.v[2]].lRef;
              const VertexData * cpX = (r0 == r1) ? &mW.sVerts[rTri.v[0]] : (r1 == r2) ? &mW.sVerts[rTri.v[1]]
                                     : (r2 == r0) ? &mW.sVerts[rTri.v[2]] : nullptr;
              const VertexData * cpY = (r0 == r1) ? &mW.sVerts[rTri.v[1]] : (r1 == r2) ? &mW.sVerts[rTri.v[2]]
                                     : (r2 == r0) ? &mW.sVerts[rTri.v[0]] : nullptr;
              if (cpX && cpX->lPole >= 0 && cpX->lPole == cpY->lPole) rTri.state = kDropped;
            }
        CountInvalid(lInterior, lBoundary);
        if (lInterior > 0) { rFail = SM_TF_INVALID_TRIANGLE; return false; }
      }
    // A triangle on a loop segment can stay flipped or degenerate when the loop segments are too coarse
    // for the surface (SMLib's tessellation has the same triangle unless its Phase 2 splits the edges):
    // its loop segments are reported, for the caller to split them (Output() rejects any other), as are
    // the loop segments too near the new vertices of triangles the tolerances want refined.
    mOut.sSplitSegments.clear();
    if (mOut.bTolerancesMet) mOut.sSplitSegments = msBlocking;
    for (int t = 0; t < (int)mCdt.mT.size(); ++t)
      {
        const Tri & rTri = mCdt.mT[t];
        if (rTri.label <= 0 || rTri.state == kDropped || TriangleValid(t)) continue;
        for (int e = 0; e < 3; ++e)
          if ((rTri.c >> e) & 1)
            {
              mOut.sSplitSegments.push_back(mW.sVerts[rTri.v[Next3(e)]].lRef);
              mOut.sSplitSegments.push_back(mW.sVerts[rTri.v[Prev3(e)]].lRef);
            }
      }
    return true;
  }

  bool RefineOnce(SmTrimFaceFailure & rFail)
  {
    std::vector<int> & rQueue = mW.sQueue;
    rQueue.clear();
    for (int t = 0; t < (int)mCdt.mT.size(); ++t)
      if (mCdt.mT[t].label > 0) rQueue.push_back(t);
    size_t head = 0;
    bool bExhausted = false, bUnresolved = false;
    msBlocking.clear();
    msBlockedTris.clear();
    while (head < rQueue.size())
      {
        const int t = rQueue[head++];
        Tri & rTri = mCdt.mT[t];
        if (rTri.label <= 0 || rTri.state != kUnchecked) continue;
        SmPoint3d sCentroidP;
        bool bHaveCentroid = false;
        const Verdict eV = Judge(t, sCentroidP, bHaveCentroid);
        if (eV == kOk)      { rTri.state = kGood; continue; }
        if (eV == kDrop)    { rTri.state = kDropped; continue; }
        if (eV == kInvalid) { rTri.state = kGood; continue; }  // checked again below
        if (eV == kEvalError) { rFail = SM_TF_EVALUATION; return false; }
        if (mlNew >= mlBudget) { bExhausted = true; break; }  // best effort: checked below
        mCdt.mTouched.clear();
        const SplitResult eS = SplitTriangle(t, eV == kRefineCentroid, bHaveCentroid ? &sCentroidP : nullptr);
        if (eS == kSplitError) { rFail = SM_TF_EVALUATION; return false; }
        if (eS == kSplitCannot)  // no new vertex fits (a sliver along the loops): tolerances not met
          {
            if (mlBlockA < 0) { bUnresolved = true; break; }   // nor a loop segment to split for it
            msBlocking.push_back(mlBlockA);   // the caller splits that segment, then tries again
            msBlocking.push_back(mlBlockB);
            msBlockedTris.push_back(TriIds(mCdt.mT[t]));
            mCdt.mT[t].state = kGood;
            continue;
          }
        ++mOut.lRefinePoints;
        for (int tt : mCdt.mTouched)
          if (mCdt.mT[tt].label > 0) rQueue.push_back(tt);
        if (head > 4096 && head * 2 > rQueue.size())
          {
            rQueue.erase(rQueue.begin(), rQueue.begin() + head);
            head = 0;
          }
      }
    mOut.bTolerancesMet = !bExhausted && !bUnresolved;
    return true;
  }

  // Not degenerate in 3D (by mesh_checker's measure: twice the area above 2e-12 times the longest
  // edge squared), its corners distinct vertices, and wound with its corners' surface normals
  // (their sum, as mesh_checker judges flipped triangles).
  bool TriangleValid(int t) const
  {
    const Tri & rTri = mCdt.mT[t];
    const VertexData & A = mW.sVerts[rTri.v[0]];
    const VertexData & B = mW.sVerts[rTri.v[1]];
    const VertexData & C = mW.sVerts[rTri.v[2]];
    if (A.lRef == B.lRef || B.lRef == C.lRef || C.lRef == A.lRef) return false;
    const SmVector3d sAB = B.sPoint - A.sPoint;
    const SmVector3d sNT = sAB * (C.sPoint - A.sPoint);
    const double area2 = sNT.Length();
    const double lMax2 = std::max(sAB.LengthSquared(), std::max((C.sPoint - B.sPoint).LengthSquared(),
                                                                (A.sPoint - C.sPoint).LengthSquared()));
    if (!(area2 > kDegenerateArea * lMax2)) return false;
    SmVector3d sSum(0.0, 0.0, 0.0);
    const VertexData * caV[3] = {&A, &B, &C};
    for (int k = 0; k < 3; ++k)
      if (!caV[k]->bSingular) sSum += caV[k]->sNormal;
    return sNT.Dot(sSum) > 0.0;
  }

  // Whether 3D point crP keeps dSplitClearance of their lengths from the loop segments of triangle
  // t and its inside neighbors (poles excepted: one point in 3D); if not, mlBlockA and mlBlockB are
  // the loop vertex ids of a segment too near
  bool Clear3D(int t, const SmPoint3d & crP) const
  {
    const int caT[4] = {t, mCdt.mT[t].n[0], mCdt.mT[t].n[1], mCdt.mT[t].n[2]};
    for (int k = 0; k < 4; ++k)
      {
        const int tt = caT[k];
        if (tt < 0 || (k > 0 && mCdt.mT[tt].label <= 0)) continue;
        const Tri & rT = mCdt.mT[tt];
        if ((rT.c & 7) == 0) continue;
        for (int e = 0; e < 3; ++e)
          {
            if (!((rT.c >> e) & 1)) continue;
            const VertexData & P = mW.sVerts[rT.v[Next3(e)]];
            const VertexData & Q = mW.sVerts[rT.v[Prev3(e)]];
            if (P.lRef == Q.lRef) continue;
            const SmVector3d sD = Q.sPoint - P.sPoint;
            const double len2 = sD.LengthSquared();
            const SmVector3d sR = crP - P.sPoint;
            double s = (len2 > 0.0) ? sR.Dot(sD) / len2 : 0.0;
            s = std::min(1.0, std::max(0.0, s));
            const SmVector3d sE = sR - sD * s;
            if (sE.LengthSquared() < g_sTuning.dSplitClearance * g_sTuning.dSplitClearance * len2)
              {
                mlBlockA = P.lRef;
                mlBlockB = Q.lRef;
                return false;
              }
          }
      }
    return true;
  }

  // New vertex for triangle t: its working-plane circumcenter when that is inside the face and
  // clear of the loop segments in 3D, else its centroid when that is clear; kSplitCannot when
  // neither is (a sliver along the boundary that new vertices can't improve).
  SplitResult SplitTriangle(int t, bool bForceCentroid, const SmPoint3d * cpCentroidPoint)
  {
    mlBlockA = mlBlockB = -1;
    const Tri sTri = mCdt.mT[t];
    const int a = sTri.v[0], b = sTri.v[1], c = sTri.v[2];
    const double ax = mCdt.mX[a], ay = mCdt.mY[a];
    const double bx = mCdt.mX[b] - ax, by = mCdt.mY[b] - ay, cx = mCdt.mX[c] - ax, cy = mCdt.mY[c] - ay;
    if (!bForceCentroid)
      {
        const double d = 2.0 * (bx * cy - by * cx);
        if (d != 0.0)
          {
            const double b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
            const double ox = ax + (cy * b2 - by * c2) / d, oy = ay + (bx * c2 - cx * b2) / d;
            int tt = -1, idx = 0;
            const int loc = (IsFinite(ox) && IsFinite(oy)) ? mCdt.Locate(ox, oy, t, tt, idx, true) : kLocFail;
            if (   (loc == kLocIn || loc == kLocEdge) && mCdt.mT[tt].label > 0
                && !(loc == kLocEdge && ((mCdt.mT[tt].c >> idx) & 1)))
              {
                bool bNear = false;
                const Tri & rT2 = mCdt.mT[tt];
                for (int k = 0; k < 3 && !bNear; ++k)
                  {
                    const int v = rT2.v[k];
                    if (std::hypot(mCdt.mX[v] - ox, mCdt.mY[v] - oy) < 1.0e-9 * mdWorkDiag) bNear = true;
                  }
                if (!bNear)
                  {
                    double u, v;
                    mMap.Inverse(ox, oy, u, v);
                    SmPoint3d sP;
                    if (mpSurface->EvaluatePoint(SmPoint2d(u, v), sP) != SM_SUCCESS) return kSplitError;
                    if (Clear3D(tt, sP) && Clear3D(t, sP))
                      return AddVertexAt(tt, loc, idx, ox, oy, u, v, &sP) ? kSplitDone : kSplitError;
                  }
              }
          }
      }
    // centroid (in UV, pole corners as in CornerUV())
    const VertexData & A = mW.sVerts[a], & B = mW.sVerts[b], & C = mW.sVerts[c];
    const SmPoint2d sG = CentroidUV(A, B, C);
    const double uc = sG.x, vc = sG.y;
    double gx, gy;
    mMap.Forward(uc, vc, gx, gy);
    int tt = -1, idx = 0;
    const int loc = mCdt.Locate(gx, gy, t, tt, idx, true);
    mlBlockA = mlBlockB = -1;   // (the circumcenter's clearance doesn't count)
    if (loc != kLocIn && loc != kLocEdge) return kSplitCannot;
    if (mCdt.mT[tt].label <= 0) return kSplitCannot;
    if (loc == kLocEdge && ((mCdt.mT[tt].c >> idx) & 1))   // on a loop segment: that one
      {
        mlBlockA = mW.sVerts[mCdt.mT[tt].v[Next3(idx)]].lRef;
        mlBlockB = mW.sVerts[mCdt.mT[tt].v[Prev3(idx)]].lRef;
        if (mlBlockA == mlBlockB) mlBlockA = mlBlockB = -1;
        return kSplitCannot;
      }
    SmPoint3d sP;
    if (cpCentroidPoint) sP = *cpCentroidPoint;
    else if (mpSurface->EvaluatePoint(SmPoint2d(uc, vc), sP) != SM_SUCCESS) return kSplitError;
    if (!bForceCentroid && !(Clear3D(tt, sP) && Clear3D(t, sP))) return kSplitCannot;
    return AddVertexAt(tt, loc, idx, gx, gy, uc, vc, &sP) ? kSplitDone : kSplitError;
  }

  bool Output(SmTrimFaceFailure & rFail)
  {
    rFail = SM_TF_INTERNAL;
    const double dSign = mbReversed ? -1.0 : 1.0;
    mOut.sNewUV.resize(mlNew);
    mOut.sNewPoints.resize(mlNew);
    mOut.sNewNormals.resize(mlNew);
    for (size_t v = 3; v < mW.sVerts.size(); ++v)
      {
        const VertexData & crV = mW.sVerts[v];
        if (crV.lRef >= 0) continue;
        const ULONG k = (ULONG)~crV.lRef;
        mOut.sNewUV[k] = crV.sUV;
        mOut.sNewPoints[k] = crV.sPoint;
        mOut.sNewNormals[k] = crV.sNormal * dSign;
      }
    mOut.sLoopNormals.resize(mIn.sLoops.size());
    ULONG k = 0;
    for (size_t l = 0; l < mIn.sLoops.size(); ++l)
      {
        mOut.sLoopNormals[l].resize(mIn.sLoops[l].size());
        for (size_t i = 0; i < mIn.sLoops[l].size(); ++i, ++k) mOut.sLoopNormals[l][i] = mW.sLoopNormal[k] * dSign;
      }
    // counterclockwise in the working plane, as in UV (the direction of the loops), reversed for a
    // reversed face
    const int i1 = mbReversed ? 2 : 1, i2 = mbReversed ? 1 : 2;
    mOut.sTriangles.clear();
    mOut.sTriangles.reserve(3 * mCdt.mT.size() / 2);
    mOut.sTriangleUVs.clear();
    for (int t = 0; t < (int)mCdt.mT.size(); ++t)
      {
        const Tri & crT = mCdt.mT[t];
        if (crT.label <= 0 || crT.state == kDropped) continue;
        // valid, or reported for its loop segment to be split (after the last edge flips)
        if (!TriangleValid(t) && !(SplitActionable(crT) && !mOut.sSplitSegments.empty()))
          { rFail = SM_TF_INVALID_TRIANGLE; return false; }
        const int v0 = crT.v[0], v1 = crT.v[i1], v2 = crT.v[i2];
        const long r0 = mW.sVerts[v0].lRef, r1 = mW.sVerts[v1].lRef, r2 = mW.sVerts[v2].lRef;
        if (r0 == r1 || r1 == r2 || r2 == r0) return false;
        mOut.sTriangles.insert(mOut.sTriangles.end(), {r0, r1, r2});
        if (mIn.bTriangleUVs)
          mOut.sTriangleUVs.insert(mOut.sTriangleUVs.end(), {mW.sVerts[v0].sUV, mW.sVerts[v1].sUV, mW.sVerts[v2].sUV});
      }
    if (mOut.sTriangles.empty()) return false;
    rFail = SM_TF_NONE;
    return true;
  }
};

} // namespace

/*******************************************************************//**
PURPOSE: Triangulate one trimmed face from its boundary loops.

NOTES: See SmFastFace.h. Deterministic and thread-safe; uses a
   thread-local workspace that keeps its memory between calls.
***********************************************************************/
SmStatus SmTriangulateTrimmedFace(const SmTrimFaceInput & crIn, SmTrimFaceOutput & rOut)
{
  rOut.eStatus = SM_ERR;
  rOut.eFailure = SM_TF_NONE;
  rOut.sNewUV.clear();
  rOut.sNewPoints.clear();
  rOut.sNewNormals.clear();
  rOut.sTriangles.clear();
  rOut.sTriangleUVs.clear();
  rOut.sLoopNormals.clear();
  rOut.sSplitSegments.clear();
  rOut.bTolerancesMet = true;
  rOut.lSamplesUsed = 0;
  rOut.lRefinePoints = 0;
  Workspace & rW = GetWorkspace();
  rW.Clear();
  FaceTriangulator sTriangulator(crIn, rOut, rW);
  return sTriangulator.Run();
}

/*******************************************************************//**
PURPOSE: Name of a SmTrimFaceFailure category.
***********************************************************************/
const char * SmTrimFaceFailureName(SmTrimFaceFailure eFailure)
{
  switch (eFailure)
    {
      case SM_TF_NONE:              return "none";
      case SM_TF_BAD_INPUT:         return "bad_input";
      case SM_TF_DUPLICATE_POINT:   return "duplicate_point";
      case SM_TF_LOOP_INTERSECTION: return "loop_intersection";
      case SM_TF_LOOP_ORIENTATION:  return "loop_orientation";
      case SM_TF_EVALUATION:        return "evaluation";
      case SM_TF_INVALID_TRIANGLE:  return "invalid_triangle";
      case SM_TF_INTERNAL:          return "internal";
    }
  return "unknown";
}

#ifdef SM_FASTFACE_DEBUG
/*******************************************************************//**
PURPOSE: Test hook: sets a tuning parameter by name; false when unknown.
***********************************************************************/
bool SmTrimFaceDebugTuning(const std::string & crName, double dValue)
{
  struct Entry { const char * pszName; double * pValue; };
  const Entry aEntries[] = {
    {"angle_factor", &g_sTuning.dAngleFactor},         {"deviation_factor", &g_sTuning.dDeviationFactor},
    {"leaf_clearance", &g_sTuning.dLeafClearance},
    {"leaf_angle_factor", &g_sTuning.dLeafAngleFactor},
    {"boundary_margin", &g_sTuning.dBoundaryMargin},   {"sample_clearance", &g_sTuning.dSampleClearance},
    {"split_clearance", &g_sTuning.dSplitClearance},
    {"min_budget", &g_sTuning.dMinBudget},             {"budget_per_vertex", &g_sTuning.dBudgetPerVertex},
    {"leaf_budget_per_vertex", &g_sTuning.dLeafBudgetPerVertex}, {"bulge_flip", &g_sTuning.dBulgeFlip},
    {"leaf_deviation", &g_sTuning.dLeafDeviation}, {"leaf_span", &g_sTuning.dLeafSpan},
    {"chord_factor", &g_sTuning.dChordFactor}};
  for (const Entry & crE : aEntries)
    if (crName == crE.pszName) { *crE.pValue = dValue; return true; }
  return false;
}
#endif
