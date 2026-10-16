// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmFastTessSrfCache.h
* PURPOSE: The refinement tree cache of SmComputeFastBoundaries()
*    (SmFastEdges.cpp), with what SmTessSrfCache::BuildTreeFast()
*    (SmTessFast.cpp) needs beyond an SmTessSrfCache. A derived class, so
*    that the exported SmTessSrfCache keeps its layout. Used inside SMLib
*    only: not exported (its members are inline).
**********************************************************************/

#ifndef __SMFASTTESSSRFCACHE_H__
#define __SMFASTTESSSRFCACHE_H__

#include <SmTess.h>

class SmFastTessSrfCache : public SmTessSrfCache
{
public:
  SmFastTessSrfCache
  (
    SmFace              * pFace,                 ///< [in ]: face to tessellate, as SmTessSrfCache's                 <br>
    SmSurfaceTessDriver * pSurfaceTessDriver,    ///< [in ]: as SmTessSrfCache's                                     <br>
    double                dChordHeightTolerance, ///< [in ]: as SmTessSrfCache's                                     <br>
    double                dAngleTolerance,       ///< [in ]: as SmTessSrfCache's (radians)                           <br>
    double                dMaxEdgeLength3D,      ///< [in ]: as SmTessSrfCache's                                     <br>
    double                dMinEdgeLength3D,      ///< [in ]: as SmTessSrfCache's                                     <br>
    double                dMinEdgeLengthRatioUV, ///< [in ]: as SmTessSrfCache's                                     <br>
    double                dMaxAspectRatio        ///< [in ]: as SmTessSrfCache's                                     <br>
  )
  : SmTessSrfCache(pFace, NULL, pSurfaceTessDriver, dChordHeightTolerance, dAngleTolerance, dMaxEdgeLength3D,
                   dMinEdgeLength3D, dMinEdgeLengthRatioUV, dMaxAspectRatio)
  {}

  virtual SM_TYPE       GetType()           const { return SmFastTessSrfCache_TYPE; }
  virtual const TCHAR * GetTypeString()     const { return _T("SmFastTessSrfCache_TYPE"); }
  virtual const TCHAR * GetClassString()    const { return _T("SmFastTessSrfCache"); }
  virtual SmBoolean     IsKindOf(SM_TYPE t) const
    { return (t == SmFastTessSrfCache_TYPE) ? TRUE : SmTessSrfCache::IsKindOf(t); }

  // BuildTreeFast(): with m_bFastRegion, nodes that don't meet m_sFastRegion (where the face lies on its
  // surface, shifted by whole periods m_adFastPeriod) stay leaves
  SmBoolean  m_bFastRegion     = FALSE;
  SmExtent2d m_sFastRegion;
  double     m_adFastPeriod[2] = {0.0, 0.0};
};

#endif // !__SMFASTTESSSRFCACHE_H__
