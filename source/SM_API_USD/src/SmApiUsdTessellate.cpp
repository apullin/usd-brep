// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/******************************************************************//**
* FILE NAME --- SmApiUsdTessellate.cpp
* PURPOSE: Tessellate a BrepArray prim into UsdGeomMesh-ready arrays, or a whole
*          USD file into UsdGeomMesh prims.
**********************************************************************/

#include "SmApiUsdTessellate.h"
#include "SmApiUsd.h"

// USD includes
#include "UsdBrepHeaders.h"
#include "UsdBrepSuppressPixarWarningsPush.h"
#include <pxr/usd/sdf/copyUtils.h>
#include <pxr/usd/sdf/layerUtils.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdUtils/dependencies.h>
#include "UsdBrepSuppressPixarWarningsPop.h"

// BREP_SM_USD conversion layer
#include "SmuAttribute.h"
#include "SmuConvert.h"
#include "SmuTessellate.h"
#include "UsdBrepUtilities.h"

// SMLib / SM_API
#include "SmApiBrep.h"
#include "SmBrep.h"
#include "SmContext.h"
#include "SmFace.h"
#include "SmMessages.h"

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <unordered_map>

SmApiUsdTessellator SmApiUsdGetDefaultTessellator()
{
    static const SmApiUsdTessellator seDefault = []
    {
        const char* pValue = std::getenv("SMLIB_FAST_TESSELLATION");
        return (pValue && std::strcmp(pValue, "1") == 0) ? SM_API_USD_TESSELLATOR_FAST : SM_API_USD_TESSELLATOR_SMLIB;
    }();
    return seDefault;
}

// GetTessellatedMeshData's arrays from SmApiTessellateFast's output, with its conventions: float points and
// extent, one USD face per triangle, face-varying normals indexed per corner with one normal per vertex and
// Brep face, and one GeomSubset per face material attribute with its first path. GetMeshData orders the
// subsets by an unordered_map; here they come in the order of their first face.
static SmStatus sm_GetFastMeshData(const SmBrep& crBrep, const SmFastTessellation& crMesh, SmApiUsdTessellatedBrep& rResult)
{
    const ULONG lVertices  = crMesh.sPoints.GetSize() / 3;
    const ULONG lTriangles = crMesh.sTriangles.GetSize() / 3;
    if (crMesh.sPoints.GetSize() != 3 * lVertices || crMesh.sTriangles.GetSize() != 3 * lTriangles
        || crMesh.sTriangleFaces.GetSize() != lTriangles || crMesh.sCornerNormals.GetSize() != 9 * lTriangles
        || lVertices > static_cast<ULONG>(INT_MAX) || lTriangles > static_cast<ULONG>(INT_MAX / 3))
        return SM_ERR;
    SmTArray<SmFace*> sFaces;
    crBrep.GetFaces(sFaces);

    SmExtent3d sBox;
    pxr::VtArray<pxr::GfVec3f> sPoints(lVertices);
    for (ULONG ii = 0; ii < lVertices; ++ii)
    {
        const SmPoint3d sPoint(crMesh.sPoints[3 * ii], crMesh.sPoints[3 * ii + 1], crMesh.sPoints[3 * ii + 2]);
        sBox.AddPoint3d(sPoint);
        sPoints[ii] = pxr::GfVec3f(static_cast<float>(sPoint.x), static_cast<float>(sPoint.y), static_cast<float>(sPoint.z));
    }
    pxr::VtArray<pxr::GfVec3f> sExtent;
    if (!sBox.IsInit())
    {
        const SmPoint3d sMin = sBox.GetMin(), sMax = sBox.GetMax();
        sExtent = { pxr::GfVec3f(static_cast<float>(sMin.x), static_cast<float>(sMin.y), static_cast<float>(sMin.z)),
                    pxr::GfVec3f(static_cast<float>(sMax.x), static_cast<float>(sMax.y), static_cast<float>(sMax.z)) };
    }

    pxr::VtIntArray sCounts(lTriangles, 3);
    pxr::VtIntArray sIndices(3 * lTriangles);
    pxr::VtIntArray sNormalIndices(3 * lTriangles);
    pxr::VtArray<pxr::GfVec3f> sNormals;
    sNormals.reserve(lVertices);
    // The normal of each vertex for the last face that used it. A face's triangles are consecutive, so a
    // vertex gets one normal per face, unless its corners in a face have different normals.
    std::vector<int> sLastFace(lVertices, -1), sLastNormal(lVertices, -1);
    std::vector<SmSdfPathAttribute*> sMaterials;
    std::vector<std::vector<int>> sSubsets;
    std::unordered_map<SmSdfPathAttribute*, size_t> sSubsetOfMaterial;
    for (ULONG tt = 0; tt < lTriangles; ++tt)
    {
        const int iFace = crMesh.sTriangleFaces[tt];
        if (iFace < 0 || static_cast<ULONG>(iFace) >= sFaces.GetSize())
            return SM_ERR;
        for (ULONG cc = 3 * tt; cc < 3 * tt + 3; ++cc)
        {
            const int iVertex = crMesh.sTriangles[cc];
            if (iVertex < 0 || static_cast<ULONG>(iVertex) >= lVertices)
                return SM_ERR;
            pxr::GfVec3f sNormal(crMesh.sCornerNormals[3 * cc], crMesh.sCornerNormals[3 * cc + 1], crMesh.sCornerNormals[3 * cc + 2]);
            sNormal.Normalize();
            if (sLastFace[iVertex] != iFace || sNormals[sLastNormal[iVertex]] != sNormal)
            {
                sLastFace[iVertex] = iFace;
                sLastNormal[iVertex] = static_cast<int>(sNormals.size());
                sNormals.push_back(sNormal);
            }
            sIndices[cc] = iVertex;
            sNormalIndices[cc] = sLastNormal[iVertex];
        }
        SmSdfPathAttribute* pMaterial = SM_CAST_PTR(SmSdfPathAttribute, sFaces[iFace]->FindAttribute(SM_AI_MATERIAL_BINDING));
        if (pMaterial)
        {
            const auto sFound = sSubsetOfMaterial.emplace(pMaterial, sSubsets.size());
            if (sFound.second)
            {
                sMaterials.push_back(pMaterial);
                sSubsets.emplace_back();
            }
            sSubsets[sFound.first->second].push_back(static_cast<int>(tt));
        }
    }

    rResult.m_sExtent = std::move(sExtent);
    rResult.m_sPoints = std::move(sPoints);
    rResult.m_sFaceVertexCounts = std::move(sCounts);
    rResult.m_sFaceVertexIndices = std::move(sIndices);
    rResult.m_sNormals = std::move(sNormals);
    rResult.m_sNormalsIndices = std::move(sNormalIndices);
    rResult.m_sNormalsInterpolation = pxr::UsdGeomTokens->faceVarying;
    rResult.m_sSubdivisionScheme = pxr::UsdGeomTokens->none;
    rResult.m_sSubsetIndices.clear();
    rResult.m_sSubsetMaterialPaths.clear();
    for (size_t ii = 0; ii < sSubsets.size(); ++ii)
    {
        rResult.m_sSubsetIndices.push_back(pxr::VtIntArray(sSubsets[ii].begin(), sSubsets[ii].end()));
        const pxr::SdfPathVector* pPaths = sMaterials[ii]->GetMaterialPaths();
        rResult.m_sSubsetMaterialPaths.push_back(pPaths && !pPaths->empty() ? pPaths->front() : pxr::SdfPath());
    }
    return SM_SUCCESS;
}

// The mesh part of one SmApiUsdTessellateBrepArray result, through SmApiTessellateFast, with the same
// partial-mesh policy, failed faces, statuses and messages as the SmApiTessellate path.
static void sm_TessellateFast(const SmBrep& crBrep, const SmTessellationParams& crParams, const std::string& crIndex,
                              SmApiUsdTessellatedBrep& rResult)
{
    SmFastTessellation sMesh;
    const SmStatus tessStatus = SmApiTessellateFast(&crBrep, sMesh, crParams, TRUE, TRUE);
    for (ULONG ii = 0; ii < sMesh.sFaceKinds.GetSize(); ++ii)
        if (sMesh.sFaceKinds[ii] == SM_FAST_FACE_FAILED)
            rResult.m_sFailedFaces.push_back(static_cast<int>(ii));
    if (tessStatus != SM_SUCCESS)
    {
        rResult.m_sStatus = tessStatus;
        rResult.m_sMessage = "Tessellation failed on Brep " + crIndex;
        return;
    }
    const SmStatus meshStatus = sm_GetFastMeshData(crBrep, sMesh, rResult);
    if (meshStatus != SM_SUCCESS)
    {
        rResult.m_sPoints.clear();
        rResult.m_sFaceVertexCounts.clear();
        rResult.m_sFaceVertexIndices.clear();
        rResult.m_sStatus = meshStatus;
        rResult.m_sMessage = "GetMeshData failed on Brep " + crIndex;
        return;
    }
    rResult.m_sStatus = SM_SUCCESS;
}

SmStatus SmApiUsdTessellateBrepArray
(
    const pxr::UsdPrim                   & crBrepArray,
    const SmTessellationParams           & crParams,
    SmBoolean                              bHealerIsEnabled,
    double                                 dBoundaryAngleTolDeg,
    std::vector<SmApiUsdTessellatedBrep> & rResults
)
{
    return SmApiUsdTessellateBrepArray(crBrepArray, crParams, bHealerIsEnabled, dBoundaryAngleTolDeg,
                                       SM_API_USD_TESSELLATOR_DEFAULT, rResults);
}

SmStatus SmApiUsdTessellateBrepArray
(
    const pxr::UsdPrim                   & crBrepArray,
    const SmTessellationParams           & crParams,
    SmBoolean                              bHealerIsEnabled,
    double                                 dBoundaryAngleTolDeg,
    SmApiUsdTessellator                    eTessellator,
    std::vector<SmApiUsdTessellatedBrep> & rResults
)
{
    rResults.clear();
    auto failAll = [&](const std::string& sMessage)
    {
        rResults.emplace_back().m_sMessage = sMessage + crBrepArray.GetPath().GetString();
        return SM_ERR;
    };

    if (eTessellator == SM_API_USD_TESSELLATOR_DEFAULT)
        eTessellator = SmApiUsdGetDefaultTessellator();
    if (eTessellator != SM_API_USD_TESSELLATOR_SMLIB && eTessellator != SM_API_USD_TESSELLATOR_FAST)
    {
        failAll("invalid tessellator for ");
        return SM_ERR_INVALID_INPUT;
    }

    const pxr::UsdGeomGprim gprim(crBrepArray);
    if (!gprim)
        return failAll("prim is not a UsdGeomGprim: ");

    // Declared first so it outlives the Breps.
    SmContext sContext;
    std::vector<std::unique_ptr<SmBrep>> sOwnedBreps;
    // Per-member import: one bad member does not discard the rest.
    std::vector<SMU_BrepConvert::BrepImportResult> sImports;
    if (SMU_BrepConvert::BrepMove_UsdToSMLibWithResults(sContext, gprim, sImports, bHealerIsEnabled, FALSE) != SM_SUCCESS)
        return failAll("BrepMove_UsdToSMLibWithResults failed on ");
    for (const SMU_BrepConvert::BrepImportResult& rImport : sImports)
        sOwnedBreps.emplace_back(rImport.pBrep);  // null if the import failed

    // The import gives each Brep its "brep" GeomSubset material, else the BrepArray's own
    // binding; report only the former, so consumers keep their resolved BrepArray binding.
    const pxr::SdfPath arrayMaterial = pxr::UsdShadeMaterialBindingAPI(crBrepArray).GetDirectBinding().GetMaterialPath();

    for (const SMU_BrepConvert::BrepImportResult& rImport : sImports)
    {
        SmApiUsdTessellatedBrep& rResult = rResults.emplace_back();
        rResult.m_iPackedBrepIndex = static_cast<std::int64_t>(rImport.iPackedBrepIndex);
        const std::string sIndex = std::to_string(rImport.iPackedBrepIndex);
        SmBrep* pBrep = rImport.pBrep;
        if (!pBrep)
        {
            rResult.m_sStatus = (rImport.status != SM_SUCCESS) ? rImport.status : SM_ERR;
            rResult.m_sMessage = "Import failed on Brep " + sIndex
                                 + (rImport.bRemainingMembersSkipped ? "; remaining Breps skipped" : "");
            continue;
        }

        SmSdfPathAttribute* pMaterial = SM_CAST_PTR(SmSdfPathAttribute, pBrep->FindAttribute(SM_AI_MATERIAL_BINDING));
        if (pMaterial && !pMaterial->GetMaterialPaths()->empty() && pMaterial->GetMaterialPaths()->front() != arrayMaterial)
            rResult.m_sMaterialPath = pMaterial->GetMaterialPaths()->front();

        if (dBoundaryAngleTolDeg > 0.0)
            rResult.m_sBoundaryStatus = SMU_BrepConvert::GetBoundaryData(pBrep, dBoundaryAngleTolDeg,
                                                                         rResult.m_sBoundaryPoints,
                                                                         rResult.m_sBoundaryVertexCounts);

        if (eTessellator == SM_API_USD_TESSELLATOR_FAST)
        {
            sm_TessellateFast(*pBrep, crParams, sIndex, rResult);
            continue;
        }

        SmPolyBrep* pMesh = nullptr;
        SmTArray<SmTessellationFailure> sFailures;
        const SmStatus tessStatus = SmApiTessellate(pBrep, pMesh, crParams, TRUE, &sFailures);
        SMU_BrepConvert::TessellatedMeshPtr pOwnedMesh(pMesh);
        for (const SmTessellationFailure& rFailure : sFailures)
            rResult.m_sFailedFaces.push_back(static_cast<int>(rFailure.lFaceIndex));
        if (tessStatus != SM_SUCCESS || !pMesh)
        {
            rResult.m_sStatus = (tessStatus != SM_SUCCESS) ? tessStatus : SM_ERR;
            rResult.m_sMessage = "Tessellation failed on Brep " + sIndex;
            continue;
        }

        const SmStatus meshStatus = SMU_BrepConvert::GetTessellatedMeshData(
            *pMesh, rResult.m_sExtent, rResult.m_sPoints, rResult.m_sFaceVertexCounts, rResult.m_sFaceVertexIndices,
            rResult.m_sNormals, rResult.m_sNormalsIndices, rResult.m_sNormalsInterpolation, rResult.m_sSubdivisionScheme,
            rResult.m_sSubsetIndices, rResult.m_sSubsetMaterialPaths);
        if (meshStatus != SM_SUCCESS)
        {
            rResult.m_sPoints.clear();
            rResult.m_sFaceVertexCounts.clear();
            rResult.m_sFaceVertexIndices.clear();
            rResult.m_sStatus = meshStatus;
            rResult.m_sMessage = "GetMeshData failed on Brep " + sIndex;
            continue;
        }
        rResult.m_sStatus = SM_SUCCESS;
    }
    return SM_SUCCESS;
}

// First free path: crPreferred, then crPreferred_<n>.
static pxr::SdfPath sm_UnusedPath(const pxr::UsdStageRefPtr& crStage, const pxr::SdfPath& crPreferred)
{
    pxr::SdfPath sPath = crPreferred;
    for (int n = 1; crStage->GetPrimAtPath(sPath); ++n)
        sPath = crPreferred.ReplaceName(pxr::TfToken(crPreferred.GetName() + "_" + std::to_string(n)));
    return sPath;
}

// Author one Brep's mesh at crMeshPath on the stage's edit target. crTransform moves its Brep-local points
// to the Brep; crMaterial, if valid, is bound to the mesh; crVisibility and crPurpose are the Brep's
// computed values, which the mesh cannot inherit from outside the Brep's hierarchy. On failure the mesh
// prim is removed.
static void sm_AuthorMesh(const pxr::UsdStageRefPtr& crStage, const pxr::SdfPath& crMeshPath,
                          const pxr::GfMatrix4d& crTransform, const pxr::UsdShadeMaterial& crMaterial,
                          const pxr::TfToken& crVisibility, const pxr::TfToken& crPurpose,
                          const SmApiUsdTessellatedBrep& crBrep, SmApiUsdMeshTessellationResult& rResult)
{
    rResult.m_sStatus = SM_ERR;
    if (crBrep.m_sPoints.empty() || crBrep.m_sFaceVertexCounts.empty())
    {
        rResult.m_sMessage = "Tessellation produced an empty mesh";
        return;
    }
    auto fail = [&](const char* pMessage) {
        crStage->RemovePrim(crMeshPath);
        rResult.m_sMessage = pMessage;
    };
    pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(crStage, crMeshPath);
    pxr::SdfPrimSpecHandle spec = crStage->GetEditTarget().GetLayer()->GetPrimAtPath(crMeshPath);
    if (!mesh || !spec
        || SMU_BrepConvert::PopulateMeshAttr(crBrep.m_sExtent, crBrep.m_sPoints, crBrep.m_sFaceVertexCounts,
                                             crBrep.m_sFaceVertexIndices, crBrep.m_sNormals, crBrep.m_sNormalsIndices,
                                             crBrep.m_sNormalsInterpolation, crBrep.m_sSubdivisionScheme,
                                             crBrep.m_sSubsetIndices, crBrep.m_sSubsetMaterialPaths, spec) != SM_SUCCESS)
        return fail("Failed to author the USD mesh");
    if (!pxr::GfIsClose(crTransform, pxr::GfMatrix4d(1.0), 1e-12))
    {
        pxr::UsdGeomXformOp transformOp = mesh.MakeMatrixXform();
        if (!transformOp || !transformOp.Set(crTransform))
            return fail("Failed to author the mesh transform");
    }
    if (crMaterial && !pxr::UsdShadeMaterialBindingAPI::Apply(mesh.GetPrim()).Bind(crMaterial))
        return fail("Failed to bind the mesh material");
    if (crVisibility == pxr::UsdGeomTokens->invisible && !mesh.CreateVisibilityAttr().Set(crVisibility))
        return fail("Failed to author the mesh visibility");
    if (crPurpose != pxr::UsdGeomTokens->default_ && !mesh.CreatePurposeAttr().Set(crPurpose))
        return fail("Failed to author the mesh purpose");
    rResult.m_sStatus = SM_SUCCESS;
    rResult.m_sMeshPath = crMeshPath.GetString();
    rResult.m_lPointCount = crBrep.m_sPoints.size();
    if (rResult.m_lFailedFaceCount > 0)
        rResult.m_sMessage = std::to_string(rResult.m_lFailedFaceCount) + " face(s) failed; partial mesh kept";
}

// Copy crFrom's prim specs under crParent that crTo lacks, descending through those it already has.
static bool sm_CopyNewPrims(const pxr::SdfLayerHandle& crFrom, const pxr::SdfLayerHandle& crTo,
                            const pxr::SdfPrimSpecHandle& crParent)
{
    for (const pxr::SdfPrimSpecHandle& child : crParent->GetNameChildren())
    {
        const bool bCopied = crTo->GetPrimAtPath(child->GetPath())
            ? sm_CopyNewPrims(crFrom, crTo, child)
            : pxr::SdfCopySpec(crFrom, child->GetPath(), crTo, child->GetPath());
        if (!bCopied)
            return false;
    }
    return true;
}

// Export crRoot plus the prims authored on crSession to crOutputFile, with relative asset paths
// re-anchored for the output directory. Both are copied, so neither layer nor the stage changes.
static bool sm_ExportWithMeshes(const pxr::SdfLayerHandle& crRoot, const pxr::SdfLayerHandle& crSession,
                                const std::string& crOutputFile)
{
    namespace fs = std::filesystem;
    pxr::SdfLayerRefPtr sCopy = pxr::SdfLayer::CreateAnonymous();
    sCopy->TransferContent(crRoot);
    if (!sm_CopyNewPrims(crSession, sCopy, crSession->GetPseudoRoot()))
        return false;
    const fs::path sOutputDir = fs::absolute(fs::path(crOutputFile)).parent_path().lexically_normal();
    if (sOutputDir != fs::path(crRoot->GetRealPath()).parent_path().lexically_normal())
    {
        pxr::UsdUtilsModifyAssetPaths(sCopy, [&](const std::string& crAssetPath) {
            if (crAssetPath.empty() || crAssetPath.find("://") != std::string::npos || fs::path(crAssetPath).is_absolute())
                return crAssetPath;
            const fs::path sAnchored = pxr::SdfComputeAssetPathRelativeToLayer(crRoot, crAssetPath);
            if (!sAnchored.is_absolute())
                return crAssetPath;
            const fs::path sRelative = sAnchored.lexically_normal().lexically_relative(sOutputDir);
            return sRelative.empty() ? sAnchored.generic_string() : sRelative.generic_string();
        });
    }
    return sCopy->Export(crOutputFile);
}

SmStatus SmApiUsdTessellateFile
(
    const char                                  * pInputFile,
    const char                                  * pOutputFile,
    const SmTessellationParams                  & crParams,
    SmBoolean                                     bHealerIsEnabled,
    int                                           iThreadCount,
    std::vector<SmApiUsdMeshTessellationResult> & rResults
)
{
    rResults.clear();
    if (!pInputFile || !pOutputFile)
        return SM_ERR_NULL_POINTER;
    const SmStatus pluginStatus = SmApiUsdEnsurePluginRegistered();
    if (pluginStatus != SM_SUCCESS)
        return pluginStatus;
    if (!UsdBrepData::AreOmniSolidSchemasKnown())
    {
        ERR_MSG(_T("SmApiUsdTessellateFile: omniSolid schema was registered after a USD stage was opened"));
        return SM_ERR;
    }

    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(pInputFile);
    if (!stage)
        return SM_ERR;
    // Author on the session layer: the root layer may be shared with stages the caller has open.
    stage->SetEditTarget(stage->GetSessionLayer());
    std::vector<pxr::UsdPrim> brepArrays;
    for (const pxr::UsdPrim& prim : stage->Traverse(pxr::UsdTraverseInstanceProxies()))
    {
        if (prim.IsValid() && prim.IsActive()
            && (prim.HasAttribute(pxr::TfToken("brep:regionCount")) || prim.GetTypeName() == "BrepArray"))
            brepArrays.push_back(prim);
    }
    if (brepArrays.empty())
        return SM_ERR_INVALID_INPUT;

    pxr::UsdPrim outputParent = stage->GetDefaultPrim();
    if (!outputParent.IsValid())
        outputParent = stage->DefinePrim(pxr::SdfPath("/Output"));
    // Mesh transform = Brep world transform relative to outputParent.
    pxr::UsdGeomXformCache xformCache;
    double dParentDet = 0.0;
    const pxr::GfMatrix4d sParentFromWorld = xformCache.GetLocalToWorldTransform(outputParent).GetInverse(&dParentDet);

    // Tessellate a batch of prims in parallel (each call reads the stage and uses its own
    // context), then author the batch serially so no stage writes overlap the reads.
    tbb::task_arena arena(iThreadCount > 0 ? iThreadCount : tbb::task_arena::automatic);
    const size_t lBatchSize = 4 * static_cast<size_t>(arena.max_concurrency());
    std::vector<std::vector<SmApiUsdTessellatedBrep>> sBatch;
    std::vector<SmStatus> sStatuses;
    bool bAnyMesh = false;
    for (size_t lStart = 0; lStart < brepArrays.size(); lStart += lBatchSize)
    {
        const size_t lEnd = std::min(lStart + lBatchSize, brepArrays.size());
        sBatch.assign(lEnd - lStart, {});
        sStatuses.assign(lEnd - lStart, SM_ERR);
        arena.execute([&] {
            tbb::parallel_for(lStart, lEnd, [&](size_t ai) {
                sStatuses[ai - lStart] = SmApiUsdTessellateBrepArray(brepArrays[ai], crParams, bHealerIsEnabled, 0.0,
                                                                     sBatch[ai - lStart]);
            });
        });

        for (size_t ai = lStart; ai < lEnd; ++ai)
        {
            const std::string sPrimPath = brepArrays[ai].GetPath().GetString();
            if (sStatuses[ai - lStart] != SM_SUCCESS)
            {
                SmApiUsdMeshTessellationResult& rResult = rResults.emplace_back();
                rResult.m_sPrimPath = sPrimPath;
                rResult.m_sStatus = sStatuses[ai - lStart];
                rResult.m_sMessage = sBatch[ai - lStart].empty() ? "tessellation failed" : sBatch[ai - lStart].front().m_sMessage;
                continue;
            }
            // A Brep's own material comes from its "brep" GeomSubset (m_sMaterialPath); the rest take
            // the BrepArray's. The meshes are not under the BrepArray, so they cannot inherit it.
            const pxr::UsdShadeMaterial arrayMaterial =
                pxr::UsdShadeMaterialBindingAPI(brepArrays[ai]).ComputeBoundMaterial();
            const pxr::UsdGeomImageable imageable(brepArrays[ai]);
            const pxr::TfToken visibility = imageable.ComputeVisibility();
            const pxr::TfToken purpose = imageable.ComputePurpose();
            for (const SmApiUsdTessellatedBrep& brep : sBatch[ai - lStart])
            {
                SmApiUsdMeshTessellationResult& rResult = rResults.emplace_back();
                rResult.m_sPrimPath = sPrimPath;
                rResult.m_iPackedBrepIndex = brep.m_iPackedBrepIndex;
                rResult.m_lFailedFaceCount = brep.m_sFailedFaces.size();
                if (brep.m_sStatus != SM_SUCCESS)
                {
                    rResult.m_sStatus = brep.m_sStatus;
                    rResult.m_sMessage = brep.m_sMessage;
                    continue;
                }
                if (dParentDet == 0.0) // a singular parent cannot hold a mesh at its Brep's place
                {
                    rResult.m_sStatus = SM_ERR;
                    rResult.m_sMessage = "Cannot place the mesh: " + outputParent.GetPath().GetString() + " has a singular transform";
                    continue;
                }
                const pxr::SdfPath meshPath = sm_UnusedPath(stage, outputParent.GetPath().AppendChild(pxr::TfToken(
                    "tess_" + std::to_string(ai) + "_" + std::to_string(brep.m_iPackedBrepIndex))));
                const pxr::UsdShadeMaterial material = brep.m_sMaterialPath.IsEmpty()
                    ? arrayMaterial : pxr::UsdShadeMaterial(stage->GetPrimAtPath(brep.m_sMaterialPath));
                sm_AuthorMesh(stage, meshPath, xformCache.GetLocalToWorldTransform(brepArrays[ai]) * sParentFromWorld,
                              material, visibility, purpose, brep, rResult);
                bAnyMesh |= rResult.m_sStatus == SM_SUCCESS;
            }
        }
    }

    if (!bAnyMesh)
        return SM_ERR;
    if (!sm_ExportWithMeshes(stage->GetRootLayer(), stage->GetSessionLayer(), pOutputFile))
        return SM_ERR;
    return SM_SUCCESS;
}
