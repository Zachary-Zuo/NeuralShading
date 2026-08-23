#include "Scene/SceneDefines.slangh"
import Scene.RaytracingInline;
import Utils.Geometry.GeometryHelpers;

#ifndef NCLS_REFERENCE_FAMILY_MASK
#define NCLS_REFERENCE_FAMILY_MASK 15
#endif

#include "ViewerCommon.slang"
#include "../../../shaders/ncls/contracts/layer_stack_ir.slang"
#include "../../../shaders/ncls/reference/sampling.slang"
#if NCLS_REFERENCE_FAMILY_MASK == 1
#include "../../../shaders/ncls/reference/random_walk_reference.slang"
#elif NCLS_REFERENCE_FAMILY_MASK == 2
#include "../../../shaders/ncls/reference/merl_reference.slang"
#elif NCLS_REFERENCE_FAMILY_MASK == 4
#include "../../../shaders/ncls/reference/openpbr_reference.slang"
#elif NCLS_REFERENCE_FAMILY_MASK == 8
#include "../../../shaders/ncls/reference/materialx_standard_surface_reference.slang"
#else
#include "../../../shaders/ncls/reference/random_walk_reference.slang"
#include "../../../shaders/ncls/reference/merl_reference.slang"
#include "../../../shaders/ncls/reference/openpbr_reference.slang"
#include "../../../shaders/ncls/reference/materialx_standard_surface_reference.slang"
#endif

#ifndef NCLS_MAX_SCENE_MATERIALS
#define NCLS_MAX_SCENE_MATERIALS 64
#endif

// Source-material resources remain family-specific. The array index is the
// Falcor material ID, which lets a single scene path cross arbitrary material
// slots without converting any source family to Falcor's native material ABI.
StructuredBuffer<uint4> gMaterialMetadata;
StructuredBuffer<NclsLayerStackIR> gLayerStacks[NCLS_MAX_SCENE_MATERIALS];
StructuredBuffer<float3> gMerlBrdfs[NCLS_MAX_SCENE_MATERIALS];
StructuredBuffer<float> gOpenPbrInputs[NCLS_MAX_SCENE_MATERIALS];
StructuredBuffer<float> gMaterialXInputs[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXBaseColors[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXRoughnesses[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXMetalnesses[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gMaterialXNormalMaps[NCLS_MAX_SCENE_MATERIALS];
Texture2D<float4> gPreviousReference;
Texture2D<float4> gEnvironment;
StructuredBuffer<float> gEnvironmentMarginalCdf;
StructuredBuffer<float> gEnvironmentConditionalCdf;
SamplerState gLinearSampler;
SamplerState gMaterialXSampler;
RWTexture2D<float4> gNextReference;
RWStructuredBuffer<uint> gNoiseStats;

cbuffer ReferencePathTracerCB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gMaterialCount;
    uint gReferenceSpp;
    uint gSamplesThisFrame;
    uint gMaxSceneBounces;
    uint gMaxLayerWalkDepth;
    uint gResetAccumulation;
    uint gAccumulate;
    uint gFlipTexCoordV;
    uint2 gEnvironmentSamplingDimensions;
    uint gUseEnvironment;
    float gEnvironmentRotation;
    float gEnvironmentIntensity;
    uint gUseSun;
    float3 gSunDirection;
    float gSunIntensity;
    float3 gSunColor;
    uint gUsePoint;
    float3 gPointPosition;
    float gPointIntensity;
    float3 gPointColor;
    uint gUseRectangle;
    float3 gRectangleCenter;
    float _padding0;
    float3 gRectangleAxisU;
    float _padding1;
    float3 gRectangleAxisV;
    float gRectangleIntensity;
    float3 gRectangleColor;
    float _padding2;
};

struct NclsReferenceSurface
{
    uint materialId;
    uint family;
    uint openPbrColorSpace;
    float3 position;
    float3 faceNormal;
    float3 normal;
    float3 tangent;
    float2 texCoord;
    #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
    NclsMaterialXResolvedSurface materialX;
    #endif
};

struct NclsReferenceSample
{
    uint valid;
    float3 directionLocal;
    float3 weight;
    float pdf;
};

float nclsReferencePowerHeuristic(float firstPdf, float secondPdf)
{
    const float firstSquared = firstPdf * firstPdf;
    const float secondSquared = secondPdf * secondPdf;
    return firstSquared / max(firstSquared + secondSquared, 1e-20f);
}

float3 nclsOpenPbrAcesCgToLinearSrgbPath(float3 value)
{
    return float3(
        1.70505099f * value.x - 0.62179212f * value.y - 0.08325887f * value.z,
        -0.13025642f * value.x + 1.14080474f * value.y - 0.01054832f * value.z,
        -0.02400336f * value.x - 0.12896898f * value.y + 1.15297233f * value.z);
}

float3 nclsOpenPbrLinearSrgbToAcesCgPath(float3 value)
{
    return float3(
        0.61309740f * value.x + 0.33952315f * value.y + 0.04737945f * value.z,
        0.07019372f * value.x + 0.91635388f * value.y + 0.01345240f * value.z,
        0.02061560f * value.x + 0.10956978f * value.y + 0.86981464f * value.z);
}

float3 nclsReferenceIlluminatePath(
    const NclsReferenceSurface surface,
    float3 response,
    float3 linearSrgbRadiance)
{
    if (surface.family != 2u || surface.openPbrColorSpace == 0u)
        return response * linearSrgbRadiance;
    return nclsOpenPbrAcesCgToLinearSrgbPath(
        response * nclsOpenPbrLinearSrgbToAcesCgPath(linearSrgbRadiance));
}

bool nclsLoadVertexData(
    const HitInfo hit,
    const Ray ray,
    out VertexData vertex,
    out uint materialId)
{
    vertex = {};
    materialId = 0u;
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        // Retain Falcor's per-triangle UV/world-space Jacobian so textured
        // MaterialX inputs can select a stable mip from the path ray cone.
        vertex = gScene.getVertexDataRayCones(
            triangleHit.instanceID,
            triangleHit.primitiveIndex,
            triangleHit.getBarycentricWeights());
        materialId = gScene.getMaterialID(triangleHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedHit = hit.getDisplacedTriangleHit();
        vertex = gScene.getVertexData(displacedHit, -ray.dir);
        materialId = gScene.getMaterialID(displacedHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        vertex = gScene.getVertexDataFromCurve(curveHit);
        materialId = gScene.getMaterialID(curveHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfHit = hit.getSDFGridHit();
        vertex = gScene.getVertexDataFromSDFGrid(sdfHit, ray.origin, ray.dir);
        materialId = gScene.getMaterialID(sdfHit.instanceID);
        return true;
    }
#endif
    return false;
}

NclsReferenceSurface nclsPrepareReferenceSurface(
    const VertexData vertex,
    uint materialId,
    float3 viewWorld,
    float3 rayDirection,
    float rayConeWidth)
{
    NclsReferenceSurface surface;
    surface.materialId = min(materialId, gMaterialCount - 1u);
    const uint4 metadata = gMaterialMetadata[surface.materialId];
    #if NCLS_REFERENCE_FAMILY_MASK == 1
    surface.family = 0u;
    #elif NCLS_REFERENCE_FAMILY_MASK == 2
    surface.family = 1u;
    #elif NCLS_REFERENCE_FAMILY_MASK == 4
    surface.family = 2u;
    #elif NCLS_REFERENCE_FAMILY_MASK == 8
    surface.family = 3u;
    #else
    surface.family = metadata.x;
    #endif
    surface.openPbrColorSpace = metadata.y;
    surface.position = vertex.posW;
    surface.faceNormal = normalize(vertex.faceNormalW);
    surface.normal = normalize(vertex.normalW);
    surface.tangent = vertex.tangentW.xyz;
    surface.texCoord = vertex.texC;
    if (gFlipTexCoordV != 0u) surface.texCoord.y = 1.0f - surface.texCoord.y;

    // All reference closures are prepared in a view-facing local frame. OpenPBR
    // may subsequently sample the opposite hemisphere for transmission.
    if (dot(viewWorld, surface.faceNormal) < 0.0f) surface.faceNormal = -surface.faceNormal;
    if (dot(viewWorld, surface.normal) < 0.0f) surface.normal = -surface.normal;
    surface.tangent -= surface.normal * dot(surface.normal, surface.tangent);
    surface.tangent = dot(surface.tangent, surface.tangent) > 1e-10f
        ? normalize(surface.tangent)
        : nclsViewerTangent(surface.normal);

    #if (NCLS_REFERENCE_FAMILY_MASK & 8) != 0
    surface.materialX.baseColor = float3(0.0f);
    surface.materialX.metalness = 0.0f;
    surface.materialX.specularRoughness = 0.0f;
    if (surface.family == 3u)
    {
        // coneTexLODValue converts the world-space cone footprint to a
        // normalized UV footprint. SampleGrad then adds each texture's own
        // dimensions, so the same gradients remain valid for all connected
        // MaterialX inputs.
        const float normalProjection = max(abs(dot(rayDirection, surface.faceNormal)), 1e-4f);
        const float lambda = vertex.coneTexLODValue
            + log2(max(abs(rayConeWidth) / normalProjection, 1e-8f));
        const float uvFootprint = clamp(exp2(lambda), 1e-8f, 1.0f);
        const float2 uvDx = float2(uvFootprint, 0.0f);
        const float2 uvDy = float2(0.0f, uvFootprint);
        surface.materialX = nclsResolveMaterialXSurface(
            gMaterialXInputs[surface.materialId],
            gMaterialXBaseColors[surface.materialId],
            gMaterialXRoughnesses[surface.materialId],
            gMaterialXMetalnesses[surface.materialId],
            gMaterialXSampler,
            surface.texCoord,
            uvDx,
            uvDy);
        surface.normal = nclsMaterialXShadingNormal(
            gMaterialXInputs[surface.materialId],
            gMaterialXNormalMaps[surface.materialId],
            gMaterialXSampler,
            surface.texCoord,
            uvDx,
            uvDy,
            surface.normal,
            surface.tangent);
        surface.normal = nclsMxForwardFacingNormal(surface.normal, viewWorld);
        surface.tangent -= surface.normal * dot(surface.normal, surface.tangent);
        surface.tangent = dot(surface.tangent, surface.tangent) > 1e-10f
            ? normalize(surface.tangent)
            : nclsViewerTangent(surface.normal);
    }
    #endif
    return surface;
}

float3 nclsEvalReferencePath(
    const NclsReferenceSurface surface,
    float3 viewLocal,
    float3 lightLocal,
    uint2 key)
{
    #if NCLS_REFERENCE_FAMILY_MASK == 1
    if (lightLocal.z <= 0.0f) return float3(0.0f);
    return max(nclsEvalRandomWalkReference(
        gLayerStacks[surface.materialId][0], viewLocal, lightLocal, key, gMaxLayerWalkDepth), 0.0f);
    #elif NCLS_REFERENCE_FAMILY_MASK == 2
    return nclsEvalMerlReference(gMerlBrdfs[surface.materialId], viewLocal, lightLocal);
    #elif NCLS_REFERENCE_FAMILY_MASK == 4
    return nclsEvalOpenPbrReference(gOpenPbrInputs[surface.materialId], viewLocal, lightLocal);
    #elif NCLS_REFERENCE_FAMILY_MASK == 8
    if (lightLocal.z <= 0.0f) return float3(0.0f);
    return nclsEvalMaterialXStandardSurface(
        gMaterialXInputs[surface.materialId], surface.materialX, viewLocal, lightLocal);
    #else
    if (surface.family == 1u)
        return nclsEvalMerlReference(gMerlBrdfs[surface.materialId], viewLocal, lightLocal);
    if (surface.family == 2u)
        return nclsEvalOpenPbrReference(gOpenPbrInputs[surface.materialId], viewLocal, lightLocal);
    if (surface.family == 3u)
    {
        if (lightLocal.z <= 0.0f) return float3(0.0f);
        return nclsEvalMaterialXStandardSurface(
            gMaterialXInputs[surface.materialId], surface.materialX, viewLocal, lightLocal);
    }
    if (lightLocal.z <= 0.0f) return float3(0.0f);
    return max(nclsEvalRandomWalkReference(
        gLayerStacks[surface.materialId][0], viewLocal, lightLocal, key, gMaxLayerWalkDepth), 0.0f);
    #endif
}

float nclsReflectionProposalPdf(
    const NclsReferenceSurface surface,
    float3 viewLocal,
    float3 lightLocal)
{
    if (lightLocal.z <= 0.0f) return 0.0f;
    const float cosinePdf = lightLocal.z / NCLS_VIEWER_PI;
    #if NCLS_REFERENCE_FAMILY_MASK == 1
    const NclsLayerInterfaceIR top = gLayerStacks[surface.materialId][0].interfaces[0];
    const float2 alpha = max(float2(top.alphaX, top.alphaY), float2(0.001f));
    #elif NCLS_REFERENCE_FAMILY_MASK == 8
    const float anisotropy = gMaterialXInputs[surface.materialId][15];
    const float2 alpha = nclsMxRoughnessAnisotropy(
        surface.materialX.specularRoughness, anisotropy);
    #elif NCLS_REFERENCE_FAMILY_MASK == 2 || NCLS_REFERENCE_FAMILY_MASK == 4
    const float2 alpha = float2(0.2f);
    #else
    float2 alpha = float2(0.2f);
    if (surface.family == 0u)
    {
        const NclsLayerInterfaceIR top = gLayerStacks[surface.materialId][0].interfaces[0];
        alpha = max(float2(top.alphaX, top.alphaY), float2(0.001f));
    }
    else if (surface.family == 3u)
    {
        const float anisotropy = gMaterialXInputs[surface.materialId][15];
        alpha = nclsMxRoughnessAnisotropy(surface.materialX.specularRoughness, anisotropy);
    }
    #endif
    const float3 halfVector = nclsSafeNormalize(viewLocal + lightLocal);
    const float specularPdf = nclsGgxVisibleNormalPdf(viewLocal, halfVector, alpha)
        / max(4.0f * abs(dot(viewLocal, halfVector)), 1e-8f);
    return 0.5f * cosinePdf + 0.5f * specularPdf;
}

#if (NCLS_REFERENCE_FAMILY_MASK & 1) != 0
NclsReferenceSample nclsSampleLayerStackPath(
    const NclsReferenceSurface surface,
    float3 viewLocal,
    inout NclsRng rng)
{
    NclsReferenceSample result = {};
    const NclsLayerStackIR stack = gLayerStacks[surface.materialId][0];
    if (stack.interfaceCount == 0u || stack.interfaceCount > NCLS_MAX_INTERFACES) return result;

    // A single interface is a reflection closure in the current LayerStack
    // contract. Multi-interface stacks enter through a dielectric top coat and
    // then perform the same analog interface/medium walk as the directional
    // reference, but return the sampled exterior direction and path weight.
    if (stack.interfaceCount == 1u)
    {
        const NclsLayerInterfaceIR layer = stack.interfaces[0];
        const float etaT = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
            ? layer.relativeIor : 1.0f;
        const LayerBsdfSample scatter = nclsSampleInterfaceTransport(
            layer, viewLocal, 1.0f, etaT, NCLS_SAMPLE_REFLECTION, true, rng);
        if (scatter.valid == 0u) return result;
        result.valid = 1u;
        result.directionLocal = scatter.direction;
        result.weight = scatter.weight;
        result.pdf = scatter.pdf;
        return result;
    }

    if (stack.interfaces[0].kind != (uint)NclsInterfaceKind::RoughDielectric) return result;
    LayerBsdfSample entrance = nclsSampleInterfaceTransport(
        stack.interfaces[0], viewLocal, 1.0f, stack.interfaces[0].relativeIor,
        NCLS_SAMPLE_ALL, true, rng);
    if (entrance.valid == 0u) return result;
    if (entrance.isReflection != 0u)
    {
        result.valid = 1u;
        result.directionLocal = entrance.direction;
        result.weight = entrance.weight;
        result.pdf = 0.0f;
        return result;
    }

    float3 throughput = entrance.weight;
    float3 direction = entrance.direction;
    uint mediumIndex = 0u;
    float z = stack.media[0].thickness;
    [loop]
    for (uint walkDepth = 0u; walkDepth < gMaxLayerWalkDepth; ++walkDepth)
    {
        if (!all(isfinite(throughput)) || !any(throughput > 0.0f)) return result;
        if (walkDepth >= 3u)
        {
            const float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng.nextFloat() >= survival) return result;
            throughput /= survival;
        }
        if (abs(direction.z) <= NCLS_MIN_COS) return result;

        const NclsHomogeneousMediumIR medium = stack.media[mediumIndex];
        const float3 sigmaS = nclsMediumSigmaS(medium);
        const float3 sigmaT = nclsMediumSigmaA(medium) + sigmaS;
        const bool hasScattering = any(sigmaS > 0.0f);
        if (hasScattering && !nclsHasAchromaticExtinction(sigmaT)) return result;
        const float scalarSigmaT = (sigmaT.x + sigmaT.y + sigmaT.z) / 3.0f;
        const bool travelingToTop = direction.z > 0.0f;
        const float boundaryZ = travelingToTop ? medium.thickness : 0.0f;
        const float distanceToBoundary = abs(boundaryZ - z) / abs(direction.z);
        if (!hasScattering)
        {
            throughput *= exp(-sigmaT * distanceToBoundary);
            z = boundaryZ;
        }
        else if (scalarSigmaT > 0.0f)
        {
            const float freeFlight = -log(max(1.0f - rng.nextFloat(), 1e-7f)) / scalarSigmaT;
            if (freeFlight < distanceToBoundary)
            {
                z += direction.z * freeFlight;
                throughput *= sigmaS / scalarSigmaT;
                direction = nclsSampleHenyeyGreenstein(direction, medium.g, rng);
                continue;
            }
            z = boundaryZ;
        }
        else z = boundaryZ;

        const uint interfaceIndex = travelingToTop ? mediumIndex : mediumIndex + 1u;
        const NclsLayerInterfaceIR layer = stack.interfaces[interfaceIndex];
        const float etaAbove = nclsEtaAboveInterface(stack, interfaceIndex);
        const float etaBelow = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
            ? etaAbove * layer.relativeIor : etaAbove;
        const float etaI = travelingToTop ? etaBelow : etaAbove;
        const float etaT = travelingToTop ? etaAbove : etaBelow;
        const float3 incident = -direction;

        if (interfaceIndex + 1u == stack.interfaceCount)
        {
            const LayerBsdfSample reflection = nclsSampleInterface(
                layer, incident, etaI, etaT, NCLS_SAMPLE_REFLECTION, rng);
            if (reflection.valid == 0u) return result;
            throughput *= reflection.weight;
            direction = reflection.direction;
            continue;
        }

        const LayerBsdfSample scatter = nclsSampleInterfaceTransport(
            layer, incident, etaI, etaT, NCLS_SAMPLE_ALL, true, rng);
        if (scatter.valid == 0u) return result;
        throughput *= scatter.weight;
        direction = scatter.direction;
        if (interfaceIndex == 0u && scatter.isReflection == 0u)
        {
            result.valid = all(isfinite(throughput)) && any(throughput > 0.0f) ? 1u : 0u;
            result.directionLocal = direction;
            result.weight = throughput;
            // The marginal PDF of a layered random walk is intentionally not
            // approximated. LayerStack therefore uses BSDF-only environment
            // transport instead of an inconsistent MIS weight.
            result.pdf = 0.0f;
            return result;
        }
        if (scatter.isReflection == 0u)
        {
            if (travelingToTop)
            {
                mediumIndex--;
                z = 0.0f;
            }
            else
            {
                mediumIndex++;
                z = stack.media[mediumIndex].thickness;
            }
        }
    }
    return result;
}
#endif

#if (NCLS_REFERENCE_FAMILY_MASK & 4) != 0
NclsReferenceSample nclsSampleOpenPbrPath(
    const NclsReferenceSurface surface,
    float3 viewLocal,
    inout NclsRng rng)
{
    NclsReferenceSample result = {};
    const OpenPBR_PreparedBsdf prepared = nclsPrepareOpenPbrReference(
        gOpenPbrInputs[surface.materialId], viewLocal);
    OpenPBR_DiffuseSpecular weight;
    OpenPBR_BsdfLobeType sampledType;
    const float3 randomValue = float3(rng.nextFloat(), rng.nextFloat(), rng.nextFloat());
    openpbr_sample(prepared, randomValue, result.directionLocal, weight, result.pdf, sampledType);
    result.weight = openpbr_get_sum_of_diffuse_specular(weight);
    if (surface.openPbrColorSpace != 0u)
        result.weight = nclsOpenPbrAcesCgToLinearSrgbPath(result.weight);
    result.valid = result.pdf > 0.0f && isfinite(result.pdf)
        && all(isfinite(result.weight)) && any(result.weight > 0.0f)
        ? 1u : 0u;
    return result;
}
#endif

NclsReferenceSample nclsSampleReferencePath(
    const NclsReferenceSurface surface,
    float3 viewLocal,
    inout NclsRng rng,
    uint2 key)
{
    NclsReferenceSample result = {};
    #if NCLS_REFERENCE_FAMILY_MASK == 1
    return nclsSampleLayerStackPath(surface, viewLocal, rng);
    #elif NCLS_REFERENCE_FAMILY_MASK == 4
    return nclsSampleOpenPbrPath(surface, viewLocal, rng);
    #else
    #if NCLS_REFERENCE_FAMILY_MASK != 2 && NCLS_REFERENCE_FAMILY_MASK != 8
    if (surface.family == 0u) return nclsSampleLayerStackPath(surface, viewLocal, rng);
    if (surface.family == 2u) return nclsSampleOpenPbrPath(surface, viewLocal, rng);
    #endif

    float3 direction;
    if (rng.nextFloat() < 0.5f)
    {
        direction = nclsSampleCosineHemisphere(rng.nextFloat2());
    }
    else
    {
        #if NCLS_REFERENCE_FAMILY_MASK == 8
        const float anisotropy = gMaterialXInputs[surface.materialId][15];
        float2 alpha = nclsMxRoughnessAnisotropy(
            surface.materialX.specularRoughness, anisotropy);
        #elif NCLS_REFERENCE_FAMILY_MASK == 2
        float2 alpha = float2(0.2f);
        #else
        float2 alpha = float2(0.2f);
        if (surface.family == 0u)
        {
            const NclsLayerInterfaceIR top = gLayerStacks[surface.materialId][0].interfaces[0];
            alpha = max(float2(top.alphaX, top.alphaY), float2(0.001f));
        }
        else if (surface.family == 3u)
        {
            const float anisotropy = gMaterialXInputs[surface.materialId][15];
            alpha = nclsMxRoughnessAnisotropy(surface.materialX.specularRoughness, anisotropy);
        }
        #endif
        const float3 halfVector = nclsSampleGgxVisibleNormal(viewLocal, alpha, rng);
        direction = reflect(-viewLocal, halfVector);
        if (direction.z <= 0.0f) direction = nclsSampleCosineHemisphere(rng.nextFloat2());
    }
    result.directionLocal = direction;
    result.pdf = nclsReflectionProposalPdf(surface, viewLocal, direction);
    const float3 response = nclsEvalReferencePath(surface, viewLocal, direction, key);
    result.weight = response / max(result.pdf, 1e-8f);
    result.valid = result.pdf > 0.0f && isfinite(result.pdf)
        && all(isfinite(result.weight)) && any(result.weight > 0.0f)
        ? 1u : 0u;
    return result;
    #endif
}

float nclsReferencePdfPath(
    const NclsReferenceSurface surface,
    float3 viewLocal,
    float3 lightLocal)
{
    #if NCLS_REFERENCE_FAMILY_MASK == 1
    const NclsLayerStackIR stack = gLayerStacks[surface.materialId][0];
    if (stack.interfaceCount != 1u) return 0.0f;
    const NclsLayerInterfaceIR layer = stack.interfaces[0];
    const float etaT = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
        ? layer.relativeIor : 1.0f;
    return nclsInterfacePdf(layer, viewLocal, lightLocal, 1.0f, etaT, NCLS_SAMPLE_REFLECTION);
    #elif NCLS_REFERENCE_FAMILY_MASK == 4
    const OpenPBR_PreparedBsdf prepared = nclsPrepareOpenPbrReference(
        gOpenPbrInputs[surface.materialId], viewLocal);
    return openpbr_pdf(prepared, lightLocal);
    #elif NCLS_REFERENCE_FAMILY_MASK == 2 || NCLS_REFERENCE_FAMILY_MASK == 8
    return nclsReflectionProposalPdf(surface, viewLocal, lightLocal);
    #else
    if (surface.family == 2u)
    {
        const OpenPBR_PreparedBsdf prepared = nclsPrepareOpenPbrReference(
            gOpenPbrInputs[surface.materialId], viewLocal);
        return openpbr_pdf(prepared, lightLocal);
    }
    return nclsReflectionProposalPdf(surface, viewLocal, lightLocal);
    #endif
}

float3 nclsReferenceEmissionPath(const NclsReferenceSurface surface, float3 viewLocal)
{
    #if NCLS_REFERENCE_FAMILY_MASK == 4
    const float3 emission = nclsOpenPbrEmission(gOpenPbrInputs[surface.materialId], viewLocal);
    return surface.openPbrColorSpace == 0u
        ? emission : nclsOpenPbrAcesCgToLinearSrgbPath(emission);
    #elif NCLS_REFERENCE_FAMILY_MASK == 8
    return nclsMaterialXEmission(gMaterialXInputs[surface.materialId]);
    #elif (NCLS_REFERENCE_FAMILY_MASK & 12) == 0
    return float3(0.0f);
    #else
    if (surface.family == 2u)
    {
        const float3 emission = nclsOpenPbrEmission(gOpenPbrInputs[surface.materialId], viewLocal);
        return surface.openPbrColorSpace == 0u
            ? emission : nclsOpenPbrAcesCgToLinearSrgbPath(emission);
    }
    if (surface.family == 3u)
        return nclsMaterialXEmission(gMaterialXInputs[surface.materialId]);
    return float3(0.0f);
    #endif
}

float3 nclsEnvironmentPath(float3 direction)
{
    return gEnvironment.SampleLevel(
        gLinearSampler, nclsViewerLatLongUv(direction, gEnvironmentRotation), 0.0f).rgb
        * gEnvironmentIntensity;
}

uint nclsSearchMarginalCdf(float value)
{
    uint lower = 0u;
    uint upper = gEnvironmentSamplingDimensions.y;
    [loop]
    while (lower + 1u < upper)
    {
        const uint middle = (lower + upper) >> 1u;
        if (gEnvironmentMarginalCdf[middle] <= value) lower = middle;
        else upper = middle;
    }
    return min(lower, gEnvironmentSamplingDimensions.y - 1u);
}

uint nclsSearchConditionalCdf(uint row, float value)
{
    const uint width = gEnvironmentSamplingDimensions.x;
    const uint offset = row * (width + 1u);
    uint lower = 0u;
    uint upper = width;
    [loop]
    while (lower + 1u < upper)
    {
        const uint middle = (lower + upper) >> 1u;
        if (gEnvironmentConditionalCdf[offset + middle] <= value) lower = middle;
        else upper = middle;
    }
    return min(lower, width - 1u);
}

float nclsEnvironmentPdfPath(float3 direction)
{
    if (any(gEnvironmentSamplingDimensions == uint2(0u)))
        return 1.0f / (4.0f * NCLS_VIEWER_PI);
    const float2 uv = nclsViewerLatLongUv(direction, gEnvironmentRotation);
    const uint width = gEnvironmentSamplingDimensions.x;
    const uint height = gEnvironmentSamplingDimensions.y;
    const uint column = min(uint(uv.x * float(width)), width - 1u);
    const uint row = min(uint(uv.y * float(height)), height - 1u);
    const float rowProbability = max(
        gEnvironmentMarginalCdf[row + 1u] - gEnvironmentMarginalCdf[row], 0.0f);
    const uint offset = row * (width + 1u);
    const float columnProbability = max(
        gEnvironmentConditionalCdf[offset + column + 1u]
        - gEnvironmentConditionalCdf[offset + column], 0.0f);
    const float sineTheta = max(sin(NCLS_VIEWER_PI * uv.y), 1e-7f);
    return rowProbability * columnProbability * float(width * height)
        / (2.0f * NCLS_VIEWER_PI * NCLS_VIEWER_PI * sineTheta);
}

void nclsSampleEnvironmentPath(inout NclsRng rng, out float3 direction, out float pdf)
{
    if (any(gEnvironmentSamplingDimensions == uint2(0u)))
    {
        const float2 u = rng.nextFloat2();
        const float y = 2.0f * u.x - 1.0f;
        const float radius = sqrt(max(1.0f - y * y, 0.0f));
        const float longitude = 2.0f * NCLS_VIEWER_PI * u.y;
        direction = float3(radius * cos(longitude), y, radius * sin(longitude));
        pdf = 1.0f / (4.0f * NCLS_VIEWER_PI);
        return;
    }
    const uint row = nclsSearchMarginalCdf(rng.nextFloat());
    const uint column = nclsSearchConditionalCdf(row, rng.nextFloat());
    const float2 uv = (float2(column, row) + rng.nextFloat2())
        / float2(gEnvironmentSamplingDimensions);
    const float theta = NCLS_VIEWER_PI * uv.y;
    const float longitude = 2.0f * NCLS_VIEWER_PI * (uv.x - 0.5f) - gEnvironmentRotation;
    const float sineTheta = sin(theta);
    direction = float3(sineTheta * cos(longitude), cos(theta), sineTheta * sin(longitude));
    pdf = nclsEnvironmentPdfPath(direction);
}

bool nclsVisible(SceneRayQuery<1> query, float3 origin, float3 direction, float maximumDistance)
{
    return query.traceVisibilityRay(Ray(origin, direction, 0.0f, maximumDistance));
}

float3 nclsDirectRayOrigin(const NclsReferenceSurface surface, float3 direction)
{
    return computeRayOrigin(
        surface.position,
        dot(direction, surface.faceNormal) >= 0.0f ? surface.faceNormal : -surface.faceNormal);
}

float3 nclsDirectLighting(
    SceneRayQuery<1> query,
    const NclsReferenceSurface surface,
    float3 viewLocal,
    uint globalSample,
    uint depth,
    inout NclsRng rng)
{
    float3 radiance = float3(0.0f);
    uint queryIndex = 0u;
    if (gUseSun != 0u)
    {
        const float3 lightWorld = normalize(gSunDirection);
        const float3 lightLocal = nclsViewerToLocal(lightWorld, surface.normal, surface.tangent);
        if ((surface.family == 2u || lightLocal.z > 0.0f)
            && nclsVisible(query, nclsDirectRayOrigin(surface, lightWorld), lightWorld, 3.402823466e+38f))
        {
            const float3 response = nclsEvalReferencePath(
                surface, viewLocal, lightLocal,
                uint2(nclsViewerHash(globalSample ^ depth ^ queryIndex++), globalSample));
            radiance += nclsReferenceIlluminatePath(surface, response, gSunColor * gSunIntensity);
        }
    }
    if (gUsePoint != 0u)
    {
        const float3 toLight = gPointPosition - surface.position;
        const float distanceSquared = max(dot(toLight, toLight), 1e-8f);
        const float distance = sqrt(distanceSquared);
        const float3 lightWorld = toLight / distance;
        const float3 lightLocal = nclsViewerToLocal(lightWorld, surface.normal, surface.tangent);
        if ((surface.family == 2u || lightLocal.z > 0.0f)
            && nclsVisible(query, nclsDirectRayOrigin(surface, lightWorld), lightWorld, max(distance - 1e-4f, 0.0f)))
        {
            const float3 response = nclsEvalReferencePath(
                surface, viewLocal, lightLocal,
                uint2(nclsViewerHash(globalSample ^ depth ^ queryIndex++), globalSample));
            radiance += nclsReferenceIlluminatePath(
                surface, response, gPointColor * (gPointIntensity / distanceSquared));
        }
    }
    if (gUseRectangle != 0u)
    {
        const float2 u = rng.nextFloat2();
        const float3 lightPoint = gRectangleCenter
            + (2.0f * u.x - 1.0f) * gRectangleAxisU
            + (2.0f * u.y - 1.0f) * gRectangleAxisV;
        const float3 toLight = lightPoint - surface.position;
        const float distanceSquared = max(dot(toLight, toLight), 1e-8f);
        const float distance = sqrt(distanceSquared);
        const float3 lightWorld = toLight / distance;
        const float3 lightNormal = -normalize(cross(gRectangleAxisU, gRectangleAxisV));
        const float lightCosine = max(dot(lightNormal, -lightWorld), 0.0f);
        const float area = 4.0f * length(cross(gRectangleAxisU, gRectangleAxisV));
        const float3 lightLocal = nclsViewerToLocal(lightWorld, surface.normal, surface.tangent);
        if (lightCosine > 0.0f && area > 0.0f
            && (surface.family == 2u || lightLocal.z > 0.0f)
            && nclsVisible(query, nclsDirectRayOrigin(surface, lightWorld), lightWorld, max(distance - 1e-4f, 0.0f)))
        {
            const float3 response = nclsEvalReferencePath(
                surface, viewLocal, lightLocal,
                uint2(nclsViewerHash(globalSample ^ depth ^ queryIndex++), globalSample));
            const float inversePdf = area * lightCosine / distanceSquared;
            radiance += nclsReferenceIlluminatePath(
                surface, response, gRectangleColor * gRectangleIntensity) * inversePdf;
        }
    }
    const bool supportsEnvironmentNee = surface.family != 0u
        || gLayerStacks[surface.materialId][0].interfaceCount == 1u;
    if (gUseEnvironment != 0u && supportsEnvironmentNee)
    {
        float3 lightWorld;
        float lightPdf;
        nclsSampleEnvironmentPath(rng, lightWorld, lightPdf);
        const float3 lightLocal = nclsViewerToLocal(lightWorld, surface.normal, surface.tangent);
        if ((surface.family == 2u || lightLocal.z > 0.0f)
            && nclsVisible(query, nclsDirectRayOrigin(surface, lightWorld), lightWorld, 3.402823466e+38f))
        {
            const float3 response = nclsEvalReferencePath(
                surface, viewLocal, lightLocal,
                uint2(nclsViewerHash(globalSample ^ depth ^ queryIndex++), globalSample));
            const float bsdfPdf = nclsReferencePdfPath(surface, viewLocal, lightLocal);
            const float misWeight = nclsReferencePowerHeuristic(lightPdf, bsdfPdf);
            radiance += nclsReferenceIlluminatePath(surface, response, nclsEnvironmentPath(lightWorld))
                * (misWeight / max(lightPdf, 1e-8f));
        }
    }
    return radiance;
}

float2 nclsR2Sample(uint index, uint scramble)
{
    const float2 shift = float2(
        nclsViewerHashFloat(scramble),
        nclsViewerHashFloat(scramble ^ 0x68bc21ebu));
    return frac(shift + (float(index) + 0.5f) * float2(0.7548776662466927f, 0.5698402909980532f));
}

Ray nclsPrimaryRay(uint2 pixel, uint globalSample)
{
    const uint linearPixel = pixel.y * gFrameDim.x + pixel.x;
    const float2 subpixel = nclsR2Sample(globalSample, nclsViewerHash(linearPixel));
    const float2 screen = (float2(pixel) + subpixel) / float2(gFrameDim);
    const float2 ndc = float2(2.0f, -2.0f) * screen + float2(-1.0f, 1.0f);
    const float3 direction = normalize(
        ndc.x * gScene.camera.data.cameraU
        + ndc.y * gScene.camera.data.cameraV
        + gScene.camera.data.cameraW);
    const float inverseCosine = 1.0f / dot(normalize(gScene.camera.data.cameraW), direction);
    return Ray(
        gScene.camera.getPosition(), direction,
        gScene.camera.data.nearZ * inverseCosine,
        gScene.camera.data.farZ * inverseCosine);
}

float3 nclsTraceReferencePath(uint2 pixel, uint globalSample)
{
    const uint linearPixel = pixel.y * gFrameDim.x + pixel.x;
    const uint seed = nclsViewerHash(linearPixel ^ nclsViewerHash(globalSample + 0x9e3779b9u));
    NclsRng rng = NclsRng(uint2(seed, globalSample));
    SceneRayQuery<1> query;
    Ray ray = nclsPrimaryRay(pixel, globalSample);
    float3 throughput = float3(1.0f);
    float3 radiance = float3(0.0f);
    float previousBsdfPdf = 0.0f;
    float previousEnvironmentPdf = 0.0f;
    float rayConeWidth = 0.0f;
    // cameraV spans half of the vertical image plane in Falcor's camera
    // convention. This is the angular footprint of one primary pixel.
    const float rayConeSpreadAngle =
        2.0f * length(gScene.camera.data.cameraV) / float(max(gFrameDim.y, 1u));

    [loop]
    for (uint depth = 0u; depth <= gMaxSceneBounces; ++depth)
    {
        float hitDistance;
        const HitInfo hit = query.traceRay(ray, hitDistance);
        if (!hit.isValid())
        {
            if (gUseEnvironment != 0u)
            {
                float misWeight = 1.0f;
                if (depth > 0u && previousEnvironmentPdf > 0.0f)
                    misWeight = nclsReferencePowerHeuristic(previousBsdfPdf, previousEnvironmentPdf);
                radiance += throughput * nclsEnvironmentPath(ray.dir) * misWeight;
            }
            else if (depth == 0u)
            {
                radiance += throughput * float3(0.025f, 0.03f, 0.04f);
            }
            break;
        }

        VertexData vertex;
        uint materialId;
        if (!nclsLoadVertexData(hit, ray, vertex, materialId) || materialId >= gMaterialCount) break;
        rayConeWidth += rayConeSpreadAngle * hitDistance;
        const float3 viewWorld = -ray.dir;
        const NclsReferenceSurface surface = nclsPrepareReferenceSurface(
            vertex, materialId, viewWorld, ray.dir, rayConeWidth);
        const float3 viewLocal = nclsViewerToLocal(viewWorld, surface.normal, surface.tangent);
        radiance += throughput * nclsReferenceEmissionPath(surface, viewLocal);
        radiance += throughput * nclsDirectLighting(
            query, surface, viewLocal, globalSample, depth, rng);
        if (depth >= gMaxSceneBounces) break;

        NclsReferenceSample scatter = nclsSampleReferencePath(
            surface, viewLocal, rng,
            uint2(nclsViewerHash(seed ^ depth ^ 0x51ed270bu), globalSample));
        if (scatter.valid == 0u) break;
        throughput *= scatter.weight;
        if (!all(isfinite(throughput)) || !any(throughput > 0.0f)) break;

        if (depth >= 3u)
        {
            const float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05f, 0.95f);
            if (rng.nextFloat() >= survival) break;
            throughput /= survival;
        }

        const float3 directionWorld = normalize(nclsViewerToWorld(
            scatter.directionLocal, surface.normal, surface.tangent));
        const bool transmission = scatter.directionLocal.z < 0.0f;
        const float3 nextOrigin = computeRayOrigin(
            surface.position, transmission ? -surface.faceNormal : surface.faceNormal);
        ray = Ray(nextOrigin, directionWorld);
        previousBsdfPdf = scatter.pdf;
        previousEnvironmentPdf = nclsEnvironmentPdfPath(directionWorld);
    }
    return max(radiance, 0.0f);
}

groupshared float gRelativeStandardError[64];

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    const uint2 pixel = dispatchThreadID.xy;
    const bool validPixel = all(pixel < gFrameDim);
    float3 batchSum = float3(0.0f);
    float batchLuminanceSquare = 0.0f;
    if (validPixel)
    {
        [loop]
        for (uint sampleIndex = 0u; sampleIndex < gSamplesThisFrame; ++sampleIndex)
        {
            const uint globalSample = gReferenceSpp + sampleIndex;
            const float3 sampleRadiance = nclsTraceReferencePath(pixel, globalSample);
            batchSum += sampleRadiance;
            const float luminance = nclsViewerLuminance(sampleRadiance);
            batchLuminanceSquare += luminance * luminance;
        }
    }

    float4 accumulated = float4(0.0f);
    if (validPixel)
    {
        const float inverseBatch = 1.0f / max(1u, gSamplesThisFrame);
        if (gAccumulate == 0u || gResetAccumulation != 0u || gReferenceSpp == 0u)
        {
            accumulated = float4(batchSum * inverseBatch, batchLuminanceSquare * inverseBatch);
        }
        else
        {
            const float4 previous = gPreviousReference[pixel];
            const float oldCount = float(gReferenceSpp);
            const float newCount = oldCount + float(gSamplesThisFrame);
            accumulated = float4(
                (previous.rgb * oldCount + batchSum) / newCount,
                (previous.a * oldCount + batchLuminanceSquare) / newCount);
        }
        gNextReference[pixel] = accumulated;
    }

    const float sampleCount = float(
        (gAccumulate == 0u || gResetAccumulation != 0u || gReferenceSpp == 0u)
        ? gSamplesThisFrame : gReferenceSpp + gSamplesThisFrame);
    const float meanLuminance = nclsViewerLuminance(accumulated.rgb);
    const float variance = max(accumulated.a - meanLuminance * meanLuminance, 0.0f);
    const float standardError = sqrt(variance / max(sampleCount, 1.0f));
    gRelativeStandardError[groupIndex] = validPixel
        ? min(standardError / max(abs(meanLuminance), 0.01f), 16.0f)
        : 0.0f;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u)
    {
        if (groupIndex < stride) gRelativeStandardError[groupIndex] += gRelativeStandardError[groupIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0u)
    {
        const uint groupPixels = min(8u, gFrameDim.x - dispatchThreadID.x)
            * min(8u, gFrameDim.y - dispatchThreadID.y);
        const float groupMean = gRelativeStandardError[0] / max(groupPixels, 1u);
        InterlockedAdd(gNoiseStats[0], uint(groupMean * 4096.0f));
        InterlockedAdd(gNoiseStats[1], 1u);
    }
}
