#ifndef NCLS_RANDOM_WALK_REFERENCE_SLANG
#define NCLS_RANDOM_WALK_REFERENCE_SLANG

#include "interfaces.slang"

float nclsMaxComponent(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

float3 nclsMediumSigmaA(const NclsHomogeneousMediumIR medium)
{
    return float3(medium.sigmaAR, medium.sigmaAG, medium.sigmaAB);
}

float3 nclsMediumSigmaS(const NclsHomogeneousMediumIR medium)
{
    return float3(medium.sigmaSR, medium.sigmaSG, medium.sigmaSB);
}

bool nclsHasAchromaticExtinction(float3 sigmaT)
{
    const float maximum = nclsMaxComponent(sigmaT);
    const float minimum = min(sigmaT.x, min(sigmaT.y, sigmaT.z));
    return maximum - minimum <= max(1e-5f, maximum * 1e-4f);
}

float nclsPowerHeuristic(float firstPdf, float secondPdf)
{
    const float firstSquared = firstPdf * firstPdf;
    const float secondSquared = secondPdf * secondPdf;
    return firstSquared / max(firstSquared + secondSquared, 1e-20f);
}

float3 nclsMediumTransmittance(const NclsHomogeneousMediumIR medium, float verticalDistance, float3 direction)
{
    if (abs(direction.z) <= NCLS_MIN_COS) return float3(0.0f);
    const float3 sigmaT = nclsMediumSigmaA(medium) + nclsMediumSigmaS(medium);
    return exp(-sigmaT * abs(verticalDistance / direction.z));
}

float3 nclsEvalTwoLayerWalk(
    const NclsLayerStackIR stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey,
    const uint maxDepth)
{
    if (stack.interfaceCount != 2u || stack.mediumCount != 1u) return float3(0.0f);
    if (viewDirection.z <= NCLS_MIN_COS || lightDirection.z <= NCLS_MIN_COS) return float3(0.0f);

    const NclsLayerInterfaceIR coat = stack.interfaces[0];
    const NclsLayerInterfaceIR base = stack.interfaces[1];
    const NclsHomogeneousMediumIR medium = stack.media[0];
    if (coat.kind != (uint)NclsInterfaceKind::RoughDielectric) return float3(0.0f);
    if (base.kind != (uint)NclsInterfaceKind::Diffuse && base.kind != (uint)NclsInterfaceKind::RoughConductor
        && base.kind != (uint)NclsInterfaceKind::Sheen)
        return float3(0.0f);

    const float outsideEta = 1.0f;
    const float slabEta = coat.relativeIor;
    const float3 sigmaA = nclsMediumSigmaA(medium);
    const float3 sigmaS = nclsMediumSigmaS(medium);
    const float3 sigmaT = sigmaA + sigmaS;
    const bool hasScattering = any(sigmaS > 0.0f);
    if (hasScattering && !nclsHasAchromaticExtinction(sigmaT))
    {
        // v0 supports RGB albedo but uses one shared extinction/free-flight
        // distribution. A negative sentinel is surfaced as a driver error.
        return float3(-1.0f);
    }
    const float scalarSigmaT = (sigmaT.x + sigmaT.y + sigmaT.z) / 3.0f;
    NclsRng rng = NclsRng(rngKey);

    // Direct reflection at the coat is analytic. The remaining terms are
    // estimated by walking inside the slab and connecting to lightDirection
    // every time the path reaches the coat from below.
    float3 result = nclsEvalInterface(coat, viewDirection, lightDirection, outsideEta, slabEta)
        * lightDirection.z;

    LayerBsdfSample entrance = nclsSampleInterface(
        coat,
        viewDirection,
        outsideEta,
        slabEta,
        NCLS_SAMPLE_TRANSMISSION,
        rng);
    if (entrance.valid == 0u) return result;

    float3 throughput = entrance.weight;
    float3 w = entrance.direction;
    float z = medium.thickness;

    for (uint depth = 0u; depth < maxDepth; ++depth)
    {
        if (depth > 3u)
        {
            const float survival = min(nclsMaxComponent(throughput), 1.0f);
            if (survival < 0.25f)
            {
                if (rng.nextFloat() >= survival) break;
                throughput /= max(survival, 1e-6f);
            }
        }

        if (abs(w.z) <= NCLS_MIN_COS) break;
        const bool travelingToTop = w.z > 0.0f;
        const float boundaryZ = travelingToTop ? medium.thickness : 0.0f;
        const float distanceToBoundary = abs(boundaryZ - z) / abs(w.z);

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
                z += w.z * freeFlight;
                throughput *= sigmaS / scalarSigmaT;
                if (!any(throughput > 0.0f)) break;
                w = nclsSampleHenyeyGreenstein(w, medium.g, rng);
                continue;
            }
            z = boundaryZ;
        }
        else
        {
            z = boundaryZ;
        }

        if (!travelingToTop)
        {
            const float3 incidentAtBase = -w;
            LayerBsdfSample bounce = nclsSampleInterface(
                base,
                incidentAtBase,
                slabEta,
                slabEta,
                NCLS_SAMPLE_REFLECTION,
                rng);
            if (bounce.valid == 0u) break;
            throughput *= bounce.weight;
            w = bounce.direction;
        }
        else
        {
            const float3 incidentAtCoat = -w;
            result += throughput * nclsEvalInterface(
                coat,
                incidentAtCoat,
                lightDirection,
                slabEta,
                outsideEta) * lightDirection.z;

            LayerBsdfSample bounce = nclsSampleInterface(
                coat,
                incidentAtCoat,
                slabEta,
                outsideEta,
                NCLS_SAMPLE_REFLECTION,
                rng);
            if (bounce.valid == 0u) break;
            throughput *= bounce.weight;
            w = bounce.direction;
        }
    }
    return result;
}

float3 nclsEvalTwoLayerPbrtWalk(
    const NclsLayerStackIR stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey,
    const uint maxDepth)
{
    if (stack.interfaceCount != 2u || stack.mediumCount != 1u) return float3(0.0f);
    if (viewDirection.z <= NCLS_MIN_COS || lightDirection.z <= NCLS_MIN_COS) return float3(0.0f);

    const NclsLayerInterfaceIR coat = stack.interfaces[0];
    const NclsLayerInterfaceIR base = stack.interfaces[1];
    const NclsHomogeneousMediumIR medium = stack.media[0];
    if (coat.kind != (uint)NclsInterfaceKind::RoughDielectric) return float3(0.0f);
    if (base.kind != (uint)NclsInterfaceKind::Diffuse && base.kind != (uint)NclsInterfaceKind::RoughConductor
        && base.kind != (uint)NclsInterfaceKind::Sheen)
        return float3(0.0f);

    const float3 sigmaS = nclsMediumSigmaS(medium);
    const float3 sigmaT = nclsMediumSigmaA(medium) + sigmaS;
    const bool hasScattering = any(sigmaS > 0.0f);
    if (hasScattering && !nclsHasAchromaticExtinction(sigmaT)) return float3(-1.0f);
    const float scalarSigmaT = (sigmaT.x + sigmaT.y + sigmaT.z) / 3.0f;
    const float3 mediumAlbedo = scalarSigmaT > 0.0f ? sigmaS / scalarSigmaT : float3(0.0f);

    const float outsideEta = 1.0f;
    const float slabEta = coat.relativeIor;
    NclsRng rng = NclsRng(rngKey);
    float3 result = nclsEvalInterfaceTransport(
        coat, viewDirection, lightDirection, outsideEta, slabEta, true);

    LayerBsdfSample entrance = nclsSampleInterfaceTransport(
        coat,
        viewDirection,
        outsideEta,
        slabEta,
        NCLS_SAMPLE_TRANSMISSION,
        true,
        rng);
    if (entrance.valid == 0u) return result * lightDirection.z;

    LayerBsdfSample lightEntrance = nclsSampleInterfaceTransport(
        coat,
        lightDirection,
        outsideEta,
        slabEta,
        NCLS_SAMPLE_TRANSMISSION,
        false,
        rng);
    if (lightEntrance.valid == 0u) return result * lightDirection.z;

    float3 throughput = entrance.weight;
    float3 w = entrance.direction;
    float z = medium.thickness;

    for (uint depth = 0u; depth < maxDepth; ++depth)
    {
        if (depth > 3u && nclsMaxComponent(throughput) < 0.25f)
        {
            const float survival = min(nclsMaxComponent(throughput), 1.0f);
            if (rng.nextFloat() >= survival) break;
            throughput /= max(survival, 1e-6f);
        }
        if (abs(w.z) <= NCLS_MIN_COS) break;

        const bool travelingToTop = w.z > 0.0f;
        const float boundaryZ = travelingToTop ? medium.thickness : 0.0f;
        const float distanceToBoundary = abs(boundaryZ - z) / abs(w.z);
        if (hasScattering && scalarSigmaT > 0.0f)
        {
            const float freeFlight = -log(max(1.0f - rng.nextFloat(), 1e-7f)) / scalarSigmaT;
            if (freeFlight < distanceToBoundary)
            {
                const float scatteringZ = z + w.z * freeFlight;
                const float3 directionToLightEntrance = -lightEntrance.direction;
                const float phaseConnectionPdf = nclsHenyeyGreensteinPdf(
                    w, directionToLightEntrance, medium.g);
                const float lightMisWeight = nclsPowerHeuristic(
                    lightEntrance.pdf, phaseConnectionPdf);
                result += throughput * mediumAlbedo * phaseConnectionPdf * lightMisWeight
                    * nclsMediumTransmittance(
                        medium, medium.thickness - scatteringZ, lightEntrance.direction)
                    * lightEntrance.weight / max(abs(lightEntrance.direction.z), NCLS_MIN_COS);

                const float3 sampledDirection = nclsSampleHenyeyGreenstein(w, medium.g, rng);
                const float phaseSamplePdf = nclsHenyeyGreensteinPdf(w, sampledDirection, medium.g);
                throughput *= mediumAlbedo;
                w = sampledDirection;
                z = scatteringZ;

                if (w.z > 0.0f)
                {
                    const float3 exitConnection = nclsEvalInterfaceTransport(
                        coat,
                        -w,
                        lightDirection,
                        slabEta,
                        outsideEta,
                        true);
                    if (any(exitConnection > 0.0f))
                    {
                        const float exitPdf = nclsInterfacePdf(
                            coat,
                            -w,
                            lightDirection,
                            slabEta,
                            outsideEta,
                            NCLS_SAMPLE_TRANSMISSION);
                        const float phaseMisWeight = nclsPowerHeuristic(phaseSamplePdf, exitPdf);
                        result += throughput
                            * nclsMediumTransmittance(
                                medium, medium.thickness - scatteringZ, w)
                            * exitConnection * phaseMisWeight;
                    }
                }
                continue;
            }
            z = boundaryZ;
        }
        else
        {
            throughput *= nclsMediumTransmittance(medium, boundaryZ - z, w);
            z = boundaryZ;
        }

        if (travelingToTop)
        {
            const float3 incidentAtCoat = -w;
            LayerBsdfSample reflection = nclsSampleInterfaceTransport(
                coat,
                incidentAtCoat,
                slabEta,
                outsideEta,
                NCLS_SAMPLE_REFLECTION,
                true,
                rng);
            if (reflection.valid == 0u) break;
            throughput *= reflection.weight;
            w = reflection.direction;
        }
        else
        {
            const float3 incidentAtBase = -w;
            const float3 directionToLightEntrance = -lightEntrance.direction;
            const float3 baseConnection = nclsEvalInterface(
                base,
                incidentAtBase,
                directionToLightEntrance,
                slabEta,
                slabEta);
            if (any(baseConnection > 0.0f))
            {
                const float baseConnectionPdf = nclsInterfacePdf(
                    base,
                    incidentAtBase,
                    directionToLightEntrance,
                    slabEta,
                    slabEta,
                    NCLS_SAMPLE_REFLECTION);
                const float misWeight = nclsPowerHeuristic(lightEntrance.pdf, baseConnectionPdf);
                result += throughput * baseConnection * misWeight
                    * nclsMediumTransmittance(medium, medium.thickness, lightEntrance.direction)
                    * lightEntrance.weight;
            }

            LayerBsdfSample reflection = nclsSampleInterface(
                base,
                incidentAtBase,
                slabEta,
                slabEta,
                NCLS_SAMPLE_REFLECTION,
                rng);
            if (reflection.valid == 0u) break;
            throughput *= reflection.weight;
            w = reflection.direction;

            const float3 exitConnection = nclsEvalInterfaceTransport(
                coat,
                -w,
                lightDirection,
                slabEta,
                outsideEta,
                true);
            if (any(exitConnection > 0.0f))
            {
                const float exitPdf = nclsInterfacePdf(
                    coat,
                    -w,
                    lightDirection,
                    slabEta,
                    outsideEta,
                    NCLS_SAMPLE_TRANSMISSION);
                const float misWeight = nclsPowerHeuristic(reflection.pdf, exitPdf);
                result += throughput
                    * nclsMediumTransmittance(medium, medium.thickness, w)
                    * exitConnection * misWeight;
            }
        }
    }
    return result * lightDirection.z;
}

float nclsEtaAboveInterface(const NclsLayerStackIR stack, uint interfaceIndex)
{
    float eta = 1.0f;
    for (uint index = 0u; index < interfaceIndex; ++index)
    {
        if (stack.interfaces[index].kind == (uint)NclsInterfaceKind::RoughDielectric)
            eta *= stack.interfaces[index].relativeIor;
    }
    return eta;
}

struct NclsBallisticPath
{
    uint valid;
    float3 directionAtBase;
    float3 throughput;
};

NclsBallisticPath nclsSampleBallisticToBase(
    const NclsLayerStackIR stack,
    const float3 externalDirection,
    const bool radianceTransport,
    inout NclsRng rng)
{
    NclsBallisticPath path;
    path.valid = 0u;
    path.directionAtBase = float3(0.0f, 0.0f, 1.0f);
    path.throughput = float3(0.0f);
    float3 incident = externalDirection;
    float3 throughput = float3(1.0f);

    for (uint interfaceIndex = 0u; interfaceIndex + 1u < stack.interfaceCount; ++interfaceIndex)
    {
        const NclsLayerInterfaceIR layer = stack.interfaces[interfaceIndex];
        if (layer.kind != (uint)NclsInterfaceKind::RoughDielectric) return path;
        const float etaAbove = nclsEtaAboveInterface(stack, interfaceIndex);
        const float etaBelow = etaAbove * layer.relativeIor;
        LayerBsdfSample transmission = nclsSampleInterfaceTransport(
            layer,
            incident,
            etaAbove,
            etaBelow,
            NCLS_SAMPLE_TRANSMISSION,
            radianceTransport,
            rng);
        if (transmission.valid == 0u || transmission.direction.z >= -NCLS_MIN_COS) return path;
        throughput *= transmission.weight;
        throughput *= nclsMediumTransmittance(
            stack.media[interfaceIndex],
            stack.media[interfaceIndex].thickness,
            transmission.direction);
        if (!any(throughput > 0.0f)) return path;
        incident = -transmission.direction;
    }

    path.valid = 1u;
    path.directionAtBase = incident;
    path.throughput = throughput;
    return path;
}

float3 nclsEvalRandomWalkReferenceDiagnostic(
    const NclsLayerStackIR stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey,
    const uint maxDepth,
    out uint debugCode)
{
    debugCode = 0u;
    if (stack.interfaceCount < 1u || stack.interfaceCount > NCLS_MAX_INTERFACES) return float3(0.0f);
    if (viewDirection.z <= NCLS_MIN_COS || lightDirection.z <= NCLS_MIN_COS) return float3(0.0f);

    const NclsLayerInterfaceIR top = stack.interfaces[0];
    if (stack.interfaceCount == 1u)
    {
        const float etaT = top.kind == (uint)NclsInterfaceKind::RoughDielectric ? top.relativeIor : 1.0f;
        return nclsEvalInterfaceTransport(top, viewDirection, lightDirection, 1.0f, etaT, true)
            * lightDirection.z;
    }

    if (top.kind != (uint)NclsInterfaceKind::RoughDielectric) return float3(0.0f);
    for (uint index = 1u; index + 1u < stack.interfaceCount; ++index)
    {
        if (stack.interfaces[index].kind != (uint)NclsInterfaceKind::RoughDielectric) return float3(0.0f);
    }
    const NclsLayerInterfaceIR base = stack.interfaces[stack.interfaceCount - 1u];
    if (base.kind != (uint)NclsInterfaceKind::Diffuse && base.kind != (uint)NclsInterfaceKind::RoughConductor
        && base.kind != (uint)NclsInterfaceKind::Sheen)
        return float3(0.0f);

    for (uint index = 0u; index < stack.mediumCount; ++index)
    {
        const float3 sigmaS = nclsMediumSigmaS(stack.media[index]);
        const float3 sigmaT = nclsMediumSigmaA(stack.media[index]) + sigmaS;
        if (any(sigmaS > 0.0f) && !nclsHasAchromaticExtinction(sigmaT)) return float3(-1.0f);
    }

    const float etaBelowTop = top.relativeIor;
    float3 result = nclsEvalInterfaceTransport(
        top, viewDirection, lightDirection, 1.0f, etaBelowTop, true) * lightDirection.z;
    if (!all(isfinite(result))) { debugCode = 101u; return result; }

    // Rao-Blackwellize the fully ballistic base-reflection path. Both sides
    // force transmission through every dielectric interface, so this path no
    // longer depends on the forward walk randomly returning through all coats.
    // The forward estimator below omits only this exact path class.
    NclsRng ballisticRng = NclsRng(uint2(
        rngKey.x ^ 0x6a09e667u,
        rngKey.y ^ 0xbb67ae85u));
    const NclsBallisticPath viewBallistic = nclsSampleBallisticToBase(
        stack, viewDirection, true, ballisticRng);
    const NclsBallisticPath lightBallistic = nclsSampleBallisticToBase(
        stack, lightDirection, false, ballisticRng);
    if (viewBallistic.valid != 0u && !all(isfinite(viewBallistic.throughput)))
    { debugCode = 102u; return viewBallistic.throughput; }
    if (lightBallistic.valid != 0u && !all(isfinite(lightBallistic.throughput)))
    { debugCode = 103u; return lightBallistic.throughput; }
    if (viewBallistic.valid != 0u && lightBallistic.valid != 0u)
    {
        const float baseEta = nclsEtaAboveInterface(stack, stack.interfaceCount - 1u);
        const float3 baseConnection = nclsEvalInterface(
            base,
            viewBallistic.directionAtBase,
            lightBallistic.directionAtBase,
            baseEta,
            baseEta);
        if (!all(isfinite(baseConnection))) { debugCode = 104u; return baseConnection; }
        result += viewBallistic.throughput * baseConnection * lightBallistic.throughput
            * lightDirection.z;
        if (!all(isfinite(result))) { debugCode = 105u; return result; }
    }

    NclsRng rng = NclsRng(rngKey);
    LayerBsdfSample entrance = nclsSampleInterfaceTransport(
        top,
        viewDirection,
        1.0f,
        etaBelowTop,
        NCLS_SAMPLE_TRANSMISSION,
        true,
        rng);
    if (entrance.valid == 0u) return result;

    float3 throughput = entrance.weight;
    if (!all(isfinite(throughput))) { debugCode = 106u; return throughput; }
    float3 w = entrance.direction;
    uint mediumIndex = 0u;
    float z = stack.media[0].thickness;
    bool nonBallisticEvent = false;

    for (uint depth = 0u; depth < maxDepth; ++depth)
    {
        if (depth > 3u && nclsMaxComponent(throughput) < 0.25f)
        {
            const float survival = min(nclsMaxComponent(throughput), 1.0f);
            if (rng.nextFloat() >= survival) break;
            throughput /= max(survival, 1e-6f);
            if (!all(isfinite(throughput))) { debugCode = 107u; return throughput; }
        }
        if (abs(w.z) <= NCLS_MIN_COS) break;

        const NclsHomogeneousMediumIR medium = stack.media[mediumIndex];
        const float3 sigmaS = nclsMediumSigmaS(medium);
        const float3 sigmaT = nclsMediumSigmaA(medium) + sigmaS;
        const bool hasScattering = any(sigmaS > 0.0f);
        const float scalarSigmaT = (sigmaT.x + sigmaT.y + sigmaT.z) / 3.0f;
        const bool travelingToTop = w.z > 0.0f;
        const float boundaryZ = travelingToTop ? medium.thickness : 0.0f;
        const float distanceToBoundary = abs(boundaryZ - z) / abs(w.z);

        if (!hasScattering)
        {
            throughput *= exp(-sigmaT * distanceToBoundary);
            if (!all(isfinite(throughput))) { debugCode = 108u; return throughput; }
            z = boundaryZ;
        }
        else if (scalarSigmaT > 0.0f)
        {
            const float freeFlight = -log(max(1.0f - rng.nextFloat(), 1e-7f)) / scalarSigmaT;
            if (freeFlight < distanceToBoundary)
            {
                z += w.z * freeFlight;
                throughput *= sigmaS / scalarSigmaT;
                if (!all(isfinite(throughput))) { debugCode = 109u; return throughput; }
                if (!any(throughput > 0.0f)) break;
                w = nclsSampleHenyeyGreenstein(w, medium.g, rng);
                nonBallisticEvent = true;
                continue;
            }
            z = boundaryZ;
        }
        else
        {
            z = boundaryZ;
        }

        const uint interfaceIndex = travelingToTop ? mediumIndex : mediumIndex + 1u;
        const NclsLayerInterfaceIR layer = stack.interfaces[interfaceIndex];
        const float etaAbove = nclsEtaAboveInterface(stack, interfaceIndex);
        const float etaBelow = layer.kind == (uint)NclsInterfaceKind::RoughDielectric
            ? etaAbove * layer.relativeIor
            : etaAbove;
        const float etaI = travelingToTop ? etaBelow : etaAbove;
        const float etaT = travelingToTop ? etaAbove : etaBelow;
        const float3 incidentAtInterface = -w;

        if (interfaceIndex == 0u)
        {
            if (nonBallisticEvent)
            {
                result += throughput * nclsEvalInterfaceTransport(
                    layer, incidentAtInterface, lightDirection, etaI, etaT, true) * lightDirection.z;
                if (!all(isfinite(result))) { debugCode = 110u; return result; }
            }
            LayerBsdfSample reflection = nclsSampleInterfaceTransport(
                layer,
                incidentAtInterface,
                etaI,
                etaT,
                NCLS_SAMPLE_REFLECTION,
                true,
                rng);
            if (reflection.valid == 0u) break;
            throughput *= reflection.weight;
            if (!all(isfinite(throughput))) { debugCode = 112u; return throughput; }
            w = reflection.direction;
            nonBallisticEvent = true;
            continue;
        }

        if (interfaceIndex + 1u == stack.interfaceCount)
        {
            LayerBsdfSample reflection = nclsSampleInterface(
                layer,
                incidentAtInterface,
                etaI,
                etaT,
                NCLS_SAMPLE_REFLECTION,
                rng);
            if (reflection.valid == 0u) break;
            throughput *= reflection.weight;
            if (!all(isfinite(throughput))) { debugCode = 114u; return throughput; }
            w = reflection.direction;
            continue;
        }

        LayerBsdfSample scatter = nclsSampleInterfaceTransport(
            layer,
            incidentAtInterface,
            etaI,
            etaT,
            NCLS_SAMPLE_ALL,
            true,
            rng);
        if (scatter.valid == 0u) break;
        throughput *= scatter.weight;
        if (!all(isfinite(throughput))) { debugCode = 116u; return throughput; }
        w = scatter.direction;
        if (scatter.isReflection != 0u) nonBallisticEvent = true;
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

float3 nclsEvalRandomWalkReferenceGeneral(
    const NclsLayerStackIR stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey,
    const uint maxDepth)
{
    uint debugCode;
    return nclsEvalRandomWalkReferenceDiagnostic(
        stack, viewDirection, lightDirection, rngKey, maxDepth, debugCode);
}

float3 nclsEvalRandomWalkReference(
    const NclsLayerStackIR stack,
    const float3 viewDirection,
    const float3 lightDirection,
    const uint2 rngKey,
    const uint maxDepth)
{
    return stack.interfaceCount == 2u
        ? nclsEvalTwoLayerPbrtWalk(stack, viewDirection, lightDirection, rngKey, maxDepth)
        : nclsEvalRandomWalkReferenceGeneral(stack, viewDirection, lightDirection, rngKey, maxDepth);
}

#endif
