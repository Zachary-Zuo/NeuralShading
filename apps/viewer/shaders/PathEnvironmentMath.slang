#ifndef NCLS_VIEWER_PATH_ENVIRONMENT_MATH_SLANG
#define NCLS_VIEWER_PATH_ENVIRONMENT_MATH_SLANG

#include "ViewerCommon.slang"

float nclsViewerEnvironmentPowerHeuristic(float firstTechniquePdf, float secondTechniquePdf)
{
    const float firstSquared = firstTechniquePdf * firstTechniquePdf;
    const float secondSquared = secondTechniquePdf * secondTechniquePdf;
    return firstSquared / max(firstSquared + secondSquared, 1e-20f);
}

float nclsViewerEnvironmentLightTechniquePdf(float lightPdf)
{
    return float(NCLS_VIEWER_ENVIRONMENT_NEE_SAMPLE_COUNT) * lightPdf;
}

float nclsViewerEnvironmentBsdfTechniquePdf(float bsdfPdf)
{
    return float(NCLS_VIEWER_ENVIRONMENT_BSDF_SAMPLE_COUNT) * bsdfPdf;
}

float nclsViewerEnvironmentLightMisWeight(float lightPdf, float bsdfPdf)
{
    return nclsViewerEnvironmentPowerHeuristic(
        nclsViewerEnvironmentLightTechniquePdf(lightPdf),
        nclsViewerEnvironmentBsdfTechniquePdf(bsdfPdf));
}

float nclsViewerEnvironmentBsdfMisWeight(float bsdfPdf, float lightPdf, bool delta)
{
    return delta ? 1.0f : nclsViewerEnvironmentPowerHeuristic(
        nclsViewerEnvironmentBsdfTechniquePdf(bsdfPdf),
        nclsViewerEnvironmentLightTechniquePdf(lightPdf));
}

#endif
