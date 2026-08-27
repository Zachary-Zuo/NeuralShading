// 此文件由 ncls.core.scattering.abi_layout 生成；不要手工编辑。
#ifndef NCLS_SCATTERING_CONTRACT_SLANG
#define NCLS_SCATTERING_CONTRACT_SLANG

static const uint NCLS_SCATTERING_CONTRACT_VERSION = 1u;

enum NclsTransportMode : uint
{
    Radiance = 0u,
    Importance = 1u,
}

enum NclsScatteringEvent : uint
{
    None = 0u,
    Reflection = 1u,
    Transmission = 2u,
    Diffuse = 4u,
    Glossy = 8u,
    Delta = 16u,
    FrontSide = 32u,
    BackSide = 64u,
    VolumeBoundary = 128u,
}

enum NclsBackendCapability : uint
{
    None = 0u,
    Prepare = 1u,
    Evaluate = 2u,
    Sample = 4u,
    Pdf = 8u,
    AnisotropicFrame = 16u,
    ReversePdf = 32u,
    AnalyticPolygonIntegration = 64u,
    PrefilteredEnvironmentIntegration = 128u,
    NeuralEnvironmentIntegration = 256u,
    DeltaEvents = 512u,
    Transmission = 1024u,
    HomogeneousVolume = 2048u,
}

struct NclsShadingFrame
{
    float3 normal;
    float3 tangent;
    float3 bitangent;
};

struct NclsSurfaceInteraction
{
    float3 position;
    float3 geometricNormal;
    NclsShadingFrame shadingFrame;
    float2 uv;
    float2 uvDx;
    float2 uvDy;
    uint materialInstanceId;
    uint primitiveId;
    uint frontFacing;
};

struct NclsScatteringContext
{
    NclsSurfaceInteraction surface;
    float3 woWorld;
    uint transportMode;
    uint componentMask;
    float filterRandom;
};

struct NclsScatteringPdf
{
    float forward;
    float reverse;
};

struct NclsScatteringEval
{
    float3 f;
    NclsScatteringPdf pdf;
    uint eventFlags;
    uint valid;
};

struct NclsScatteringSample
{
    float3 wiWorld;
    float3 weight;
    NclsScatteringPdf pdf;
    float eta;
    uint eventFlags;
    uint valid;
};

#endif
