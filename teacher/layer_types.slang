#ifndef NCLS_LAYER_TYPES_SLANG
#define NCLS_LAYER_TYPES_SLANG

static const uint NCLS_SCHEMA_MAGIC = 0x4e434c53;
static const uint NCLS_SCHEMA_VERSION = 1;
static const uint NCLS_MAX_LAYERS = 8;
static const uint NCLS_MAX_MEDIA = 7;

enum LayerType : uint
{
    RoughDielectric = 0,
    RoughConductor = 1,
    Diffuse = 2,
    Sheen = 3,
}

// Scalar fields make the StructuredBuffer layout explicit: 64 bytes.
struct LayerInterface
{
    uint type;
    uint flags;
    float roughnessX;
    float roughnessY;

    float etaR;
    float etaG;
    float etaB;
    float kR;

    float kG;
    float kB;
    float albedoR;
    float albedoG;

    float albedoB;
    float tangentRotation;
    float reserved0;
    float reserved1;
};

// 32 bytes.
struct LayerMedium
{
    float sigmaAR;
    float sigmaAG;
    float sigmaAB;
    float sigmaSR;
    float sigmaSG;
    float sigmaSB;
    float g;
    float thickness;
};

// Fixed-size payload matching schema/stack.py: 752 bytes.
struct LayerStack
{
    uint magic;
    uint version;
    uint layerCount;
    uint mediumCount;
    LayerInterface layers[NCLS_MAX_LAYERS];
    LayerMedium media[NCLS_MAX_MEDIA];
};

#endif
