// 此文件由 ncls.core.material.abi_layout 生成；不要手工编辑。
#ifndef NCLS_LAYER_STACK_IR_SLANG
#define NCLS_LAYER_STACK_IR_SLANG

static const uint NCLS_LAYER_STACK_IR_MAGIC = 0x52494c4eu;
static const uint NCLS_LAYER_STACK_IR_VERSION = 1u;
static const uint NCLS_MAX_INTERFACES = 8u;
static const uint NCLS_MAX_MEDIA = 7u;

enum NclsInterfaceKind : uint
{
    RoughDielectric = 0,
    RoughConductor = 1,
    Diffuse = 2,
    Sheen = 3,
}

// 64 bytes. 字段是类型明确的 tagged union；未使用字段必须为零。
struct NclsLayerInterfaceIR
{
    uint kind;
    uint flags;
    float alphaX;
    float alphaY;

    float relativeIor;
    float etaR;
    float etaG;
    float etaB;

    float kR;
    float kG;
    float kB;
    float colorR;

    float colorG;
    float colorB;
    float tangentRotation;
    float reserved;
};

// 32 bytes.
struct NclsHomogeneousMediumIR
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

// 752 bytes.
struct NclsLayerStackIR
{
    uint magic;
    uint abiVersion;
    uint interfaceCount;
    uint mediumCount;
    NclsLayerInterfaceIR interfaces[NCLS_MAX_INTERFACES];
    NclsHomogeneousMediumIR media[NCLS_MAX_MEDIA];
};

#endif
