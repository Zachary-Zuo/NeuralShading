RWTexture2D<float4> gPositionDepth;
RWTexture2D<float4> gNormal;
RWTexture2D<float4> gTangent;
RWTexture2D<float4> gViewDirection;
RWTexture2D<float4> gMaterialXTexCoord;
RWTexture2D<float4> gMaterialXTexCoordGrad;
RWTexture2D<uint> gInstanceId;
RWTexture2D<uint> gMaterialId;

cbuffer ClearVisibilityCB
{
    uint2 gFrameDim;
};

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gFrameDim)) return;
    gPositionDepth[pixel] = float4(0.0f, 0.0f, 0.0f, -1.0f);
    gNormal[pixel] = float4(0.0f, 1.0f, 0.0f, 0.0f);
    gTangent[pixel] = float4(1.0f, 0.0f, 0.0f, 0.0f);
    gViewDirection[pixel] = float4(0.0f);
    gMaterialXTexCoord[pixel] = float4(0.0f);
    gMaterialXTexCoordGrad[pixel] = float4(0.0f);
    gInstanceId[pixel] = 0u;
    gMaterialId[pixel] = 0u;
}
