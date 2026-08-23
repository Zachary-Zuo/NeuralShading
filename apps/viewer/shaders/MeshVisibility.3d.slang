struct VSIn
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float3 tangent : TANGENT;
    float2 texCoord : TEXCOORD;
};

struct VSOut
{
    float4 positionH : SV_POSITION;
    float3 positionW : TEXCOORD0;
    float3 normalW : TEXCOORD1;
    float3 tangentW : TEXCOORD2;
    float2 texCoord : TEXCOORD3;
};

cbuffer MeshVisibilityCB
{
    uint2 gFrameDim;
    float gVerticalFovRadians;
    float _padding0;
    float3 gCameraPosition;
    float _padding1;
    float3 gCameraTarget;
    float _padding2;
};

VSOut vsMain(VSIn input)
{
    const float3 forward = normalize(gCameraTarget - gCameraPosition);
    const float3 right = normalize(cross(forward, float3(0.0f, 1.0f, 0.0f)));
    const float3 up = cross(right, forward);
    const float3 relative = input.position - gCameraPosition;
    const float viewDepth = dot(relative, forward);
    const float tangentFov = tan(0.5f * gVerticalFovRadians);
    const float aspect = float(gFrameDim.x) / float(gFrameDim.y);
    const float nearPlane = 0.05f;
    const float farPlane = 100.0f;

    VSOut output;
    output.positionH = float4(
        dot(relative, right) / (tangentFov * aspect),
        dot(relative, up) / tangentFov,
        farPlane * viewDepth / (farPlane - nearPlane)
            - nearPlane * farPlane / (farPlane - nearPlane),
        viewDepth);
    output.positionW = input.position;
    output.normalW = normalize(input.normal);
    output.tangentW = normalize(input.tangent);
    output.texCoord = input.texCoord;
    return output;
}

struct PSOut
{
    float4 positionDepth : SV_TARGET0;
    float4 normal : SV_TARGET1;
    float4 tangent : SV_TARGET2;
    float4 viewDirection : SV_TARGET3;
    float4 texCoord : SV_TARGET4;
    float4 texCoordGrad : SV_TARGET5;
};

[earlydepthstencil]
PSOut psMain(VSOut input)
{
    PSOut output;
    output.positionDepth = float4(input.positionW, length(gCameraPosition - input.positionW));
    output.normal = float4(normalize(input.normalW), 0.0f);
    output.tangent = float4(normalize(input.tangentW), 0.0f);
    output.viewDirection = float4(normalize(gCameraPosition - input.positionW), 0.0f);
    output.texCoord = float4(input.texCoord, 0.0f, 0.0f);
    output.texCoordGrad = float4(ddx(input.texCoord), ddy(input.texCoord));
    return output;
}
