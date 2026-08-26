RWStructuredBuffer<float4> gOutput;
uniform uint gCount;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint index = dispatchThreadId.x;
    if (index >= gCount) return;
    const float value = float(index + 1u);
    gOutput[index] = float4(value, value * 2.0f, value * 3.0f, 1.0f);
}
