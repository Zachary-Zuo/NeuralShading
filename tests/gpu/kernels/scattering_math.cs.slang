#include "../../../shaders/ncls/scattering/nvidia_proposal.slang"
#include "../../../shaders/ncls/scattering/ltc.slang"
#include "../../../shaders/ncls/backends/ltc_k2_analytic_control/ltc_k2_analytic_control_core.slang"
#include "../../../shaders/ncls/backends/lobe_residual/lobe_residual_mlp.slang"

StructuredBuffer<float4> gInput;
RWStructuredBuffer<float4> gOutput0;
RWStructuredBuffer<float4> gOutput1;
uniform uint gCount;
uniform uint gMode;

static const uint NCLS_TEST_COSINE = 0u;
static const uint NCLS_TEST_TILTED = 1u;
static const uint NCLS_TEST_LTC = 2u;
static const uint NCLS_TEST_NONCENTERED_GGX = 3u;
static const uint NCLS_TEST_NVIDIA = 4u;
static const uint NCLS_TEST_GGX_VNDF = 5u;

float3 nclsTestWo()
{
    return normalize(float3(0.35f, -0.2f, 0.915f));
}

float2 nclsTestTiltedSlope()
{
    return float2(0.8f, -0.45f);
}

NclsLtcTransform nclsTestLtc()
{
    NclsLtcTransform transform;
    transform.inverseScale = float2(1.3f, 0.7f);
    transform.shear = float3(0.25f, -0.15f, 0.2f);
    transform.angle = 0.35f;
    return transform;
}

NclsNoncenteredGgxParams nclsTestNoncenteredGgx()
{
    NclsNoncenteredGgxParams params;
    params.alpha = float2(0.28f, 0.62f);
    params.rho = 0.55f;
    params.meanSlope = float2(0.35f, -0.22f);
    return params;
}

NclsNvidiaProposalParams nclsTestNvidiaProposal()
{
    float raw[NCLS_NVIDIA_PROPOSAL_RAW_COUNT];
    raw[0] = -0.45f;
    raw[1] = 0.3f;
    raw[2] = 0.7f;
    raw[3] = 0.2f;
    raw[4] = -0.15f;
    raw[5] = 0.55f;
    raw[6] = -0.4f;
    raw[7] = 0.6f;
    raw[8] = -0.25f;
    return nclsDecodeNvidiaProposal(raw);
}

float nclsTestPdf(uint mode, float3 direction)
{
    if (mode == NCLS_TEST_COSINE) return nclsCosineHemispherePdf(direction);
    if (mode == NCLS_TEST_TILTED) return nclsTiltedCosinePdf(direction, nclsTestTiltedSlope());
    if (mode == NCLS_TEST_LTC) return nclsLtcPdf(nclsTestLtc(), direction);
    if (mode == NCLS_TEST_NONCENTERED_GGX)
        return nclsNoncenteredGgxPdf(nclsTestNoncenteredGgx(), nclsTestWo(), direction);
    if (mode == NCLS_TEST_NVIDIA)
        return nclsNvidiaProposalPdf(nclsTestNvidiaProposal(), nclsTestWo(), direction);
    if (mode == NCLS_TEST_GGX_VNDF)
        return nclsGgxVisibleNormalPdf(nclsTestWo(), direction, float2(0.2f, 0.65f));
    return 0.0f;
}

[numthreads(64, 1, 1)]
void queryScatteringMath(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float value = nclsTestPdf(gMode, normalize(gInput[index].xyz));
    gOutput0[index] = float4(value, 0.0f, 0.0f, 0.0f);
}

[numthreads(64, 1, 1)]
void sampleScatteringMath(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 u = gInput[index].xyz;
    NclsDirectionalSample sample = nclsInvalidDirectionalSample();
    uint component = 0u;
    if (gMode == NCLS_TEST_COSINE)
        sample = nclsSampleCosineDirection(u.xy);
    else if (gMode == NCLS_TEST_TILTED)
        sample = nclsSampleTiltedCosine(nclsTestTiltedSlope(), u.xy);
    else if (gMode == NCLS_TEST_LTC)
        sample = nclsSampleLtc(nclsTestLtc(), u.xy);
    else if (gMode == NCLS_TEST_NONCENTERED_GGX)
        sample = nclsSampleNoncenteredGgx(nclsTestNoncenteredGgx(), nclsTestWo(), u.xy);
    else if (gMode == NCLS_TEST_NVIDIA)
        sample = nclsSampleNvidiaProposal(nclsTestNvidiaProposal(), nclsTestWo(), u, component);
    else if (gMode == NCLS_TEST_GGX_VNDF)
    {
        const float3 halfVector = nclsSampleGgxVisibleNormal(
            nclsTestWo(), float2(0.2f, 0.65f), u.xy);
        const float pdf = nclsGgxVisibleNormalPdf(
            nclsTestWo(), halfVector, float2(0.2f, 0.65f));
        sample = nclsContinuousDirectionalSample(halfVector, pdf);
        const float3 reflected = reflect(-nclsTestWo(), halfVector);
        sample.nullEvent = reflected.z <= NCLS_MIN_COS ? 1u : 0u;
    }
    const float reevaluated = sample.valid != 0u && sample.nullEvent == 0u
        ? nclsTestPdf(gMode, sample.direction)
        : 0.0f;
    gOutput0[index] = float4(sample.direction, sample.pdf);
    gOutput1[index] = float4(
        float(sample.valid), float(sample.nullEvent), reevaluated, float(component));
}

[numthreads(64, 1, 1)]
void evaluateLtcK2AnalyticControl(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    NclsLtcK2AnalyticControlState state = {};
    state.top.kind = (uint)NclsInterfaceKind::Diffuse;
    state.top.colorR = 0.3f;
    state.top.colorG = 0.2f;
    state.top.colorB = 0.1f;
    [unroll]
    for (uint lobe = 0u; lobe < NCLS_LTC_K2_ANALYTIC_CONTROL_LOBE_COUNT; ++lobe)
    {
        state.residual[lobe].amplitude = (lobe == 0u)
            ? float3(0.2f, 0.1f, 0.05f)
            : float3(0.05f, 0.08f, 0.12f);
        state.residual[lobe].transform = nclsTestLtc();
    }
    const float3 value = nclsLtcK2AnalyticControlEvaluateF(
        state, nclsTestWo(), normalize(gInput[index].xyz));
    gOutput0[index] = float4(value, all(isfinite(value)) ? 1.0f : 0.0f);
}

[numthreads(64, 1, 1)]
void checkScatteringBoundaries(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    NclsDirectionalSample sample = nclsInvalidDirectionalSample();
    float reevaluated = 0.0f;
    if (index < 2u)
    {
        NclsLtcTransform transform;
        transform.inverseScale = index == 0u
            ? float2(0.049787068f, 20.085537f)
            : float2(20.085537f, 0.049787068f);
        transform.shear = index == 0u
            ? float3(3.0f, -3.0f, 3.0f)
            : float3(-3.0f, 3.0f, -3.0f);
        transform.angle = index == 0u ? NCLS_PI : -NCLS_PI;
        sample = nclsSampleLtc(transform, float2(0.99999988f, 0.99999988f));
        if (sample.valid != 0u && sample.nullEvent == 0u)
            reevaluated = nclsLtcPdf(transform, sample.direction);
    }
    else if (index < 4u)
    {
        NclsNoncenteredGgxParams params;
        params.alpha = index == 2u ? float2(1e-4f, 1.0001f) : float2(1.0001f, 1e-4f);
        params.rho = index == 2u ? 0.999999f : -0.999999f;
        params.meanSlope = index == 2u ? float2(256.0f, -256.0f) : float2(-256.0f, 256.0f);
        const float3 grazingWo = normalize(float3(1.0f, 0.0f, 1.1e-6f));
        sample = nclsSampleNoncenteredGgx(params, grazingWo, float2(0.75f, 0.375f));
        if (sample.valid != 0u && sample.nullEvent == 0u)
            reevaluated = nclsNoncenteredGgxPdf(params, grazingWo, sample.direction);
    }
    else
    {
        float raw[NCLS_NVIDIA_PROPOSAL_RAW_COUNT];
        raw[0] = index == 4u ? -10000.0f : 10000.0f;
        raw[1] = -raw[0];
        raw[2] = index == 4u ? 10000.0f : -10000.0f;
        raw[3] = 16.0f;
        raw[4] = -16.0f;
        raw[5] = -16.0f;
        raw[6] = 16.0f;
        raw[7] = index == 4u ? 80.0f : -80.0f;
        raw[8] = -raw[7];
        const NclsNvidiaProposalParams params = nclsDecodeNvidiaProposal(raw);
        uint component = 0u;
        sample = nclsSampleNvidiaProposal(
            params,
            normalize(float3(1.0f, 0.0f, 1.1e-6f)),
            float3(0.5f, 0.75f, 0.375f),
            component);
        if (sample.valid != 0u && sample.nullEvent == 0u)
            reevaluated = nclsNvidiaProposalPdf(
                params, normalize(float3(1.0f, 0.0f, 1.1e-6f)), sample.direction);
    }
    const bool finite = nclsIsFiniteFloat3(sample.direction)
        && isfinite(sample.pdf) && isfinite(reevaluated);
    gOutput0[index] = float4(sample.direction, sample.pdf);
    gOutput1[index] = float4(
        float(sample.valid), float(sample.nullEvent), reevaluated, finite ? 1.0f : 0.0f);
}

[numthreads(64, 1, 1)]
void checkMixture3Selection(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const NclsMixture3Selection selection = nclsSelectMixture3(
        float3(0.2f, 0.3f, 0.5f), gInput[index].x);
    gOutput0[index] = float4(
        float(selection.component), selection.remapped, float(selection.valid), 0.0f);
}

[numthreads(64, 1, 1)]
void checkNvidiaProposalDecode(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const NclsNvidiaProposalParams params = nclsTestNvidiaProposal();
    const float3 mixtureWeights = nclsNvidiaProposalMixtureWeights(params);
    if (index == 0u)
        gOutput0[index] = float4(params.alpha, params.rho, float(params.valid));
    else if (index == 1u)
        gOutput0[index] = float4(params.specularSlope, params.diffuseSlope);
    else
        gOutput0[index] = float4(
            params.learnedWeights, mixtureWeights.y, mixtureWeights.z);
}

[numthreads(64, 1, 1)]
void checkDirectionalSampleStates(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    NclsDirectionalSample sample;
    if (index == 0u)
        sample = nclsContinuousDirectionalSample(float3(0.0f, 0.0f, 1.0f), NCLS_INV_PI);
    else
        sample = nclsNullDirectionalSample(gInput[index].xyz);
    gOutput0[index] = float4(sample.direction, sample.pdf);
    gOutput1[index] = float4(float(sample.valid), float(sample.nullEvent), 0.0f, 0.0f);
}

[numthreads(64, 1, 1)]
void checkLtcK2AnalyticControlIdentity(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint index = dispatchThreadID.x;
    if (index >= gCount) return;
    const float3 wi = normalize(gInput[index].xyz);

    NclsLtcK2AnalyticControlState state = {};
    state.top.kind = (uint)NclsInterfaceKind::Diffuse;
    state.top.colorR = 0.3f;
    state.top.colorG = 0.2f;
    state.top.colorB = 0.1f;
    [unroll]
    for (uint lobe = 0u; lobe < NCLS_LTC_K2_ANALYTIC_CONTROL_LOBE_COUNT; ++lobe)
        state.residual[lobe].transform.inverseScale = float2(1.0f);
    const float3 diffuse = nclsLtcK2AnalyticControlEvaluateF(
        state, nclsTestWo(), wi);

    state.top.colorR = 0.0f;
    state.top.colorG = 0.0f;
    state.top.colorB = 0.0f;
    state.residual[0].amplitude = float3(0.24f, 0.12f, 0.06f);
    const float3 identityLtc = nclsLtcK2AnalyticControlEvaluateF(
        state, nclsTestWo(), wi);
    gOutput0[index] = float4(diffuse, all(isfinite(diffuse)) ? 1.0f : 0.0f);
    gOutput1[index] = float4(identityLtc, all(isfinite(identityLtc)) ? 1.0f : 0.0f);
}
