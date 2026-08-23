#include "NclsViewer.h"

#include "Core/Platform/OS.h"
#include "Utils/Image/Bitmap.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace Falcor;

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
constexpr uint32_t kLegacyStateBytes = 176;
const Gui::DropdownList kObjectModes = {{0, "球体"}, {1, "Shader ball"}, {2, "细节 hero 物体"}};
const Gui::DropdownList kComparisonModes = {
    {0, "左 reference / 右方法"},
    {1, "线性绝对差"},
    {2, "线性相对差"},
    {3, "放大绝对差"},
};
const Gui::DropdownList kBaseKinds = {{1, "粗糙导体"}, {2, "漫反射"}, {3, "Sheen"}};

float3 normalizedOr(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-12f ? value / std::sqrt(lengthSquared) : fallback;
}

std::string shortId(const std::string& value)
{
    return value.size() > 12 ? value.substr(0, 12) : value;
}

ViewerOptions parseOptions(int argc, char** argv)
{
    ViewerOptions options;
    for (int index = 1; index + 1 < argc; ++index)
    {
        if (std::string(argv[index]) != "--replay") continue;
        options.replayPath = argv[index + 1];
        std::ifstream stream(options.replayPath);
        if (!stream) throw std::runtime_error("cannot open replay manifest: " + options.replayPath.string());
        const nlohmann::json replay = nlohmann::json::parse(stream);
        if (replay.value("format_name", "") != "ncls.viewer-capture" || replay.value("format_version", 0u) != 1u)
            throw std::runtime_error("unsupported replay manifest");
        const auto base = std::filesystem::absolute(options.replayPath).parent_path();
        const auto resolve = [&](const std::string& value) {
            if (value.empty()) return std::filesystem::path();
            const std::filesystem::path path(value);
            return path.is_absolute() ? path : base / path;
        };
        options.bundleRoot = resolve(replay.value("bundle_root", std::string()));
        options.materialPath = resolve(replay.value("material_program", std::string()));
        options.environmentPath = resolve(replay.value("environment", std::string()));
        options.requestedMethodId = replay.value("method_id", std::string());
        const auto resolution = replay.at("resolution");
        options.width = resolution.at(0).get<uint32_t>();
        options.height = resolution.at(1).get<uint32_t>();
        const uint32_t samples = replay.value("reference_samples_per_frame", 1u);
        options.frameCount = std::max(1u, (replay.value("reference_spp", 1u) + samples - 1u) / samples);
        break;
    }
    auto value = [&](int& index, const char* name) -> std::string {
        if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--bundle-root") options.bundleRoot = value(index, "--bundle-root");
        else if (argument == "--material") options.materialPath = value(index, "--material");
        else if (argument == "--replay") { options.replayPath = value(index, "--replay"); }
        else if (argument == "--capture") options.captureManifest = value(index, "--capture");
        else if (argument == "--frames") options.frameCount = static_cast<uint32_t>(std::stoul(value(index, "--frames")));
        else if (argument == "--width") options.width = static_cast<uint32_t>(std::stoul(value(index, "--width")));
        else if (argument == "--height") options.height = static_cast<uint32_t>(std::stoul(value(index, "--height")));
        else if (argument == "--headless") options.headless = true;
        else if (argument == "--help")
        {
            std::cout
                << "NclsViewer [--bundle-root DIR] [--material FILE] [--replay CAPTURE.json] "
                   "[--headless --frames N --capture FILE] "
                   "[--width W --height H]\n";
            std::exit(0);
        }
        else throw std::runtime_error("unknown argument: " + argument);
    }
    if (options.width < 320 || options.height < 240 || options.frameCount < 1)
        throw std::runtime_error("viewer dimensions/frame count are outside supported bounds");
    return options;
}
} // namespace

NclsViewer::NclsViewer(const SampleAppConfig& config, ViewerOptions options)
    : SampleApp(config), mOptions(std::move(options))
{}

void NclsViewer::onLoad(RenderContext* pRenderContext)
{
    createPasses();
    Sampler::Desc samplerDesc;
    samplerDesc
        .setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    mpLinearSampler = getDevice()->createSampler(samplerDesc);
    createDefaultEnvironment();
    mpMaterial = getDevice()->createStructuredBuffer(
        sizeof(ncls::LayerStackIR), 1, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &mMaterial);
    const float zero = 0.f;
    mpWeights = getDevice()->createStructuredBuffer(
        sizeof(float), 1, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &zero);
    auto initializeTiming = [&](PassTiming& timing) {
        for (auto& timer : timing.timers) timer = GpuTimer::create(getDevice());
    };
    initializeTiming(mVisibilityTiming);
    initializeTiming(mReferenceTiming);
    initializeTiming(mPrepareTiming);
    initializeTiming(mLightingTiming);
    initializeTiming(mCompositeTiming);

    if (!mOptions.replayPath.empty()) applyReplaySettings(mOptions.replayPath);
    if (!mOptions.materialPath.empty()) loadMaterial(mOptions.materialPath);
    if (!mOptions.environmentPath.empty()) loadEnvironment(mOptions.environmentPath);
    resizeResources(getTargetFbo()->getWidth(), getTargetFbo()->getHeight());
    scanBundles();
    if (!mOptions.requestedMethodId.empty())
    {
        int32_t requested = -1;
        if (mOptions.requestedMethodId != "diagnostic-exact-top-only")
            for (uint32_t index = 0; index < mMethods.size(); ++index)
                if (mMethods[index].methodId == mOptions.requestedMethodId) requested = static_cast<int32_t>(index);
        if (requested < 0 && mOptions.requestedMethodId != "diagnostic-exact-top-only" && mOptions.headless)
            throw std::runtime_error("replay MethodBundle did not pass compatibility/parity: " + mOptions.requestedMethodId);
        selectMethod(requested);
    }
    mStatus = mMethods.empty()
        ? "未发现兼容 realtime bundle；右侧显示“仅顶层界面”诊断模式。"
        : "已加载并通过 GPU parity 的 MethodBundle。";
}

void NclsViewer::createPasses()
{
    mpVisibilityPass = ComputePass::create(getDevice(), "NclsViewer/shaders/Visibility.cs.slang");
    mpReferencePass = ComputePass::create(getDevice(), "NclsViewer/shaders/Reference.cs.slang");
    mpPreparePass = ComputePass::create(getDevice(), "NclsViewer/shaders/Prepare.cs.slang");
    mpApproximationPass = ComputePass::create(getDevice(), "NclsViewer/shaders/Approximation.cs.slang");
    mpCompositePass = ComputePass::create(getDevice(), "NclsViewer/shaders/Composite.cs.slang");
    mpParityPass = ComputePass::create(getDevice(), "NclsViewer/shaders/Parity.cs.slang");
}

void NclsViewer::createDefaultEnvironment()
{
    constexpr uint32_t width = 16;
    constexpr uint32_t height = 8;
    std::vector<float4> pixels(width * height);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            const float v = float(y) / float(height - 1);
            float3 color = (1.f - v) * float3(0.18f, 0.24f, 0.34f) + v * float3(0.035f, 0.04f, 0.05f);
            if (x >= 10 && x <= 12 && y >= 2 && y <= 3) color += float3(5.0f, 3.2f, 1.8f);
            if (x >= 2 && x <= 4 && y >= 3 && y <= 5) color += float3(0.4f, 0.75f, 1.4f);
            pixels[y * width + x] = float4(color, 1.f);
        }
    }
    mpEnvironment = getDevice()->createTexture2D(
        width, height, ResourceFormat::RGBA32Float, 1, 1, pixels.data(), ResourceBindFlags::ShaderResource);
    mEnvironmentPath.clear();
}

void NclsViewer::onResize(uint32_t width, uint32_t height)
{
    if (width > 0 && height > 0 && (width != mOutputWidth || height != mOutputHeight)) resizeResources(width, height);
}

void NclsViewer::resizeResources(uint32_t width, uint32_t height)
{
    mOutputWidth = std::max(width, 2u);
    mOutputHeight = std::max(height, 1u);
    mViewWidth = std::max(mOutputWidth / 2u, 1u);
    const auto shaderUav = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto viewTexture = [&]() {
        return getDevice()->createTexture2D(mViewWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, shaderUav);
    };
    mpPositionDepth = viewTexture();
    mpNormal = viewTexture();
    mpTangent = viewTexture();
    mpViewDirection = viewTexture();
    mpReference[0] = viewTexture();
    mpReference[1] = viewTexture();
    mpApproximation = viewTexture();
    mpComparisonLinear = getDevice()->createTexture2D(
        mOutputWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, shaderUav);
    mpDisplay = getDevice()->createTexture2D(
        mOutputWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, shaderUav);
    const uint64_t stateCount64 = uint64_t(mViewWidth) * uint64_t(mOutputHeight);
    if (stateCount64 > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("viewer state buffer is too large");
    mpStates = getDevice()->createStructuredBuffer(
        kLegacyStateBytes,
        static_cast<uint32_t>(stateCount64),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    mReferencePing = 0;
    resetReference(true, true);
}

void NclsViewer::updateMaterialBuffer()
{
    ncls::validateLayerStack(mMaterial);
    mpMaterial->setBlob(&mMaterial, 0, sizeof(mMaterial));
    resetReference(false, true);
}

void NclsViewer::resetReference(bool visibilityChanged, bool prepareChanged)
{
    mReferenceSpp = 0;
    mAccumulationSeconds = 0.0;
    mResetAccumulation = true;
    mVisibilityDirty |= visibilityChanged;
    mPrepareDirty |= prepareChanged || visibilityChanged;
}

void NclsViewer::resetCamera()
{
    mCamera = {};
    resetReference(true, true);
}

float3 NclsViewer::cameraPosition() const
{
    const float cosinePitch = std::cos(mCamera.pitch);
    return mCamera.target + mCamera.distance * float3(
        cosinePitch * std::sin(mCamera.yaw),
        std::sin(mCamera.pitch),
        cosinePitch * std::cos(mCamera.yaw));
}

void NclsViewer::scanBundles()
{
    const std::string previousId = mSelectedMethod >= 0 && mSelectedMethod < static_cast<int32_t>(mMethods.size())
        ? mMethods[mSelectedMethod].methodId
        : std::string();
    auto scan = ncls::scanMethodBundles(mOptions.bundleRoot, getRuntimeDirectory() / "shaders");
    std::vector<ncls::ViewerMethod> accepted;
    for (auto& method : scan.methods)
    {
        std::string error;
        if (runParityProbe(method, error))
        {
            logInfo("Accepted MethodBundle '{}' ({})", method.displayName, shortId(method.methodId));
            accepted.push_back(std::move(method));
        }
        else scan.failures.push_back({method.root, "GPU parity failed: " + error});
    }
    mMethods = std::move(accepted);
    mBundleFailures = std::move(scan.failures);
    for (const auto& failure : mBundleFailures)
        logWarning("Rejected MethodBundle '{}': {}", failure.path, failure.reason);
    int32_t selection = -1;
    if (!previousId.empty())
        for (uint32_t index = 0; index < mMethods.size(); ++index)
            if (mMethods[index].methodId == previousId) selection = static_cast<int32_t>(index);
    if (selection < 0 && !mMethods.empty()) selection = 0;
    selectMethod(selection);
}

bool NclsViewer::runParityProbe(const ncls::ViewerMethod& method, std::string& error)
{
    try
    {
        const auto flags = ResourceBindFlags::ShaderResource;
        auto material = getDevice()->createStructuredBuffer(752, 1, flags, MemoryType::DeviceLocal, method.parity.material.data());
        auto weights = getDevice()->createStructuredBuffer(
            sizeof(float), static_cast<uint32_t>(method.weights.size()), flags, MemoryType::DeviceLocal, method.weights.data());
        const float4 view(method.parity.view[0], method.parity.view[1], method.parity.view[2], 0.f);
        std::vector<float4> lights;
        for (const auto& item : method.parity.lights) lights.emplace_back(item[0], item[1], item[2], 0.f);
        auto viewBuffer = getDevice()->createStructuredBuffer(sizeof(float4), 1, flags, MemoryType::DeviceLocal, &view);
        auto lightBuffer = getDevice()->createStructuredBuffer(
            sizeof(float4), static_cast<uint32_t>(lights.size()), flags, MemoryType::DeviceLocal, lights.data());
        auto output = getDevice()->createStructuredBuffer(
            sizeof(float4),
            static_cast<uint32_t>(lights.size()),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        auto root = mpParityPass->getRootVar();
        root["gMaterials"] = material;
        root["gWeights"] = weights;
        root["gViews"] = viewBuffer;
        root["gLights"] = lightBuffer;
        root["gOutput"] = output;
        root["gWidth"] = method.width;
        root["gLightCount"] = static_cast<uint32_t>(lights.size());
        mpParityPass->execute(getRenderContext(), static_cast<uint32_t>(lights.size()), 1, 1);
        std::vector<float4> actual(lights.size());
        output->getBlob(actual.data(), 0, actual.size() * sizeof(float4));
        for (size_t light = 0; light < actual.size(); ++light)
        {
            for (size_t channel = 0; channel < 3; ++channel)
            {
                const float observed = actual[light][channel];
                const float expected = method.parity.expectedResponseCos[light][channel];
                const float tolerance = method.parity.absoluteTolerance + method.parity.relativeTolerance * std::abs(expected);
                if (!std::isfinite(observed) || std::abs(observed - expected) > tolerance)
                {
                    error = "light " + std::to_string(light) + ", channel " + std::to_string(channel)
                        + ": expected " + std::to_string(expected) + ", got " + std::to_string(observed);
                    return false;
                }
            }
        }
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

void NclsViewer::selectMethod(int32_t methodIndex)
{
    mSelectedMethod = methodIndex >= 0 && methodIndex < static_cast<int32_t>(mMethods.size()) ? methodIndex : -1;
    mMethodUiValue = mSelectedMethod >= 0 ? static_cast<uint32_t>(mSelectedMethod + 1) : 0u;
    if (mSelectedMethod >= 0)
    {
        const auto& method = mMethods[mSelectedMethod];
        mpWeights = getDevice()->createStructuredBuffer(
            sizeof(float),
            static_cast<uint32_t>(method.weights.size()),
            ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            method.weights.data());
    }
    else
    {
        const float zero = 0.f;
        mpWeights = getDevice()->createStructuredBuffer(
            sizeof(float), 1, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &zero);
    }
    mPrepareDirty = true;
}

void NclsViewer::bindLighting(ShaderVar root, const char* constantBufferName)
{
    auto constants = root[constantBufferName];
    constants["gUseEnvironment"] = uint32_t(mLighting.useEnvironment);
    constants["gEnvironmentRotation"] = mLighting.environmentRotation;
    constants["gEnvironmentIntensity"] = mLighting.environmentIntensity;
    constants["gUseSun"] = uint32_t(mLighting.useSun);
    constants["gSunDirection"] = normalizedOr(mLighting.sunDirection, float3(0.f, 1.f, 0.f));
    constants["gSunIntensity"] = mLighting.sunIntensity;
    constants["gSunColor"] = mLighting.sunColor;
    constants["gUsePoint"] = uint32_t(mLighting.usePoint);
    constants["gPointPosition"] = mLighting.pointPosition;
    constants["gPointIntensity"] = mLighting.pointIntensity;
    constants["gPointColor"] = mLighting.pointColor;
    constants["gUseRectangle"] = uint32_t(mLighting.useRectangle);
    constants["gRectangleCenter"] = mLighting.rectangleCenter;
    constants["gRectangleAxisU"] = mLighting.rectangleAxisU;
    constants["gRectangleAxisV"] = mLighting.rectangleAxisV;
    constants["gRectangleIntensity"] = mLighting.rectangleIntensity;
    constants["gRectangleColor"] = mLighting.rectangleColor;
}

void NclsViewer::beginTiming(PassTiming& timing)
{
    timing.activeSlot = static_cast<uint32_t>(timing.sampleIndex % timing.timers.size());
    if (timing.sampleIndex >= timing.timers.size()) timing.milliseconds = timing.timers[timing.activeSlot]->getElapsedTime();
    timing.timers[timing.activeSlot]->begin();
}

void NclsViewer::endTiming(PassTiming& timing)
{
    timing.timers[timing.activeSlot]->end();
    timing.timers[timing.activeSlot]->resolve();
    ++timing.sampleIndex;
}

void NclsViewer::renderVisibility(RenderContext* pRenderContext)
{
    auto root = mpVisibilityPass->getRootVar();
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    auto constants = root["VisibilityCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gObjectMode"] = mObjectMode;
    constants["gVerticalFovRadians"] = mCamera.verticalFovDegrees * (3.14159265358979323846f / 180.f);
    constants["gCameraPosition"] = cameraPosition();
    constants["gCameraTarget"] = mCamera.target;
    beginTiming(mVisibilityTiming);
    mpVisibilityPass->execute(pRenderContext, mViewWidth, mOutputHeight);
    endTiming(mVisibilityTiming);
    mVisibilityDirty = false;
}

void NclsViewer::renderReference(RenderContext* pRenderContext)
{
    const uint32_t next = 1u - mReferencePing;
    auto root = mpReferencePass->getRootVar();
    root["gMaterials"] = mpMaterial;
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gPreviousReference"] = mpReference[mReferencePing];
    root["gNextReference"] = mpReference[next];
    root["gEnvironment"] = mpEnvironment;
    root["gLinearSampler"] = mpLinearSampler;
    auto constants = root["ReferenceCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gFrameIndex"] = mFrameIndex;
    constants["gReferenceSpp"] = mReferenceSpp;
    constants["gSamplesThisFrame"] = mSamplesPerFrame;
    constants["gMaxDepth"] = mMaxReferenceDepth;
    constants["gResetAccumulation"] = uint32_t(mResetAccumulation);
    const bool accumulate = !mCameraDragging && !mPanDragging;
    constants["gAccumulate"] = uint32_t(accumulate);
    bindLighting(root, "ReferenceCB");
    beginTiming(mReferenceTiming);
    mpReferencePass->execute(pRenderContext, mViewWidth, mOutputHeight);
    endTiming(mReferenceTiming);
    mReferencePing = next;
    if (accumulate)
    {
        mReferenceSpp += mSamplesPerFrame;
        mAccumulationSeconds += getFrameRate().getLastFrameTime();
    }
    else mReferenceSpp = 0;
    mResetAccumulation = false;
}

void NclsViewer::renderPrepare(RenderContext* pRenderContext)
{
    auto root = mpPreparePass->getRootVar();
    root["gMaterials"] = mpMaterial;
    root["gWeights"] = mpWeights;
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gStates"] = mpStates;
    auto constants = root["PrepareCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gMethodMode"] = mSelectedMethod >= 0 ? 1u : 0u;
    constants["gWidth"] = mSelectedMethod >= 0 ? mMethods[mSelectedMethod].width : 8u;
    beginTiming(mPrepareTiming);
    mpPreparePass->execute(pRenderContext, mViewWidth, mOutputHeight);
    endTiming(mPrepareTiming);
    mPrepareDirty = false;
}

void NclsViewer::renderApproximation(RenderContext* pRenderContext)
{
    auto root = mpApproximationPass->getRootVar();
    root["gStates"] = mpStates;
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gEnvironment"] = mpEnvironment;
    root["gLinearSampler"] = mpLinearSampler;
    root["gApproximation"] = mpApproximation;
    root["ApproximationCB"]["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    bindLighting(root, "ApproximationCB");
    beginTiming(mLightingTiming);
    mpApproximationPass->execute(pRenderContext, mViewWidth, mOutputHeight);
    endTiming(mLightingTiming);
}

void NclsViewer::renderComposite(RenderContext* pRenderContext)
{
    auto root = mpCompositePass->getRootVar();
    root["gReference"] = mpReference[mReferencePing];
    root["gApproximation"] = mpApproximation;
    root["gLinearSampler"] = mpLinearSampler;
    root["gComparisonLinear"] = mpComparisonLinear;
    root["gDisplay"] = mpDisplay;
    auto constants = root["CompositeCB"];
    constants["gOutputDim"] = uint2(mOutputWidth, mOutputHeight);
    constants["gSplit"] = mSplit;
    constants["gComparisonMode"] = mComparisonMode;
    constants["gExposure"] = mExposure;
    constants["gDifferenceScale"] = mDifferenceScale;
    beginTiming(mCompositeTiming);
    mpCompositePass->execute(pRenderContext, mOutputWidth, mOutputHeight);
    endTiming(mCompositeTiming);
}

void NclsViewer::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (mVisibilityDirty) renderVisibility(pRenderContext);
    if (!mFreezeReference) renderReference(pRenderContext);
    if (mPrepareDirty) renderPrepare(pRenderContext);
    renderApproximation(pRenderContext);
    renderComposite(pRenderContext);
    pRenderContext->blit(mpDisplay->getSRV(), pTargetFbo->getRenderTargetView(0));
    ++mFrameIndex;

    if (mOptions.headless && ++mRenderedFrames >= mOptions.frameCount)
    {
        capture(mOptions.captureManifest);
        shutdown(0);
    }
}

void NclsViewer::renderMaterialUi(Gui::Window& window)
{
    window.text("材质程序（编辑的是 LayerStackIR 的规范化输入，不是 K2 packet）");
    mSelectedInterface = std::min(mSelectedInterface, mMaterial.interfaceCount - 1);
    Gui::DropdownList layers;
    for (uint32_t index = 0; index < mMaterial.interfaceCount; ++index)
    {
        const bool base = index + 1 == mMaterial.interfaceCount;
        layers.push_back({index, (base ? "基底 " : "涂层 ") + std::to_string(index)});
    }
    window.dropdown("当前界面", layers, mSelectedInterface);
    bool changed = false;
    if (window.button("新增 dielectric 涂层"))
    {
        changed |= ncls::addDielectricCoat(mMaterial);
        mSelectedInterface = mMaterial.interfaceCount - 2;
    }
    if (mSelectedInterface + 1 < mMaterial.interfaceCount)
    {
        if (window.button("删除当前涂层", true))
        {
            changed |= ncls::removeCoat(mMaterial, mSelectedInterface);
            mSelectedInterface = std::min(mSelectedInterface, mMaterial.interfaceCount - 1);
        }
        if (window.button("上移", true)) changed |= ncls::moveCoat(mMaterial, mSelectedInterface, -1);
        if (window.button("下移", true)) changed |= ncls::moveCoat(mMaterial, mSelectedInterface, 1);
    }
    auto& interfaceValue = mMaterial.interfaces[mSelectedInterface];
    const bool isBase = mSelectedInterface + 1 == mMaterial.interfaceCount;
    if (isBase)
    {
        uint32_t kind = interfaceValue.kind;
        if (window.dropdown("基底类型", kBaseKinds, kind))
        {
            interfaceValue = {};
            interfaceValue.kind = kind;
            interfaceValue.alphaX = interfaceValue.alphaY = 0.3f;
            interfaceValue.etaR = 0.2f; interfaceValue.etaG = 0.9f; interfaceValue.etaB = 1.1f;
            interfaceValue.kR = 3.9f; interfaceValue.kG = 2.5f; interfaceValue.kB = 2.1f;
            interfaceValue.colorR = 0.55f; interfaceValue.colorG = 0.22f; interfaceValue.colorB = 0.08f;
            changed = true;
        }
    }
    const auto kind = static_cast<ncls::InterfaceKind>(interfaceValue.kind);
    if (kind == ncls::InterfaceKind::RoughDielectric || kind == ncls::InterfaceKind::RoughConductor)
    {
        changed |= window.var("alpha X", interfaceValue.alphaX, 0.001f, 1.f, 0.005f);
        changed |= window.var("alpha Y", interfaceValue.alphaY, 0.001f, 1.f, 0.005f);
        changed |= window.var("切线旋转（rad）", interfaceValue.tangentRotation, -3.14159f, 3.14159f, 0.01f);
    }
    if (kind == ncls::InterfaceKind::RoughDielectric)
        changed |= window.var("相对 IOR", interfaceValue.relativeIor, 1.001f, 3.f, 0.01f);
    else if (kind == ncls::InterfaceKind::RoughConductor)
    {
        float3 eta(interfaceValue.etaR, interfaceValue.etaG, interfaceValue.etaB);
        float3 k(interfaceValue.kR, interfaceValue.kG, interfaceValue.kB);
        if (window.var("eta RGB", eta, 0.f, 5.f, 0.01f))
        {
            interfaceValue.etaR = eta.x; interfaceValue.etaG = eta.y; interfaceValue.etaB = eta.z; changed = true;
        }
        if (window.var("k RGB", k, 0.f, 10.f, 0.01f))
        {
            interfaceValue.kR = k.x; interfaceValue.kG = k.y; interfaceValue.kB = k.z; changed = true;
        }
    }
    else if (kind == ncls::InterfaceKind::Diffuse || kind == ncls::InterfaceKind::Sheen)
    {
        float3 color(interfaceValue.colorR, interfaceValue.colorG, interfaceValue.colorB);
        if (window.rgbColor("颜色", color))
        {
            interfaceValue.colorR = color.x; interfaceValue.colorG = color.y; interfaceValue.colorB = color.z; changed = true;
        }
        if (kind == ncls::InterfaceKind::Sheen)
        {
            if (window.var("Sheen roughness", interfaceValue.alphaX, 0.001f, 1.f, 0.005f))
            {
                interfaceValue.alphaY = interfaceValue.alphaX;
                changed = true;
            }
        }
    }
    if (mSelectedInterface < mMaterial.mediumCount)
    {
        auto& medium = mMaterial.media[mSelectedInterface];
        const float3 sigmaA(medium.sigmaAR, medium.sigmaAG, medium.sigmaAB);
        const float3 sigmaS(medium.sigmaSR, medium.sigmaSG, medium.sigmaSB);
        float extinction = std::max((sigmaA.x + sigmaS.x + sigmaA.y + sigmaS.y + sigmaA.z + sigmaS.z) / 3.f, 0.f);
        float3 albedo = extinction > 1e-6f ? sigmaS / extinction : float3(0.f);
        bool mediumChanged = false;
        mediumChanged |= window.var("介质总消光（1/unit）", extinction, 0.f, 6.f, 0.01f);
        mediumChanged |= window.rgbColor("介质散射反照率", albedo);
        mediumChanged |= window.var("相函数 g", medium.g, -0.95f, 0.95f, 0.01f);
        mediumChanged |= window.var("厚度", medium.thickness, 0.f, 2.f, 0.01f);
        if (mediumChanged)
        {
            albedo = clamp(albedo, float3(0.f), float3(1.f));
            const float3 newSigmaS = extinction * albedo;
            const float3 newSigmaA = extinction * (float3(1.f) - albedo);
            medium.sigmaAR = newSigmaA.x; medium.sigmaAG = newSigmaA.y; medium.sigmaAB = newSigmaA.z;
            medium.sigmaSR = newSigmaS.x; medium.sigmaSG = newSigmaS.y; medium.sigmaSB = newSigmaS.z;
            changed = true;
        }
    }
    if (changed)
    {
        try { updateMaterialBuffer(); mStatus = "材质已更新，reference 累积已清空。"; }
        catch (const std::exception& error) { mStatus = error.what(); }
    }
    window.text("IR SHA-256: " + shortId(ncls::layerStackHash(mMaterial)));
}

void NclsViewer::onGuiRender(Gui* pGui)
{
    Gui::Window window(pGui, "NeuralShading 材质比较", {410, 850}, {12, 12});
    Gui::DropdownList methodList = {{0, "诊断：仅精确顶层界面（不是完整拟合）"}};
    for (uint32_t index = 0; index < mMethods.size(); ++index)
        methodList.push_back({index + 1, mMethods[index].displayName + " [" + shortId(mMethods[index].methodId) + "]"});
    if (window.dropdown("右侧方法", methodList, mMethodUiValue)) selectMethod(int32_t(mMethodUiValue) - 1);
    if (window.button("重新扫描 MethodBundle")) scanBundles();
    if (mSelectedMethod >= 0)
    {
        const auto& method = mMethods[mSelectedMethod];
        window.text("method: " + shortId(method.methodId) + " / backend v" + std::to_string(method.backendVersion));
        window.text("参数: " + std::to_string(method.parameterCount) + ", state: " + std::to_string(method.stateBytesPerPixel) + " B/pixel");
    }
    if (!mBundleFailures.empty()) window.text("未加载 bundle: " + std::to_string(mBundleFailures.size()) + "（原因见日志/状态）");

    window.text("Reference 随机游走");
    window.var("每帧 samples", mSamplesPerFrame, 1u, 16u);
    if (window.var("最大随机游走深度", mMaxReferenceDepth, 4u, 128u)) resetReference(false, false);
    window.checkbox("冻结 reference", mFreezeReference);
    if (window.button("清空累积")) resetReference(false, false);
    window.text("spp: " + std::to_string(mReferenceSpp) + ", 累积: " + fmt::format("{:.2f}s", mAccumulationSeconds));
    window.text("噪声 proxy 1/sqrt(spp): " + fmt::format("{:.4f}", 1.f / std::sqrt(float(std::max(mReferenceSpp, 1u)))));

    bool physicalChanged = false;
    physicalChanged |= window.dropdown("观察物体", kObjectModes, mObjectMode);
    physicalChanged |= window.var("相机距离", mCamera.distance, 1.25f, 9.f, 0.02f);
    physicalChanged |= window.var("垂直 FOV", mCamera.verticalFovDegrees, 12.f, 90.f, 0.5f);
    if (physicalChanged) resetReference(true, true);
    if (window.button("重置相机")) resetCamera();

    window.dropdown("比较显示", kComparisonModes, mComparisonMode);
    window.var("分割位置", mSplit, 0.1f, 0.9f, 0.005f);
    window.var("共同曝光 EV", mExposure, -8.f, 8.f, 0.05f);
    if (mComparisonMode == 3u) window.var("误差放大", mDifferenceScale, 1.f, 100.f, 0.5f);

    bool lightChanged = false;
    lightChanged |= window.checkbox("HDRI/环境", mLighting.useEnvironment);
    lightChanged |= window.var("环境旋转", mLighting.environmentRotation, -3.14159f, 3.14159f, 0.01f);
    lightChanged |= window.var("环境强度", mLighting.environmentIntensity, 0.f, 20.f, 0.02f);
    if (window.button("加载 HDRI"))
    {
        std::filesystem::path path;
        if (openFileDialog(Bitmap::getFileDialogFilters(ResourceFormat::RGBA32Float), path)) loadEnvironment(path);
    }
    lightChanged |= window.checkbox("方向光", mLighting.useSun);
    lightChanged |= window.var("方向光方向", mLighting.sunDirection, -1.f, 1.f, 0.01f);
    lightChanged |= window.var("方向光强度", mLighting.sunIntensity, 0.f, 50.f, 0.05f);
    lightChanged |= window.rgbColor("方向光颜色", mLighting.sunColor);
    lightChanged |= window.checkbox("点光", mLighting.usePoint);
    lightChanged |= window.var("点光位置", mLighting.pointPosition, -10.f, 10.f, 0.02f);
    lightChanged |= window.var("点光强度", mLighting.pointIntensity, 0.f, 100.f, 0.1f);
    lightChanged |= window.rgbColor("点光颜色", mLighting.pointColor);
    lightChanged |= window.checkbox("矩形面光", mLighting.useRectangle);
    lightChanged |= window.var("面光中心", mLighting.rectangleCenter, -10.f, 10.f, 0.02f);
    lightChanged |= window.var("面光半轴 U", mLighting.rectangleAxisU, -3.f, 3.f, 0.02f);
    lightChanged |= window.var("面光半轴 V", mLighting.rectangleAxisV, -3.f, 3.f, 0.02f);
    lightChanged |= window.var("面光强度", mLighting.rectangleIntensity, 0.f, 100.f, 0.1f);
    lightChanged |= window.rgbColor("面光颜色", mLighting.rectangleColor);
    if (lightChanged) resetReference(false, false);

    renderMaterialUi(window);
    if (window.button("打开 MaterialProgram"))
    {
        std::filesystem::path path;
        if (openFileDialog({{"json", "MaterialProgram JSON"}}, path)) loadMaterial(path);
    }
    if (window.button("保存 MaterialProgram", true))
    {
        std::filesystem::path path = mMaterialPath;
        if (saveFileDialog({{"json", "MaterialProgram JSON"}}, path)) saveMaterial(path);
    }
    if (window.button("保存完整 capture"))
    {
        std::filesystem::path path = "capture.json";
        if (saveFileDialog({{"json", "Capture manifest"}}, path)) capture(path);
    }

    window.text("GPU ms（异步时间戳）");
    window.text(fmt::format(
        "visibility {:.3f} | reference {:.3f}\nprepare {:.3f} | lighting {:.3f} | composite {:.3f}",
        mVisibilityTiming.milliseconds,
        mReferenceTiming.milliseconds,
        mPrepareTiming.milliseconds,
        mLightingTiming.milliseconds,
        mCompositeTiming.milliseconds));
    window.text("交互：左键 orbit；中/右键 pan；滚轮 dolly；拖分割线；Space 冻结 reference。");
    if (!mStatus.empty()) window.text("状态：" + mStatus);
}

void NclsViewer::loadMaterial(const std::filesystem::path& path)
{
    try
    {
        mMaterial = ncls::loadMaterialProgram(path, &mMaterialDisplayName);
        mMaterialPath = path;
        updateMaterialBuffer();
        mStatus = "已加载 MaterialProgram: " + path.string();
    }
    catch (const std::exception& error)
    {
        mStatus = "MaterialProgram 加载失败: " + std::string(error.what());
        if (mOptions.headless) throw;
    }
}

void NclsViewer::saveMaterial(const std::filesystem::path& path)
{
    try
    {
        ncls::saveMaterialProgram(path, mMaterial, mMaterialDisplayName);
        mMaterialPath = path;
        mStatus = "已保存 MaterialProgram: " + path.string();
    }
    catch (const std::exception& error) { mStatus = "保存失败: " + std::string(error.what()); }
}

void NclsViewer::loadEnvironment(const std::filesystem::path& path)
{
    auto texture = Texture::createFromFile(getDevice(), path, true, false);
    if (!texture)
    {
        mStatus = "HDRI 加载失败: " + path.string();
        return;
    }
    mpEnvironment = texture;
    mEnvironmentPath = path;
    resetReference(false, false);
    mStatus = "已加载 HDRI: " + path.string();
}

void NclsViewer::applyReplaySettings(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open replay manifest: " + path.string());
    const nlohmann::json replay = nlohmann::json::parse(stream);
    const auto vector3 = [](const nlohmann::json& value) {
        if (!value.is_array() || value.size() != 3) throw std::runtime_error("replay vector must contain three values");
        return float3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    };
    mObjectMode = replay.value("object_mode", 0u);
    mSamplesPerFrame = replay.value("reference_samples_per_frame", 1u);
    mMaxReferenceDepth = replay.value("reference_max_depth", 24u);
    const auto& camera = replay.at("camera");
    mCamera.target = vector3(camera.at("target"));
    mCamera.yaw = camera.at("yaw").get<float>();
    mCamera.pitch = camera.at("pitch").get<float>();
    mCamera.distance = camera.at("distance").get<float>();
    mCamera.verticalFovDegrees = camera.at("vertical_fov_degrees").get<float>();
    const auto& display = replay.at("display");
    mComparisonMode = display.at("comparison_mode").get<uint32_t>();
    mSplit = display.at("split").get<float>();
    mExposure = display.at("exposure_ev").get<float>();
    mDifferenceScale = display.at("difference_scale").get<float>();
    const auto& lighting = replay.at("lighting");
    mLighting.useEnvironment = lighting.at("use_environment").get<bool>();
    mLighting.environmentRotation = lighting.at("environment_rotation").get<float>();
    mLighting.environmentIntensity = lighting.at("environment_intensity").get<float>();
    mLighting.useSun = lighting.at("use_sun").get<bool>();
    mLighting.sunDirection = vector3(lighting.at("sun_direction"));
    mLighting.sunIntensity = lighting.at("sun_intensity").get<float>();
    mLighting.sunColor = vector3(lighting.at("sun_color"));
    mLighting.usePoint = lighting.at("use_point").get<bool>();
    mLighting.pointPosition = vector3(lighting.at("point_position"));
    mLighting.pointIntensity = lighting.at("point_intensity").get<float>();
    mLighting.pointColor = vector3(lighting.at("point_color"));
    mLighting.useRectangle = lighting.at("use_rectangle").get<bool>();
    mLighting.rectangleCenter = vector3(lighting.at("rectangle_center"));
    mLighting.rectangleAxisU = vector3(lighting.at("rectangle_axis_u"));
    mLighting.rectangleAxisV = vector3(lighting.at("rectangle_axis_v"));
    mLighting.rectangleIntensity = lighting.at("rectangle_intensity").get<float>();
    mLighting.rectangleColor = vector3(lighting.at("rectangle_color"));
}

void NclsViewer::capture(const std::filesystem::path& requestedManifestPath)
{
    namespace fs = std::filesystem;
    fs::path manifestPath = requestedManifestPath;
    if (manifestPath.extension() != ".json") manifestPath /= "capture.json";
    if (!manifestPath.parent_path().empty()) fs::create_directories(manifestPath.parent_path());
    const fs::path stem = manifestPath.parent_path() / manifestPath.stem();
    const fs::path referencePath = stem.string() + "-reference.exr";
    const fs::path approximationPath = stem.string() + "-approximation.exr";
    const fs::path comparisonPath = stem.string() + "-comparison.exr";
    const fs::path displayPath = stem.string() + "-display.png";
    const fs::path differencePath = stem.string() + "-difference.exr";
    const fs::path differenceDisplayPath = stem.string() + "-difference.png";
    const fs::path materialPath = stem.string() + "-material.json";
    const fs::path metricsPath = stem.string() + "-metrics.csv";

    getDevice()->wait();
    const auto refreshTiming = [](PassTiming& timing) {
        if (timing.sampleIndex > 0) timing.milliseconds = timing.timers[timing.activeSlot]->getElapsedTime();
    };
    refreshTiming(mVisibilityTiming);
    refreshTiming(mReferenceTiming);
    refreshTiming(mPrepareTiming);
    refreshTiming(mLightingTiming);
    refreshTiming(mCompositeTiming);
    ncls::saveMaterialProgram(materialPath, mMaterial, mMaterialDisplayName);
    mpReference[mReferencePing]->captureToFile(0, 0, referencePath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    mpApproximation->captureToFile(0, 0, approximationPath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    mpComparisonLinear->captureToFile(0, 0, comparisonPath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    getTargetFbo()->getColorTexture(0)->captureToFile(
        0, 0, displayPath, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
    const uint32_t originalComparisonMode = mComparisonMode;
    mComparisonMode = 1u;
    renderComposite(getRenderContext());
    getRenderContext()->blit(mpDisplay->getSRV(), getTargetFbo()->getRenderTargetView(0));
    getDevice()->wait();
    mpComparisonLinear->captureToFile(0, 0, differencePath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    getTargetFbo()->getColorTexture(0)->captureToFile(
        0, 0, differenceDisplayPath, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
    mComparisonMode = originalComparisonMode;
    renderComposite(getRenderContext());
    getRenderContext()->blit(mpDisplay->getSRV(), getTargetFbo()->getRenderTargetView(0));
    const std::string methodId = mSelectedMethod >= 0 ? mMethods[mSelectedMethod].methodId : "diagnostic-exact-top-only";
    const std::string methodRoot = mSelectedMethod >= 0 ? mMethods[mSelectedMethod].root.string() : std::string();
    nlohmann::json manifest = {
        {"format_name", "ncls.viewer-capture"},
        {"format_version", 1},
        {"method_id", methodId},
        {"method_bundle", methodRoot},
        {"bundle_root", std::filesystem::absolute(mOptions.bundleRoot).string()},
        {"material_ir_sha256", ncls::layerStackHash(mMaterial)},
        {"material_program", materialPath.filename().string()},
        {"source_material_program", mMaterialPath.string()},
        {"environment", mEnvironmentPath.empty() ? std::string() : std::filesystem::absolute(mEnvironmentPath).string()},
        {"resolution", {mOutputWidth, mOutputHeight}},
        {"view_resolution", {mViewWidth, mOutputHeight}},
        {"object_mode", mObjectMode},
        {"reference_spp", mReferenceSpp},
        {"reference_samples_per_frame", mSamplesPerFrame},
        {"reference_max_depth", mMaxReferenceDepth},
        {"camera", {
            {"target", {mCamera.target.x, mCamera.target.y, mCamera.target.z}},
            {"yaw", mCamera.yaw}, {"pitch", mCamera.pitch}, {"distance", mCamera.distance},
            {"vertical_fov_degrees", mCamera.verticalFovDegrees},
        }},
        {"display", {{"comparison_mode", mComparisonMode}, {"split", mSplit}, {"exposure_ev", mExposure}, {"difference_scale", mDifferenceScale}}},
        {"lighting", {
            {"use_environment", mLighting.useEnvironment},
            {"environment_rotation", mLighting.environmentRotation},
            {"environment_intensity", mLighting.environmentIntensity},
            {"use_sun", mLighting.useSun},
            {"sun_direction", {mLighting.sunDirection.x, mLighting.sunDirection.y, mLighting.sunDirection.z}},
            {"sun_intensity", mLighting.sunIntensity},
            {"sun_color", {mLighting.sunColor.x, mLighting.sunColor.y, mLighting.sunColor.z}},
            {"use_point", mLighting.usePoint},
            {"point_position", {mLighting.pointPosition.x, mLighting.pointPosition.y, mLighting.pointPosition.z}},
            {"point_intensity", mLighting.pointIntensity},
            {"point_color", {mLighting.pointColor.x, mLighting.pointColor.y, mLighting.pointColor.z}},
            {"use_rectangle", mLighting.useRectangle},
            {"rectangle_center", {mLighting.rectangleCenter.x, mLighting.rectangleCenter.y, mLighting.rectangleCenter.z}},
            {"rectangle_axis_u", {mLighting.rectangleAxisU.x, mLighting.rectangleAxisU.y, mLighting.rectangleAxisU.z}},
            {"rectangle_axis_v", {mLighting.rectangleAxisV.x, mLighting.rectangleAxisV.y, mLighting.rectangleAxisV.z}},
            {"rectangle_intensity", mLighting.rectangleIntensity},
            {"rectangle_color", {mLighting.rectangleColor.x, mLighting.rectangleColor.y, mLighting.rectangleColor.z}},
        }},
        {"gpu_ms", {
            {"visibility", mVisibilityTiming.milliseconds}, {"reference", mReferenceTiming.milliseconds},
            {"prepare", mPrepareTiming.milliseconds}, {"lighting", mLightingTiming.milliseconds},
            {"composite", mCompositeTiming.milliseconds},
        }},
        {"files", {
            {"reference_linear", referencePath.filename().string()},
            {"approximation_linear", approximationPath.filename().string()},
            {"comparison_linear", comparisonPath.filename().string()},
            {"display", displayPath.filename().string()},
            {"difference_linear", differencePath.filename().string()},
            {"difference_display", differenceDisplayPath.filename().string()},
            {"material_program", materialPath.filename().string()},
            {"metrics_csv", metricsPath.filename().string()},
        }},
    };
    std::ofstream stream(manifestPath, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write capture manifest: " + manifestPath.string());
    stream << manifest.dump(2) << '\n';
    std::ofstream metrics(metricsPath, std::ios::binary | std::ios::trunc);
    if (!metrics) throw std::runtime_error("cannot write capture metrics: " + metricsPath.string());
    metrics << "method_id,width,height,reference_spp,visibility_ms,reference_ms,prepare_ms,lighting_ms,composite_ms\n";
    metrics << methodId << ',' << mOutputWidth << ',' << mOutputHeight << ',' << mReferenceSpp << ','
            << mVisibilityTiming.milliseconds << ',' << mReferenceTiming.milliseconds << ','
            << mPrepareTiming.milliseconds << ',' << mLightingTiming.milliseconds << ','
            << mCompositeTiming.milliseconds << '\n';
    mStatus = "capture 已保存: " + manifestPath.string();
}

bool NclsViewer::onKeyEvent(const KeyboardEvent& event)
{
    if (event.type != KeyboardEvent::Type::KeyPressed) return false;
    if (event.key == Input::Key::R) { resetCamera(); return true; }
    if (event.key == Input::Key::Space) { mFreezeReference = !mFreezeReference; return true; }
    return false;
}

bool NclsViewer::onMouseEvent(const MouseEvent& event)
{
    const float dividerX = mSplit * float(mOutputWidth);
    if (event.type == MouseEvent::Type::ButtonDown)
    {
        mLastMouse = event.pos;
        if (event.button == Input::MouseButton::Left && std::abs(event.screenPos.x - dividerX) < 8.f)
        {
            mDividerDragging = true;
            return true;
        }
        if (event.button == Input::MouseButton::Left) { mCameraDragging = true; return true; }
        if (event.button == Input::MouseButton::Middle || event.button == Input::MouseButton::Right)
        {
            mPanDragging = true;
            return true;
        }
    }
    else if (event.type == MouseEvent::Type::ButtonUp)
    {
        const bool handled = mDividerDragging || mCameraDragging || mPanDragging;
        if (event.button == Input::MouseButton::Left) { mDividerDragging = false; mCameraDragging = false; }
        if (event.button == Input::MouseButton::Middle || event.button == Input::MouseButton::Right) mPanDragging = false;
        return handled;
    }
    else if (event.type == MouseEvent::Type::Wheel)
    {
        mCamera.distance = std::clamp(mCamera.distance * std::exp(-0.12f * event.wheelDelta.y), 1.25f, 9.f);
        resetReference(true, true);
        return true;
    }
    else if (event.type == MouseEvent::Type::Move)
    {
        const float2 delta = event.pos - mLastMouse;
        mLastMouse = event.pos;
        if (mDividerDragging)
        {
            mSplit = std::clamp(event.screenPos.x / float(mOutputWidth), 0.1f, 0.9f);
            return true;
        }
        if (mCameraDragging)
        {
            mCamera.yaw -= delta.x * 6.f;
            mCamera.pitch = std::clamp(mCamera.pitch - delta.y * 3.f, -1.35f, 1.35f);
            resetReference(true, true);
            return true;
        }
        if (mPanDragging)
        {
            const float3 position = cameraPosition();
            const float3 forward = normalizedOr(mCamera.target - position, float3(0.f, 0.f, -1.f));
            const float3 right = normalizedOr(cross(forward, float3(0.f, 1.f, 0.f)), float3(1.f, 0.f, 0.f));
            const float3 up = cross(right, forward);
            mCamera.target += (-delta.x * right + delta.y * up) * (1.5f * mCamera.distance);
            resetReference(true, true);
            return true;
        }
    }
    return false;
}

void NclsViewer::onDroppedFile(const std::filesystem::path& path)
{
    if (std::filesystem::is_directory(path) || path.filename() == "manifest.json")
    {
        mOptions.bundleRoot = std::filesystem::is_directory(path) ? path : path.parent_path();
        scanBundles();
    }
    else if (path.extension() == ".json") loadMaterial(path);
    else loadEnvironment(path);
}

int runMain(int argc, char** argv)
{
    ViewerOptions options = parseOptions(argc, argv);
    SampleAppConfig config;
    config.deviceDesc.type = Device::Type::D3D12;
    config.windowDesc.title = "NeuralShading — 随机游走 reference / MethodBundle";
    config.windowDesc.width = options.width;
    config.windowDesc.height = options.height;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.enableVSync = true;
    config.colorFormat = ResourceFormat::BGRA8UnormSrgb;
    config.headless = options.headless;
    config.showUI = !options.headless;
    NclsViewer viewer(config, std::move(options));
    return viewer.run();
}

int main(int argc, char** argv)
{
    try
    {
        return runMain(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "NclsViewer fatal error: " << error.what() << '\n';
        return 1;
    }
    catch (...)
    {
        std::cerr << "NclsViewer fatal error: unknown exception\n";
        return 1;
    }
}
