#include "NclsViewer.h"

#include "Hash.h"

#include "Core/Platform/OS.h"
#include "Utils/Image/Bitmap.h"
#include "Utils/Math/FalcorMath.h"

#include <nlohmann/json.hpp>
#include <ImfRgbaFile.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace Falcor;

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
constexpr uint32_t kLegacyStateBytes = 176;
const Gui::DropdownList kObjectModes = {{0, "Sphere"}, {1, "Shader ball"}, {2, "Detail hero"}};
const Gui::DropdownList kComparisonModes = {
    {0, "Reference / method split"},
    {1, "Linear absolute error"},
    {2, "Linear relative error"},
    {3, "Amplified absolute error"},
};
const Gui::DropdownList kBaseKinds = {{1, "Rough conductor"}, {2, "Diffuse"}, {3, "Sheen"}};

float3 normalizedOr(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-12f ? value / std::sqrt(lengthSquared) : fallback;
}

std::string shortId(const std::string& value)
{
    return value.size() > 12 ? value.substr(0, 12) : value;
}

bool isSceneFile(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    for (const auto& filter : Scene::getFileExtensionFilters())
    {
        std::string supported = filter.ext;
        std::transform(supported.begin(), supported.end(), supported.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (!supported.empty() && supported.front() != '.') supported.insert(supported.begin(), '.');
        if (extension == supported) return true;
    }
    return false;
}

ref<Texture> loadHalfRgbaExr(ref<Device> device, const std::filesystem::path& path, bool generateMips)
{
    OPENEXR_IMF_NAMESPACE::RgbaInputFile file(path.string().c_str());
    const auto window = file.dataWindow();
    if (window.min.x != 0 || window.min.y != 0 || window.max.x < 0 || window.max.y < 0)
        throw std::runtime_error("MaterialX normal EXR requires a zero-origin data window: " + path.string());
    const uint32_t width = static_cast<uint32_t>(window.max.x + 1);
    const uint32_t height = static_cast<uint32_t>(window.max.y + 1);
    std::vector<OPENEXR_IMF_NAMESPACE::Rgba> pixels(size_t(width) * height);
    file.setFrameBuffer(pixels.data(), 1, width);
    file.readPixels(window.min.y, window.max.y);
    auto texture = device->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA16Float,
        1,
        generateMips ? Texture::kMaxPossible : 1,
        pixels.data(),
        ResourceBindFlags::ShaderResource);
    if (!texture) throw std::runtime_error("Falcor failed to upload MaterialX EXR: " + path.string());
    texture->setSourcePath(path);
    return texture;
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
        const uint32_t replayVersion = replay.value("format_version", 0u);
        if (replay.value("format_name", "") != "ncls.viewer-capture" || (replayVersion != 1u && replayVersion != 2u))
            throw std::runtime_error("unsupported replay manifest");
        const auto base = std::filesystem::absolute(options.replayPath).parent_path();
        const auto resolve = [&](const std::string& value) {
            if (value.empty()) return std::filesystem::path();
            const std::filesystem::path path(value);
            return path.is_absolute() ? path : base / path;
        };
        options.bundleRoot = resolve(replay.value("bundle_root", std::string()));
        options.materialPath = resolve(replay.value(
            "source_material", replay.value("material_program", std::string())));
        options.environmentPath = resolve(replay.value("environment", std::string()));
        options.referenceGeometryPath = resolve(replay.value("reference_geometry", std::string()));
        options.referenceGeometrySha256 = replay.value("reference_geometry_sha256", std::string());
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
        else if (argument == "--reference-geometry") options.referenceGeometryPath = value(index, "--reference-geometry");
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
                   "[--reference-geometry SCENE] "
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
    Sampler::Desc materialSamplerDesc;
    materialSamplerDesc
        .setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear)
        .setMaxAnisotropy(16)
        .setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    mpMaterialXSampler = getDevice()->createSampler(materialSamplerDesc);
    createDefaultEnvironment();
    mSourceGpu = createSourceGpuResources(mReferenceSource);
    mOpenPbrLuts = ncls::OpenPbrLuts::create(getDevice());
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
    if (!mOptions.referenceGeometryPath.empty()) loadScene(mOptions.referenceGeometryPath);
    if (!mOptions.materialPath.empty()) loadMaterial(mOptions.materialPath);
    if (!mOptions.environmentPath.empty()) loadEnvironment(mOptions.environmentPath);
    resizeResources(getTargetFbo()->getWidth(), getTargetFbo()->getHeight());
    scanBundles();
    if (!mOptions.requestedMethodId.empty())
    {
        int32_t requested = -1;
        if (mOptions.requestedMethodId != "none")
            for (uint32_t index = 0; index < mMethods.size(); ++index)
                if (mMethods[index].methodId == mOptions.requestedMethodId) requested = static_cast<int32_t>(index);
        if (requested < 0 && mOptions.requestedMethodId != "none" && mOptions.headless)
            throw std::runtime_error("replay MethodBundle did not pass compatibility/parity: " + mOptions.requestedMethodId);
        selectMethod(requested);
    }
    mStatus = mMethods.empty()
        ? "No compatible realtime bundle was found; showing a full-width reference."
        : "GPU-parity-validated MethodBundles found; the method selection starts empty.";
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

NclsViewer::SourceGpuResources NclsViewer::createSourceGpuResources(const ncls::ReferenceSource& source)
{
    const auto shaderResource = ResourceBindFlags::ShaderResource;
    SourceGpuResources resources;
    resources.pMaterial = getDevice()->createStructuredBuffer(
        sizeof(ncls::LayerStackIR), 1, shaderResource, MemoryType::DeviceLocal, &source.layerStack);

    const float3 zeroMerl(0.f);
    resources.pMerlBrdf = source.merlBrdf.empty()
        ? getDevice()->createStructuredBuffer(sizeof(float3), 1, shaderResource, MemoryType::DeviceLocal, &zeroMerl)
        : getDevice()->createStructuredBuffer(
            sizeof(std::array<float, 3>), static_cast<uint32_t>(source.merlBrdf.size()),
            shaderResource, MemoryType::DeviceLocal, source.merlBrdf.data());
    resources.pOpenPbrInputs = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(source.openPbrInputs.size()),
        shaderResource, MemoryType::DeviceLocal, source.openPbrInputs.data());
    resources.pMaterialXInputs = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(source.materialXInputs.size()),
        shaderResource, MemoryType::DeviceLocal, source.materialXInputs.data());

    auto loadTexture = [&](const std::filesystem::path& texturePath, bool srgb, bool generateMips,
                           bool convertToFloat16, const float4& fallback) {
        if (!texturePath.empty())
        {
            logInfo("Loading MaterialX texture '{}' (sRGB={})", texturePath, srgb);
            auto texture = convertToFloat16
                ? loadHalfRgbaExr(getDevice(), texturePath, generateMips)
                : Texture::createFromFile(getDevice(), texturePath, generateMips, srgb);
            if (!texture) throw std::runtime_error("Falcor failed to load MaterialX texture: " + texturePath.string());
            logInfo("Loaded MaterialX texture '{}' as {}x{} with {} mip(s)",
                texturePath.filename(), texture->getWidth(), texture->getHeight(), texture->getMipCount());
            return texture;
        }
        return getDevice()->createTexture2D(
            1, 1, ResourceFormat::RGBA32Float, 1, 1, &fallback, shaderResource);
    };
    // MaterialX samples srgb_texture in the encoded domain and applies the
    // explicit srgb_texture_to_lin_rec709 transform in the reference shader.
    resources.pMaterialXBaseColor = loadTexture(
        source.materialXBaseColorTexture, false, true, false, float4(.8f, .8f, .8f, 1.f));
    resources.pMaterialXRoughness = loadTexture(
        source.materialXRoughnessTexture, false, true, true, float4(.2f, .2f, .2f, 1.f));
    resources.pMaterialXMetalness = loadTexture(
        source.materialXMetalnessTexture, false, true, true, float4(0.f, 0.f, 0.f, 1.f));
    resources.pMaterialXNormalMap = loadTexture(
        source.materialXNormalTexture, false, true, true, float4(.5f, .5f, 1.f, 1.f));
    return resources;
}

void NclsViewer::onResize(uint32_t width, uint32_t height)
{
    if (width > 0 && height > 0 && (width != mOutputWidth || height != mOutputHeight)) resizeResources(width, height);
}

void NclsViewer::resizeResources(uint32_t width, uint32_t height)
{
    mOutputWidth = std::max(width, 2u);
    mOutputHeight = std::max(height, 1u);
    mViewWidth = hasActiveMethod() ? std::max(mOutputWidth / 2u, 1u) : mOutputWidth;
    const auto shaderUav = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    const auto gBufferFlags = shaderUav | ResourceBindFlags::RenderTarget;
    auto viewTexture = [&]() {
        return getDevice()->createTexture2D(mViewWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, shaderUav);
    };
    auto gBufferTexture = [&]() {
        return getDevice()->createTexture2D(
            mViewWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, gBufferFlags);
    };
    mpPositionDepth = gBufferTexture();
    mpNormal = gBufferTexture();
    mpTangent = gBufferTexture();
    mpViewDirection = gBufferTexture();
    mpMaterialXTexCoord = gBufferTexture();
    mpMaterialXTexCoordGrad = gBufferTexture();
    mpInstanceId = getDevice()->createTexture2D(
        mViewWidth, mOutputHeight, ResourceFormat::R32Uint, 1, 1, nullptr, gBufferFlags);
    mpSceneMaterialId = getDevice()->createTexture2D(
        mViewWidth, mOutputHeight, ResourceFormat::R32Uint, 1, 1, nullptr, gBufferFlags);
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
    rebuildSceneFbo();
    mReferencePing = 0;
    resetReference(true, true);
}

void NclsViewer::loadScene(const std::filesystem::path& requestedPath)
{
    namespace fs = std::filesystem;
    const fs::path path = fs::absolute(requestedPath).lexically_normal();
    if (!fs::is_regular_file(path)) throw std::runtime_error("scene does not exist: " + path.string());
    const std::string geometrySha256 = ncls::sha256FileHex(path);
    if (!mOptions.referenceGeometrySha256.empty()
        && mOptions.referenceGeometrySha256 != geometrySha256)
        throw std::runtime_error("scene SHA-256 does not match replay: " + path.string());

    auto scene = Scene::create(getDevice(), path);
    if (!scene || scene->getGeometryInstanceCount() == 0u)
        throw std::runtime_error("Falcor loaded a scene without renderable geometry: " + path.string());

    ProgramDesc program;
    program.addShaderModules(scene->getShaderModules());
    program.addShaderLibrary("NclsViewer/shaders/SceneVisibility.3d.slang").vsEntry("vsMain").psEntry("psMain");
    program.addTypeConformances(scene->getTypeConformances());
    auto visibilityPass = RasterPass::create(getDevice(), program, scene->getSceneDefines());

    mpScene = std::move(scene);
    mpSceneVisibilityPass = std::move(visibilityPass);
    mReferenceGeometryPath = path;
    mReferenceGeometrySha256 = geometrySha256;
    mInactiveSceneMaterials.clear();
    const auto& firstInstance = mpScene->getGeometryInstance(0u);
    mActiveSceneMaterial = firstInstance.materialID;
    mSelectedSceneInstance = 0u;
    mSelectedSceneMaterial = firstInstance.materialID;
    mSelectedSceneGeometryName = (firstInstance.getType() == Scene::GeometryType::TriangleMesh
        || firstInstance.getType() == Scene::GeometryType::DisplacedTriangleMesh)
        ? mpScene->getMeshName(firstInstance.geometryID)
        : "Geometry #" + std::to_string(firstInstance.geometryID);
    const auto firstMaterial = mpScene->getMaterial(MaterialID::fromSlang(firstInstance.materialID));
    mSelectedSceneMaterialName = firstMaterial ? firstMaterial->getName() : "Unnamed material";
    for (uint32_t materialId = 0u; materialId < mpScene->getMaterialCount(); ++materialId)
    {
        if (materialId == mActiveSceneMaterial) continue;
        MaterialSlotBinding binding;
        binding.source = ncls::makeDefaultReferenceSource();
        binding.gpu = createSourceGpuResources(binding.source);
        mInactiveSceneMaterials.emplace(materialId, std::move(binding));
    }

    const auto& bounds = mpScene->getSceneBounds();
    mCamera.target = bounds.center();
    mCamera.distance = std::max(2.4f * bounds.radius(), 0.01f);
    rebuildSceneFbo();
    resetReference(true, true);
    mStatus = "Loaded Falcor scene: " + path.string();
    logInfo("Loaded Falcor scene '{}' ({} instances, {} materials, SHA-256 {})",
        path, mpScene->getGeometryInstanceCount(), mpScene->getMaterialCount(), shortId(mReferenceGeometrySha256));
}

void NclsViewer::rebuildSceneFbo()
{
    if (!mpScene || !mpSceneVisibilityPass || !mpPositionDepth || mViewWidth == 0u || mOutputHeight == 0u)
    {
        mpSceneFbo.reset();
        mpSceneDepth.reset();
        return;
    }
    mpSceneDepth = getDevice()->createTexture2D(
        mViewWidth,
        mOutputHeight,
        ResourceFormat::D32Float,
        1,
        1,
        nullptr,
        ResourceBindFlags::DepthStencil);
    mpSceneFbo = Fbo::create(getDevice(), {
        mpPositionDepth,
        mpNormal,
        mpTangent,
        mpViewDirection,
        mpMaterialXTexCoord,
        mpMaterialXTexCoordGrad,
        mpInstanceId,
        mpSceneMaterialId,
    }, mpSceneDepth);
    mpSceneVisibilityPass->getState()->setFbo(mpSceneFbo);
}

void NclsViewer::syncSceneCamera()
{
    if (!mpScene) return;
    auto camera = mpScene->getCamera();
    camera->setPosition(cameraPosition());
    camera->setTarget(mCamera.target);
    camera->setUpVector(float3(0.f, 1.f, 0.f));
    camera->setAspectRatio(float(mViewWidth) / float(std::max(mOutputHeight, 1u)));
    camera->setFocalLength(fovYToFocalLength(
        mCamera.verticalFovDegrees * (3.14159265358979323846f / 180.f), camera->getFrameHeight()));
    const float radius = std::max(mpScene->getSceneBounds().radius(), 0.01f);
    camera->setDepthRange(std::max(radius / 1000.f, 0.0001f), std::max(20.f * radius, mCamera.distance + 4.f * radius));
}

const NclsViewer::MaterialSlotBinding* NclsViewer::inactiveSceneMaterial(uint32_t materialId) const
{
    const auto found = mInactiveSceneMaterials.find(materialId);
    return found == mInactiveSceneMaterials.end() ? nullptr : &found->second;
}

void NclsViewer::activateSceneMaterial(uint32_t materialId)
{
    if (!mpScene || materialId == mActiveSceneMaterial || materialId >= mpScene->getMaterialCount()) return;
    auto found = mInactiveSceneMaterials.find(materialId);
    if (found == mInactiveSceneMaterials.end()) return;

    MaterialSlotBinding previous;
    previous.source = std::move(mReferenceSource);
    previous.gpu = std::move(mSourceGpu);
    previous.materialPath = std::move(mMaterialPath);
    previous.displayName = std::move(mMaterialDisplayName);

    mReferenceSource = std::move(found->second.source);
    mSourceGpu = std::move(found->second.gpu);
    mMaterialPath = std::move(found->second.materialPath);
    mMaterialDisplayName = std::move(found->second.displayName);
    mInactiveSceneMaterials.erase(found);
    mInactiveSceneMaterials.emplace(mActiveSceneMaterial, std::move(previous));
    mActiveSceneMaterial = materialId;
    mSelectedInterface = 0u;
    resetReference(false, true);

    if (!allMaterialsSupportCurrentCompiler()) selectMethod(-1);
}

void NclsViewer::updateMaterialBuffer()
{
    if (mReferenceSource.family != ncls::ReferenceFamily::LayerStack)
        throw std::runtime_error("current source material is not a LayerStack material");
    ncls::validateLayerStack(mMaterial);
    mReferenceSource.sourceSha256 = ncls::layerStackHash(mMaterial);
    mSourceGpu.pMaterial->setBlob(&mMaterial, 0, sizeof(mMaterial));
    resetReference(false, true);
}

void NclsViewer::updateReferenceSourceBuffer()
{
    switch (mReferenceSource.family)
    {
    case ncls::ReferenceFamily::LayerStack:
        updateMaterialBuffer();
        return;
    case ncls::ReferenceFamily::OpenPbr:
        mSourceGpu.pOpenPbrInputs->setBlob(
            mReferenceSource.openPbrInputs.data(), 0, sizeof(mReferenceSource.openPbrInputs));
        break;
    case ncls::ReferenceFamily::MaterialX:
        mSourceGpu.pMaterialXInputs->setBlob(
            mReferenceSource.materialXInputs.data(), 0, sizeof(mReferenceSource.materialXInputs));
        break;
    case ncls::ReferenceFamily::Merl:
        return;
    }
    resetReference(false, false);
}

bool NclsViewer::allMaterialsSupportCurrentCompiler() const
{
    if (!mReferenceSource.supportsCurrentCompiler()) return false;
    for (const auto& [materialId, binding] : mInactiveSceneMaterials)
        if (!binding.source.supportsCurrentCompiler()) return false;
    return true;
}

bool NclsViewer::hasActiveMethod() const
{
    return allMaterialsSupportCurrentCompiler()
        && mSelectedMethod >= 0
        && mSelectedMethod < static_cast<int32_t>(mMethods.size());
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
    if (mpScene)
    {
        const auto& bounds = mpScene->getSceneBounds();
        mCamera.target = bounds.center();
        mCamera.distance = std::max(2.4f * bounds.radius(), 0.01f);
    }
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
    selectMethod(allMaterialsSupportCurrentCompiler() ? selection : -1);
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
    const bool previouslyActive = hasActiveMethod();
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
    if (previouslyActive != hasActiveMethod() && mOutputWidth > 0u && mOutputHeight > 0u)
        resizeResources(mOutputWidth, mOutputHeight);
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
    const bool useScene = mpSceneFbo && mpScene && mpSceneVisibilityPass;
    auto root = mpVisibilityPass->getRootVar();
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gMaterialXTexCoord"] = mpMaterialXTexCoord;
    root["gMaterialXTexCoordGrad"] = mpMaterialXTexCoordGrad;
    root["gInstanceId"] = mpInstanceId;
    root["gMaterialId"] = mpSceneMaterialId;
    auto constants = root["VisibilityCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gObjectMode"] = mObjectMode;
    constants["gClearOnly"] = uint32_t(useScene);
    constants["gVerticalFovRadians"] = mCamera.verticalFovDegrees * (3.14159265358979323846f / 180.f);
    constants["gCameraPosition"] = cameraPosition();
    constants["gCameraTarget"] = mCamera.target;
    beginTiming(mVisibilityTiming);
    mpVisibilityPass->execute(pRenderContext, mViewWidth, mOutputHeight);
    if (useScene)
    {
        syncSceneCamera();
        mpScene->update(pRenderContext, getGlobalClock().getTime());
        pRenderContext->clearDsv(mpSceneFbo->getDepthStencilView().get(), 1.f, 0u, true, false);
        std::string extension = mReferenceGeometryPath.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        mpSceneVisibilityPass->getRootVar()["SceneVisibilityCB"]["gFlipTexCoordV"] = uint32_t(extension == ".obj");
        mpScene->rasterize(
            pRenderContext, mpSceneVisibilityPass->getState().get(), mpSceneVisibilityPass->getVars().get(),
            RasterizerState::CullMode::None);
    }
    endTiming(mVisibilityTiming);
    mVisibilityDirty = false;
}

void NclsViewer::renderReference(RenderContext* pRenderContext)
{
    const uint32_t next = 1u - mReferencePing;
    auto root = mpReferencePass->getRootVar();
    root["gMaterialXSampler"] = mpMaterialXSampler;
    mOpenPbrLuts.bind(root);
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gMaterialXTexCoord"] = mpMaterialXTexCoord;
    root["gMaterialXTexCoordGrad"] = mpMaterialXTexCoordGrad;
    root["gMaterialId"] = mpSceneMaterialId;
    root["gPreviousReference"] = mpReference[mReferencePing];
    root["gNextReference"] = mpReference[next];
    root["gEnvironment"] = mpEnvironment;
    root["gLinearSampler"] = mpLinearSampler;
    auto constants = root["ReferenceCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gFrameIndex"] = mFrameIndex;
    constants["gUseRasterGeometry"] = uint32_t(mpScene != nullptr);
    constants["gReferenceSpp"] = mReferenceSpp;
    constants["gSamplesThisFrame"] = mSamplesPerFrame;
    constants["gMaxDepth"] = mMaxReferenceDepth;
    constants["gResetAccumulation"] = uint32_t(mResetAccumulation);
    const bool accumulate = !mCameraDragging && !mPanDragging;
    constants["gAccumulate"] = uint32_t(accumulate);
    bindLighting(root, "ReferenceCB");

    auto executeSource = [&](const ncls::ReferenceSource& source, const SourceGpuResources& gpu,
                             uint32_t targetMaterialId, bool initializeOutput) {
        root["gMaterials"] = gpu.pMaterial;
        root["gMerlBrdf"] = gpu.pMerlBrdf;
        root["gOpenPbrInputs"] = gpu.pOpenPbrInputs;
        root["gMaterialXInputs"] = gpu.pMaterialXInputs;
        root["gMaterialXBaseColor"] = gpu.pMaterialXBaseColor;
        root["gMaterialXRoughness"] = gpu.pMaterialXRoughness;
        root["gMaterialXMetalness"] = gpu.pMaterialXMetalness;
        root["gMaterialXNormalMap"] = gpu.pMaterialXNormalMap;
        constants["gReferenceFamily"] = static_cast<uint32_t>(source.family);
        constants["gOpenPbrColorSpace"] = source.openPbrColorSpace;
        constants["gTargetMaterialId"] = targetMaterialId;
        constants["gInitializeOutput"] = uint32_t(initializeOutput);
        mpReferencePass->execute(pRenderContext, mViewWidth, mOutputHeight);
    };

    beginTiming(mReferenceTiming);
    if (!mpScene) executeSource(mReferenceSource, mSourceGpu, 0u, true);
    else
    {
        executeSource(mReferenceSource, mSourceGpu, mActiveSceneMaterial, true);
        for (const auto& [materialId, binding] : mInactiveSceneMaterials)
            executeSource(binding.source, binding.gpu, materialId, false);
    }
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
    root["gWeights"] = mpWeights;
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gMaterialId"] = mpSceneMaterialId;
    root["gStates"] = mpStates;
    auto constants = root["PrepareCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gMethodMode"] = mSelectedMethod >= 0 ? 1u : 0u;
    constants["gWidth"] = mSelectedMethod >= 0 ? mMethods[mSelectedMethod].width : 8u;
    constants["gUseScene"] = uint32_t(mpScene != nullptr);
    auto executeMaterial = [&](const SourceGpuResources& gpu, uint32_t materialId) {
        root["gMaterials"] = gpu.pMaterial;
        constants["gTargetMaterialId"] = materialId;
        mpPreparePass->execute(pRenderContext, mViewWidth, mOutputHeight);
    };
    beginTiming(mPrepareTiming);
    if (!mpScene) executeMaterial(mSourceGpu, 0u);
    else
    {
        executeMaterial(mSourceGpu, mActiveSceneMaterial);
        for (const auto& [materialId, binding] : mInactiveSceneMaterials)
            executeMaterial(binding.gpu, materialId);
    }
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
    constants["gHasApproximation"] = uint32_t(hasActiveMethod());
    beginTiming(mCompositeTiming);
    mpCompositePass->execute(pRenderContext, mOutputWidth, mOutputHeight);
    endTiming(mCompositeTiming);
}

void NclsViewer::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (mVisibilityDirty) renderVisibility(pRenderContext);
    if (!mFreezeReference) renderReference(pRenderContext);
    if (hasActiveMethod())
    {
        if (mPrepareDirty) renderPrepare(pRenderContext);
        renderApproximation(pRenderContext);
    }
    renderComposite(pRenderContext);
    pRenderContext->blit(mpDisplay->getSRV(), pTargetFbo->getRenderTargetView(0));
    ++mFrameIndex;

    if (mOptions.headless && ++mRenderedFrames >= mOptions.frameCount)
    {
        capture(mOptions.captureManifest);
        shutdown(0);
    }
}

void NclsViewer::renderOpenPbrUi(Gui::Window& window)
{
    auto& values = mReferenceSource.openPbrInputs;
    bool changed = false;
    auto color = [&](const char* label, uint32_t offset) {
        float3 value(values[offset], values[offset + 1u], values[offset + 2u]);
        if (!window.rgbColor(label, value)) return false;
        values[offset] = value.x;
        values[offset + 1u] = value.y;
        values[offset + 2u] = value.z;
        return true;
    };
    auto scalar = [&](const char* label, uint32_t offset, float minimum, float maximum, float step) {
        return window.var(label, values[offset], minimum, maximum, step);
    };

    window.text("OpenPBR 1.1.1 native parameters (in-memory edit)");
    window.text("Base");
    changed |= scalar("base_weight", 0, 0.f, 1.f, .01f);
    changed |= color("base_color", 1);
    changed |= scalar("base_diffuse_roughness", 4, 0.f, 1.f, .01f);
    changed |= scalar("base_metalness", 5, 0.f, 1.f, .01f);

    window.text("Subsurface");
    changed |= scalar("subsurface_weight", 6, 0.f, 1.f, .01f);
    changed |= color("subsurface_color", 7);
    changed |= scalar("subsurface_radius", 10, 0.f, 100.f, .01f);
    changed |= color("subsurface_radius_scale", 11);
    changed |= scalar("subsurface_scatter_anisotropy", 14, -1.f, 1.f, .01f);

    window.text("Specular");
    changed |= scalar("specular_weight", 15, 0.f, 1.f, .01f);
    changed |= color("specular_color", 16);
    changed |= scalar("specular_roughness", 19, .001f, 1.f, .005f);
    changed |= scalar("specular_roughness_anisotropy", 20, 0.f, 1.f, .01f);
    changed |= scalar("specular_ior", 21, 1.001f, 3.f, .01f);
    float specularRotation = std::atan2(values[23], values[22]);
    if (window.var("specular_anisotropy_rotation", specularRotation, -3.14159f, 3.14159f, .01f))
    {
        values[22] = std::cos(specularRotation);
        values[23] = std::sin(specularRotation);
        changed = true;
    }

    window.text("Coat / Fuzz");
    changed |= scalar("coat_weight", 24, 0.f, 1.f, .01f);
    changed |= color("coat_color", 25);
    changed |= scalar("coat_roughness", 28, .001f, 1.f, .005f);
    changed |= scalar("coat_roughness_anisotropy", 29, 0.f, 1.f, .01f);
    changed |= scalar("coat_ior", 30, 1.001f, 3.f, .01f);
    changed |= scalar("coat_darkening", 31, 0.f, 1.f, .01f);
    float coatRotation = std::atan2(values[33], values[32]);
    if (window.var("coat_anisotropy_rotation", coatRotation, -3.14159f, 3.14159f, .01f))
    {
        values[32] = std::cos(coatRotation);
        values[33] = std::sin(coatRotation);
        changed = true;
    }
    changed |= scalar("fuzz_weight", 34, 0.f, 1.f, .01f);
    changed |= color("fuzz_color", 35);
    changed |= scalar("fuzz_roughness", 38, .001f, 1.f, .005f);

    window.text("Transmission / Thin film");
    changed |= scalar("transmission_weight", 39, 0.f, 1.f, .01f);
    changed |= color("transmission_color", 40);
    changed |= scalar("transmission_depth", 43, 0.f, 100.f, .01f);
    changed |= color("transmission_scatter", 44);
    changed |= scalar("transmission_scatter_anisotropy", 47, -1.f, 1.f, .01f);
    changed |= scalar("transmission_dispersion_scale", 48, 0.f, 1.f, .01f);
    changed |= scalar("transmission_dispersion_abbe_number", 49, 1.f, 100.f, .1f);
    changed |= scalar("thin_film_weight", 50, 0.f, 1.f, .01f);
    changed |= scalar("thin_film_thickness", 51, 0.f, 2000.f, 1.f);
    changed |= scalar("thin_film_ior", 52, 1.001f, 3.f, .01f);

    window.text("Emission / Geometry");
    changed |= scalar("emission_luminance", 53, 0.f, 100000.f, 1.f);
    changed |= color("emission_color", 54);
    changed |= scalar("geometry_opacity", 57, 0.f, 1.f, .01f);
    bool thinWalled = values[58] != 0.f;
    if (window.checkbox("geometry_thin_walled", thinWalled))
    {
        values[58] = thinWalled ? 1.f : 0.f;
        changed = true;
    }
    if (changed)
    {
        updateReferenceSourceBuffer();
        mStatus = "OpenPBR parameters updated; reference accumulation reset.";
    }
}

void NclsViewer::renderMaterialXUi(Gui::Window& window)
{
    auto& values = mReferenceSource.materialXInputs;
    bool changed = false;
    auto color = [&](const char* label, uint32_t offset) {
        float3 value(values[offset], values[offset + 1u], values[offset + 2u]);
        if (!window.rgbColor(label, value)) return false;
        values[offset] = value.x;
        values[offset + 1u] = value.y;
        values[offset + 2u] = value.z;
        return true;
    };

    window.text("MaterialX standard_surface native inputs (in-memory edit)");
    changed |= window.var("base", values[0], 0.f, 1.f, .01f);
    if (values[6] == 0.f) changed |= color("base_color", 1);
    else window.text("base_color: driven by the source texture");
    changed |= window.var("diffuse_roughness", values[4], 0.f, 1.f, .01f);
    if (values[7] == 0.f) changed |= window.var("metalness", values[5], 0.f, 1.f, .01f);
    else window.text("metalness: driven by the source texture");
    changed |= window.var("specular", values[8], 0.f, 1.f, .01f);
    changed |= color("specular_color", 9);
    if (values[13] == 0.f) changed |= window.var("specular_roughness", values[12], .001f, 1.f, .005f);
    else window.text("specular_roughness: driven by the source texture");
    changed |= window.var("specular_IOR", values[14], 1.001f, 3.f, .01f);
    changed |= window.var("specular_anisotropy", values[15], 0.f, .98f, .01f);
    changed |= window.var("specular_rotation", values[16], 0.f, 1.f, .01f);
    if (values[18] != 0.f) changed |= window.var("normal scale", values[17], 0.f, 4.f, .01f);
    else window.text("normal: no texture connected");
    changed |= window.var("emission", values[19], 0.f, 1000.f, .01f);
    changed |= color("emission_color", 20);
    changed |= window.var("opacity", values[23], 0.f, 1.f, .01f);
    if (changed)
    {
        updateReferenceSourceBuffer();
        mStatus = "MaterialX parameters updated; reference accumulation reset.";
    }
}

void NclsViewer::renderMaterialUi(Gui::Window& window)
{
    if (mReferenceSource.family != ncls::ReferenceFamily::LayerStack)
    {
        window.text("Source family: " + std::string(mReferenceSource.familyId()));
        window.text("Native reference: " + mReferenceSource.displayName);
        window.text("Source file: " + mReferenceSource.sourcePath.string());
        window.text("SHA-256: " + shortId(mReferenceSource.sourceSha256));
        if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX)
        {
            window.text("Falcor query: MaterialX 1.39.4 standard_surface + source textures");
            window.text(mReferenceGeometryPath.empty()
                ? "Geometry contract: no scene specified; using the analytic-sphere fallback"
                : "Geometry contract: Falcor rasterizes the replay scene; SHA-256 " + shortId(mReferenceGeometrySha256));
            window.text("The displacement graph remains in the source document and is outside this surface-response query");
        }
        if (mReferenceSource.family == ncls::ReferenceFamily::OpenPbr) renderOpenPbrUi(window);
        else if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX) renderMaterialXUi(window);
        else window.text("MERL is a measured BRDF table with no continuous native controls; select another measurement to switch material.");
        window.text("This source material retains its native representation; no compatible approximation compiler is available yet.");
        return;
    }
    window.text("Material program (edits normalized LayerStackIR inputs, not a K2 packet)");
    mSelectedInterface = std::min(mSelectedInterface, mMaterial.interfaceCount - 1);
    Gui::DropdownList layers;
    for (uint32_t index = 0; index < mMaterial.interfaceCount; ++index)
    {
        const bool base = index + 1 == mMaterial.interfaceCount;
        layers.push_back({index, (base ? "Base " : "Coat ") + std::to_string(index)});
    }
    window.dropdown("Current interface", layers, mSelectedInterface);
    bool changed = false;
    if (window.button("Add dielectric coat"))
    {
        changed |= ncls::addDielectricCoat(mMaterial);
        mSelectedInterface = mMaterial.interfaceCount - 2;
    }
    if (mSelectedInterface + 1 < mMaterial.interfaceCount)
    {
        if (window.button("Delete current coat", true))
        {
            changed |= ncls::removeCoat(mMaterial, mSelectedInterface);
            mSelectedInterface = std::min(mSelectedInterface, mMaterial.interfaceCount - 1);
        }
        if (window.button("Move up", true)) changed |= ncls::moveCoat(mMaterial, mSelectedInterface, -1);
        if (window.button("Move down", true)) changed |= ncls::moveCoat(mMaterial, mSelectedInterface, 1);
    }
    auto& interfaceValue = mMaterial.interfaces[mSelectedInterface];
    const bool isBase = mSelectedInterface + 1 == mMaterial.interfaceCount;
    if (isBase)
    {
        uint32_t kind = interfaceValue.kind;
        if (window.dropdown("Base type", kBaseKinds, kind))
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
        changed |= window.var("Tangent rotation (rad)", interfaceValue.tangentRotation, -3.14159f, 3.14159f, 0.01f);
    }
    if (kind == ncls::InterfaceKind::RoughDielectric)
        changed |= window.var("Relative IOR", interfaceValue.relativeIor, 1.001f, 3.f, 0.01f);
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
        if (window.rgbColor("Color", color))
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
        mediumChanged |= window.var("Medium extinction (1/unit)", extinction, 0.f, 6.f, 0.01f);
        mediumChanged |= window.rgbColor("Medium scattering albedo", albedo);
        mediumChanged |= window.var("Phase-function g", medium.g, -0.95f, 0.95f, 0.01f);
        mediumChanged |= window.var("Thickness", medium.thickness, 0.f, 2.f, 0.01f);
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
        try { updateMaterialBuffer(); mStatus = "Material updated; reference accumulation reset."; }
        catch (const std::exception& error) { mStatus = error.what(); }
    }
    window.text("IR SHA-256: " + shortId(ncls::layerStackHash(mMaterial)));
}

void NclsViewer::onGuiRender(Gui* pGui)
{
    Gui::Window window(pGui, "NeuralShading Material Comparison", {410, 850}, {12, 12});
    if (allMaterialsSupportCurrentCompiler())
    {
        Gui::DropdownList methodList = {{0, "None (reference only)"}};
        for (uint32_t index = 0; index < mMethods.size(); ++index)
            methodList.push_back({index + 1, mMethods[index].displayName + " [" + shortId(mMethods[index].methodId) + "]"});
        if (window.dropdown("Right-side method", methodList, mMethodUiValue)) selectMethod(int32_t(mMethodUiValue) - 1);
        if (window.button("Rescan MethodBundles")) scanBundles();
    }
    else window.text("Right-side method: one or more source-material slots have no compatible compiler; showing reference only.");
    if (mSelectedMethod >= 0)
    {
        const auto& method = mMethods[mSelectedMethod];
        window.text("method: " + shortId(method.methodId) + " / backend v" + std::to_string(method.backendVersion));
        window.text("Parameters: " + std::to_string(method.parameterCount) + ", state: " + std::to_string(method.stateBytesPerPixel) + " B/pixel");
    }
    if (!mBundleFailures.empty()) window.text("Rejected bundles: " + std::to_string(mBundleFailures.size()) + " (see log/status for details)");

    window.text("Reference: " + std::string(mReferenceSource.familyId()));
    window.var("Samples per frame", mSamplesPerFrame, 1u, 16u);
    if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack
        && window.var("Max random-walk depth", mMaxReferenceDepth, 4u, 128u)) resetReference(false, false);
    window.checkbox("Freeze reference", mFreezeReference);
    if (window.button("Clear accumulation")) resetReference(false, false);
    window.text("spp: " + std::to_string(mReferenceSpp) + ", elapsed: " + fmt::format("{:.2f}s", mAccumulationSeconds));
    window.text("Noise proxy 1/sqrt(spp): " + fmt::format("{:.4f}", 1.f / std::sqrt(float(std::max(mReferenceSpp, 1u)))));

    bool physicalChanged = false;
    if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX)
    {
        if (mObjectMode != 0u) { mObjectMode = 0u; physicalChanged = true; }
        window.text(mReferenceGeometryPath.empty()
            ? "Preview object: analytic-sphere fallback (parity validation requires an explicit scene)"
            : "Preview object: Falcor scene " + mReferenceGeometryPath.filename().string());
    }
    else if (!mpScene) physicalChanged |= window.dropdown("Preview object", kObjectModes, mObjectMode);
    if (window.button("Load scene"))
    {
        std::filesystem::path path;
        if (openFileDialog(Scene::getFileExtensionFilters(), path))
        {
            try { loadScene(path); }
            catch (const std::exception& error) { mStatus = "Failed to load scene: " + std::string(error.what()); }
        }
    }
    if (mpScene)
    {
        window.text("Scene: " + mReferenceGeometryPath.filename().string());
        if (mSelectedSceneInstance != std::numeric_limits<uint32_t>::max())
        {
            window.text("Selected instance: " + std::to_string(mSelectedSceneInstance));
            window.text("Geometry: " + mSelectedSceneGeometryName);
            window.text("Scene material: " + mSelectedSceneMaterialName + " (#" + std::to_string(mSelectedSceneMaterial) + ")");
        }
        else window.text("Click an object to select its material slot.");
    }
    const float sceneRadius = mpScene ? std::max(mpScene->getSceneBounds().radius(), 0.01f) : 1.f;
    const float minimumDistance = mpScene ? sceneRadius * 0.01f : 1.25f;
    const float maximumDistance = mpScene ? sceneRadius * 20.f : 9.f;
    physicalChanged |= window.var("Camera distance", mCamera.distance, minimumDistance, maximumDistance, sceneRadius * 0.002f);
    physicalChanged |= window.var("Vertical FOV", mCamera.verticalFovDegrees, 12.f, 90.f, 0.5f);
    if (physicalChanged) resetReference(true, true);
    if (window.button("Reset camera")) resetCamera();

    if (hasActiveMethod())
    {
        window.dropdown("Comparison display", kComparisonModes, mComparisonMode);
        window.var("Split position", mSplit, 0.1f, 0.9f, 0.005f);
    }
    else
    {
        mComparisonMode = 0u;
        window.text("Display mode: full-width reference preview");
    }
    window.var("Shared exposure EV", mExposure, -8.f, 8.f, 0.05f);
    if (mComparisonMode == 3u) window.var("Error amplification", mDifferenceScale, 1.f, 100.f, 0.5f);

    bool lightChanged = false;
    lightChanged |= window.checkbox("HDRI / environment", mLighting.useEnvironment);
    lightChanged |= window.var("Environment rotation", mLighting.environmentRotation, -3.14159f, 3.14159f, 0.01f);
    lightChanged |= window.var("Environment intensity", mLighting.environmentIntensity, 0.f, 20.f, 0.02f);
    if (window.button("Load HDRI"))
    {
        std::filesystem::path path;
        if (openFileDialog(Bitmap::getFileDialogFilters(ResourceFormat::RGBA32Float), path)) loadEnvironment(path);
    }
    lightChanged |= window.checkbox("Directional light", mLighting.useSun);
    lightChanged |= window.var("Directional-light direction", mLighting.sunDirection, -1.f, 1.f, 0.01f);
    lightChanged |= window.var("Directional-light intensity", mLighting.sunIntensity, 0.f, 50.f, 0.05f);
    lightChanged |= window.rgbColor("Directional-light color", mLighting.sunColor);
    lightChanged |= window.checkbox("Point light", mLighting.usePoint);
    lightChanged |= window.var("Point-light position", mLighting.pointPosition, -10.f, 10.f, 0.02f);
    lightChanged |= window.var("Point-light intensity", mLighting.pointIntensity, 0.f, 100.f, 0.1f);
    lightChanged |= window.rgbColor("Point-light color", mLighting.pointColor);
    lightChanged |= window.checkbox("Rectangle light", mLighting.useRectangle);
    lightChanged |= window.var("Rectangle-light center", mLighting.rectangleCenter, -10.f, 10.f, 0.02f);
    lightChanged |= window.var("Rectangle-light half-axis U", mLighting.rectangleAxisU, -3.f, 3.f, 0.02f);
    lightChanged |= window.var("Rectangle-light half-axis V", mLighting.rectangleAxisV, -3.f, 3.f, 0.02f);
    lightChanged |= window.var("Rectangle-light intensity", mLighting.rectangleIntensity, 0.f, 100.f, 0.1f);
    lightChanged |= window.rgbColor("Rectangle-light color", mLighting.rectangleColor);
    if (lightChanged) resetReference(false, false);

    renderMaterialUi(window);
    if (window.button("Switch source material / family"))
    {
        std::filesystem::path path;
        if (openFileDialog({
                {"json", "MaterialProgram / OpenPBR JSON"},
                {"binary", "MERL BRDF table"},
                {"mtlx", "MaterialX document"}}, path))
            loadMaterial(path);
    }
    if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack
        && window.button("Save MaterialProgram", true))
    {
        std::filesystem::path path = mMaterialPath;
        if (saveFileDialog({{"json", "MaterialProgram JSON"}}, path)) saveMaterial(path);
    }
    if (window.button("Save full capture"))
    {
        std::filesystem::path path = "capture.json";
        if (saveFileDialog({{"json", "Capture manifest"}}, path)) capture(path);
    }

    window.text("GPU ms (asynchronous timestamp)");
    window.text(fmt::format(
        "visibility {:.3f} | reference {:.3f}\nprepare {:.3f} | lighting {:.3f} | composite {:.3f}",
        mVisibilityTiming.milliseconds,
        mReferenceTiming.milliseconds,
        mPrepareTiming.milliseconds,
        mLightingTiming.milliseconds,
        mCompositeTiming.milliseconds));
    window.text("Controls: left-drag orbit; middle/right-drag pan; wheel dolly; drag divider; Space freezes reference.");
    if (!mStatus.empty()) window.text("Status: " + mStatus);
}

void NclsViewer::loadMaterial(const std::filesystem::path& path)
{
    const bool viewWasSplit = hasActiveMethod();
    try
    {
        logInfo("Loading source material '{}'", path);
        auto source = ncls::loadReferenceSource(path);
        auto gpu = createSourceGpuResources(source);
        mReferenceSource = std::move(source);
        mSourceGpu = std::move(gpu);
        logInfo("Loaded source metadata: family='{}' hash='{}'", mReferenceSource.familyId(), shortId(mReferenceSource.sourceSha256));
        mMaterialDisplayName = mReferenceSource.displayName;
        mMaterialPath = path;
        if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack)
        {
            updateMaterialBuffer();
            mStatus = "Loaded LayerStack MaterialProgram: " + path.string();
        }
        else if (mReferenceSource.family == ncls::ReferenceFamily::Merl)
        {
            selectMethod(-1);
            resetReference(false, true);
            mStatus = "Loaded native MERL measurement into the Falcor reference pass: " + path.string();
        }
        else if (mReferenceSource.family == ncls::ReferenceFamily::OpenPbr)
        {
            selectMethod(-1);
            resetReference(false, true);
            mStatus = "Loaded OpenPBR 1.1.1 resolved-input material into the Falcor reference pass: " + path.string();
        }
        else if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX)
        {
            mObjectMode = 0u; // Only used by the analytic fallback when no replay scene is present.
            selectMethod(-1);
            resetReference(true, true);
            mStatus = "Loaded MaterialX standard_surface from the source .mtlx and 4K textures into the Falcor reference pass; "
                + (mReferenceGeometryPath.empty() ? std::string("currently using the analytic-sphere fallback: ") : std::string("currently rasterizing the replay scene: "))
                + path.string();
        }
        if (!allMaterialsSupportCurrentCompiler()) selectMethod(-1);
        if (viewWasSplit != hasActiveMethod() && mOutputWidth > 0u && mOutputHeight > 0u)
            resizeResources(mOutputWidth, mOutputHeight);
    }
    catch (const std::exception& error)
    {
        mStatus = "Failed to load source material: " + std::string(error.what());
        if (mOptions.headless) throw;
    }
}

void NclsViewer::saveMaterial(const std::filesystem::path& path)
{
    try
    {
        ncls::saveMaterialProgram(path, mMaterial, mMaterialDisplayName);
        mMaterialPath = path;
        mStatus = "Saved MaterialProgram: " + path.string();
    }
    catch (const std::exception& error) { mStatus = "Save failed: " + std::string(error.what()); }
}

void NclsViewer::loadEnvironment(const std::filesystem::path& path)
{
    auto texture = Texture::createFromFile(getDevice(), path, true, false);
    if (!texture)
    {
        mStatus = "Failed to load HDRI: " + path.string();
        return;
    }
    mpEnvironment = texture;
    mEnvironmentPath = path;
    resetReference(false, false);
    mStatus = "Loaded HDRI: " + path.string();
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
    const bool approximationAvailable = hasActiveMethod();
    if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack)
        ncls::saveMaterialProgram(materialPath, mMaterial, mMaterialDisplayName);
    mpReference[mReferencePing]->captureToFile(0, 0, referencePath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    if (approximationAvailable)
        mpApproximation->captureToFile(0, 0, approximationPath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    mpComparisonLinear->captureToFile(0, 0, comparisonPath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    getTargetFbo()->getColorTexture(0)->captureToFile(
        0, 0, displayPath, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
    if (approximationAvailable)
    {
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
    }
    const std::string methodId = !approximationAvailable
        ? "none"
        : mMethods[mSelectedMethod].methodId;
    const std::string methodRoot = approximationAvailable ? mMethods[mSelectedMethod].root.string() : std::string();
    nlohmann::json sceneMaterialBindings = nlohmann::json::array();
    auto appendBinding = [&](uint32_t materialId, const ncls::ReferenceSource& source, bool active) {
        std::string sceneMaterialName;
        if (mpScene && materialId < mpScene->getMaterialCount())
        {
            const auto sceneMaterial = mpScene->getMaterial(MaterialID(materialId));
            if (sceneMaterial) sceneMaterialName = sceneMaterial->getName();
        }
        sceneMaterialBindings.push_back({
            {"material_id", materialId},
            {"scene_material_name", sceneMaterialName},
            {"active", active},
            {"source_family_id", source.familyId()},
            {"source_sha256", source.family == ncls::ReferenceFamily::LayerStack
                ? ncls::layerStackHash(source.layerStack) : source.sourceSha256},
            {"source_path", source.sourcePath.string()},
        });
    };
    if (mpScene)
    {
        appendBinding(mActiveSceneMaterial, mReferenceSource, true);
        for (const auto& [materialId, binding] : mInactiveSceneMaterials)
            appendBinding(materialId, binding.source, false);
    }
    nlohmann::json manifest = {
        {"format_name", "ncls.viewer-capture"},
        {"format_version", 2},
        {"method_id", methodId},
        {"method_bundle", methodRoot},
        {"bundle_root", std::filesystem::absolute(mOptions.bundleRoot).string()},
        {"source_material_family_id", mReferenceSource.familyId()},
        {"source_material_sha256", mReferenceSource.sourceSha256},
        {"source_material", mReferenceSource.family == ncls::ReferenceFamily::LayerStack
            ? materialPath.filename().string()
            : mReferenceSource.sourcePath.string()},
        {"material_ir_sha256", mReferenceSource.family == ncls::ReferenceFamily::LayerStack ? ncls::layerStackHash(mMaterial) : std::string()},
        {"material_program", mReferenceSource.family == ncls::ReferenceFamily::LayerStack ? materialPath.filename().string() : std::string()},
        {"approximation_available", approximationAvailable},
        {"environment", mEnvironmentPath.empty() ? std::string() : std::filesystem::absolute(mEnvironmentPath).string()},
        {"reference_geometry", mReferenceGeometryPath.empty() ? std::string() : std::filesystem::absolute(mReferenceGeometryPath).string()},
        {"reference_geometry_sha256", mReferenceGeometrySha256},
        {"scene_material_bindings", sceneMaterialBindings},
        {"scene_material_bindings_replayable", false},
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
            {"approximation_linear", approximationAvailable ? approximationPath.filename().string() : std::string()},
            {"comparison_linear", comparisonPath.filename().string()},
            {"display", displayPath.filename().string()},
            {"difference_linear", approximationAvailable ? differencePath.filename().string() : std::string()},
            {"difference_display", approximationAvailable ? differenceDisplayPath.filename().string() : std::string()},
            {"material_program", mReferenceSource.family == ncls::ReferenceFamily::LayerStack ? materialPath.filename().string() : std::string()},
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
    mStatus = "Capture saved: " + manifestPath.string();
}

bool NclsViewer::pickSceneObject(const float2& screenPosition)
{
    if (!mpScene || !mpInstanceId || mOutputWidth == 0u || mOutputHeight == 0u) return false;

    const float outputX = std::clamp(screenPosition.x, 0.f, float(mOutputWidth - 1u));
    const float outputY = std::clamp(screenPosition.y, 0.f, float(mOutputHeight - 1u));
    float sourceU = (outputX + 0.5f) / float(mOutputWidth);
    if (hasActiveMethod() && mComparisonMode == 0u)
    {
        sourceU = sourceU < mSplit
            ? sourceU / std::max(mSplit, 1e-4f)
            : (sourceU - mSplit) / std::max(1.f - mSplit, 1e-4f);
    }
    const uint32_t x = std::min(uint32_t(std::clamp(sourceU, 0.f, 0.999999f) * float(mViewWidth)), mViewWidth - 1u);
    const uint32_t y = std::min(uint32_t(outputY), mOutputHeight - 1u);
    const auto pixels = getRenderContext()->readTextureSubresource(mpInstanceId.get(), 0u);
    const size_t byteOffset = (size_t(y) * mViewWidth + x) * sizeof(uint32_t);
    if (byteOffset + sizeof(uint32_t) > pixels.size()) return false;

    uint32_t encodedInstance = 0u;
    std::memcpy(&encodedInstance, pixels.data() + byteOffset, sizeof(encodedInstance));
    if (encodedInstance == 0u)
    {
        mStatus = "No scene object at the selected pixel; the previous material slot remains active.";
        return true;
    }

    const uint32_t instanceId = encodedInstance - 1u;
    if (instanceId >= mpScene->getGeometryInstanceCount()) return false;
    const auto& instance = mpScene->getGeometryInstance(instanceId);
    mSelectedSceneInstance = instanceId;
    mSelectedSceneMaterial = instance.materialID;
    if (instance.getType() == Scene::GeometryType::TriangleMesh
        || instance.getType() == Scene::GeometryType::DisplacedTriangleMesh)
        mSelectedSceneGeometryName = mpScene->getMeshName(instance.geometryID);
    else mSelectedSceneGeometryName = "Geometry #" + std::to_string(instance.geometryID);
    const auto material = mpScene->getMaterial(MaterialID::fromSlang(instance.materialID));
    mSelectedSceneMaterialName = material ? material->getName() : "Unnamed material";
    activateSceneMaterial(instance.materialID);
    mStatus = "Selected scene material '" + mSelectedSceneMaterialName + "'. Source-family edits now target this material slot.";
    return true;
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
        mMousePressScreen = event.screenPos;
        if (event.button == Input::MouseButton::Left && hasActiveMethod() && mComparisonMode == 0u
            && std::abs(event.screenPos.x - dividerX) < 8.f)
        {
            mDividerDragging = true;
            return true;
        }
        if (event.button == Input::MouseButton::Left)
        {
            mCameraDragging = true;
            mCameraDragMoved = false;
            return true;
        }
        if (event.button == Input::MouseButton::Middle || event.button == Input::MouseButton::Right)
        {
            mPanDragging = true;
            return true;
        }
    }
    else if (event.type == MouseEvent::Type::ButtonUp)
    {
        const bool handled = mDividerDragging || mCameraDragging || mPanDragging;
        const bool shouldPick = event.button == Input::MouseButton::Left
            && mCameraDragging && !mCameraDragMoved && !mDividerDragging;
        if (event.button == Input::MouseButton::Left)
        {
            mDividerDragging = false;
            mCameraDragging = false;
            mCameraDragMoved = false;
        }
        if (event.button == Input::MouseButton::Middle || event.button == Input::MouseButton::Right) mPanDragging = false;
        if (shouldPick && pickSceneObject(event.screenPos)) return true;
        return handled;
    }
    else if (event.type == MouseEvent::Type::Wheel)
    {
        const float radius = mpScene ? std::max(mpScene->getSceneBounds().radius(), 0.01f) : 1.f;
        const float minimumDistance = mpScene ? radius * 0.01f : 1.25f;
        const float maximumDistance = mpScene ? radius * 20.f : 9.f;
        mCamera.distance = std::clamp(
            mCamera.distance * std::exp(-0.12f * event.wheelDelta.y), minimumDistance, maximumDistance);
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
            const float2 dragDistance = event.screenPos - mMousePressScreen;
            mCameraDragMoved |= dot(dragDistance, dragDistance) > 9.f;
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
    else if (path.extension() == ".json" || path.extension() == ".binary" || path.extension() == ".mtlx") loadMaterial(path);
    else if (isSceneFile(path))
    {
        try { loadScene(path); }
        catch (const std::exception& error) { mStatus = "Failed to load scene: " + std::string(error.what()); }
    }
    else loadEnvironment(path);
}

int runMain(int argc, char** argv)
{
    ViewerOptions options = parseOptions(argc, argv);
    SampleAppConfig config;
    config.deviceDesc.type = Device::Type::D3D12;
    config.windowDesc.title = "NeuralShading - Multi-Family Reference / MethodBundle";
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
