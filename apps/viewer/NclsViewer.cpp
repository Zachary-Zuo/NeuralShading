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
constexpr uint32_t kMaximumSceneMaterials = 64;
constexpr uint32_t kCapturePathTracingSpp = 1024;
const Gui::DropdownList kComparisonModes = {
    {0, "Reference / method split"},
    {1, "Linear absolute error"},
    {2, "Linear relative error"},
    {3, "Amplified absolute error"},
};
const Gui::DropdownList kSlotModes = {
    {0, "Path tracing"},
    {1, "Deferred"},
};
const Gui::DropdownList kBaseKinds = {{1, "Rough conductor"}, {2, "Diffuse"}, {3, "Sheen"}};
const Gui::DropdownList kReferenceFamilies = {
    {0, "LayerStack / homogeneous slab"},
    {1, "MERL measured BRDF"},
    {2, "OpenPBR 1.1.1"},
    {3, "MaterialX standard_surface"},
};

float3 normalizedOr(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1e-12f ? value / std::sqrt(lengthSquared) : fallback;
}

bool isSha256(const std::string& value)
{
    return value.size() == 64u && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
    });
}

std::string shortId(const std::string& value)
{
    return value.size() > 12 ? value.substr(0, 12) : value;
}

std::string portableUri(const std::filesystem::path& path, const std::filesystem::path& baseDirectory)
{
    if (path.empty()) return {};
    const auto absolutePath = std::filesystem::absolute(path).lexically_normal();
    std::error_code error;
    const auto relative = std::filesystem::relative(
        absolutePath,
        std::filesystem::absolute(baseDirectory).lexically_normal(),
        error);
    return error ? absolutePath.generic_string() : relative.generic_string();
}

std::filesystem::path resolveUri(const std::string& uri, const std::filesystem::path& baseDirectory)
{
    if (uri.empty()) return {};
    const std::filesystem::path path(uri);
    return std::filesystem::absolute(path.is_absolute() ? path : baseDirectory / path).lexically_normal();
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

std::vector<float4> readHalfRgbaExrPixels(
    const std::filesystem::path& path, uint32_t& width, uint32_t& height)
{
    OPENEXR_IMF_NAMESPACE::RgbaInputFile file(path.string().c_str());
    const auto window = file.dataWindow();
    if (window.min.x != 0 || window.min.y != 0 || window.max.x < 0 || window.max.y < 0)
        throw std::runtime_error("environment EXR requires a zero-origin data window: " + path.string());
    width = static_cast<uint32_t>(window.max.x + 1);
    height = static_cast<uint32_t>(window.max.y + 1);
    std::vector<OPENEXR_IMF_NAMESPACE::Rgba> halfPixels(size_t(width) * height);
    file.setFrameBuffer(halfPixels.data(), 1, width);
    file.readPixels(window.min.y, window.max.y);
    std::vector<float4> pixels(halfPixels.size());
    for (size_t index = 0; index < halfPixels.size(); ++index)
        pixels[index] = float4(
            float(halfPixels[index].r), float(halfPixels[index].g),
            float(halfPixels[index].b), float(halfPixels[index].a));
    return pixels;
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
        if (replay.value("format_name", "") != "ncls.viewer-capture"
            || (replayVersion != 3u && replayVersion != 4u))
            throw std::runtime_error("unsupported replay manifest");
        if (replay.value("reference_integrator", "") != "ncls.scene-path-tracer@1")
            throw std::runtime_error("capture v3 requires reference_integrator ncls.scene-path-tracer@1");
        const auto base = std::filesystem::absolute(options.replayPath).parent_path();
        const auto resolve = [&](const std::string& value) {
            if (value.empty()) return std::filesystem::path();
            const std::filesystem::path path(value);
            return path.is_absolute() ? path : base / path;
        };
        options.packageRoot = resolve(replay.value("bundle_root", std::string()));
        options.materialPath = resolve(replay.value("source_material", std::string()));
        options.environmentPath = resolve(replay.value("environment", std::string()));
        options.environmentSha256 = replay.value("environment_sha256", std::string());
        options.referenceGeometryPath = resolve(replay.value("reference_geometry", std::string()));
        options.referenceGeometrySha256 = replay.value("reference_geometry_sha256", std::string());
        options.viewerScenePath = resolve(replay.value("viewer_scene", std::string()));
        options.requestedPackageId = replay.value("method_id", std::string());
        if (replayVersion == 4u)
        {
            const auto& slots = replay.at("slots");
            if (!slots.is_array() || slots.size() != 2u)
                throw std::runtime_error("capture v4 requires exactly two slots");
            for (uint32_t slotIndex = 0u; slotIndex < 2u; ++slotIndex)
            {
                options.requestedSlotPackages[slotIndex] = slots[slotIndex].value("package_id", std::string());
                const std::string mode = slots[slotIndex].value("mode", "path-tracing");
                if (mode == "path-tracing") options.requestedSlotModes[slotIndex] = ncls::SlotMode::PathTracing;
                else if (mode == "deferred") options.requestedSlotModes[slotIndex] = ncls::SlotMode::Deferred;
                else throw std::runtime_error("capture v4 slot mode is invalid");
            }
            options.hasRequestedSlots = true;
        }
        const auto resolution = replay.at("resolution");
        options.width = resolution.at(0).get<uint32_t>();
        options.height = resolution.at(1).get<uint32_t>();
        const uint32_t samples = std::clamp(replay.value("reference_samples_per_frame", 1u), 1u, 16u);
        options.frameCount = (kCapturePathTracingSpp + samples - 1u) / samples;
        break;
    }
    auto value = [&](int& index, const char* name) -> std::string {
        if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
        return argv[index];
    };
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--bundle-root") options.packageRoot = value(index, "--bundle-root");
        else if (argument == "--material") options.materialPath = value(index, "--material");
        else if (argument == "--method") options.requestedPackageId = value(index, "--method");
        else if (argument == "--environment") options.environmentPath = value(index, "--environment");
        else if (argument == "--reference-geometry") options.referenceGeometryPath = value(index, "--reference-geometry");
        else if (argument == "--replay") { options.replayPath = value(index, "--replay"); }
        else if (argument == "--viewer-scene") options.viewerScenePath = value(index, "--viewer-scene");
        else if (argument == "--capture") options.captureManifest = value(index, "--capture");
        else if (argument == "--frames") options.frameCount = static_cast<uint32_t>(std::stoul(value(index, "--frames")));
        else if (argument == "--width") options.width = static_cast<uint32_t>(std::stoul(value(index, "--width")));
        else if (argument == "--height") options.height = static_cast<uint32_t>(std::stoul(value(index, "--height")));
        else if (argument == "--headless") options.headless = true;
        else if (argument == "--evaluator-preview-lighting") options.evaluatorPreviewLighting = true;
        else if (argument == "--verbose-console") options.verboseConsole = true;
        else if (argument == "--help")
        {
            std::cout
                << "NclsViewer [--bundle-root DIR] [--material FILE] [--replay CAPTURE.json] [--viewer-scene FILE] "
                   "[--reference-geometry SCENE] [--environment HDRI] "
                   "[--headless --frames N --capture FILE] "
                   "[--method SHA256] [--evaluator-preview-lighting] "
                   "[--width W --height H] [--verbose-console]\n";
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
{
    mComparisonSlots[0].sourceReference = true;
    mComparisonSlots[0].uiValue = 1u;
    mComparisonSlots[0].contract.mode = ncls::SlotMode::PathTracing;
    mComparisonSlots[0].contract.status = ncls::SlotStatus::Ready;
    mComparisonSlots[1].contract.mode = ncls::SlotMode::PathTracing;
}

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
    mFallbackSourceGpu = createFallbackSourceGpuResources();
    mSourceGpu = createSourceGpuResources(mReferenceSource);
    mOpenPbrLuts = ncls::OpenPbrLuts::create(getDevice());
    auto initializeTiming = [&](PassTiming& timing) {
        for (auto& timer : timing.timers) timer = GpuTimer::create(getDevice());
    };
    initializeTiming(mVisibilityTiming);
    initializeTiming(mCompositeTiming);

    if (!mOptions.viewerScenePath.empty())
    {
        loadViewerScene(mOptions.viewerScenePath);
        if (!mOptions.replayPath.empty()) applyReplaySettings(mOptions.replayPath);
    }
    else if (!mOptions.replayPath.empty()) applyReplaySettings(mOptions.replayPath);
    else if (mOptions.referenceGeometryPath.empty())
    {
        const auto presetPath = getRuntimeDirectory() / "data/ncls-viewer/studio-v1.json";
        std::ifstream stream(presetPath, std::ios::binary);
        if (!stream) throw std::runtime_error(
            "fixed viewer studio preset is missing; rebuild with scripts/build_viewer.ps1: " + presetPath.string());
        const nlohmann::json preset = nlohmann::json::parse(stream);
        if (preset.value("format_name", "") != "ncls.viewer-studio" || preset.value("format_version", 0u) != 1u)
            throw std::runtime_error("unsupported fixed viewer studio preset: " + presetPath.string());
        const auto assetRoot = presetPath.parent_path();
        mOptions.referenceGeometryPath = assetRoot / preset.at("reference_geometry").get<std::string>();
        mOptions.referenceGeometrySha256 = preset.at("reference_geometry_sha256").get<std::string>();
        if (mOptions.environmentPath.empty())
        {
            mOptions.environmentPath = assetRoot / preset.at("environment").get<std::string>();
            mOptions.environmentSha256 = preset.at("environment_sha256").get<std::string>();
        }
        if (mOptions.materialPath.empty())
            mOptions.materialPath = assetRoot / preset.at("source_material").get<std::string>();
        applyReplaySettings(presetPath);
    }
    if (mOptions.viewerScenePath.empty())
    {
        if (!mOptions.referenceGeometryPath.empty())
            loadScene(mOptions.referenceGeometryPath, mOptions.referenceGeometrySha256);
        if (!mOptions.materialPath.empty()) loadMaterial(mOptions.materialPath);
        if (!mOptions.environmentPath.empty())
            loadEnvironment(mOptions.environmentPath, mOptions.environmentSha256);
    }
    if (mOptions.evaluatorPreviewLighting)
    {
        // 首屏只保留一个方向光，让视觉比较隔离局部 evaluator，
        // 同时避免把环境积分 query 数隐藏在启动延迟里。
        mLighting.useEnvironment = false;
        mLighting.usePoint = false;
        mLighting.useRectangle = false;
        mLighting.useSun = true;
        mLighting.sunDirection = float3(0.35f, 0.55f, 0.76f);
        mMaxSceneBounces = 0u;
    }
    if (!mpScene || !mpReferencePathPass)
        throw std::runtime_error("the viewer requires a loaded scene and the unified scene reference path");
    resizeResources(getTargetFbo()->getWidth(), getTargetFbo()->getHeight());
    scanPackages();
    if (mOptions.hasRequestedSlots)
    {
        for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
        {
            mComparisonSlots[slotIndex].contract.mode = mOptions.requestedSlotModes[slotIndex];
            const std::string& packageId = mOptions.requestedSlotPackages[slotIndex];
            uint32_t selection = packageId == "source-reference" ? 1u : 0u;
            if (!packageId.empty() && packageId != "source-reference")
                for (uint32_t index = 0u; index < mPrograms.size(); ++index)
                    if (mPrograms[index].packageId == packageId) selection = index + 2u;
            activateComparisonSlot(slotIndex, selection);
            if (!packageId.empty() && packageId != "source-reference" && selection == 0u && mOptions.headless)
                throw std::runtime_error("capture v4 slot package did not pass compatibility/parity: " + packageId);
        }
    }
    if (!mOptions.hasRequestedSlots && !mOptions.requestedPackageId.empty())
    {
        int32_t requested = -1;
        if (mOptions.requestedPackageId != "none")
            for (uint32_t index = 0; index < mPrograms.size(); ++index)
                if (mPrograms[index].packageId == mOptions.requestedPackageId) requested = static_cast<int32_t>(index);
        if (requested < 0 && mOptions.requestedPackageId != "none" && mOptions.headless)
            throw std::runtime_error("replay ScatteringPackage did not pass compatibility/parity: " + mOptions.requestedPackageId);
        selectProgram(requested);
    }
    mStatus = mPrograms.empty()
        ? "No compatible neural evaluator bundle was found; showing a full-width reference."
        : "GPU-parity-validated neural evaluator bundles found; the method selection starts empty.";
}

void NclsViewer::createPasses()
{
    mpVisibilityClearPass = ComputePass::create(getDevice(), "NclsViewer/shaders/ClearVisibility.cs.slang");
    mpCompositePass = ComputePass::create(getDevice(), "NclsViewer/shaders/Composite.cs.slang");
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
    rebuildEnvironmentSampling(pixels, width, height);
    mEnvironmentPath.clear();
    mEnvironmentSha256.clear();
}

void NclsViewer::rebuildEnvironmentSampling(
    const std::vector<float4>& pixels, uint32_t width, uint32_t height)
{
    if (width == 0u || height == 0u || pixels.size() != size_t(width) * height)
    {
        const std::array<float, 2> fallback{0.f, 1.f};
        mpEnvironmentMarginalCdf = getDevice()->createStructuredBuffer(
            sizeof(float), 2u, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, fallback.data());
        mpEnvironmentConditionalCdf = getDevice()->createStructuredBuffer(
            sizeof(float), 2u, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, fallback.data());
        mEnvironmentSamplingDimensions = uint2(0u);
        return;
    }

    std::vector<float> marginal(height + 1u, 0.f);
    std::vector<float> conditional(size_t(height) * (width + 1u), 0.f);
    std::vector<double> rowWeights(height, 0.0);
    double totalWeight = 0.0;
    for (uint32_t y = 0u; y < height; ++y)
    {
        const double theta = 3.14159265358979323846 * (double(y) + 0.5) / double(height);
        const double solidAngleFactor = std::sin(theta);
        double rowWeight = 0.0;
        const size_t cdfBase = size_t(y) * (width + 1u);
        for (uint32_t x = 0u; x < width; ++x)
        {
            const float4 pixel = pixels[size_t(y) * width + x];
            const double luminance = 0.2126 * std::max(pixel.x, 0.f)
                + 0.7152 * std::max(pixel.y, 0.f) + 0.0722 * std::max(pixel.z, 0.f);
            rowWeight += luminance * solidAngleFactor;
            conditional[cdfBase + x + 1u] = float(rowWeight);
        }
        rowWeights[y] = rowWeight;
        totalWeight += rowWeight;
        if (rowWeight > 0.0)
            for (uint32_t x = 1u; x <= width; ++x)
                conditional[cdfBase + x] = float(double(conditional[cdfBase + x]) / rowWeight);
        else
            for (uint32_t x = 1u; x <= width; ++x)
                conditional[cdfBase + x] = float(x) / float(width);
    }
    if (!(totalWeight > 0.0) || !std::isfinite(totalWeight))
    {
        rebuildEnvironmentSampling({}, 0u, 0u);
        return;
    }
    double runningWeight = 0.0;
    for (uint32_t y = 0u; y < height; ++y)
    {
        runningWeight += rowWeights[y];
        marginal[y + 1u] = float(runningWeight / totalWeight);
    }
    marginal.back() = 1.f;
    for (uint32_t y = 0u; y < height; ++y)
        conditional[size_t(y) * (width + 1u) + width] = 1.f;
    mpEnvironmentMarginalCdf = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(marginal.size()), ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal, marginal.data());
    mpEnvironmentConditionalCdf = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(conditional.size()), ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal, conditional.data());
    mEnvironmentSamplingDimensions = uint2(width, height);
}

NclsViewer::SourceGpuResources NclsViewer::createFallbackSourceGpuResources()
{
    const auto shaderResource = ResourceBindFlags::ShaderResource;
    SourceGpuResources resources;
    const auto layerStack = ncls::makeDefaultMaterial();
    resources.pMaterial = getDevice()->createStructuredBuffer(
        sizeof(ncls::LayerStackIR), 1, shaderResource, MemoryType::DeviceLocal, &layerStack);
    const float3 zeroMerl(0.f);
    resources.pMerlBrdf = getDevice()->createStructuredBuffer(
        sizeof(float3), 1, shaderResource, MemoryType::DeviceLocal, &zeroMerl);
    const std::array<float, 77> zeroOpenPbr{};
    resources.pOpenPbrInputs = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(zeroOpenPbr.size()),
        shaderResource, MemoryType::DeviceLocal, zeroOpenPbr.data());
    const std::array<float, 24> zeroMaterialX{};
    resources.pMaterialXInputs = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(zeroMaterialX.size()),
        shaderResource, MemoryType::DeviceLocal, zeroMaterialX.data());
    const float4 baseColor(.8f, .8f, .8f, 1.f);
    const float4 roughness(.2f, .2f, .2f, 1.f);
    const float4 metalness(0.f, 0.f, 0.f, 1.f);
    const float4 normal(.5f, .5f, 1.f, 1.f);
    resources.pMaterialXBaseColor = getDevice()->createTexture2D(
        1, 1, ResourceFormat::RGBA32Float, 1, 1, &baseColor, shaderResource);
    resources.pMaterialXRoughness = getDevice()->createTexture2D(
        1, 1, ResourceFormat::RGBA32Float, 1, 1, &roughness, shaderResource);
    resources.pMaterialXMetalness = getDevice()->createTexture2D(
        1, 1, ResourceFormat::RGBA32Float, 1, 1, &metalness, shaderResource);
    resources.pMaterialXNormalMap = getDevice()->createTexture2D(
        1, 1, ResourceFormat::RGBA32Float, 1, 1, &normal, shaderResource);
    return resources;
}

NclsViewer::SourceGpuResources NclsViewer::createSourceGpuResources(const ncls::ReferenceSource& source)
{
    const auto shaderResource = ResourceBindFlags::ShaderResource;
    SourceGpuResources resources = mFallbackSourceGpu;
    if (source.family == ncls::ReferenceFamily::LayerStack)
        resources.pMaterial = getDevice()->createStructuredBuffer(
            sizeof(ncls::LayerStackIR), 1, shaderResource, MemoryType::DeviceLocal, &source.layerStack);
    else if (source.family == ncls::ReferenceFamily::Merl)
        resources.pMerlBrdf = getDevice()->createStructuredBuffer(
            sizeof(std::array<float, 3>), static_cast<uint32_t>(source.merlBrdf.size()),
            shaderResource, MemoryType::DeviceLocal, source.merlBrdf.data());
    else if (source.family == ncls::ReferenceFamily::OpenPbr)
        resources.pOpenPbrInputs = getDevice()->createStructuredBuffer(
            sizeof(float), static_cast<uint32_t>(source.openPbrInputs.size()),
            shaderResource, MemoryType::DeviceLocal, source.openPbrInputs.data());
    else if (source.family != ncls::ReferenceFamily::MaterialX)
        throw std::runtime_error("unsupported reference source family");

    if (source.family != ncls::ReferenceFamily::MaterialX) return resources;
    resources.pMaterialXInputs = getDevice()->createStructuredBuffer(
        sizeof(float), static_cast<uint32_t>(source.materialXInputs.size()),
        shaderResource, MemoryType::DeviceLocal, source.materialXInputs.data());

    auto loadTexture = [&](const std::filesystem::path& texturePath, bool srgb, bool generateMips,
                           bool convertToFloat16, const ref<Texture>& fallback) {
        if (texturePath.empty()) return fallback;
        logInfo("Loading MaterialX texture '{}' (sRGB={})", texturePath, srgb);
        auto texture = convertToFloat16
            ? loadHalfRgbaExr(getDevice(), texturePath, generateMips)
            : Texture::createFromFile(getDevice(), texturePath, generateMips, srgb);
        if (!texture) throw std::runtime_error("Falcor failed to load MaterialX texture: " + texturePath.string());
        logInfo("Loaded MaterialX texture '{}' as {}x{} with {} mip(s)",
            texturePath.filename(), texture->getWidth(), texture->getHeight(), texture->getMipCount());
        return texture;
    };
    // MaterialX samples srgb_texture in the encoded domain and applies the
    // explicit srgb_texture_to_lin_rec709 transform in the reference shader.
    resources.pMaterialXBaseColor = loadTexture(
        source.materialXBaseColorTexture, false, true, false, mFallbackSourceGpu.pMaterialXBaseColor);
    resources.pMaterialXRoughness = loadTexture(
        source.materialXRoughnessTexture, false, true, true, mFallbackSourceGpu.pMaterialXRoughness);
    resources.pMaterialXMetalness = loadTexture(
        source.materialXMetalnessTexture, false, true, true, mFallbackSourceGpu.pMaterialXMetalness);
    resources.pMaterialXNormalMap = loadTexture(
        source.materialXNormalTexture, false, true, true, mFallbackSourceGpu.pMaterialXNormalMap);
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
    mViewWidth = std::max(mOutputWidth / 2u, 1u);
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
    mpEmptySlot = viewTexture();
    mpComparisonLinear = getDevice()->createTexture2D(
        mOutputWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, shaderUav);
    mpDisplay = getDevice()->createTexture2D(
        mOutputWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, shaderUav);
    mpDifferenceLinear = viewTexture();
    mpDifferenceDisplay = viewTexture();
    rebuildSceneFbo();
    for (auto& slot : mComparisonSlots) resizeComparisonSlot(slot);
    resetReference(true);
}

void NclsViewer::loadScene(const std::filesystem::path& requestedPath, const std::string& expectedSha256)
{
    namespace fs = std::filesystem;
    const fs::path path = fs::absolute(requestedPath).lexically_normal();
    if (!fs::is_regular_file(path)) throw std::runtime_error("scene does not exist: " + path.string());
    const std::string geometrySha256 = ncls::sha256FileHex(path);
    if (!expectedSha256.empty() && expectedSha256 != geometrySha256)
        throw std::runtime_error("scene SHA-256 does not match replay: " + path.string());

    auto scene = Scene::create(getDevice(), path);
    if (!scene || scene->getGeometryInstanceCount() == 0u)
        throw std::runtime_error("Falcor loaded a scene without renderable geometry: " + path.string());
    if (scene->getMaterialCount() == 0u || scene->getMaterialCount() > kMaximumSceneMaterials)
        throw std::runtime_error("scene material count must be in [1, "
            + std::to_string(kMaximumSceneMaterials) + "]: " + path.string());

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
    rebuildReferenceMaterialMetadata();
    createSceneReferencePass();

    const auto& bounds = mpScene->getSceneBounds();
    mCamera.target = bounds.center();
    mCamera.distance = std::max(2.4f * bounds.radius(), 0.01f);
    rebuildSceneFbo();
    resetReference(true);
    mStatus = "Loaded Falcor scene: " + path.string();
    logInfo("Loaded Falcor scene '{}' ({} instances, {} materials, SHA-256 {})",
        path, mpScene->getGeometryInstanceCount(), mpScene->getMaterialCount(), shortId(mReferenceGeometrySha256));
}

void NclsViewer::createSceneReferencePass()
{
    if (!mpScene)
    {
        mpReferencePathPass.reset();
        return;
    }
    ProgramDesc program;
    program.addShaderModules(mpScene->getShaderModules());
    program.addShaderLibrary("NclsViewer/shaders/ReferencePathTracer.cs.slang").csEntry("main");
    program.addTypeConformances(mpScene->getTypeConformances());
    DefineList defines = mpScene->getSceneDefines();
    defines.add("NCLS_MAX_SCENE_MATERIALS", std::to_string(mpScene->getMaterialCount()));
    uint32_t familyMask = 1u << static_cast<uint32_t>(mReferenceSource.family);
    for (const auto& [materialId, binding] : mInactiveSceneMaterials)
        familyMask |= 1u << static_cast<uint32_t>(binding.source.family);
    defines.add("NCLS_REFERENCE_FAMILY_MASK", std::to_string(familyMask));
    mpReferencePathPass = ComputePass::create(getDevice(), program, defines, true);
}

void NclsViewer::rebuildReferenceMaterialMetadata()
{
    const uint32_t materialCount = mpScene ? mpScene->getMaterialCount() : 1u;
    std::vector<uint4> metadata(materialCount, uint4(0u));
    for (uint32_t materialId = 0u; materialId < materialCount; ++materialId)
    {
        const ncls::ReferenceSource* source = nullptr;
        if (!mpScene || materialId == mActiveSceneMaterial) source = &mReferenceSource;
        else if (const auto* binding = inactiveSceneMaterial(materialId)) source = &binding->source;
        if (!source) throw std::runtime_error("scene material slot has no source reference binding");
        metadata[materialId] = uint4(
            static_cast<uint32_t>(source->family), source->openPbrColorSpace, 0u, 0u);
    }
    mpReferenceMaterialMetadata = getDevice()->createStructuredBuffer(
        sizeof(uint4), materialCount, ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal, metadata.data());
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
    rebuildReferenceMaterialMetadata();
    createSceneReferencePass();
    mSelectedInterface = 0u;
    resetReference(false);

    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
    {
        const uint32_t selection = mComparisonSlots[slotIndex].uiValue;
        if (selection >= 2u) activateComparisonSlot(slotIndex, selection);
    }
}

void NclsViewer::updateMaterialBuffer()
{
    if (mReferenceSource.family != ncls::ReferenceFamily::LayerStack)
        throw std::runtime_error("current source material is not a LayerStack material");
    ncls::validateLayerStack(mMaterial);
    mReferenceSource.sourceSha256 = ncls::layerStackHash(mMaterial);
    mSourceGpu.pMaterial->setBlob(&mMaterial, 0, sizeof(mMaterial));
    resetReference(false);
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
    resetReference(false);
}

bool NclsViewer::allMaterialsSupportedBy(const ncls::ViewerProgram& method) const
{
    const auto supports = [&](const ncls::ReferenceSource& source) {
        return source.familyId() == method.sourceFamilyId
            && source.sourceSha256 == method.sourceAssetSha256;
    };
    if (!supports(mReferenceSource)) return false;
    for (const auto& [materialId, binding] : mInactiveSceneMaterials)
        if (!supports(binding.source)) return false;
    return true;
}

bool NclsViewer::hasActiveProgram() const
{
    return std::any_of(
        mComparisonSlots.begin(), mComparisonSlots.end(),
        [](const ComparisonSlotRuntime& slot) { return slot.ready() && !slot.sourceReference; });
}

void NclsViewer::resetReference(bool visibilityChanged)
{
    mAccumulationSeconds = 0.0;
    mVisibilityDirty |= visibilityChanged;
    for (auto& slot : mComparisonSlots)
    {
        slot.spp = 0u;
        slot.ping = 0u;
        slot.resetAccumulation = true;
    }
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
    resetReference(true);
}

float3 NclsViewer::cameraPosition() const
{
    const float cosinePitch = std::cos(mCamera.pitch);
    return mCamera.target + mCamera.distance * float3(
        cosinePitch * std::sin(mCamera.yaw),
        std::sin(mCamera.pitch),
        cosinePitch * std::cos(mCamera.yaw));
}

void NclsViewer::scanPackages()
{
    std::array<std::string, 2> previousIds;
    std::array<bool, 2> previousSources{};
    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
    {
        previousSources[slotIndex] = mComparisonSlots[slotIndex].sourceReference;
        if (const auto* program = slotProgram(mComparisonSlots[slotIndex]))
            previousIds[slotIndex] = program->packageId;
    }
    if (!std::filesystem::is_directory(mOptions.packageRoot))
    {
        mPrograms.clear();
        mPackageFailures.clear();
        activateComparisonSlot(0u, previousSources[0] ? 1u : 0u);
        activateComparisonSlot(1u, previousSources[1] ? 1u : 0u);
        mStatus = "ScatteringPackage directory is absent; running reference-only: " + mOptions.packageRoot.string();
        logInfo("No ScatteringPackage directory at '{}'; running reference-only", mOptions.packageRoot);
        return;
    }
    auto scan = ncls::scanScatteringPackages(mOptions.packageRoot, getRuntimeDirectory() / "shaders");
    std::vector<ncls::ViewerProgram> accepted;
    for (auto& method : scan.programs)
    {
        std::string error;
        if (runParityProbe(method, error))
        {
            logInfo("Accepted ScatteringPackage '{}' ({})", method.displayName, shortId(method.packageId));
            accepted.push_back(std::move(method));
        }
        else scan.failures.push_back({method.root, "GPU parity failed: " + error});
    }
    mPrograms = std::move(accepted);
    mPackageFailures = std::move(scan.failures);
    for (const auto& failure : mPackageFailures)
        logWarning("Rejected ScatteringPackage '{}': {}", failure.path, failure.reason);
    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
    {
        uint32_t selection = previousSources[slotIndex] ? 1u : 0u;
        if (!previousIds[slotIndex].empty())
            for (uint32_t index = 0u; index < mPrograms.size(); ++index)
                if (mPrograms[index].packageId == previousIds[slotIndex]) selection = index + 2u;
        activateComparisonSlot(slotIndex, selection);
    }
}

ref<ComputePass> NclsViewer::createProgramPass(
    const char* shaderPath,
    const ncls::ViewerProgram& method)
{
    ProgramDesc program;
    program.addShaderLibrary(shaderPath).csEntry("main");
    DefineList defines;
    defines.add(
        "NCLS_PACKAGE_PROGRAM_HEADER",
        "\"" + std::filesystem::path(method.shaderModule).generic_string() + "\"");
    for (const auto& [name, value] : method.shaderDefines) defines.add(name, value);
    return ComputePass::create(getDevice(), program, defines, true);
}

ref<ComputePass> NclsViewer::createProgramPathPass(const ncls::ViewerProgram& method)
{
    if (!mpScene) throw std::runtime_error("package path tracer requires a loaded scene");
    ProgramDesc program;
    program.addShaderModules(mpScene->getShaderModules());
    program.addShaderLibrary("NclsViewer/shaders/PackagePathTracer.cs.slang").csEntry("main");
    program.addTypeConformances(mpScene->getTypeConformances());
    DefineList defines = mpScene->getSceneDefines();
    defines.add(
        "NCLS_PACKAGE_PROGRAM_HEADER",
        "\"" + std::filesystem::path(method.shaderModule).generic_string() + "\"");
    for (const auto& [name, value] : method.shaderDefines) defines.add(name, value);
    return ComputePass::create(getDevice(), program, defines, true);
}

bool NclsViewer::runParityProbe(const ncls::ViewerProgram& method, std::string& error)
{
    try
    {
        const auto flags = ResourceBindFlags::ShaderResource;
        auto weights = getDevice()->createStructuredBuffer(
            sizeof(uint32_t), static_cast<uint32_t>(method.sharedWeightWords.size()),
            flags, MemoryType::DeviceLocal, method.sharedWeightWords.data());
        auto compiledMaterials = getDevice()->createStructuredBuffer(
            method.compiledMaterialBytes, method.compiledMaterialCount,
            flags, MemoryType::DeviceLocal, method.compiledMaterials.data());
        const float4 view(method.parity.view[0], method.parity.view[1], method.parity.view[2], 0.f);
        const std::vector<std::array<float, 3>> probeLights = method.parity.lights.empty()
            ? std::vector<std::array<float, 3>>{
                {0.f, 0.f, 1.f}, {.6f, 0.f, .8f}, {0.f, .8f, .6f}, {-.55f, .35f, .757f}}
            : method.parity.lights;
        std::vector<float4> lights;
        for (const auto& item : probeLights) lights.emplace_back(item[0], item[1], item[2], 0.f);
        auto viewBuffer = getDevice()->createStructuredBuffer(sizeof(float4), 1, flags, MemoryType::DeviceLocal, &view);
        auto lightBuffer = getDevice()->createStructuredBuffer(
            sizeof(float4), static_cast<uint32_t>(lights.size()), flags, MemoryType::DeviceLocal, lights.data());
        auto output = getDevice()->createStructuredBuffer(
            sizeof(float4),
            static_cast<uint32_t>(lights.size()),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        auto parityPass = createProgramPass("NclsViewer/shaders/PackageParity.cs.slang", method);
        auto root = parityPass->getRootVar();
        root["gNclsRuntimeWeights"] = weights;
        root["gNclsCompiledMaterials"] = compiledMaterials;
        std::vector<ref<Texture>> textures;
        std::vector<ref<Sampler>> samplers;
        for (const auto& resource : method.resources)
        {
            if (resource.dtype == "texture2d-rgba16float-dds@1")
            {
                auto texture = Texture::createFromFile(getDevice(), resource.path, false, false);
                if (!texture || texture->getWidth() != resource.shape.at(0)
                    || texture->getHeight() != resource.shape.at(1)
                    || texture->getMipCount() != resource.shape.at(2))
                    throw std::runtime_error("package texture load disagrees with typed descriptor");
                root[resource.usage] = texture;
                textures.push_back(std::move(texture));
            }
            else if (resource.dtype == "sampler-linear-wrap-explicit-lod@1")
            {
                Sampler::Desc desc;
                desc.setFilterMode(
                        TextureFilteringMode::Linear,
                        TextureFilteringMode::Linear,
                        TextureFilteringMode::Point)
                    .setAddressingMode(
                        TextureAddressingMode::Wrap,
                        TextureAddressingMode::Wrap,
                        TextureAddressingMode::Wrap);
                auto sampler = getDevice()->createSampler(desc);
                root[resource.usage] = sampler;
                samplers.push_back(std::move(sampler));
            }
        }
        root["gViews"] = viewBuffer;
        root["gLights"] = lightBuffer;
        root["gOutput"] = output;
        root["gLightCount"] = static_cast<uint32_t>(lights.size());
        root["gCompiledMaterialIndex"] = method.compiledMaterialIndex;
        parityPass->execute(getRenderContext(), static_cast<uint32_t>(lights.size()), 1, 1);
        std::vector<float4> actual(lights.size());
        output->getBlob(actual.data(), 0, actual.size() * sizeof(float4));
        const bool hasExpected = method.parity.expectedResponseCos.size() == actual.size();
        for (size_t light = 0; light < actual.size(); ++light)
        {
            for (size_t channel = 0; channel < 3; ++channel)
            {
                const float observed = actual[light][channel];
                if (!std::isfinite(observed) || observed < 0.f)
                {
                    error = "GPU contract probe produced a non-finite or negative response";
                    return false;
                }
                if (!hasExpected) continue;
                const float expected = method.parity.expectedResponseCos[light][channel];
                const float tolerance = method.parity.absoluteTolerance + method.parity.relativeTolerance * std::abs(expected);
                if (std::abs(observed - expected) > tolerance)
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

void NclsViewer::selectProgram(int32_t methodIndex)
{
    activateComparisonSlot(
        1u,
        methodIndex >= 0 && methodIndex < static_cast<int32_t>(mPrograms.size())
            ? static_cast<uint32_t>(methodIndex) + 2u : 0u);
}

const ncls::ViewerProgram* NclsViewer::slotProgram(const ComparisonSlotRuntime& slot) const
{
    return slot.programIndex >= 0 && slot.programIndex < static_cast<int32_t>(mPrograms.size())
        ? &mPrograms[slot.programIndex] : nullptr;
}

void NclsViewer::resizeComparisonSlot(ComparisonSlotRuntime& slot)
{
    if (mViewWidth == 0u || mOutputHeight == 0u) return;
    const auto flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto texture = [&]() {
        return getDevice()->createTexture2D(
            mViewWidth, mOutputHeight, ResourceFormat::RGBA32Float, 1, 1, nullptr, flags);
    };
    slot.pAccumulated[0] = texture();
    slot.pAccumulated[1] = texture();
    slot.pDeferred = texture();
    const std::array<uint32_t, 2> zero{};
    slot.pNoiseStats = getDevice()->createStructuredBuffer(
        sizeof(uint32_t), 2u, flags, MemoryType::DeviceLocal, zero.data());
    slot.ping = 0u;
    slot.spp = 0u;
    slot.resetAccumulation = true;
}

void NclsViewer::activateComparisonSlot(uint32_t slotIndex, uint32_t selection)
{
    if (slotIndex >= mComparisonSlots.size()) throw std::runtime_error("comparison slot index is invalid");
    ComparisonSlotRuntime candidate;
    candidate.contract.mode = mComparisonSlots[slotIndex].contract.mode;
    candidate.uiValue = selection;
    auto initializeTiming = [&](PassTiming& timing) {
        for (auto& timer : timing.timers) timer = GpuTimer::create(getDevice());
    };
    initializeTiming(candidate.timing);
    try
    {
        if (selection == 0u)
        {
            candidate.contract.status = ncls::SlotStatus::Empty;
        }
        else if (selection == 1u)
        {
            candidate.sourceReference = true;
            if (candidate.contract.mode == ncls::SlotMode::PathTracing)
                candidate.contract.status = ncls::SlotStatus::Ready;
            else
            {
                candidate.contract.status = ncls::SlotStatus::Unsupported;
                candidate.contract.diagnostic = "source reference currently exposes the scene path integrator only";
            }
        }
        else
        {
            const uint32_t programIndex = selection - 2u;
            if (programIndex >= mPrograms.size()) throw std::runtime_error("comparison slot package index is invalid");
            const auto& method = mPrograms[programIndex];
            candidate.programIndex = static_cast<int32_t>(programIndex);
            candidate.contract.bind(&method);
            if (!allMaterialsSupportedBy(method))
            {
                candidate.contract.status = ncls::SlotStatus::Unsupported;
                candidate.contract.diagnostic = "package material asset does not match every scene material slot";
            }
            if (candidate.contract.status == ncls::SlotStatus::Ready)
            {
                candidate.pWeights = getDevice()->createStructuredBuffer(
                    sizeof(uint32_t), static_cast<uint32_t>(method.sharedWeightWords.size()),
                    ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal,
                    method.sharedWeightWords.data());
                candidate.pCompiledMaterials = getDevice()->createStructuredBuffer(
                    method.compiledMaterialBytes, method.compiledMaterialCount,
                    ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal,
                    method.compiledMaterials.data());
                candidate.pDeferredPass = createProgramPass(
                    "NclsViewer/shaders/DeferredRenderer.cs.slang", method);
                if ((method.capabilities & (4u | 8u)) == (4u | 8u))
                    candidate.pPathPass = createProgramPathPass(method);
                for (const auto& resource : method.resources)
                {
                    if (resource.dtype == "texture2d-rgba16float-dds@1")
                    {
                        auto value = Texture::createFromFile(getDevice(), resource.path, false, false);
                        if (!value || value->getWidth() != resource.shape.at(0)
                            || value->getHeight() != resource.shape.at(1)
                            || value->getMipCount() != resource.shape.at(2))
                            throw std::runtime_error("package texture load disagrees with typed descriptor");
                        candidate.textures.emplace(resource.usage, std::move(value));
                    }
                    else if (resource.dtype == "sampler-linear-wrap-explicit-lod@1")
                    {
                        Sampler::Desc desc;
                        desc.setFilterMode(
                                TextureFilteringMode::Linear,
                                TextureFilteringMode::Linear,
                                TextureFilteringMode::Point)
                            .setAddressingMode(
                                TextureAddressingMode::Wrap,
                                TextureAddressingMode::Wrap,
                                TextureAddressingMode::Wrap);
                        candidate.samplers.emplace(resource.usage, getDevice()->createSampler(desc));
                    }
                }
            }
        }
        resizeComparisonSlot(candidate);
    }
    catch (const std::exception& exception)
    {
        candidate.contract.status = ncls::SlotStatus::Error;
        candidate.contract.diagnostic = exception.what();
        logWarning("Comparison slot {} failed: {}", slotIndex, exception.what());
    }
    mComparisonSlots[slotIndex] = std::move(candidate);
}

ref<Texture> NclsViewer::slotOutput(const ComparisonSlotRuntime& slot) const
{
    if (!slot.ready()) return {};
    if (slot.sourceReference || slot.contract.mode == ncls::SlotMode::PathTracing)
        return slot.pAccumulated[slot.ping];
    return slot.pDeferred;
}

void NclsViewer::bindProgramResources(ShaderVar root, const ComparisonSlotRuntime& slot) const
{
    root["gNclsRuntimeWeights"] = slot.pWeights;
    root["gNclsCompiledMaterials"] = slot.pCompiledMaterials;
    for (const auto& [usage, texture] : slot.textures) root[usage] = texture;
    for (const auto& [usage, sampler] : slot.samplers) root[usage] = sampler;
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
    if (!mpSceneFbo || !mpScene || !mpSceneVisibilityPass)
        throw std::runtime_error("scene visibility resources are unavailable");
    auto root = mpVisibilityClearPass->getRootVar();
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gMaterialXTexCoord"] = mpMaterialXTexCoord;
    root["gMaterialXTexCoordGrad"] = mpMaterialXTexCoordGrad;
    root["gInstanceId"] = mpInstanceId;
    root["gMaterialId"] = mpSceneMaterialId;
    auto constants = root["ClearVisibilityCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    beginTiming(mVisibilityTiming);
    mpVisibilityClearPass->execute(pRenderContext, mViewWidth, mOutputHeight);
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
    endTiming(mVisibilityTiming);
    mVisibilityDirty = false;
}

void NclsViewer::renderReference(RenderContext* pRenderContext, ComparisonSlotRuntime& slot)
{
    const uint32_t samplesThisFrame = std::min(
        mSamplesPerFrame, kCapturePathTracingSpp - std::min(slot.spp, kCapturePathTracingSpp));
    if (samplesThisFrame == 0u) return;
    const uint32_t next = 1u - slot.ping;
    const bool accumulate = !mCameraDragging && !mPanDragging;
    beginTiming(slot.timing);
    {
        auto root = mpReferencePathPass->getRootVar();
        mpScene->bindShaderDataForRaytracing(pRenderContext, root["gScene"]);
        root["gMaterialMetadata"] = mpReferenceMaterialMetadata;
        root["gMaterialXSampler"] = mpMaterialXSampler;
        root["gPreviousReference"] = slot.pAccumulated[slot.ping];
        root["gNextReference"] = slot.pAccumulated[next];
        root["gEnvironment"] = mpEnvironment;
        root["gEnvironmentMarginalCdf"] = mpEnvironmentMarginalCdf;
        root["gEnvironmentConditionalCdf"] = mpEnvironmentConditionalCdf;
        root["gLinearSampler"] = mpLinearSampler;
        root["gNoiseStats"] = slot.pNoiseStats;
        bool hasOpenPbr = mReferenceSource.family == ncls::ReferenceFamily::OpenPbr;
        for (const auto& [materialId, binding] : mInactiveSceneMaterials)
            hasOpenPbr |= binding.source.family == ncls::ReferenceFamily::OpenPbr;
        if (hasOpenPbr) mOpenPbrLuts.bind(root);

        auto bindSource = [&](uint32_t materialId, const SourceGpuResources& gpu) {
            root["gLayerStacks"][materialId] = gpu.pMaterial;
            root["gMerlBrdfs"][materialId] = gpu.pMerlBrdf;
            root["gOpenPbrInputs"][materialId] = gpu.pOpenPbrInputs;
            root["gMaterialXInputs"][materialId] = gpu.pMaterialXInputs;
            root["gMaterialXBaseColors"][materialId] = gpu.pMaterialXBaseColor;
            root["gMaterialXRoughnesses"][materialId] = gpu.pMaterialXRoughness;
            root["gMaterialXMetalnesses"][materialId] = gpu.pMaterialXMetalness;
            root["gMaterialXNormalMaps"][materialId] = gpu.pMaterialXNormalMap;
        };
        bindSource(mActiveSceneMaterial, mSourceGpu);
        for (const auto& [materialId, binding] : mInactiveSceneMaterials)
            bindSource(materialId, binding.gpu);

        auto constants = root["ReferencePathTracerCB"];
        constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
        constants["gFrameIndex"] = mFrameIndex;
        constants["gMaterialCount"] = mpScene->getMaterialCount();
        constants["gReferenceSpp"] = slot.spp;
        constants["gSamplesThisFrame"] = samplesThisFrame;
        constants["gMaxSceneBounces"] = mMaxSceneBounces;
        constants["gMaxLayerWalkDepth"] = mMaxLayerWalkDepth;
        constants["gResetAccumulation"] = uint32_t(slot.resetAccumulation);
        constants["gAccumulate"] = uint32_t(accumulate);
        std::string extension = mReferenceGeometryPath.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        constants["gFlipTexCoordV"] = uint32_t(extension == ".obj");
        constants["gEnvironmentSamplingDimensions"] = mEnvironmentSamplingDimensions;
        bindLighting(root, "ReferencePathTracerCB");
        pRenderContext->clearUAV(slot.pNoiseStats->getUAV().get(), uint4(0u));
        mpReferencePathPass->execute(pRenderContext, mViewWidth, mOutputHeight);
    }
    endTiming(slot.timing);
    if (mFrameIndex % 8u == 0u || slot.resetAccumulation)
    {
        const auto stats = pRenderContext->readBuffer<uint32_t>(slot.pNoiseStats.get(), 0u, 2u);
        if (stats.size() == 2u && stats[1] > 0u)
            mEstimatedRelativeStandardError = float(stats[0]) / (4096.f * float(stats[1]));
    }
    slot.ping = next;
    if (accumulate)
    {
        slot.spp += samplesThisFrame;
        mAccumulationSeconds += getFrameRate().getLastFrameTime();
    }
    else slot.spp = 0;
    slot.resetAccumulation = false;
}

void NclsViewer::renderApproximation(RenderContext* pRenderContext, ComparisonSlotRuntime& slot)
{
    const auto* method = slotProgram(slot);
    if (!method || !slot.pDeferredPass) return;
    auto root = slot.pDeferredPass->getRootVar();
    root["gPositionDepth"] = mpPositionDepth;
    root["gNormal"] = mpNormal;
    root["gTangent"] = mpTangent;
    root["gViewDirection"] = mpViewDirection;
    root["gTexCoord"] = mpMaterialXTexCoord;
    root["gTexCoordGrad"] = mpMaterialXTexCoordGrad;
    root["gMaterialId"] = mpSceneMaterialId;
    root["gEnvironment"] = mpEnvironment;
    root["gLinearSampler"] = mpLinearSampler;
    root["gApproximation"] = slot.pDeferred;
    bindProgramResources(root, slot);
    root["ApproximationCB"]["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    root["ApproximationCB"]["gCompiledMaterialIndex"] = method->compiledMaterialIndex;
    root["ApproximationCB"]["gEnvironmentQueryBudget"] = method->environmentQueryBudget;
    root["ApproximationCB"]["gRectangleQueryBudget"] = method->rectangleQueryBudget;
    bindLighting(root, "ApproximationCB");
    beginTiming(slot.timing);
    slot.pDeferredPass->execute(pRenderContext, mViewWidth, mOutputHeight);
    endTiming(slot.timing);
}

void NclsViewer::renderPackagePath(RenderContext* pRenderContext, ComparisonSlotRuntime& slot)
{
    const auto* method = slotProgram(slot);
    if (!method || !slot.pPathPass) return;
    const uint32_t samplesThisFrame = std::min(
        mSamplesPerFrame, kCapturePathTracingSpp - std::min(slot.spp, kCapturePathTracingSpp));
    if (samplesThisFrame == 0u) return;
    const uint32_t next = 1u - slot.ping;
    const bool accumulate = !mCameraDragging && !mPanDragging;
    auto root = slot.pPathPass->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, root["gScene"]);
    root["gPreviousPackage"] = slot.pAccumulated[slot.ping];
    root["gNextPackage"] = slot.pAccumulated[next];
    root["gEnvironment"] = mpEnvironment;
    root["gEnvironmentMarginalCdf"] = mpEnvironmentMarginalCdf;
    root["gEnvironmentConditionalCdf"] = mpEnvironmentConditionalCdf;
    root["gLinearSampler"] = mpLinearSampler;
    root["gPackageNoiseStats"] = slot.pNoiseStats;
    bindProgramResources(root, slot);
    auto constants = root["PackagePathTracerCB"];
    constants["gFrameDim"] = uint2(mViewWidth, mOutputHeight);
    constants["gFrameIndex"] = mFrameIndex;
    constants["gPackageSpp"] = slot.spp;
    constants["gSamplesThisFrame"] = samplesThisFrame;
    constants["gMaxSceneBounces"] = mMaxSceneBounces;
    constants["gResetAccumulation"] = uint32_t(slot.resetAccumulation);
    constants["gAccumulate"] = uint32_t(accumulate);
    constants["gCompiledMaterialIndex"] = method->compiledMaterialIndex;
    std::string extension = mReferenceGeometryPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    constants["gFlipTexCoordV"] = uint32_t(extension == ".obj");
    constants["gEnvironmentSamplingDimensions"] = mEnvironmentSamplingDimensions;
    bindLighting(root, "PackagePathTracerCB");
    pRenderContext->clearUAV(slot.pNoiseStats->getUAV().get(), uint4(0u));
    beginTiming(slot.timing);
    slot.pPathPass->execute(pRenderContext, mViewWidth, mOutputHeight);
    endTiming(slot.timing);
    slot.ping = next;
    if (accumulate) slot.spp += samplesThisFrame;
    else slot.spp = 0u;
    slot.resetAccumulation = false;
}

void NclsViewer::renderComposite(RenderContext* pRenderContext)
{
    auto root = mpCompositePass->getRootVar();
    const auto slot0 = slotOutput(mComparisonSlots[0]);
    const auto slot1 = slotOutput(mComparisonSlots[1]);
    root["gSlot0"] = slot0 ? slot0 : mpEmptySlot;
    root["gSlot1"] = slot1 ? slot1 : mpEmptySlot;
    root["gLinearSampler"] = mpLinearSampler;
    root["gComparisonLinear"] = mpComparisonLinear;
    root["gDisplay"] = mpDisplay;
    root["gDifferenceLinear"] = mpDifferenceLinear;
    root["gDifferenceDisplay"] = mpDifferenceDisplay;
    auto constants = root["CompositeCB"];
    constants["gOutputDim"] = uint2(mOutputWidth, mOutputHeight);
    constants["gComparisonMode"] = mComparisonMode;
    constants["gExposure"] = mExposure;
    constants["gDifferenceScale"] = mDifferenceScale;
    constants["gSlotReady"] = uint2(
        uint32_t(mComparisonSlots[0].ready()), uint32_t(mComparisonSlots[1].ready()));
    beginTiming(mCompositeTiming);
    mpCompositePass->execute(pRenderContext, mOutputWidth, mOutputHeight);
    endTiming(mCompositeTiming);
}

void NclsViewer::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    if (mVisibilityDirty) renderVisibility(pRenderContext);
    for (auto& slot : mComparisonSlots)
    {
        if (!slot.ready()) continue;
        if (slot.sourceReference)
        {
            if (!mFreezeReference) renderReference(pRenderContext, slot);
        }
        else if (slot.contract.mode == ncls::SlotMode::PathTracing)
        {
            if (!mFreezeReference) renderPackagePath(pRenderContext, slot);
        }
        else renderApproximation(pRenderContext, slot);
    }
    renderComposite(pRenderContext);
    pRenderContext->blit(mpDisplay->getSRV(), pTargetFbo->getRenderTargetView(0));
    ++mFrameIndex;

    const bool captureSppReady = std::all_of(
        mComparisonSlots.begin(), mComparisonSlots.end(), [](const ComparisonSlotRuntime& slot) {
            return !slot.ready() || slot.contract.mode != ncls::SlotMode::PathTracing
                || slot.spp == kCapturePathTracingSpp;
        });
    if (mOptions.headless && ++mRenderedFrames >= mOptions.frameCount && captureSppReady)
    {
        capture(mOptions.captureManifest);
        shutdown(0);
    }
}

void NclsViewer::renderOpenPbrUi(Gui::Widgets& widgets)
{
    auto& values = mReferenceSource.openPbrInputs;
    bool changed = false;
    auto color = [&](const char* label, uint32_t offset) {
        float3 value(values[offset], values[offset + 1u], values[offset + 2u]);
        if (!widgets.rgbColor(label, value)) return false;
        values[offset] = value.x;
        values[offset + 1u] = value.y;
        values[offset + 2u] = value.z;
        return true;
    };
    auto scalar = [&](const char* label, uint32_t offset, float minimum, float maximum, float step) {
        return widgets.var(label, values[offset], minimum, maximum, step);
    };

    widgets.text("OpenPBR 1.1.1 resolved native parameters");
    {
        Gui::Group group = widgets.group("Base", true);
        if (group)
        {
            changed |= scalar("base_weight", 0, 0.f, 1.f, .01f);
            changed |= color("base_color (linear RGB)", 1);
            changed |= scalar("base_diffuse_roughness", 4, 0.f, 1.f, .01f);
            changed |= scalar("base_metalness", 5, 0.f, 1.f, .01f);
        }
    }
    {
        Gui::Group group = widgets.group("Subsurface", false);
        if (group)
        {
            changed |= scalar("subsurface_weight", 6, 0.f, 1.f, .01f);
            changed |= color("subsurface_color (linear RGB)", 7);
            changed |= scalar("subsurface_radius", 10, 0.f, 100.f, .01f);
            changed |= color("subsurface_radius_scale", 11);
            changed |= scalar("subsurface_scatter_anisotropy", 14, -1.f, 1.f, .01f);
        }
    }
    {
        Gui::Group group = widgets.group("Specular", true);
        if (group)
        {
            changed |= scalar("specular_weight", 15, 0.f, 1.f, .01f);
            changed |= color("specular_color (linear RGB)", 16);
            changed |= scalar("specular_roughness", 19, .001f, 1.f, .005f);
            changed |= scalar("specular_roughness_anisotropy", 20, 0.f, 1.f, .01f);
            changed |= scalar("specular_ior", 21, 1.001f, 3.f, .01f);
            float rotation = std::atan2(values[23], values[22]);
            if (group.var("specular_anisotropy_rotation", rotation, -3.14159f, 3.14159f, .01f))
            {
                values[22] = std::cos(rotation);
                values[23] = std::sin(rotation);
                changed = true;
            }
        }
    }
    {
        Gui::Group group = widgets.group("Coat and fuzz", false);
        if (group)
        {
            changed |= scalar("coat_weight", 24, 0.f, 1.f, .01f);
            changed |= color("coat_color (linear RGB)", 25);
            changed |= scalar("coat_roughness", 28, .001f, 1.f, .005f);
            changed |= scalar("coat_roughness_anisotropy", 29, 0.f, 1.f, .01f);
            changed |= scalar("coat_ior", 30, 1.001f, 3.f, .01f);
            changed |= scalar("coat_darkening", 31, 0.f, 1.f, .01f);
            float rotation = std::atan2(values[33], values[32]);
            if (group.var("coat_anisotropy_rotation", rotation, -3.14159f, 3.14159f, .01f))
            {
                values[32] = std::cos(rotation);
                values[33] = std::sin(rotation);
                changed = true;
            }
            changed |= scalar("fuzz_weight", 34, 0.f, 1.f, .01f);
            changed |= color("fuzz_color (linear RGB)", 35);
            changed |= scalar("fuzz_roughness", 38, .001f, 1.f, .005f);
        }
    }
    {
        Gui::Group group = widgets.group("Transmission and thin film", false);
        if (group)
        {
            changed |= scalar("transmission_weight", 39, 0.f, 1.f, .01f);
            changed |= color("transmission_color (linear RGB)", 40);
            changed |= scalar("transmission_depth", 43, 0.f, 100.f, .01f);
            changed |= color("transmission_scatter (linear RGB)", 44);
            changed |= scalar("transmission_scatter_anisotropy", 47, -1.f, 1.f, .01f);
            changed |= scalar("transmission_dispersion_scale", 48, 0.f, 1.f, .01f);
            changed |= scalar("transmission_dispersion_abbe_number", 49, 1.f, 100.f, .1f);
            changed |= scalar("thin_film_weight", 50, 0.f, 1.f, .01f);
            changed |= scalar("thin_film_thickness", 51, 0.f, 2000.f, 1.f);
            changed |= scalar("thin_film_ior", 52, 1.001f, 3.f, .01f);
        }
    }
    {
        Gui::Group group = widgets.group("Emission and geometry", false);
        if (group)
        {
            changed |= scalar("emission_luminance", 53, 0.f, 100000.f, 1.f);
            changed |= color("emission_color (linear RGB)", 54);
            changed |= scalar("geometry_opacity", 57, 0.f, 1.f, .01f);
            bool thinWalled = values[58] != 0.f;
            if (group.checkbox("geometry_thin_walled", thinWalled))
            {
                values[58] = thinWalled ? 1.f : 0.f;
                changed = true;
            }
        }
    }
    if (changed)
    {
        mFreezeReference = false;
        updateReferenceSourceBuffer();
        mStatus = "OpenPBR parameters applied; reference resumed and accumulation reset.";
    }
}

void NclsViewer::renderMaterialXUi(Gui::Widgets& widgets)
{
    auto& values = mReferenceSource.materialXInputs;
    bool changed = false;
    auto color = [&](const char* label, uint32_t offset) {
        float3 value(values[offset], values[offset + 1u], values[offset + 2u]);
        if (!widgets.rgbColor(label, value)) return false;
        values[offset] = value.x;
        values[offset + 1u] = value.y;
        values[offset + 2u] = value.z;
        return true;
    };

    widgets.text("MaterialX standard_surface resolved inputs");
    {
        Gui::Group group = widgets.group("Base", true);
        if (group)
        {
            changed |= group.var("base", values[0], 0.f, 1.f, .01f);
            if (values[6] == 0.f) changed |= color("base_color (linear RGB)", 1);
            else group.text("base_color: driven by the source texture");
            changed |= group.var("diffuse_roughness", values[4], 0.f, 1.f, .01f);
            if (values[7] == 0.f) changed |= group.var("metalness", values[5], 0.f, 1.f, .01f);
            else group.text("metalness: driven by the source texture");
        }
    }
    {
        Gui::Group group = widgets.group("Specular", true);
        if (group)
        {
            changed |= group.var("specular", values[8], 0.f, 1.f, .01f);
            changed |= color("specular_color (linear RGB)", 9);
            if (values[13] == 0.f) changed |= group.var("specular_roughness", values[12], .001f, 1.f, .005f);
            else group.text("specular_roughness: driven by the source texture");
            changed |= group.var("specular_IOR", values[14], 1.001f, 3.f, .01f);
            changed |= group.var("specular_anisotropy", values[15], 0.f, .98f, .01f);
            changed |= group.var("specular_rotation", values[16], 0.f, 1.f, .01f);
        }
    }
    {
        Gui::Group group = widgets.group("Normal, emission and opacity", false);
        if (group)
        {
            if (values[18] != 0.f) changed |= group.var("normal scale", values[17], 0.f, 4.f, .01f);
            else group.text("normal: no texture connected");
            changed |= group.var("emission", values[19], 0.f, 1000.f, .01f);
            changed |= color("emission_color (linear RGB)", 20);
            changed |= group.var("opacity", values[23], 0.f, 1.f, .01f);
        }
    }
    if (changed)
    {
        mFreezeReference = false;
        updateReferenceSourceBuffer();
        mStatus = "MaterialX parameters applied; reference resumed and accumulation reset.";
    }
}

void NclsViewer::renderMaterialUi(Gui::Widgets& widgets)
{
    if (mReferenceSource.family != ncls::ReferenceFamily::LayerStack)
    {
        widgets.text("Source family: " + std::string(mReferenceSource.familyId()));
        widgets.text("Native reference: " + mReferenceSource.displayName);
        widgets.text("Source file: " + mReferenceSource.sourcePath.string());
        widgets.text("Source asset SHA-256: " + shortId(mReferenceSource.sourceSha256));
        widgets.text("Edited state SHA-256: " + shortId(ncls::referenceSourceStateHash(mReferenceSource)));
        if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX)
        {
            widgets.text("Falcor query: MaterialX 1.39.4 standard_surface + source textures");
            widgets.text("Geometry contract: reference path traces the Falcor scene; SHA-256 "
                + shortId(mReferenceGeometrySha256));
            widgets.text("The displacement graph remains in the source document and is outside this surface-response query");
        }
        if (mReferenceSource.family == ncls::ReferenceFamily::OpenPbr) renderOpenPbrUi(widgets);
        else if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX) renderMaterialXUi(widgets);
        else widgets.text("MERL is a measured BRDF table with no continuous native controls; select another measurement to switch material.");
        widgets.text("This source material retains its native representation; no compatible approximation compiler is available yet.");
        return;
    }
    widgets.text("Material program (edits normalized LayerStackIR inputs, not a K2 packet)");
    mSelectedInterface = std::min(mSelectedInterface, mMaterial.interfaceCount - 1);
    Gui::DropdownList layers;
    for (uint32_t index = 0; index < mMaterial.interfaceCount; ++index)
    {
        const bool base = index + 1 == mMaterial.interfaceCount;
        layers.push_back({index, (base ? "Base " : "Coat ") + std::to_string(index)});
    }
    widgets.dropdown("Current interface", layers, mSelectedInterface);
    bool changed = false;
    if (widgets.button("Add dielectric coat"))
    {
        changed |= ncls::addDielectricCoat(mMaterial);
        mSelectedInterface = mMaterial.interfaceCount - 2;
    }
    if (mSelectedInterface + 1 < mMaterial.interfaceCount)
    {
        if (widgets.button("Delete current coat", true))
        {
            changed |= ncls::removeCoat(mMaterial, mSelectedInterface);
            mSelectedInterface = std::min(mSelectedInterface, mMaterial.interfaceCount - 1);
        }
        if (widgets.button("Move up", true)) changed |= ncls::moveCoat(mMaterial, mSelectedInterface, -1);
        if (widgets.button("Move down", true)) changed |= ncls::moveCoat(mMaterial, mSelectedInterface, 1);
    }
    auto& interfaceValue = mMaterial.interfaces[mSelectedInterface];
    const bool isBase = mSelectedInterface + 1 == mMaterial.interfaceCount;
    if (isBase)
    {
        uint32_t kind = interfaceValue.kind;
        if (widgets.dropdown("Base type", kBaseKinds, kind))
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
        changed |= widgets.var("alpha X", interfaceValue.alphaX, 0.001f, 1.f, 0.005f);
        changed |= widgets.var("alpha Y", interfaceValue.alphaY, 0.001f, 1.f, 0.005f);
        changed |= widgets.var("Tangent rotation (rad)", interfaceValue.tangentRotation, -3.14159f, 3.14159f, 0.01f);
    }
    if (kind == ncls::InterfaceKind::RoughDielectric)
        changed |= widgets.var("Relative IOR", interfaceValue.relativeIor, 1.001f, 3.f, 0.01f);
    else if (kind == ncls::InterfaceKind::RoughConductor)
    {
        widgets.text("Rough-conductor color is determined by eta/k; it has no albedo color input.");
        float3 eta(interfaceValue.etaR, interfaceValue.etaG, interfaceValue.etaB);
        float3 k(interfaceValue.kR, interfaceValue.kG, interfaceValue.kB);
        if (widgets.var("eta RGB", eta, 0.f, 5.f, 0.01f))
        {
            interfaceValue.etaR = eta.x; interfaceValue.etaG = eta.y; interfaceValue.etaB = eta.z; changed = true;
        }
        if (widgets.var("k RGB", k, 0.f, 10.f, 0.01f))
        {
            interfaceValue.kR = k.x; interfaceValue.kG = k.y; interfaceValue.kB = k.z; changed = true;
        }
    }
    else if (kind == ncls::InterfaceKind::Diffuse || kind == ncls::InterfaceKind::Sheen)
    {
        float3 color(interfaceValue.colorR, interfaceValue.colorG, interfaceValue.colorB);
        if (widgets.rgbColor("Color (linear RGB)", color))
        {
            interfaceValue.colorR = color.x; interfaceValue.colorG = color.y; interfaceValue.colorB = color.z; changed = true;
        }
        if (kind == ncls::InterfaceKind::Sheen)
        {
            if (widgets.var("Sheen roughness", interfaceValue.alphaX, 0.001f, 1.f, 0.005f))
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
        mediumChanged |= widgets.var("Medium extinction (1/unit)", extinction, 0.f, 6.f, 0.01f);
        mediumChanged |= widgets.rgbColor("Medium scattering albedo (linear RGB)", albedo);
        mediumChanged |= widgets.var("Phase-function g", medium.g, -0.95f, 0.95f, 0.01f);
        mediumChanged |= widgets.var("Thickness", medium.thickness, 0.f, 2.f, 0.01f);
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
        mFreezeReference = false;
        try { updateMaterialBuffer(); mStatus = "Material applied; reference resumed and accumulation reset."; }
        catch (const std::exception& error) { mStatus = error.what(); }
    }
    widgets.text("IR SHA-256: " + shortId(ncls::layerStackHash(mMaterial)));
    widgets.text("Scene-state SHA-256: " + shortId(ncls::referenceSourceStateHash(mReferenceSource)));
}

void NclsViewer::onGuiRender(Gui* pGui)
{
    Gui::Window window(pGui, "NeuralShading Viewer", {460, 850}, {12, 12});
    const float sceneRadius = mpScene ? std::max(mpScene->getSceneBounds().radius(), 0.01f) : 1.f;
    const float3 sceneCenter = mpScene ? mpScene->getSceneBounds().center() : float3(0.f);
    const float sceneCenterExtent = std::max({
        std::abs(sceneCenter.x),
        std::abs(sceneCenter.y),
        std::abs(sceneCenter.z),
    });
    const float lightCoordinateLimit = std::max(10.f, sceneCenterExtent + 4.f * sceneRadius);
    const float lightAxisLimit = std::max(3.f, 2.f * sceneRadius);
    const float lightPositionStep = std::max(0.02f, sceneRadius * 0.005f);
    const float minimumDistance = mpScene ? sceneRadius * 0.01f : 1.25f;
    const float maximumDistance = mpScene ? sceneRadius * 20.f : 9.f;

    {
        Gui::Group group = window.group("Scene and camera", true);
        if (group)
        {
            bool physicalChanged = false;
            if (mpScene)
            {
                group.text("Scene: " + mReferenceGeometryPath.filename().string());
                if (mSelectedSceneInstance != std::numeric_limits<uint32_t>::max())
                {
                    group.text("Selected instance: " + std::to_string(mSelectedSceneInstance));
                    group.text("Geometry: " + mSelectedSceneGeometryName);
                    group.text("Material slot: " + mSelectedSceneMaterialName
                        + " (#" + std::to_string(mSelectedSceneMaterial) + ")");
                }
                else group.text("Click an object to select its material slot.");
            }
            if (group.button("Load scene"))
            {
                std::filesystem::path path;
                if (openFileDialog(Scene::getFileExtensionFilters(), path))
                {
                    try { loadScene(path); }
                    catch (const std::exception& error) { mStatus = "Failed to load scene: " + std::string(error.what()); }
                }
            }
            if (group.button("Load viewer scene"))
            {
                std::filesystem::path path;
                if (openFileDialog({{"json", "NeuralShading viewer scene JSON"}}, path))
                {
                    try { loadViewerScene(path); }
                    catch (const std::exception& error) { mStatus = "Failed to load viewer scene: " + std::string(error.what()); }
                }
            }
            if (mpScene && group.button("Save viewer scene", true))
            {
                std::filesystem::path path = "viewer-scene.json";
                if (saveFileDialog({{"json", "NeuralShading viewer scene JSON"}}, path))
                {
                    try { saveViewerScene(path); }
                    catch (const std::exception& error) { mStatus = "Failed to save viewer scene: " + std::string(error.what()); }
                }
            }
            physicalChanged |= group.var(
                "Camera distance", mCamera.distance,
                minimumDistance, maximumDistance, sceneRadius * 0.002f);
            physicalChanged |= group.var("Vertical FOV", mCamera.verticalFovDegrees, 12.f, 90.f, 0.5f);
            if (physicalChanged) resetReference(true);
            if (group.button("Reset camera")) resetCamera();
        }
    }

    {
        Gui::Group group = window.group("Material", true);
        if (group)
        {
            group.text("Editing material slot #" + std::to_string(mActiveSceneMaterial));
            uint32_t requestedFamily = static_cast<uint32_t>(mReferenceSource.family);
            if (group.dropdown("Source material family", kReferenceFamilies, requestedFamily)
                && requestedFamily != static_cast<uint32_t>(mReferenceSource.family))
            {
                const auto family = static_cast<ncls::ReferenceFamily>(requestedFamily);
                if (family == ncls::ReferenceFamily::LayerStack)
                {
                    try
                    {
                        installReferenceSource(ncls::makeDefaultReferenceSource(family));
                        mStatus = "Created a default " + std::string(mReferenceSource.familyId())
                            + " source material for the selected scene slot.";
                    }
                    catch (const std::exception& error) { mStatus = "Failed to switch source family: " + std::string(error.what()); }
                }
                else
                {
                    std::filesystem::path path;
                    const FileDialogFilterVec filters = family == ncls::ReferenceFamily::Merl
                        ? FileDialogFilterVec{{"binary", "MERL BRDF table"}}
                        : family == ncls::ReferenceFamily::OpenPbr
                            ? FileDialogFilterVec{{"json", "OpenPBR resolved adapter"}}
                            : FileDialogFilterVec{{"mtlx", "MaterialX document"}};
                    if (openFileDialog(filters, path)) loadMaterial(path);
                    else mStatus = "Family switch cancelled; resource-backed families require a native source asset.";
                }
            }
            renderMaterialUi(group);
            if (group.button("Load source material into selected slot"))
            {
                std::filesystem::path path;
                if (openFileDialog({
                        {"json", "MaterialProgram / OpenPBR JSON"},
                        {"binary", "MERL BRDF table"},
                        {"mtlx", "MaterialX document"}}, path))
                    loadMaterial(path);
            }
            if ((mReferenceSource.family == ncls::ReferenceFamily::LayerStack
                    || mReferenceSource.family == ncls::ReferenceFamily::OpenPbr)
                && group.button(mReferenceSource.family == ncls::ReferenceFamily::LayerStack
                        ? "Save MaterialProgram" : "Save OpenPBR source JSON", true))
            {
                std::filesystem::path path = mMaterialPath;
                if (saveFileDialog({{"json", "MaterialProgram JSON"}}, path)) saveMaterial(path);
            }
        }
    }

    {
        Gui::Group group = window.group("Lighting", true);
        if (group)
        {
            group.text("Colors are linear RGB. Editing an enabled light resumes the reference.");
            bool lightChanged = false;
            {
                Gui::Group light = group.group("Environment / HDRI", true);
                if (light)
                {
                    lightChanged |= light.checkbox("Enabled", mLighting.useEnvironment);
                    if (mLighting.useEnvironment)
                    {
                        lightChanged |= light.var("Rotation", mLighting.environmentRotation, -3.14159f, 3.14159f, 0.01f);
                        lightChanged |= light.var("Intensity", mLighting.environmentIntensity, 0.f, 20.f, 0.02f);
                    }
                    else light.text("Disabled: this HDRI does not contribute to the image.");
                    if (light.button("Load HDRI"))
                    {
                        std::filesystem::path path;
                        if (openFileDialog(Bitmap::getFileDialogFilters(ResourceFormat::RGBA32Float), path))
                        {
                            try { loadEnvironment(path); }
                            catch (const std::exception& error) { mStatus = "Failed to load HDRI: " + std::string(error.what()); }
                        }
                    }
                }
            }
            {
                Gui::Group light = group.group("Directional light", true);
                if (light)
                {
                    lightChanged |= light.checkbox("Enabled", mLighting.useSun);
                    if (mLighting.useSun)
                    {
                        lightChanged |= light.direction("Direction (surface -> light)", mLighting.sunDirection);
                        lightChanged |= light.var("Intensity", mLighting.sunIntensity, 0.f, 50.f, 0.05f);
                        lightChanged |= light.rgbColor("Color (linear RGB)", mLighting.sunColor);
                    }
                    else light.text("Disabled: its direction, intensity and color do not affect the image.");
                }
            }
            {
                Gui::Group light = group.group("Point light", false);
                if (light)
                {
                    lightChanged |= light.checkbox("Enabled", mLighting.usePoint);
                    if (mLighting.usePoint)
                    {
                        lightChanged |= light.var(
                            "Position", mLighting.pointPosition,
                            -lightCoordinateLimit, lightCoordinateLimit, lightPositionStep);
                        lightChanged |= light.var("Intensity", mLighting.pointIntensity, 0.f, 100.f, 0.1f);
                        lightChanged |= light.rgbColor("Color (linear RGB)", mLighting.pointColor);
                    }
                    else light.text("Disabled: its position, intensity and color do not affect the image.");
                }
            }
            {
                Gui::Group light = group.group("Rectangle light", true);
                if (light)
                {
                    lightChanged |= light.checkbox("Enabled", mLighting.useRectangle);
                    if (mLighting.useRectangle)
                    {
                        lightChanged |= light.var(
                            "Center", mLighting.rectangleCenter,
                            -lightCoordinateLimit, lightCoordinateLimit, lightPositionStep);
                        lightChanged |= light.var(
                            "Half-axis U", mLighting.rectangleAxisU,
                            -lightAxisLimit, lightAxisLimit, lightPositionStep);
                        lightChanged |= light.var(
                            "Half-axis V", mLighting.rectangleAxisV,
                            -lightAxisLimit, lightAxisLimit, lightPositionStep);
                        light.text("Emitting normal = normalize(U x V)");
                        lightChanged |= light.var("Intensity", mLighting.rectangleIntensity, 0.f, 100.f, 0.1f);
                        lightChanged |= light.rgbColor("Color (linear RGB)", mLighting.rectangleColor);
                    }
                    else light.text("Disabled: its shape, intensity and color do not affect the image.");
                }
            }
            if (lightChanged)
            {
                mFreezeReference = false;
                resetReference(false);
                mStatus = "Lighting applied; reference resumed and accumulation reset.";
            }
        }
    }

    {
        Gui::Group group = window.group("Reference and display", true);
        if (group)
        {
            group.text("Reference: " + std::string(mReferenceSource.familyId()));
            group.var("Samples per frame", mSamplesPerFrame, 1u, 16u);
            if (mpScene && group.var("Max scene bounces", mMaxSceneBounces, 0u, 16u))
                resetReference(false);
            if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack
                && group.var("Max layer-walk depth", mMaxLayerWalkDepth, 4u, 128u))
                resetReference(false);
            group.checkbox("Freeze reference", mFreezeReference);
            group.text("Material and lighting edits automatically resume a frozen reference.");
            if (group.button("Clear accumulation")) resetReference(false);
            if (mComparisonSlots[0].ready() && mComparisonSlots[1].ready())
            {
                group.dropdown("Comparison display", kComparisonModes, mComparisonMode);
            }
            else
            {
                mComparisonMode = 0u;
                group.text("未 ready 的 slot 保持错误占位，不改变 peer extent 或 camera aspect。");
            }
            group.var("Shared exposure EV", mExposure, -8.f, 8.f, 0.05f);
            if (mComparisonMode == 3u) group.var("Error amplification", mDifferenceScale, 1.f, 100.f, 0.5f);
            group.text("slot 0 spp: " + std::to_string(mComparisonSlots[0].spp)
                + ", elapsed: " + fmt::format("{:.2f}s", mAccumulationSeconds));
            group.text("Estimated mean relative standard error: "
                + fmt::format("{:.2f}%", 100.f * mEstimatedRelativeStandardError));
        }
    }

    {
        Gui::Group group = window.group("Comparison slots", false);
        if (group)
        {
            Gui::DropdownList methodList = {{0, "Empty"}, {1, "Source reference"}};
            for (uint32_t index = 0; index < mPrograms.size(); ++index)
            {
                if (allMaterialsSupportedBy(mPrograms[index]))
                    methodList.push_back({index + 2, mPrograms[index].displayName
                        + " [" + shortId(mPrograms[index].packageId) + "]"});
            }
            for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
            {
                auto& slot = mComparisonSlots[slotIndex];
                const std::string prefix = "Slot " + std::to_string(slotIndex);
                uint32_t selection = slot.uiValue;
                const std::string programLabel = prefix + " program";
                if (group.dropdown(programLabel.c_str(), methodList, selection))
                    activateComparisonSlot(slotIndex, selection);
                uint32_t mode = slot.contract.mode == ncls::SlotMode::PathTracing ? 0u : 1u;
                const std::string rendererLabel = prefix + " renderer";
                if (group.dropdown(rendererLabel.c_str(), kSlotModes, mode))
                {
                    slot.contract.mode = mode == 0u
                        ? ncls::SlotMode::PathTracing : ncls::SlotMode::Deferred;
                    activateComparisonSlot(slotIndex, slot.uiValue);
                }
                const char* status = slot.contract.status == ncls::SlotStatus::Ready ? "ready"
                    : slot.contract.status == ncls::SlotStatus::Unsupported ? "unsupported"
                    : slot.contract.status == ncls::SlotStatus::Error ? "error" : "empty";
                group.text(prefix + ": " + status + ", spp=" + std::to_string(slot.spp)
                    + ", GPU=" + fmt::format("{:.3f} ms", slot.timing.milliseconds));
                if (!slot.contract.diagnostic.empty()) group.textWrapped(slot.contract.diagnostic);
                if (const auto* method = slotProgram(slot))
                    group.text("package " + shortId(method->packageId)
                        + " / " + method->runtimeClass);
            }
            if (group.button("Rescan ScatteringPackages")) scanPackages();
            if (!mPackageFailures.empty())
                group.text("Rejected bundles: " + std::to_string(mPackageFailures.size()) + " (details are in the log)");
        }
    }

    {
        Gui::Group group = window.group("Capture", false);
        if (group && group.button("Save full capture"))
        {
            std::filesystem::path path = "capture.json";
            if (saveFileDialog({{"json", "Capture manifest"}}, path)) capture(path);
        }
    }

    {
        Gui::Group group = window.group("Performance and status", false);
        if (group)
        {
            group.text("GPU ms (asynchronous timestamp)");
            group.text(fmt::format(
                "visibility {:.3f} | slot 0 {:.3f}\nslot 1 {:.3f} | composite {:.3f}",
                mVisibilityTiming.milliseconds,
                mComparisonSlots[0].timing.milliseconds,
                mComparisonSlots[1].timing.milliseconds,
                mCompositeTiming.milliseconds));
            group.textWrapped("Controls: left-drag orbit; middle/right-drag pan; wheel dolly; Space freezes accumulation.");
            if (!mStatus.empty()) group.textWrapped("Status: " + mStatus);
            const auto logPath = Logger::getLogFilePath();
            if (!logPath.empty()) group.textWrapped("Detailed log: " + logPath.string());
        }
    }
}

void NclsViewer::loadMaterial(const std::filesystem::path& path)
{
    try
    {
        logInfo("Loading source material '{}'", path);
        installReferenceSource(ncls::loadReferenceSource(path), path);
        logInfo("Loaded source metadata: family='{}' hash='{}'", mReferenceSource.familyId(), shortId(mReferenceSource.sourceSha256));
        if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack)
            mStatus = "Loaded LayerStack MaterialProgram: " + path.string();
        else if (mReferenceSource.family == ncls::ReferenceFamily::Merl)
            mStatus = "Loaded native MERL measurement into the Falcor reference pass: " + path.string();
        else if (mReferenceSource.family == ncls::ReferenceFamily::OpenPbr)
            mStatus = "Loaded OpenPBR 1.1.1 resolved-input material into the Falcor reference pass: " + path.string();
        else if (mReferenceSource.family == ncls::ReferenceFamily::MaterialX)
        {
            mStatus = "Loaded MaterialX standard_surface and connected source textures into the scene reference path: "
                + path.string();
        }
    }
    catch (const std::exception& error)
    {
        mStatus = "Failed to load source material: " + std::string(error.what());
        if (mOptions.headless) throw;
    }
}

void NclsViewer::installReferenceSource(ncls::ReferenceSource source, const std::filesystem::path& path)
{
    auto gpu = createSourceGpuResources(source);
    mReferenceSource = std::move(source);
    mSourceGpu = std::move(gpu);
    mMaterialDisplayName = mReferenceSource.displayName;
    mMaterialPath = path.empty() ? mReferenceSource.sourcePath : path;
    if (mpScene)
    {
        rebuildReferenceMaterialMetadata();
        createSceneReferencePass();
    }
    if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack) updateMaterialBuffer();
    else resetReference(mReferenceSource.family == ncls::ReferenceFamily::MaterialX);
    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
        activateComparisonSlot(slotIndex, mComparisonSlots[slotIndex].uiValue);
}

void NclsViewer::saveMaterial(const std::filesystem::path& path)
{
    try
    {
        if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack)
            ncls::saveMaterialProgram(path, mMaterial, mMaterialDisplayName);
        else if (mReferenceSource.family == ncls::ReferenceFamily::OpenPbr)
            ncls::saveOpenPbrReferenceSource(path, mReferenceSource);
        else throw std::runtime_error("this source family is saved through the viewer scene because it depends on an external native asset");
        mMaterialPath = path;
        mStatus = "Saved source material: " + path.string();
    }
    catch (const std::exception& error) { mStatus = "Save failed: " + std::string(error.what()); }
}

void NclsViewer::loadEnvironment(const std::filesystem::path& path, const std::string& expectedSha256)
{
    const auto resolvedPath = std::filesystem::absolute(path).lexically_normal();
    if (!std::filesystem::is_regular_file(resolvedPath))
        throw std::runtime_error("HDRI does not exist: " + resolvedPath.string());
    const std::string environmentSha256 = ncls::sha256FileHex(resolvedPath);
    if (!expectedSha256.empty() && environmentSha256 != expectedSha256)
        throw std::runtime_error("HDRI SHA-256 does not match the viewer scene/preset/replay: " + resolvedPath.string());
    auto texture = Texture::createFromFile(getDevice(), resolvedPath, true, false);
    if (!texture) throw std::runtime_error("Falcor failed to load HDRI: " + resolvedPath.string());
    mpEnvironment = texture;
    std::string extension = resolvedPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (extension == ".exr")
    {
        uint32_t width = 0u;
        uint32_t height = 0u;
        const auto pixels = readHalfRgbaExrPixels(resolvedPath, width, height);
        rebuildEnvironmentSampling(pixels, width, height);
    }
    else rebuildEnvironmentSampling({}, 0u, 0u);
    mEnvironmentPath = resolvedPath;
    mEnvironmentSha256 = environmentSha256;
    resetReference(false);
    mStatus = "Loaded HDRI: " + resolvedPath.string();
}

void NclsViewer::saveViewerScene(const std::filesystem::path& requestedPath)
{
    namespace fs = std::filesystem;
    if (!mpScene) throw std::runtime_error("a viewer scene requires an explicit Falcor scene");
    fs::path path = requestedPath;
    if (path.extension() != ".json") path += ".json";
    path = fs::absolute(path).lexically_normal();
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    const fs::path base = path.parent_path();

    nlohmann::json bindings = nlohmann::json::array();
    auto appendBinding = [&](uint32_t materialId, const ncls::ReferenceSource& source) {
        std::string sceneMaterialName;
        if (const auto material = mpScene->getMaterial(MaterialID(materialId)))
            sceneMaterialName = material->getName();
        bindings.push_back({
            {"material_id", materialId},
            {"scene_material_name", sceneMaterialName},
            {"source", ncls::serializeReferenceSourceState(source, base)},
        });
    };
    appendBinding(mActiveSceneMaterial, mReferenceSource);
    for (uint32_t materialId = 0u; materialId < mpScene->getMaterialCount(); ++materialId)
        if (materialId != mActiveSceneMaterial)
        {
            const auto* binding = inactiveSceneMaterial(materialId);
            if (!binding) throw std::runtime_error("scene material slot has no source binding");
            appendBinding(materialId, binding->source);
        }

    const nlohmann::json document = {
        {"format_name", "ncls.viewer-scene"},
        {"format_version", 1},
        {"reference_integrator", "ncls.scene-path-tracer@1"},
        {"geometry", {
            {"uri", portableUri(mReferenceGeometryPath, base)},
            {"sha256", mReferenceGeometrySha256},
        }},
        {"environment", {
            {"uri", portableUri(mEnvironmentPath, base)},
            {"sha256", mEnvironmentSha256},
        }},
        {"active_material_id", mActiveSceneMaterial},
        {"material_bindings", bindings},
        {"reference", {
            {"samples_per_frame", mSamplesPerFrame},
            {"scene_max_bounces", mMaxSceneBounces},
            {"layer_walk_max_depth", mMaxLayerWalkDepth},
        }},
        {"camera", {
            {"target", {mCamera.target.x, mCamera.target.y, mCamera.target.z}},
            {"yaw", mCamera.yaw}, {"pitch", mCamera.pitch}, {"distance", mCamera.distance},
            {"vertical_fov_degrees", mCamera.verticalFovDegrees},
        }},
        {"display", {
            {"exposure_ev", mExposure},
        }},
        {"lighting", {
            {"use_environment", mLighting.useEnvironment},
            {"environment_rotation", mLighting.environmentRotation},
            {"environment_intensity", mLighting.environmentIntensity},
            {"use_sun", mLighting.useSun},
            {"sun_direction_to_light", {mLighting.sunDirection.x, mLighting.sunDirection.y, mLighting.sunDirection.z}},
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
    };
    const auto temporary = path.string() + ".tmp";
    std::error_code error;
    fs::remove(temporary, error);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot write viewer scene: " + path.string());
        stream << document.dump(2) << '\n';
    }
    fs::remove(path, error);
    fs::rename(temporary, path);
    mStatus = "Saved viewer scene with all material-slot bindings: " + path.string();
}

void NclsViewer::loadViewerScene(const std::filesystem::path& requestedPath)
{
    namespace fs = std::filesystem;
    const fs::path path = fs::absolute(requestedPath).lexically_normal();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open viewer scene: " + path.string());
    const nlohmann::json document = nlohmann::json::parse(stream);
    if (document.value("format_name", "") != "ncls.viewer-scene" || document.value("format_version", 0u) != 1u)
        throw std::runtime_error("unsupported viewer scene format: " + path.string());
    if (document.value("reference_integrator", "") != "ncls.scene-path-tracer@1")
        throw std::runtime_error("viewer scene requires reference_integrator ncls.scene-path-tracer@1");
    const fs::path base = path.parent_path();
    const auto& geometry = document.at("geometry");
    const std::string geometryUri = geometry.at("uri").get<std::string>();
    const std::string geometryHash = geometry.at("sha256").get<std::string>();
    if (geometryUri.empty() || !isSha256(geometryHash))
        throw std::runtime_error("viewer scene geometry requires a URI and lowercase SHA-256");
    loadScene(resolveUri(geometryUri, base), geometryHash);

    const auto& environment = document.at("environment");
    const auto environmentPath = resolveUri(environment.value("uri", std::string()), base);
    const std::string environmentHash = environment.value("sha256", std::string());
    if (environmentPath.empty())
    {
        if (!environmentHash.empty()) throw std::runtime_error("viewer scene default environment must not declare a file hash");
        createDefaultEnvironment();
    }
    else
    {
        if (!isSha256(environmentHash))
            throw std::runtime_error("viewer scene environment requires a lowercase SHA-256");
        loadEnvironment(environmentPath, environmentHash);
    }

    const auto vector3 = [](const nlohmann::json& value) {
        if (!value.is_array() || value.size() != 3u)
            throw std::runtime_error("viewer scene vector must contain three values");
        return float3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    };
    const auto& reference = document.at("reference");
    mSamplesPerFrame = std::clamp(reference.at("samples_per_frame").get<uint32_t>(), 1u, 16u);
    mMaxSceneBounces = std::clamp(reference.at("scene_max_bounces").get<uint32_t>(), 0u, 16u);
    mMaxLayerWalkDepth = std::clamp(reference.at("layer_walk_max_depth").get<uint32_t>(), 4u, 128u);
    const auto& camera = document.at("camera");
    mCamera.target = vector3(camera.at("target"));
    mCamera.yaw = camera.at("yaw").get<float>();
    mCamera.pitch = camera.at("pitch").get<float>();
    mCamera.distance = camera.at("distance").get<float>();
    mCamera.verticalFovDegrees = camera.at("vertical_fov_degrees").get<float>();
    const auto& display = document.at("display");
    mExposure = display.at("exposure_ev").get<float>();
    const auto& lighting = document.at("lighting");
    mLighting.useEnvironment = lighting.at("use_environment").get<bool>();
    mLighting.environmentRotation = lighting.at("environment_rotation").get<float>();
    mLighting.environmentIntensity = lighting.at("environment_intensity").get<float>();
    mLighting.useSun = lighting.at("use_sun").get<bool>();
    mLighting.sunDirection = vector3(lighting.at("sun_direction_to_light"));
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

    std::unordered_map<uint32_t, MaterialSlotBinding> loadedBindings;
    for (const auto& bindingDocument : document.at("material_bindings"))
    {
        const uint32_t materialId = bindingDocument.at("material_id").get<uint32_t>();
        if (materialId >= mpScene->getMaterialCount())
            throw std::runtime_error("viewer scene material binding is outside the Falcor scene material range");
        MaterialSlotBinding binding;
        binding.source = ncls::deserializeReferenceSourceState(bindingDocument.at("source"), base);
        binding.gpu = createSourceGpuResources(binding.source);
        binding.materialPath = binding.source.sourcePath;
        binding.displayName = binding.source.displayName;
        if (!loadedBindings.emplace(materialId, std::move(binding)).second)
            throw std::runtime_error("viewer scene contains a duplicate material binding");
    }
    if (loadedBindings.size() != mpScene->getMaterialCount())
        throw std::runtime_error("viewer scene must bind every Falcor material slot exactly once");
    const uint32_t activeMaterialId = document.at("active_material_id").get<uint32_t>();
    auto active = loadedBindings.find(activeMaterialId);
    if (active == loadedBindings.end()) throw std::runtime_error("viewer scene active material ID has no binding");

    mReferenceSource = std::move(active->second.source);
    mSourceGpu = std::move(active->second.gpu);
    mMaterialPath = std::move(active->second.materialPath);
    mMaterialDisplayName = std::move(active->second.displayName);
    loadedBindings.erase(active);
    mInactiveSceneMaterials = std::move(loadedBindings);
    mActiveSceneMaterial = activeMaterialId;
    mSelectedSceneMaterial = activeMaterialId;
    mSelectedSceneInstance = std::numeric_limits<uint32_t>::max();
    for (uint32_t instanceId = 0u; instanceId < mpScene->getGeometryInstanceCount(); ++instanceId)
    {
        const auto& instance = mpScene->getGeometryInstance(instanceId);
        if (instance.materialID != activeMaterialId) continue;
        mSelectedSceneInstance = instanceId;
        if (instance.getType() == Scene::GeometryType::TriangleMesh
            || instance.getType() == Scene::GeometryType::DisplacedTriangleMesh)
            mSelectedSceneGeometryName = mpScene->getMeshName(instance.geometryID);
        else mSelectedSceneGeometryName = "Geometry #" + std::to_string(instance.geometryID);
        break;
    }
    if (const auto material = mpScene->getMaterial(MaterialID(activeMaterialId)))
        mSelectedSceneMaterialName = material->getName();
    rebuildReferenceMaterialMetadata();
    createSceneReferencePass();
    selectProgram(-1);
    resetReference(true);
    mStatus = "Loaded viewer scene with " + std::to_string(mpScene->getMaterialCount())
        + " material-slot binding(s): " + path.string();
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
    mSamplesPerFrame = std::clamp(replay.value("reference_samples_per_frame", 1u), 1u, 16u);
    mMaxSceneBounces = std::clamp(replay.value("reference_scene_max_bounces", 4u), 0u, 16u);
    mMaxLayerWalkDepth = std::clamp(replay.value("reference_layer_walk_max_depth", 24u), 4u, 128u);
    const auto& camera = replay.at("camera");
    mCamera.target = vector3(camera.at("target"));
    mCamera.yaw = camera.at("yaw").get<float>();
    mCamera.pitch = camera.at("pitch").get<float>();
    mCamera.distance = camera.at("distance").get<float>();
    mCamera.verticalFovDegrees = camera.at("vertical_fov_degrees").get<float>();
    const auto& display = replay.at("display");
    mComparisonMode = display.at("comparison_mode").get<uint32_t>();
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
    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
    {
        const auto& slot = mComparisonSlots[slotIndex];
        if (slot.ready() && slot.contract.mode == ncls::SlotMode::PathTracing
            && slot.spp != kCapturePathTracingSpp)
        {
            mStatus = "Capture blocked: path-tracing slot " + std::to_string(slotIndex)
                + " must reach exactly 1024 spp.";
            logWarning("{}", mStatus);
            return;
        }
    }
    fs::path manifestPath = requestedManifestPath;
    if (manifestPath.extension() != ".json") manifestPath /= "capture.json";
    if (!manifestPath.parent_path().empty()) fs::create_directories(manifestPath.parent_path());
    const fs::path stem = manifestPath.parent_path() / manifestPath.stem();
    const std::array<fs::path, 2> slotPaths{
        fs::path(stem.string() + "-slot-0.exr"),
        fs::path(stem.string() + "-slot-1.exr"),
    };
    const fs::path comparisonPath = stem.string() + "-comparison.exr";
    const fs::path displayPath = stem.string() + "-display.png";
    const fs::path differencePath = stem.string() + "-difference.exr";
    const fs::path differenceDisplayPath = stem.string() + "-difference.png";
    const fs::path materialPath = stem.string() + "-material.json";
    const fs::path viewerScenePath = stem.string() + "-scene.json";
    const fs::path metricsPath = stem.string() + "-metrics.csv";

    getDevice()->wait();
    if (mpScene && mComparisonSlots[0].pNoiseStats)
    {
        const auto stats = getRenderContext()->readBuffer<uint32_t>(
            mComparisonSlots[0].pNoiseStats.get(), 0u, 2u);
        if (stats.size() == 2u && stats[1] > 0u)
            mEstimatedRelativeStandardError = float(stats[0]) / (4096.f * float(stats[1]));
    }
    const auto refreshTiming = [](PassTiming& timing) {
        if (timing.sampleIndex > 0) timing.milliseconds = timing.timers[timing.activeSlot]->getElapsedTime();
    };
    refreshTiming(mVisibilityTiming);
    refreshTiming(mCompositeTiming);
    for (auto& slot : mComparisonSlots) refreshTiming(slot.timing);
    const bool bothSlotsReady = mComparisonSlots[0].ready() && mComparisonSlots[1].ready();
    if (mReferenceSource.family == ncls::ReferenceFamily::LayerStack)
        ncls::saveMaterialProgram(materialPath, mMaterial, mMaterialDisplayName);
    if (mpScene) saveViewerScene(viewerScenePath);
    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
        if (const auto output = slotOutput(mComparisonSlots[slotIndex]))
            output->captureToFile(
                0, 0, slotPaths[slotIndex], Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    mpComparisonLinear->captureToFile(0, 0, comparisonPath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
    getTargetFbo()->getColorTexture(0)->captureToFile(
        0, 0, displayPath, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
    if (bothSlotsReady)
    {
        mpDifferenceLinear->captureToFile(
            0, 0, differencePath, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::None, false);
        mpDifferenceDisplay->captureToFile(
            0, 0, differenceDisplayPath, Bitmap::FileFormat::PngFile, Bitmap::ExportFlags::None, false);
    }
    const auto* rightProgram = slotProgram(mComparisonSlots[1]);
    const std::string packageId = rightProgram ? rightProgram->packageId
        : mComparisonSlots[1].sourceReference ? "source-reference" : "none";
    const std::string methodRoot = rightProgram ? rightProgram->root.string() : std::string();
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
            {"source_state_sha256", ncls::referenceSourceStateHash(source)},
            {"source_path", source.sourcePath.string()},
        });
    };
    if (mpScene)
    {
        appendBinding(mActiveSceneMaterial, mReferenceSource, true);
        for (const auto& [materialId, binding] : mInactiveSceneMaterials)
            appendBinding(materialId, binding.source, false);
    }
    const bool sceneMaterialBindingsReplayable = mpScene != nullptr;
    const auto slotStatusName = [](ncls::SlotStatus status) {
        switch (status)
        {
        case ncls::SlotStatus::Ready: return "ready";
        case ncls::SlotStatus::Loading: return "loading";
        case ncls::SlotStatus::Compiling: return "compiling";
        case ncls::SlotStatus::Unsupported: return "unsupported";
        case ncls::SlotStatus::Error: return "error";
        default: return "empty";
        }
    };
    nlohmann::json slots = nlohmann::json::array();
    for (uint32_t slotIndex = 0u; slotIndex < mComparisonSlots.size(); ++slotIndex)
    {
        const auto& slot = mComparisonSlots[slotIndex];
        const auto* program = slotProgram(slot);
        slots.push_back({
            {"slot_index", slotIndex},
            {"package_id", program ? program->packageId : slot.sourceReference ? "source-reference" : ""},
            {"program_runtime_id", program ? program->programRuntimeId : slot.sourceReference ? "ncls.scene-path-tracer@1" : ""},
            {"material_asset_id", program ? program->materialAssetId : ""},
            {"source_snapshot_id", program ? program->sourceSnapshotId : ncls::referenceSourceStateHash(mReferenceSource)},
            {"mode", slot.contract.mode == ncls::SlotMode::PathTracing ? "path-tracing" : "deferred"},
            {"status", slotStatusName(slot.contract.status)},
            {"diagnostic", slot.contract.diagnostic},
            {"spp", slot.spp},
            {"gpu_ms", slot.timing.milliseconds},
            {"linear_output", slot.ready() ? slotPaths[slotIndex].filename().string() : std::string()},
        });
    }
    nlohmann::json manifest = {
        {"format_name", "ncls.viewer-capture"},
        {"format_version", 4},
        {"slots", slots},
        {"method_id", packageId},
        {"method_bundle", methodRoot},
        {"bundle_root", std::filesystem::absolute(mOptions.packageRoot).string()},
        {"source_material_family_id", mReferenceSource.familyId()},
        {"source_material_sha256", mReferenceSource.sourceSha256},
        {"source_material_state_sha256", ncls::referenceSourceStateHash(mReferenceSource)},
        {"source_material_asset_sha256", mReferenceSource.sourceSha256},
        {"source_material", mReferenceSource.family == ncls::ReferenceFamily::LayerStack
            ? materialPath.filename().string()
            : mReferenceSource.sourcePath.string()},
        {"material_ir_sha256", mReferenceSource.family == ncls::ReferenceFamily::LayerStack ? ncls::layerStackHash(mMaterial) : std::string()},
        {"material_program", mReferenceSource.family == ncls::ReferenceFamily::LayerStack ? materialPath.filename().string() : std::string()},
        {"approximation_available", mComparisonSlots[1].ready()},
        {"environment", mEnvironmentPath.empty() ? std::string() : std::filesystem::absolute(mEnvironmentPath).string()},
        {"environment_sha256", mEnvironmentSha256},
        {"reference_geometry", mReferenceGeometryPath.empty() ? std::string() : std::filesystem::absolute(mReferenceGeometryPath).string()},
        {"reference_geometry_sha256", mReferenceGeometrySha256},
        {"viewer_scene", mpScene ? viewerScenePath.filename().string() : std::string()},
        {"scene_material_bindings", sceneMaterialBindings},
        {"scene_material_bindings_replayable", sceneMaterialBindingsReplayable},
        {"resolution", {mOutputWidth, mOutputHeight}},
        {"view_resolution", {mViewWidth, mOutputHeight}},
        {"difference_resolution", {mViewWidth, mOutputHeight}},
        {"reference_spp", kCapturePathTracingSpp},
        {"reference_samples_per_frame", mSamplesPerFrame},
        {"reference_scene_max_bounces", mMaxSceneBounces},
        {"reference_layer_walk_max_depth", mMaxLayerWalkDepth},
        {"reference_integrator", "ncls.scene-path-tracer@1"},
        {"reference_estimator", {
            {"raw_authoritative", true},
            {"raw_accumulation", "arithmetic_mean_of_independent_monte_carlo_samples"},
            {"environment_sampling", "luminance_sin_theta_importance_sampling_with_mis"},
            {"finite_depth_transport", true},
            {"scene_bounce_cap", mMaxSceneBounces},
            {"layer_walk_cap", mMaxLayerWalkDepth},
        }},
        {"estimated_mean_relative_standard_error", mEstimatedRelativeStandardError},
        {"comparison_semantics", mComparisonSlots[0].ready() && mComparisonSlots[1].ready()
            ? "symmetric_slot_linear_output_difference" : "partial_slot_capture"},
        {"method_runtime_class", rightProgram ? rightProgram->runtimeClass : "none"},
        {"camera", {
            {"target", {mCamera.target.x, mCamera.target.y, mCamera.target.z}},
            {"yaw", mCamera.yaw}, {"pitch", mCamera.pitch}, {"distance", mCamera.distance},
            {"vertical_fov_degrees", mCamera.verticalFovDegrees},
        }},
        {"display", {{"comparison_mode", mComparisonMode}, {"exposure_ev", mExposure},
            {"difference_scale", mDifferenceScale}}},
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
            {"visibility", mVisibilityTiming.milliseconds},
            {"slot_0", mComparisonSlots[0].timing.milliseconds},
            {"slot_1", mComparisonSlots[1].timing.milliseconds},
            {"composite", mCompositeTiming.milliseconds},
        }},
        {"files", {
            {"slot_0_linear", mComparisonSlots[0].ready() ? slotPaths[0].filename().string() : std::string()},
            {"slot_1_linear", mComparisonSlots[1].ready() ? slotPaths[1].filename().string() : std::string()},
            {"reference_linear", mComparisonSlots[0].ready() ? slotPaths[0].filename().string() : std::string()},
            {"approximation_linear", mComparisonSlots[1].ready() ? slotPaths[1].filename().string() : std::string()},
            {"comparison_linear", comparisonPath.filename().string()},
            {"display", displayPath.filename().string()},
            {"difference_linear", bothSlotsReady ? differencePath.filename().string() : std::string()},
            {"difference_display", bothSlotsReady ? differenceDisplayPath.filename().string() : std::string()},
            {"material_program", mReferenceSource.family == ncls::ReferenceFamily::LayerStack ? materialPath.filename().string() : std::string()},
            {"viewer_scene", mpScene ? viewerScenePath.filename().string() : std::string()},
            {"metrics_csv", metricsPath.filename().string()},
        }},
    };
    std::ofstream stream(manifestPath, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot write capture manifest: " + manifestPath.string());
    stream << manifest.dump(2) << '\n';
    std::ofstream metrics(metricsPath, std::ios::binary | std::ios::trunc);
    if (!metrics) throw std::runtime_error("cannot write capture metrics: " + metricsPath.string());
    metrics << "method_id,width,height,slot_0_spp,slot_1_spp,estimated_mean_relative_standard_error,visibility_ms,slot_0_ms,slot_1_ms,composite_ms\n";
    metrics << packageId << ',' << mOutputWidth << ',' << mOutputHeight << ','
            << mComparisonSlots[0].spp << ',' << mComparisonSlots[1].spp << ','
            << mEstimatedRelativeStandardError << ',' << mVisibilityTiming.milliseconds << ','
            << mComparisonSlots[0].timing.milliseconds << ',' << mComparisonSlots[1].timing.milliseconds << ','
            << mCompositeTiming.milliseconds << '\n';
    mStatus = "Capture saved: " + manifestPath.string();
}

bool NclsViewer::pickSceneObject(const float2& screenPosition)
{
    if (!mpScene || !mpInstanceId || mOutputWidth == 0u || mOutputHeight == 0u) return false;

    const float outputX = std::clamp(screenPosition.x, 0.f, float(mOutputWidth - 1u));
    const float outputY = std::clamp(screenPosition.y, 0.f, float(mOutputHeight - 1u));
    float sourceU = (outputX + 0.5f) / float(mOutputWidth);
    if (mComparisonMode == 0u)
    {
        const float panelWidth = float(mOutputWidth / 2u);
        const float dividerWidth = float(mOutputWidth - 2u * (mOutputWidth / 2u));
        sourceU = outputX < panelWidth
            ? (outputX + 0.5f) / panelWidth
            : (outputX - panelWidth - dividerWidth + 0.5f) / panelWidth;
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
    if (event.type == MouseEvent::Type::ButtonDown)
    {
        mLastMouse = event.pos;
        mMousePressScreen = event.screenPos;
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
        const bool handled = mCameraDragging || mPanDragging;
        const bool shouldPick = event.button == Input::MouseButton::Left
            && mCameraDragging && !mCameraDragMoved;
        if (event.button == Input::MouseButton::Left)
        {
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
        resetReference(true);
        return true;
    }
    else if (event.type == MouseEvent::Type::Move)
    {
        const float2 delta = event.pos - mLastMouse;
        mLastMouse = event.pos;
        if (mCameraDragging)
        {
            const float2 dragDistance = event.screenPos - mMousePressScreen;
            mCameraDragMoved |= dot(dragDistance, dragDistance) > 9.f;
            mCamera.yaw -= delta.x * 6.f;
            mCamera.pitch = std::clamp(mCamera.pitch - delta.y * 3.f, -1.35f, 1.35f);
            resetReference(true);
            return true;
        }
        if (mPanDragging)
        {
            const float3 position = cameraPosition();
            const float3 forward = normalizedOr(mCamera.target - position, float3(0.f, 0.f, -1.f));
            const float3 right = normalizedOr(cross(forward, float3(0.f, 1.f, 0.f)), float3(1.f, 0.f, 0.f));
            const float3 up = cross(right, forward);
            mCamera.target += (-delta.x * right + delta.y * up) * (1.5f * mCamera.distance);
            resetReference(true);
            return true;
        }
    }
    return false;
}

void NclsViewer::onDroppedFile(const std::filesystem::path& path)
{
    if (std::filesystem::is_directory(path) || path.filename() == "manifest.json")
    {
        mOptions.packageRoot = std::filesystem::is_directory(path) ? path : path.parent_path();
        scanPackages();
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
    if (!options.headless && !options.verboseConsole)
        Logger::setOutputs(Logger::getOutputs() & ~Logger::OutputFlags::Console);
    SampleAppConfig config;
    config.deviceDesc.type = Device::Type::D3D12;
    config.windowDesc.title = "NeuralShading - Multi-Family Reference / ScatteringPackage";
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
