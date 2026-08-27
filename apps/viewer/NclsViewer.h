#pragma once

#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/API/GpuTimer.h"
#include "Core/Pass/RasterPass.h"
#include "Scene/Scene.h"

#include "MaterialProgram.h"
#include "ScatteringPackage.h"
#include "ComparisonSlot.h"
#include "OpenPbrLuts.h"
#include "ReferenceSource.h"

#include <array>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct ViewerOptions
{
    std::filesystem::path packageRoot = "artifacts/exports";
    std::filesystem::path materialPath;
    std::filesystem::path environmentPath;
    std::string environmentSha256;
    std::filesystem::path referenceGeometryPath;
    std::string referenceGeometrySha256;
    std::filesystem::path replayPath;
    std::filesystem::path viewerScenePath;
    std::filesystem::path captureManifest = "artifacts/captures/headless.json";
    std::string requestedPackageId;
    std::array<std::string, 2> requestedSlotPackages{"source-reference", ""};
    std::array<ncls::SlotMode, 2> requestedSlotModes{
        ncls::SlotMode::PathTracing, ncls::SlotMode::PathTracing};
    bool hasRequestedSlots = false;
    bool headless = false;
    bool verboseConsole = false;
    bool evaluatorPreviewLighting = false;
    uint32_t frameCount = 1024;
    uint32_t width = 1280;
    uint32_t height = 720;
};

class NclsViewer : public Falcor::SampleApp
{
public:
    NclsViewer(const Falcor::SampleAppConfig& config, ViewerOptions options);

    void onLoad(Falcor::RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(Falcor::RenderContext* pRenderContext, const Falcor::ref<Falcor::Fbo>& pTargetFbo) override;
    void onGuiRender(Falcor::Gui* pGui) override;
    bool onKeyEvent(const Falcor::KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const Falcor::MouseEvent& mouseEvent) override;
    void onDroppedFile(const std::filesystem::path& path) override;

private:
    struct CameraState
    {
        Falcor::float3 target{0.f, 0.f, 0.f};
        float yaw = 0.45f;
        float pitch = 0.12f;
        float distance = 3.4f;
        float verticalFovDegrees = 38.f;
    };

    struct LightingState
    {
        bool useEnvironment = true;
        float environmentRotation = 0.f;
        float environmentIntensity = 0.65f;
        bool useSun = true;
        Falcor::float3 sunDirection{-0.35f, 0.78f, 0.52f};
        float sunIntensity = 1.5f;
        Falcor::float3 sunColor{1.f, 0.92f, 0.82f};
        bool usePoint = false;
        Falcor::float3 pointPosition{-1.8f, 1.4f, 2.0f};
        float pointIntensity = 7.f;
        Falcor::float3 pointColor{0.55f, 0.72f, 1.f};
        bool useRectangle = true;
        Falcor::float3 rectangleCenter{1.3f, 2.4f, 1.2f};
        Falcor::float3 rectangleAxisU{0.65f, 0.f, 0.f};
        Falcor::float3 rectangleAxisV{0.f, 0.f, 0.45f};
        float rectangleIntensity = 3.2f;
        Falcor::float3 rectangleColor{1.f, 0.78f, 0.58f};
    };

    struct PassTiming
    {
        std::array<Falcor::ref<Falcor::GpuTimer>, 4> timers;
        double milliseconds = 0.0;
        uint64_t sampleIndex = 0;
        uint32_t activeSlot = 0;
    };

    struct SourceGpuResources
    {
        Falcor::ref<Falcor::Buffer> pMaterial;
        Falcor::ref<Falcor::Buffer> pMerlBrdf;
        Falcor::ref<Falcor::Buffer> pOpenPbrInputs;
        Falcor::ref<Falcor::Buffer> pMaterialXInputs;
        Falcor::ref<Falcor::Texture> pMaterialXBaseColor;
        Falcor::ref<Falcor::Texture> pMaterialXRoughness;
        Falcor::ref<Falcor::Texture> pMaterialXMetalness;
        Falcor::ref<Falcor::Texture> pMaterialXNormalMap;
        Falcor::ref<Falcor::Buffer> pMdlArgumentBlock;
        Falcor::ref<Falcor::Buffer> pMdlRoData;
        std::array<Falcor::ref<Falcor::Texture>, 16> pMdlTexture2D;
        std::array<Falcor::ref<Falcor::Texture>, 16> pMdlTexture3D;
        Falcor::ref<Falcor::Sampler> pMdlSampler;
    };

    struct MaterialSlotBinding
    {
        ncls::ReferenceSource source;
        SourceGpuResources gpu;
        std::filesystem::path materialPath;
        std::string displayName = "Default layered material";
    };

    struct ComparisonSlotRuntime
    {
        ncls::ComparisonSlot contract;
        bool sourceReference = false;
        int32_t programIndex = -1;
        uint32_t uiValue = 0;
        Falcor::ref<Falcor::Buffer> pWeights;
        Falcor::ref<Falcor::Buffer> pCompiledMaterials;
        std::map<std::string, Falcor::ref<Falcor::Texture>> textures;
        std::map<std::string, Falcor::ref<Falcor::Sampler>> samplers;
        Falcor::ref<Falcor::ComputePass> pDeferredPass;
        Falcor::ref<Falcor::ComputePass> pPathPass;
        std::array<Falcor::ref<Falcor::Texture>, 2> pAccumulated;
        Falcor::ref<Falcor::Texture> pDeferred;
        Falcor::ref<Falcor::Buffer> pNoiseStats;
        PassTiming timing;
        uint32_t ping = 0;
        uint32_t spp = 0;
        bool resetAccumulation = true;

        bool ready() const { return contract.status == ncls::SlotStatus::Ready; }
    };

    void createPasses();
    void createDefaultEnvironment();
    void rebuildEnvironmentSampling(const std::vector<Falcor::float4>& pixels, uint32_t width, uint32_t height);
    void resizeResources(uint32_t width, uint32_t height);
    void loadScene(const std::filesystem::path& path, const std::string& expectedSha256 = {});
    void createSceneReferencePass();
    void rebuildSceneFbo();
    void rebuildReferenceMaterialMetadata();
    void syncSceneCamera();
    bool pickSceneObject(const Falcor::float2& screenPosition);
    void updateMaterialBuffer();
    void updateReferenceSourceBuffer();
    SourceGpuResources createFallbackSourceGpuResources();
    SourceGpuResources createSourceGpuResources(const ncls::ReferenceSource& source);
    void activateSceneMaterial(uint32_t materialId);
    const MaterialSlotBinding* inactiveSceneMaterial(uint32_t materialId) const;
    void resetReference(bool visibilityChanged);
    void resetCamera();
    Falcor::float3 cameraPosition() const;
    void scanPackages();
    Falcor::ref<Falcor::ComputePass> createProgramPass(
        const char* shaderPath,
        const ncls::ViewerProgram& method);
    Falcor::ref<Falcor::ComputePass> createProgramPathPass(
        const ncls::ViewerProgram& method);
    bool runParityProbe(const ncls::ViewerProgram& method, std::string& error);
    void selectProgram(int32_t methodIndex);
    void activateComparisonSlot(uint32_t slotIndex, uint32_t selection);
    void resizeComparisonSlot(ComparisonSlotRuntime& slot);
    const ncls::ViewerProgram* slotProgram(const ComparisonSlotRuntime& slot) const;
    Falcor::ref<Falcor::Texture> slotOutput(const ComparisonSlotRuntime& slot) const;
    void bindProgramResources(Falcor::ShaderVar root, const ComparisonSlotRuntime& slot) const;
    void bindLighting(Falcor::ShaderVar root, const char* constantBufferName);
    void renderVisibility(Falcor::RenderContext* pRenderContext);
    void renderReference(Falcor::RenderContext* pRenderContext, ComparisonSlotRuntime& slot);
    void renderApproximation(Falcor::RenderContext* pRenderContext, ComparisonSlotRuntime& slot);
    void renderPackagePath(Falcor::RenderContext* pRenderContext, ComparisonSlotRuntime& slot);
    void renderComposite(Falcor::RenderContext* pRenderContext);
    void beginTiming(PassTiming& timing);
    void endTiming(PassTiming& timing);
    void loadMaterial(const std::filesystem::path& path);
    void installReferenceSource(ncls::ReferenceSource source, const std::filesystem::path& path = {});
    void saveMaterial(const std::filesystem::path& path);
    void loadEnvironment(const std::filesystem::path& path, const std::string& expectedSha256 = {});
    void loadViewerScene(const std::filesystem::path& path);
    void saveViewerScene(const std::filesystem::path& path);
    void applyReplaySettings(const std::filesystem::path& path);
    void capture(const std::filesystem::path& manifestPath);
    void renderMaterialUi(Falcor::Gui::Widgets& widgets);
    void renderOpenPbrUi(Falcor::Gui::Widgets& widgets);
    void renderMaterialXUi(Falcor::Gui::Widgets& widgets);
    void renderMdlUi(Falcor::Gui::Widgets& widgets);
    bool allMaterialsSupportedBy(const ncls::ViewerProgram& method) const;
    bool hasActiveProgram() const;

    ViewerOptions mOptions;
    CameraState mCamera;
    LightingState mLighting;
    ncls::ReferenceSource mReferenceSource = ncls::makeDefaultReferenceSource();
    ncls::LayerStackIR& mMaterial = mReferenceSource.layerStack;
    SourceGpuResources mSourceGpu;
    SourceGpuResources mFallbackSourceGpu;
    std::unordered_map<uint32_t, MaterialSlotBinding> mInactiveSceneMaterials;
    uint32_t mActiveSceneMaterial = std::numeric_limits<uint32_t>::max();
    std::filesystem::path mMaterialPath;
    std::filesystem::path mEnvironmentPath;
    std::string mEnvironmentSha256;
    std::filesystem::path mReferenceGeometryPath;
    std::string mReferenceGeometrySha256;
    std::string mMaterialDisplayName = "Default layered material";
    std::string mStatus;

    std::vector<ncls::ViewerProgram> mPrograms;
    std::vector<ncls::PackageFailure> mPackageFailures;
    std::array<ComparisonSlotRuntime, 2> mComparisonSlots;
    Falcor::ref<Falcor::Buffer> mpReferenceMaterialMetadata;
    Falcor::ref<Falcor::Buffer> mpEnvironmentMarginalCdf;
    Falcor::ref<Falcor::Buffer> mpEnvironmentConditionalCdf;

    Falcor::ref<Falcor::ComputePass> mpVisibilityClearPass;
    Falcor::ref<Falcor::Scene> mpScene;
    Falcor::ref<Falcor::RasterPass> mpSceneVisibilityPass;
    Falcor::ref<Falcor::ComputePass> mpReferencePathPass;
    Falcor::ref<Falcor::ComputePass> mpCompositePass;

    ncls::OpenPbrLuts mOpenPbrLuts;
    Falcor::ref<Falcor::Texture> mpPositionDepth;
    Falcor::ref<Falcor::Texture> mpNormal;
    Falcor::ref<Falcor::Texture> mpTangent;
    Falcor::ref<Falcor::Texture> mpViewDirection;
    Falcor::ref<Falcor::Texture> mpMaterialXTexCoord;
    Falcor::ref<Falcor::Texture> mpMaterialXTexCoordGrad;
    Falcor::ref<Falcor::Texture> mpInstanceId;
    Falcor::ref<Falcor::Texture> mpSceneMaterialId;
    Falcor::ref<Falcor::Texture> mpSceneDepth;
    Falcor::ref<Falcor::Fbo> mpSceneFbo;
    Falcor::ref<Falcor::Texture> mpEmptySlot;
    Falcor::ref<Falcor::Texture> mpComparisonLinear;
    Falcor::ref<Falcor::Texture> mpDisplay;
    Falcor::ref<Falcor::Texture> mpDifferenceLinear;
    Falcor::ref<Falcor::Texture> mpDifferenceDisplay;
    Falcor::ref<Falcor::Texture> mpEnvironment;
    Falcor::ref<Falcor::Sampler> mpLinearSampler;
    Falcor::ref<Falcor::Sampler> mpMaterialXSampler;
    Falcor::uint2 mEnvironmentSamplingDimensions{0u, 0u};

    PassTiming mVisibilityTiming;
    PassTiming mCompositeTiming;

    uint32_t mOutputWidth = 0;
    uint32_t mOutputHeight = 0;
    uint32_t mViewWidth = 0;
    uint32_t mFrameIndex = 0;
    uint32_t mSelectedSceneInstance = std::numeric_limits<uint32_t>::max();
    uint32_t mSelectedSceneMaterial = std::numeric_limits<uint32_t>::max();
    std::string mSelectedSceneGeometryName;
    std::string mSelectedSceneMaterialName;
    uint32_t mSamplesPerFrame = 1;
    uint32_t mMaxSceneBounces = 4;
    uint32_t mMaxLayerWalkDepth = 24;
    uint32_t mSelectedInterface = 0;
    uint32_t mComparisonMode = 0;
    float mExposure = 0.f;
    float mDifferenceScale = 8.f;
    float mEstimatedRelativeStandardError = 1.f;
    double mAccumulationSeconds = 0.0;
    bool mVisibilityDirty = true;
    bool mFreezeReference = false;
    bool mCameraDragging = false;
    bool mCameraDragMoved = false;
    bool mPanDragging = false;
    Falcor::float2 mLastMouse{0.f, 0.f};
    Falcor::float2 mMousePressScreen{0.f, 0.f};
    uint32_t mRenderedFrames = 0;
};
