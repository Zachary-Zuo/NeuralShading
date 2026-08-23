#pragma once

#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/API/GpuTimer.h"
#include "Core/Pass/RasterPass.h"
#include "Scene/Scene.h"

#include "MaterialProgram.h"
#include "MethodBundle.h"
#include "OpenPbrLuts.h"
#include "ReferenceSource.h"

#include <array>
#include <filesystem>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

struct ViewerOptions
{
    std::filesystem::path bundleRoot = "artifacts/exports";
    std::filesystem::path materialPath;
    std::filesystem::path environmentPath;
    std::string environmentSha256;
    std::filesystem::path referenceGeometryPath;
    std::string referenceGeometrySha256;
    std::filesystem::path replayPath;
    std::filesystem::path viewerScenePath;
    std::filesystem::path captureManifest = "artifacts/captures/headless.json";
    std::string requestedMethodId;
    bool headless = false;
    bool verboseConsole = false;
    uint32_t frameCount = 256;
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
    };

    struct MaterialSlotBinding
    {
        ncls::ReferenceSource source;
        SourceGpuResources gpu;
        std::filesystem::path materialPath;
        std::string displayName = "Default layered material";
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
    SourceGpuResources createSourceGpuResources(const ncls::ReferenceSource& source);
    void activateSceneMaterial(uint32_t materialId);
    const MaterialSlotBinding* inactiveSceneMaterial(uint32_t materialId) const;
    void resetReference(bool visibilityChanged, bool prepareChanged = true);
    void resetCamera();
    Falcor::float3 cameraPosition() const;
    void scanBundles();
    bool runParityProbe(const ncls::ViewerMethod& method, std::string& error);
    void selectMethod(int32_t methodIndex);
    void bindLighting(Falcor::ShaderVar root, const char* constantBufferName);
    void renderVisibility(Falcor::RenderContext* pRenderContext);
    void renderReference(Falcor::RenderContext* pRenderContext);
    void renderDenoisedReference(Falcor::RenderContext* pRenderContext);
    void renderPrepare(Falcor::RenderContext* pRenderContext);
    void renderApproximation(Falcor::RenderContext* pRenderContext);
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
    bool allMaterialsSupportCurrentCompiler() const;
    bool hasActiveMethod() const;

    ViewerOptions mOptions;
    CameraState mCamera;
    LightingState mLighting;
    ncls::ReferenceSource mReferenceSource = ncls::makeDefaultReferenceSource();
    ncls::LayerStackIR& mMaterial = mReferenceSource.layerStack;
    SourceGpuResources mSourceGpu;
    std::unordered_map<uint32_t, MaterialSlotBinding> mInactiveSceneMaterials;
    uint32_t mActiveSceneMaterial = std::numeric_limits<uint32_t>::max();
    std::filesystem::path mMaterialPath;
    std::filesystem::path mEnvironmentPath;
    std::string mEnvironmentSha256;
    std::filesystem::path mReferenceGeometryPath;
    std::string mReferenceGeometrySha256;
    std::string mMaterialDisplayName = "Default layered material";
    std::string mStatus;

    std::vector<ncls::ViewerMethod> mMethods;
    std::vector<ncls::BundleFailure> mBundleFailures;
    int32_t mSelectedMethod = -1;
    uint32_t mMethodUiValue = 0;
    Falcor::ref<Falcor::Buffer> mpWeights;
    Falcor::ref<Falcor::Buffer> mpReferenceMaterialMetadata;
    Falcor::ref<Falcor::Buffer> mpReferenceNoiseStats;
    Falcor::ref<Falcor::Buffer> mpEnvironmentMarginalCdf;
    Falcor::ref<Falcor::Buffer> mpEnvironmentConditionalCdf;

    Falcor::ref<Falcor::ComputePass> mpVisibilityPass;
    Falcor::ref<Falcor::Scene> mpScene;
    Falcor::ref<Falcor::RasterPass> mpSceneVisibilityPass;
    Falcor::ref<Falcor::ComputePass> mpReferencePass;
    Falcor::ref<Falcor::ComputePass> mpReferencePathPass;
    Falcor::ref<Falcor::ComputePass> mpDenoisePass;
    Falcor::ref<Falcor::ComputePass> mpPreparePass;
    Falcor::ref<Falcor::ComputePass> mpApproximationPass;
    Falcor::ref<Falcor::ComputePass> mpCompositePass;
    Falcor::ref<Falcor::ComputePass> mpParityPass;

    ncls::OpenPbrLuts mOpenPbrLuts;
    Falcor::ref<Falcor::Buffer> mpStates;
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
    std::array<Falcor::ref<Falcor::Texture>, 2> mpReference;
    std::array<Falcor::ref<Falcor::Texture>, 2> mpDenoisedReference;
    Falcor::ref<Falcor::Texture> mpApproximation;
    Falcor::ref<Falcor::Texture> mpComparisonLinear;
    Falcor::ref<Falcor::Texture> mpDisplay;
    Falcor::ref<Falcor::Texture> mpEnvironment;
    Falcor::ref<Falcor::Sampler> mpLinearSampler;
    Falcor::ref<Falcor::Sampler> mpMaterialXSampler;
    Falcor::uint2 mEnvironmentSamplingDimensions{0u, 0u};

    PassTiming mVisibilityTiming;
    PassTiming mReferenceTiming;
    PassTiming mPrepareTiming;
    PassTiming mLightingTiming;
    PassTiming mCompositeTiming;

    uint32_t mOutputWidth = 0;
    uint32_t mOutputHeight = 0;
    uint32_t mViewWidth = 0;
    uint32_t mFrameIndex = 0;
    uint32_t mReferenceSpp = 0;
    uint32_t mReferencePing = 0;
    uint32_t mSelectedSceneInstance = std::numeric_limits<uint32_t>::max();
    uint32_t mSelectedSceneMaterial = std::numeric_limits<uint32_t>::max();
    std::string mSelectedSceneGeometryName;
    std::string mSelectedSceneMaterialName;
    uint32_t mSamplesPerFrame = 1;
    uint32_t mMaxSceneBounces = 4;
    uint32_t mMaxLayerWalkDepth = 24;
    uint32_t mObjectMode = 0;
    uint32_t mSelectedInterface = 0;
    uint32_t mComparisonMode = 0;
    float mSplit = 0.5f;
    float mExposure = 0.f;
    float mDifferenceScale = 8.f;
    float mEstimatedRelativeStandardError = 1.f;
    double mAccumulationSeconds = 0.0;
    bool mResetAccumulation = true;
    bool mVisibilityDirty = true;
    bool mPrepareDirty = true;
    bool mFreezeReference = false;
    bool mUseDenoisedPreview = true;
    bool mCameraDragging = false;
    bool mCameraDragMoved = false;
    bool mPanDragging = false;
    bool mDividerDragging = false;
    Falcor::float2 mLastMouse{0.f, 0.f};
    Falcor::float2 mMousePressScreen{0.f, 0.f};
    uint32_t mRenderedFrames = 0;
};
