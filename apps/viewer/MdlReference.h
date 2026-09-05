#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ncls
{
struct MdlTextureResource
{
    uint32_t index = 0;
    std::string shape;
    std::string gamma;
    std::string pixelType;
    std::string dataOrigin;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    std::filesystem::path dataPath;
};

struct MdlCompiledArtifact
{
    std::filesystem::path root;
    std::string artifactSha256;
    std::string mdlSdk;
    std::string compilerBridgeSha256;
    std::string generatedCode;
    std::vector<uint8_t> argumentBlock;
    std::vector<uint8_t> roData;
    std::vector<MdlTextureResource> textures;
};

struct MdlCatalogEntry
{
    std::string assetId;
    std::string displayName;
    std::string sourceSnapshotId;
    std::string artifactSha256;
    std::filesystem::path artifactRoot;
    std::string exportId;
    std::string metal;
    std::string finish;
    std::string graphId;
    std::string textureSetId;
    std::string parameterSchemaId;
    std::string packageId;
    std::filesystem::path packageRoot;
    std::string programId;
    std::string packageAssetId;
    std::string instanceId;
    nlohmann::json parameterView;

    bool linked() const { return !packageId.empty(); }
};

struct MdlViewerCatalog
{
    std::filesystem::path sourcePath;
    std::string catalogSha256;
    std::string catalogId;
    std::string registryIdentity;
    std::string registrySha256;
    uint32_t rejectedCutoutCount = 0;
    std::string mdlSdk;
    std::string defaultAssetId;
    std::filesystem::path targetCodeTypesPath;
    std::string targetCodeTypesSha256;
    std::filesystem::path rendererRuntimePath;
    std::string rendererRuntimeSha256;
    std::vector<MdlCatalogEntry> entries;

};

MdlViewerCatalog loadMdlViewerCatalog(const std::filesystem::path& path);
std::shared_ptr<const MdlCompiledArtifact> loadMdlCompiledArtifact(
    const MdlCatalogEntry& entry);
} // namespace ncls
