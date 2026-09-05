#include "MdlReference.h"

#include "Hash.h"
#include "ScatteringPackage.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <functional>
#include <regex>
#include <set>
#include <stdexcept>

namespace ncls
{
namespace
{
using json = nlohmann::json;

constexpr const char* kMdlSdk = "2025.0.0-387700.1252";
constexpr const char* kStbCommit = "013ac3beddff3dbffafd5177e7972067cd2b5083";
constexpr const char* kStbImageSha256 = "594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3";

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void requireSha256(const std::string& value, const std::string& label)
{
    static const std::regex pattern("^[0-9a-f]{64}$");
    require(std::regex_match(value, pattern), label + " must be a lowercase SHA-256 digest");
}

void requireKeys(const json& value, const std::set<std::string>& expected, const std::string& label)
{
    require(value.is_object(), label + " must be an object");
    std::set<std::string> actual;
    for (auto item = value.begin(); item != value.end(); ++item) actual.insert(item.key());
    require(actual == expected, label + " has unknown or missing fields");
}

std::filesystem::path contained(
    const std::filesystem::path& root,
    const std::filesystem::path& candidate,
    const std::string& label)
{
    const auto absoluteRoot = std::filesystem::absolute(root).lexically_normal();
    const auto absoluteCandidate = std::filesystem::absolute(candidate).lexically_normal();
    auto relative = absoluteCandidate.lexically_relative(absoluteRoot);
    require(!relative.empty() && !relative.is_absolute()
        && (relative.begin() == relative.end() || *relative.begin() != ".."),
        label + " escapes its root");
    return absoluteCandidate;
}

std::filesystem::path resolveContainedUri(
    const std::filesystem::path& base,
    const std::string& uri,
    const std::string& label)
{
    const std::filesystem::path value(uri);
    require(!value.empty() && value != "." && !value.is_absolute(),
        label + " URI must be a nonempty relative path");
    for (const auto& component : value)
        require(component != "..", label + " URI escapes its root");
    return contained(base, base / value, label);
}

std::vector<uint8_t> readBytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "cannot open MDL artifact payload: " + path.string());
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    require(size >= 0, "cannot query MDL artifact payload size: " + path.string());
    std::vector<uint8_t> result(static_cast<size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!result.empty()) stream.read(reinterpret_cast<char*>(result.data()), result.size());
    require(bool(stream) || stream.eof(), "cannot read MDL artifact payload: " + path.string());
    return result;
}

std::string readText(const std::filesystem::path& path)
{
    const auto bytes = readBytes(path);
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

json expectedCodegenOptions()
{
    return {
        {"compile_constants", true},
        {"df_handle_slot_mode", "none"},
        {"enable_auxiliary", true},
        {"fast_math", true},
        {"fold_all_bool_parameters", false},
        {"fold_all_enum_parameters", false},
        {"fold_ternary_on_df", false},
        {"ignore_noinline", true},
        {"internal_space_request", "coordinate_world (rejected by pinned HLSL backend)"},
        {"num_texture_results", 16},
        {"num_texture_spaces", 4},
        {"opt_level", 2},
        {"texture_runtime_with_derivs", false},
        {"use_renderer_adapt_normal", true},
    };
}

std::string artifactIdentity(const std::filesystem::path& root)
{
    json files = json::object();
    for (const auto& item : std::filesystem::recursive_directory_iterator(root))
    {
        if (!item.is_regular_file()) continue;
        files[item.path().lexically_relative(root).generic_string()] = sha256FileHex(item.path());
    }
    const std::string canonical = files.dump();
    return sha256Hex(canonical.data(), canonical.size());
}
} // namespace

MdlViewerCatalog loadMdlViewerCatalog(const std::filesystem::path& requestedPath)
{
    const auto path = std::filesystem::absolute(requestedPath).lexically_normal();
    require(std::filesystem::is_regular_file(path), "MDL viewer catalog is missing: " + path.string());
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "cannot open MDL viewer catalog: " + path.string());
    const json document = json::parse(stream);
    require(document.value("schema_name", "") == "ncls.viewer-material-catalog"
            && document.value("schema_version", 0u) == 2u,
        "unsupported catalog; prepare the source again with tools/reference/prepare_mdl_viewer.py or ncls export");
    requireKeys(document,
        {"schema_name", "schema_version", "catalog_id", "registry",
            "reference_runtime", "default_export_id", "entries"}, "ViewerMaterialCatalog root");
    json identityDocument = document;

    {
        const std::string declared = identityDocument.at("catalog_id").get<std::string>();
        requireSha256(declared, "catalog_id");
        identityDocument.erase("catalog_id");
        require(sha256Json(identityDocument) == declared,
            "ViewerMaterialCatalog catalog_id does not match semantics");
    }

    MdlViewerCatalog result;
    result.sourcePath = path;
    result.catalogSha256 = sha256FileHex(path);
    const auto base = path.parent_path();
    const auto& runtime = document.at("reference_runtime");

    {
        requireKeys(runtime,
            {"mdl_sdk", "target_code_types", "renderer_runtime"},
            "ViewerMaterialCatalog reference_runtime");
        requireKeys(runtime.at("target_code_types"), {"path", "sha256"},
            "ViewerMaterialCatalog target_code_types");
        requireKeys(runtime.at("renderer_runtime"), {"path", "sha256"},
            "ViewerMaterialCatalog renderer_runtime");
    }
    result.mdlSdk = runtime.at("mdl_sdk").get<std::string>();
    require(result.mdlSdk == kMdlSdk, "MDL viewer catalog uses another SDK build");
    result.defaultAssetId = document.at("default_export_id").get<std::string>();
    const std::string targetCodeTypesUri =
        runtime.at("target_code_types").at("path").get<std::string>();
    result.targetCodeTypesPath = resolveContainedUri(base, targetCodeTypesUri, "MDL target-code types");
    result.targetCodeTypesSha256 = runtime.at("target_code_types").at("sha256").get<std::string>();
    const std::string rendererRuntimeUri =
        runtime.at("renderer_runtime").at("path").get<std::string>();
    result.rendererRuntimePath = resolveContainedUri(base, rendererRuntimeUri, "MDL renderer runtime");
    result.rendererRuntimeSha256 = runtime.at("renderer_runtime").at("sha256").get<std::string>();

    {
        result.catalogId = document.at("catalog_id").get<std::string>();
        if (!document.at("registry").is_null())
        {
        const auto& registry = document.at("registry");
        requireKeys(registry,
            {"identity", "sha256", "opaque_entry_count", "rejected_cutout_count"},
            "ViewerMaterialCatalog registry");
        result.registryIdentity = registry.at("identity").get<std::string>();
        result.registrySha256 = registry.at("sha256").get<std::string>();
        result.rejectedCutoutCount = registry.at("rejected_cutout_count").get<uint32_t>();
        require(registry.at("opaque_entry_count").get<uint32_t>() > 0u,
            "ViewerMaterialCatalog registry has no opaque entries");
        requireSha256(result.registryIdentity, "registry.identity");
        requireSha256(result.registrySha256, "registry.sha256");
        }

    }
    requireSha256(result.targetCodeTypesSha256, "target_code_types.sha256");
    requireSha256(result.rendererRuntimeSha256, "renderer_runtime.sha256");
    require(std::filesystem::is_regular_file(result.targetCodeTypesPath)
        && sha256FileHex(result.targetCodeTypesPath) == result.targetCodeTypesSha256,
        "MDL target-code types file is missing or has drifted");
    require(std::filesystem::is_regular_file(result.rendererRuntimePath)
        && sha256FileHex(result.rendererRuntimePath) == result.rendererRuntimeSha256,
        "project MDL renderer runtime is missing or has drifted");

    std::set<std::string> assetIds;
    const auto& assets = document.at("entries");
    require(assets.is_array() && !assets.empty(), "MDL viewer catalog has no assets");
    if (!document.at("registry").is_null())
        require(assets.size() == document.at("registry").at("opaque_entry_count").get<size_t>(),
            "ViewerMaterialCatalog entry count differs from the registry");
    std::set<std::string> packageIds;
    std::set<std::string> instanceIds;
    std::set<std::string> sourceSnapshotIds;
    for (const auto& item : assets)
    {
        MdlCatalogEntry entry;
        entry.assetId = item.at("export_id").get<std::string>();
        entry.displayName = item.at("display_name").get<std::string>();
        entry.sourceSnapshotId = item.at("source_snapshot_id").get<std::string>();
        entry.artifactSha256 = item.at("artifact_sha256").get<std::string>();
        const std::string artifactUri = item.at("artifact_root").get<std::string>();
        entry.artifactRoot = resolveContainedUri(base, artifactUri, "MDL compiled artifact");
        require(!entry.assetId.empty() && assetIds.insert(entry.assetId).second,
            "MDL viewer catalog asset IDs must be nonempty and unique");
        requireSha256(entry.sourceSnapshotId, "source_snapshot_id");
        requireSha256(entry.artifactSha256, "compiled_artifact_sha256");
        require(std::filesystem::is_directory(entry.artifactRoot),
            "MDL compiled artifact is missing: " + entry.artifactRoot.string());

        {
            requireKeys(item,
                {"export_id", "display_name", "metal", "finish", "graph_id",
                    "texture_set_id", "parameter_schema_id", "source_snapshot_id",
                    "artifact_sha256", "artifact_root", "package_id", "package_root",
                    "program_id", "asset_id", "instance_id", "parameter_view"},
                "ViewerMaterialCatalog entry");
            entry.exportId = entry.assetId;
            entry.metal = item.at("metal").is_null() ? "" : item.at("metal").get<std::string>();
            entry.finish = item.at("finish").is_null() ? "" : item.at("finish").get<std::string>();
            entry.graphId = item.at("graph_id").is_null() ? "" : item.at("graph_id").get<std::string>();
            entry.textureSetId = item.at("texture_set_id").is_null() ? "" : item.at("texture_set_id").get<std::string>();
            entry.parameterSchemaId = item.at("parameter_schema_id").is_null() ? "" : item.at("parameter_schema_id").get<std::string>();
            const bool hasBinding = !item.at("package_id").is_null();
            for (const char* name : {"package_id", "package_root", "program_id", "asset_id", "instance_id"})
                require(!item.at(name).is_null() == hasBinding, "ViewerMaterialCatalog partial package binding");
            if (hasBinding)
            {
            entry.packageId = item.at("package_id").get<std::string>();
            entry.packageRoot = resolveContainedUri(base,
                item.at("package_root").get<std::string>(), "ScatteringPackage");
            entry.programId = item.at("program_id").get<std::string>();
            entry.packageAssetId = item.at("asset_id").get<std::string>();
            entry.instanceId = item.at("instance_id").get<std::string>();
            }
            entry.parameterView = item.at("parameter_view");
            for (const auto& [label, value] : std::vector<std::pair<std::string, std::string>>{
                    {"export_id", entry.exportId}, {"graph_id", entry.graphId},
                    {"texture_set_id", entry.textureSetId},
                    {"parameter_schema_id", entry.parameterSchemaId},
                    {"package_id", entry.packageId}, {"program_id", entry.programId},
                    {"asset_id", entry.packageAssetId}, {"instance_id", entry.instanceId}})
                if (!value.empty()) requireSha256(value, label);
            require(!entry.displayName.empty(),
                "ViewerMaterialCatalog taxonomy is incomplete");
            if (hasBinding)
            {
            require(std::filesystem::is_directory(entry.packageRoot),
                "ViewerMaterialCatalog package is missing: " + entry.packageRoot.string());
            require(packageIds.insert(entry.packageId).second,
                "ViewerMaterialCatalog package IDs must be unique");
            require(instanceIds.insert(entry.instanceId).second,
                "ViewerMaterialCatalog instance IDs must be unique");
            }
            require(sourceSnapshotIds.insert(entry.sourceSnapshotId).second,
                "ViewerMaterialCatalog source snapshot IDs must be unique");
            if (!entry.parameterView.is_null())
            {
            validateViewerTypedParameterView(entry.parameterView);
            require(entry.parameterView.at("snapshot_id") == entry.sourceSnapshotId,
                "ViewerMaterialCatalog parameter view snapshot mismatch");
            std::function<void(const json&)> validateWrites = [&](const json& node) {
                if (node.value("editable", false))
                {
                    const std::string type = node.at("value_type").get<std::string>();
                    require(type == "bool" || type == "int" || type == "enum"
                            || type == "float" || type == "vector2" || type == "color3",
                        "ViewerMaterialCatalog parameter type is unsupported");
                    const auto& metadata = node.at("metadata");
                    require(metadata.contains("responsibility") && metadata.contains("reference_write"),
                        "ViewerMaterialCatalog parameter is not linked to the reference");
                    const std::string responsibility = metadata.at("responsibility").get<std::string>();
                    require(responsibility == "coordinates" || responsibility == "frame"
                            || responsibility == "metal-core" || responsibility == "finish-microstructure"
                            || responsibility == "aging-contamination" || responsibility == "coating-composite",
                        "ViewerMaterialCatalog parameter responsibility is unsupported");
                    const auto& write = metadata.at("reference_write");
                    requireKeys(write,
                        type == "enum"
                            ? std::set<std::string>{"offset", "size", "mdl_type", "choices"}
                            : std::set<std::string>{"offset", "size", "mdl_type"},
                        "ViewerMaterialCatalog reference write");
                    const std::pair<std::string, uint32_t> expected = type == "bool"
                        ? std::pair<std::string, uint32_t>{"bool", 1u}
                        : type == "int" ? std::pair<std::string, uint32_t>{"int", 4u}
                        : type == "enum" ? std::pair<std::string, uint32_t>{"enum", 4u}
                        : type == "float" ? std::pair<std::string, uint32_t>{"float", 4u}
                        : type == "vector2" ? std::pair<std::string, uint32_t>{"float2", 8u}
                        : std::pair<std::string, uint32_t>{"color", 12u};
                    require(write.at("mdl_type") == expected.first
                            && write.at("size").get<uint32_t>() == expected.second,
                        "ViewerMaterialCatalog reference write type/size is invalid");
                    if (type == "enum")
                    {
                        const auto& choices = write.at("choices");
                        require(choices.is_object() && node.contains("choices")
                                && node.at("choices").is_array()
                                && choices.size() == node.at("choices").size(),
                            "ViewerMaterialCatalog enum reference write is invalid");
                        for (const auto& choice : node.at("choices"))
                            require(choice.is_string() && choices.contains(choice.get<std::string>())
                                    && choices.at(choice.get<std::string>()).is_number_integer(),
                                "ViewerMaterialCatalog enum reference write is invalid");
                    }
                }
                for (const auto& child : node.at("children")) validateWrites(child);
            };
            validateWrites(entry.parameterView.at("root"));
            }
        }
        result.entries.push_back(std::move(entry));
    }
    require(assetIds.count(result.defaultAssetId) == 1u, "MDL viewer default asset is not in the catalog");
    return result;
}

std::shared_ptr<const MdlCompiledArtifact> loadMdlCompiledArtifact(const MdlCatalogEntry& entry)
{
    const auto root = std::filesystem::absolute(entry.artifactRoot).lexically_normal();
    const auto manifestPath = contained(root, root / "manifest.json", "MDL artifact manifest");
    require(std::filesystem::is_regular_file(manifestPath), "MDL artifact manifest is missing");
    std::ifstream stream(manifestPath, std::ios::binary);
    require(bool(stream), "cannot open MDL artifact manifest");
    const json manifest = json::parse(stream);
    require(manifest.value("schema", "") == "ncls.mdl-compiled-artifact@1",
        "unsupported MDL compiled artifact schema");
    require(manifest.value("mdl_sdk", "") == kMdlSdk, "MDL artifact uses another SDK build");
    require(manifest.value("texture_payloads", "") == "decoded",
        "MDL viewer requires a decoded runtime texture artifact");
    require(manifest.value("capability_audit", json::object()) == json({
        {"surface_bsdf_evaluate", true}, {"emission", false},
        {"volume", false}, {"displacement", false}, {"cutout_opacity", false}}),
        "MDL artifact does not satisfy the V1 surface-evaluate capability audit");
    require(manifest.value("diagnostics", "").empty(), "MDL artifact contains compiler diagnostics");
    const auto& compiler = manifest.at("compiler_identity");
    require(compiler.value("mdl_sdk", "") == kMdlSdk
        && compiler.value("stb_commit", "") == kStbCommit
        && compiler.value("stb_image_sha256", "") == kStbImageSha256
        && compiler.value("codegen_options", json::object()) == expectedCodegenOptions(),
        "MDL artifact compiler identity differs from the formal bridge");
    const std::string bridgeSha = compiler.at("bridge_executable_sha256").get<std::string>();
    requireSha256(bridgeSha, "bridge_executable_sha256");

    const auto& declared = manifest.at("files_sha256");
    require(declared.is_object() && !declared.empty(), "MDL artifact has no finalized file hash table");
    std::set<std::string> actualFiles;
    for (const auto& item : std::filesystem::recursive_directory_iterator(root))
    {
        if (!item.is_regular_file() || item.path().filename() == "manifest.json") continue;
        const std::string relative = item.path().lexically_relative(root).generic_string();
        actualFiles.insert(relative);
        require(declared.contains(relative)
            && declared.at(relative).get<std::string>() == sha256FileHex(item.path()),
            "MDL artifact file hash mismatch: " + relative);
    }
    require(actualFiles.size() == declared.size(), "MDL artifact file set differs from its manifest");
    const std::string identity = artifactIdentity(root);
    require(identity == entry.artifactSha256, "MDL compiled artifact identity differs from the catalog");

    auto result = std::make_shared<MdlCompiledArtifact>();
    result->root = root;
    result->artifactSha256 = identity;
    result->mdlSdk = kMdlSdk;
    result->compilerBridgeSha256 = bridgeSha;
    const auto codePath = contained(root, root / manifest.at("code").get<std::string>(), "MDL generated HLSL");
    require(std::filesystem::is_regular_file(codePath), "MDL generated HLSL is missing");
    result->generatedCode = readText(codePath);

    if (!manifest.at("argument_block").is_null())
    {
        const auto& descriptor = manifest.at("argument_block");
        const auto path = contained(root, root / descriptor.at("path").get<std::string>(), "MDL argument block");
        result->argumentBlock = readBytes(path);
        require(result->argumentBlock.size() == descriptor.at("size").get<size_t>(),
            "MDL argument block has the wrong size");
    }
    const auto& roData = manifest.at("ro_data");
    require(roData.is_array() && roData.size() <= 1u, "MDL V1 supports at most one RO segment");
    if (!roData.empty())
    {
        const auto path = contained(root, root / roData[0].at("path").get<std::string>(), "MDL RO data");
        result->roData = readBytes(path);
        require(result->roData.size() == roData[0].at("size").get<size_t>(),
            "MDL RO segment has the wrong size");
    }

    uint32_t expectedIndex = 1u;
    for (const auto& descriptor : manifest.at("textures"))
    {
        MdlTextureResource texture;
        texture.index = descriptor.at("index").get<uint32_t>();
        require(texture.index == expectedIndex++, "MDL texture indices must be contiguous and one-based");
        texture.shape = descriptor.at("shape").get<std::string>();
        texture.gamma = descriptor.at("gamma").get<std::string>();
        texture.pixelType = descriptor.at("pixel_type").get<std::string>();
        texture.dataOrigin = descriptor.at("data_origin").get<std::string>();
        texture.width = descriptor.at("width").get<uint32_t>();
        texture.height = descriptor.at("height").get<uint32_t>();
        texture.depth = descriptor.at("depth").get<uint32_t>();
        require((texture.shape == "2d" || texture.shape == "bsdf_data")
            && texture.width > 0u && texture.height > 0u && texture.depth > 0u,
            "MDL texture descriptor is outside the V1 domain");
        require(texture.dataOrigin == "top_left" || texture.dataOrigin == "lower_left",
            "MDL texture has an unsupported row origin");
        require(!descriptor.at("data").is_null(), "MDL viewer requires bridge-decoded texture payloads");
        texture.dataPath = contained(
            root, root / descriptor.at("data").get<std::string>(), "MDL texture payload");
        const size_t bytesPerTexel = texture.pixelType == "Sint8" ? 1u
            : texture.pixelType == "Rgb" ? 3u
            : texture.pixelType == "Rgba" ? 4u
            : texture.pixelType == "Rgb_16" ? 6u
            : texture.pixelType == "Rgba_16" ? 8u
            : texture.pixelType == "Float32" ? 4u
            : texture.pixelType == "Float32<2>" ? 8u
            : texture.pixelType == "Float32<3>" || texture.pixelType == "Rgb_fp" ? 12u
            : texture.pixelType == "Float32<4>" || texture.pixelType == "Color" ? 16u : 0u;
        require(bytesPerTexel > 0u
            && std::filesystem::file_size(texture.dataPath)
                == size_t(texture.width) * texture.height * texture.depth * bytesPerTexel,
            "MDL texture payload has the wrong pixel type or size");
        require(texture.shape != "bsdf_data" || texture.pixelType == "Float32",
            "MDL BSDF-data texture must use Float32");
        result->textures.push_back(std::move(texture));
    }
    require(result->textures.size() <= 16u, "MDL V1 supports at most 16 texture resources");
    if (!entry.parameterView.is_null())
    {
        require(!result->argumentBlock.empty(),
            "ViewerMaterialCatalog editable artifact has no argument block");
        std::function<void(const json&)> validateWriteBounds = [&](const json& node) {
            if (node.value("editable", false))
            {
                const auto& write = node.at("metadata").at("reference_write");
                const size_t offset = write.at("offset").get<size_t>();
                const size_t size = write.at("size").get<size_t>();
                require(size > 0u && offset <= result->argumentBlock.size()
                        && size <= result->argumentBlock.size() - offset,
                    "ViewerMaterialCatalog reference write exceeds the argument block");
            }
            for (const auto& child : node.at("children")) validateWriteBounds(child);
        };
        validateWriteBounds(entry.parameterView.at("root"));
    }
    return result;
}
} // namespace ncls
