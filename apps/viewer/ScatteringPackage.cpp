#include "ScatteringPackage.h"

#include "Hash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

namespace ncls
{
namespace
{
using json = nlohmann::json;
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }

void requireKeys(const json& value, const std::set<std::string>& expected, const std::string& label)
{
    require(value.is_object(), label + " must be an object");
    std::set<std::string> actual;
    for (auto item = value.begin(); item != value.end(); ++item) actual.insert(item.key());
    require(actual == expected, label + " has unknown or missing fields");
}

std::string sha256Json(const json& value)
{
    const std::string payload = value.dump();
    return sha256Hex(payload.data(), payload.size());
}

uint32_t readU32(const std::vector<std::byte>& bytes, size_t offset)
{
    require(offset + sizeof(uint32_t) <= bytes.size(), "typed resource header is truncated");
    uint32_t result = 0;
    std::memcpy(&result, bytes.data() + offset, sizeof(result));
    return result;
}

json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open JSON: " + path.string());
    return json::parse(stream);
}

std::filesystem::path safePath(const std::filesystem::path& root, const std::string& uri)
{
    require(!uri.empty() && uri.find('\\') == std::string::npos, "package URI must be POSIX-relative");
    std::filesystem::path relative(uri);
    require(!relative.is_absolute(), "package URI must be relative");
    for (const auto& part : relative) require(part != "..", "package URI cannot contain '..'");
    const auto canonicalRoot = std::filesystem::weakly_canonical(root);
    const auto target = std::filesystem::weakly_canonical(root / relative);
    auto left = canonicalRoot.begin();
    auto right = target.begin();
    for (; left != canonicalRoot.end(); ++left, ++right)
        require(right != target.end() && *left == *right, "package URI escapes root");
    return target;
}

std::vector<std::byte> readBytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(bool(stream), "cannot open package blob: " + path.string());
    const auto length = stream.tellg();
    require(length >= 0, "cannot determine package blob length");
    std::vector<std::byte> result(static_cast<size_t>(length));
    stream.seekg(0);
    if (!result.empty()) stream.read(reinterpret_cast<char*>(result.data()), result.size());
    return result;
}

ViewerProgram loadPackage(const std::filesystem::path& root)
{
    const json manifest = readJson(root / "manifest.json");
    require(manifest.value("format_name", "") == "ncls.scattering-package" && manifest.value("format_version", 0u) == 1u,
        "unsupported ScatteringPackage format");
    require(manifest.value("scattering_contract_version", 0u) == 1u, "unsupported scattering contract");
    const auto& files = manifest.at("files");
    const auto& hashes = manifest.at("content_hashes");
    require(files.is_object() && hashes.is_object(), "package files/content_hashes must be objects");
    std::set<std::string> uris;
    for (auto item = files.begin(); item != files.end(); ++item) uris.insert(item.value().get<std::string>());
    require(uris.size() == files.size() && hashes.size() == files.size(), "package files must be unique and fully hashed");
    for (const auto& uri : uris)
    {
        require(hashes.contains(uri), "package content hash is missing: " + uri);
        const auto path = safePath(root, uri);
        require(std::filesystem::is_regular_file(path), "package content is missing: " + uri);
        require(sha256FileHex(path) == hashes.at(uri).get<std::string>(), "package content hash mismatch: " + uri);
    }
    auto logicalPath = [&](const std::string& logical) {
        require(files.contains(logical), "package logical file is missing: " + logical);
        return safePath(root, files.at(logical).get<std::string>());
    };
    const auto& program = manifest.at("program");
    const auto& material = manifest.at("material");
    requireKeys(program, {"module", "defines", "blobs"}, "package program");
    requireKeys(material, {"blobs", "resources"}, "package material");
    auto filteredHashes = [&](const std::string& prefix, const std::string& exact = {}) {
        json result = json::object();
        for (auto item = files.begin(); item != files.end(); ++item)
        {
            const std::string logical = item.key();
            if ((!exact.empty() && logical == exact) || logical.rfind(prefix, 0u) == 0u)
                result[item.value().get<std::string>()] = hashes.at(item.value().get<std::string>());
        }
        return result;
    };
    const json programIdentity = {
        {"program_kind", manifest.at("program_kind")},
        {"program_key", manifest.at("program_key")},
        {"program_version", manifest.at("program_version")},
        {"program_descriptor_sha256", manifest.at("program_descriptor_sha256")},
        {"scattering_contract_version", manifest.at("scattering_contract_version")},
        {"runtime_abi", manifest.at("runtime_abi")},
        {"capabilities", manifest.at("capabilities")},
        {"program", program},
        {"content_hashes", filteredHashes("runtime/", program.at("module").get<std::string>())},
    };
    require(sha256Json(programIdentity) == manifest.at("program_runtime_id").get<std::string>(),
        "program_runtime_id does not match runtime semantics");
    const json materialIdentity = {
        {"source_family_id", manifest.at("source_family_id")},
        {"source_contract_version", manifest.at("source_contract_version")},
        {"source_snapshot_id", manifest.at("source_snapshot_id")},
        {"program_descriptor_sha256", manifest.at("program_descriptor_sha256")},
        {"material", material},
        {"content_hashes", filteredHashes("material/")},
    };
    require(sha256Json(materialIdentity) == manifest.at("material_asset_id").get<std::string>(),
        "material_asset_id does not match material semantics");
    json packageHashes = json::object();
    for (auto item = hashes.begin(); item != hashes.end(); ++item)
        if (item.key().rfind("validation/", 0u) == 0u || item.key().rfind("provenance/", 0u) == 0u)
            packageHashes[item.key()] = item.value();
    const json packageIdentity = {
        {"format_name", manifest.at("format_name")},
        {"format_version", manifest.at("format_version")},
        {"program_runtime_id", manifest.at("program_runtime_id")},
        {"material_asset_id", manifest.at("material_asset_id")},
        {"validation", manifest.at("validation")},
        {"provenance", manifest.at("provenance")},
        {"content_hashes", packageHashes},
    };
    require(sha256Json(packageIdentity) == manifest.at("package_id").get<std::string>(),
        "package_id does not match package semantics");
    ViewerProgram result{};
    result.root = root;
    result.packageId = manifest.at("package_id").get<std::string>();
    result.programRuntimeId = manifest.at("program_runtime_id").get<std::string>();
    result.materialAssetId = manifest.at("material_asset_id").get<std::string>();
    result.sourceSnapshotId = manifest.at("source_snapshot_id").get<std::string>();
    result.sourceFamilyId = manifest.at("source_family_id").get<std::string>();
    const json sourceIdentity = readJson(logicalPath("provenance/source"));
    result.sourceAssetSha256 = sourceIdentity.at("source_asset_sha256").get<std::string>();
    result.displayName = manifest.at("program_key").get<std::string>();
    result.backendId = result.displayName;
    result.runtimeClass = manifest.at("program_kind").get<std::string>();
    result.architectureId = result.programRuntimeId;
    result.compiledStateId = result.materialAssetId;
    result.compiledMaterialIrSha256 = result.sourceSnapshotId;
    result.shaderModule = logicalPath(program.at("module").get<std::string>()).generic_string();
    result.capabilities = manifest.at("capabilities").get<uint32_t>();
    const auto& validation = manifest.at("validation");
    if (validation.contains("parity"))
    {
        const auto& parity = validation.at("parity");
        if (parity.contains("view"))
            result.parity.view = parity.at("view").get<std::array<float, 3>>();
        if (parity.contains("lights"))
            result.parity.lights = parity.at("lights").get<std::vector<std::array<float, 3>>>();
        if (parity.contains("expected_response_cos"))
            result.parity.expectedResponseCos = parity.at("expected_response_cos").get<std::vector<std::array<float, 3>>>();
        result.parity.relativeTolerance = parity.value("relative_tolerance", result.parity.relativeTolerance);
        result.parity.absoluteTolerance = parity.value("absolute_tolerance", result.parity.absoluteTolerance);
    }
    for (auto item = program.at("defines").begin(); item != program.at("defines").end(); ++item)
        result.shaderDefines.emplace(item.key(), item.value().get<std::string>());
    auto consume = [&](const json& descriptors, bool runtime) {
        for (auto item = descriptors.begin(); item != descriptors.end(); ++item)
        {
            requireKeys(item.value(), {"dtype", "shape", "stride", "alignment", "usage"},
                "typed blob descriptor");
            const auto path = logicalPath(item.key());
            auto bytes = readBytes(path);
            const std::string usage = item.value().at("usage").get<std::string>();
            if (runtime && usage == "gNclsRuntimeWeights")
            {
                require(bytes.size() % 4u == 0u, "runtime weight blob must contain complete uint words");
                result.sharedWeightWords.resize(bytes.size() / 4u);
                std::memcpy(result.sharedWeightWords.data(), bytes.data(), bytes.size());
                result.parameterCount = uint32_t(bytes.size() / 2u);
            }
            else if (!runtime)
            {
                result.compiledMaterials.insert(result.compiledMaterials.end(), bytes.begin(), bytes.end());
                result.compiledMaterialBytes = item.value().at("stride").get<uint32_t>();
            }
        }
    };
    consume(program.at("blobs"), true);
    consume(material.at("blobs"), false);
    for (auto item = material.at("resources").begin(); item != material.at("resources").end(); ++item)
    {
        requireKeys(item.value(), {"dtype", "shape", "stride", "alignment", "usage"},
            "typed resource descriptor");
        ncls::ViewerTypedResource resource;
        resource.path = logicalPath(item.key());
        resource.dtype = item.value().at("dtype").get<std::string>();
        resource.shape = item.value().at("shape").get<std::vector<uint32_t>>();
        resource.stride = item.value().at("stride").get<uint32_t>();
        resource.alignment = item.value().at("alignment").get<uint32_t>();
        resource.usage = item.value().at("usage").get<std::string>();
        require(!resource.dtype.empty() && !resource.shape.empty() && resource.stride > 0u
            && resource.alignment > 0u && !resource.usage.empty(), "typed resource descriptor is invalid");
        if (resource.dtype == "texture2d-rgba16float-dds@1")
        {
            const auto bytes = readBytes(resource.path);
            require(bytes.size() >= 148u && std::memcmp(bytes.data(), "DDS ", 4u) == 0,
                "RGBA16F resource is not a DDS file");
            require(readU32(bytes, 4u) == 124u && readU32(bytes, 76u) == 32u
                && readU32(bytes, 80u) == 4u && std::memcmp(bytes.data() + 84u, "DX10", 4u) == 0,
                "RGBA16F DDS legacy header is invalid");
            const uint32_t height = readU32(bytes, 12u);
            const uint32_t width = readU32(bytes, 16u);
            const uint32_t mipCount = readU32(bytes, 28u);
            require(width > 0u && height > 0u && mipCount > 0u
                && readU32(bytes, 20u) == width * 8u && readU32(bytes, 24u) == 0u
                && readU32(bytes, 128u) == 10u && readU32(bytes, 132u) == 3u
                && readU32(bytes, 136u) == 0u && readU32(bytes, 140u) == 1u
                && readU32(bytes, 144u) == 0u,
                "DDS resource is not a 2D RGBA16F texture");
            uint64_t expectedBytes = 148u;
            uint32_t mipWidth = width;
            uint32_t mipHeight = height;
            for (uint32_t mip = 0u; mip < mipCount; ++mip)
            {
                expectedBytes += uint64_t(mipWidth) * uint64_t(mipHeight) * 8u;
                mipWidth = std::max(1u, mipWidth / 2u);
                mipHeight = std::max(1u, mipHeight / 2u);
            }
            require(bytes.size() == expectedBytes,
                "RGBA16F DDS payload length disagrees with its mip chain");
            require(resource.shape == std::vector<uint32_t>({width, height, mipCount, 4u})
                && resource.stride == 8u && resource.alignment >= 16u,
                "RGBA16F DDS descriptor disagrees with its header");
        }
        else
        {
            require(resource.dtype == "sampler-linear-wrap-explicit-lod@1",
                "unsupported typed package resource dtype: " + resource.dtype);
            require(resource.shape == std::vector<uint32_t>({1u})
                && resource.stride == 1u && resource.alignment == 1u,
                "typed sampler descriptor is invalid");
            const json sampler = readJson(resource.path);
            requireKeys(sampler, {"address", "filter", "mip"}, "typed sampler");
            require(sampler.at("filter") == "linear" && sampler.at("mip") == "explicit"
                && sampler.at("address") == "wrap", "typed sampler state is unsupported");
        }
        result.resources.push_back(std::move(resource));
    }
    if (result.sharedWeightWords.empty()) result.sharedWeightWords.push_back(0u);
    if (result.compiledMaterials.empty()) result.compiledMaterials.resize(result.compiledMaterialBytes);
    require(result.compiledMaterialBytes > 0u && result.compiledMaterials.size() % result.compiledMaterialBytes == 0u,
        "material blob length does not match declared stride");
    result.compiledMaterialCount = uint32_t(result.compiledMaterials.size() / result.compiledMaterialBytes);
    return result;
}
} // namespace

PackageScanResult scanScatteringPackages(const std::filesystem::path& root, const std::filesystem::path&)
{
    PackageScanResult result;
    if (!std::filesystem::is_directory(root)) { result.failures.push_back({root, "package root does not exist"}); return result; }
    std::vector<std::filesystem::path> candidates;
    if (std::filesystem::is_regular_file(root / "manifest.json")) candidates.push_back(root);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::skip_permission_denied))
        if (entry.is_regular_file() && entry.path().filename() == "manifest.json") candidates.push_back(entry.path().parent_path());
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (const auto& candidate : candidates)
        try { result.programs.push_back(loadPackage(candidate)); }
        catch (const std::exception& error) { result.failures.push_back({candidate, error.what()}); }
    return result;
}
} // namespace ncls
