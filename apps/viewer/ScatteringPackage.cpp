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
    ViewerProgram result{};
    result.root = root;
    result.packageId = manifest.at("package_id").get<std::string>();
    result.programRuntimeId = manifest.at("program_runtime_id").get<std::string>();
    result.materialAssetId = manifest.at("material_asset_id").get<std::string>();
    result.sourceSnapshotId = manifest.at("source_snapshot_id").get<std::string>();
    result.displayName = manifest.at("program_key").get<std::string>();
    result.backendId = result.displayName;
    result.runtimeClass = manifest.at("program_kind").get<std::string>();
    result.architectureId = result.programRuntimeId;
    result.compiledStateId = result.materialAssetId;
    result.compiledMaterialIrSha256 = result.sourceSnapshotId;
    result.shaderModule = logicalPath(program.at("module").get<std::string>()).generic_string();
    result.capabilities = manifest.at("capabilities").get<uint32_t>();
    for (auto item = program.at("defines").begin(); item != program.at("defines").end(); ++item)
        result.shaderDefines.emplace(item.key(), item.value().get<std::string>());
    auto consume = [&](const json& descriptors, bool runtime) {
        for (auto item = descriptors.begin(); item != descriptors.end(); ++item)
        {
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
