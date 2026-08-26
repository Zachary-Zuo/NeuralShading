#include "MethodBundle.h"

#include "Hash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace ncls
{
namespace
{
using json = nlohmann::json;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open JSON: " + path.string());
    return json::parse(stream);
}

std::filesystem::path safeRelativePath(const std::filesystem::path& root, const std::string& uri)
{
    require(!uri.empty() && uri.find('\\') == std::string::npos, "URI must be POSIX-relative");
    const std::filesystem::path relative(uri);
    require(!relative.is_absolute(), "URI must be relative");
    for (const auto& part : relative) require(part != "..", "URI cannot contain '..'");
    const auto canonicalRoot = std::filesystem::weakly_canonical(root);
    const auto target = std::filesystem::weakly_canonical(root / relative);
    auto rootIterator = canonicalRoot.begin();
    auto targetIterator = target.begin();
    for (; rootIterator != canonicalRoot.end(); ++rootIterator, ++targetIterator)
        require(targetIterator != target.end() && *rootIterator == *targetIterator, "URI escapes its root");
    return target;
}

std::vector<std::byte> readBytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open binary file: " + path.string());
    const auto length = stream.tellg();
    require(length >= 0, "cannot determine binary file length");
    std::vector<std::byte> bytes(static_cast<size_t>(length));
    stream.seekg(0);
    if (!bytes.empty()) stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    require(static_cast<bool>(stream) || stream.eof(), "failed to read binary file");
    return bytes;
}

bool supportsIr(const json& values, const char* id)
{
    return values.is_array() && std::any_of(values.begin(), values.end(), [id](const json& value) { return value == id; });
}

bool validDefineName(const std::string& value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isupper(character) || std::isdigit(character) || character == '_';
    });
}

bool validDefineValue(const std::string& value)
{
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return std::isdigit(character);
    });
}

ViewerMethod loadBundle(
    const std::filesystem::path& root,
    const std::filesystem::path& runtimeShaderDirectory)
{
    const json manifest = readJson(root / "manifest.json");
    require(manifest.value("format_name", "") == "ncls.method-bundle", "unsupported MethodBundle format_name");
    require(manifest.value("format_version", 0u) == 1u, "unsupported MethodBundle format_version");
    const std::string runtimeClass = manifest.value("runtime_class", "");
    require(runtimeClass == "diagnostic" || runtimeClass == "realtime", "unsupported MethodBundle runtime_class");
    require(manifest.value("scattering_contract_version", 0u) == 1u, "unsupported scattering contract");
    require(supportsIr(manifest.at("supported_ir_ids"), "ncls.layer-stack-ir@1"),
        "bundle does not support LayerStackIR v1");

    const auto& descriptor = manifest.at("backend_descriptor");
    require(descriptor.value("backend_id", "") == manifest.value("backend_id", ""),
        "backend descriptor identity disagrees with manifest");
    require(descriptor.value("backend_version", 0u) == manifest.value("backend_version", 0u),
        "backend descriptor version disagrees with manifest");
    require(descriptor.value("bounded_execution", false), "viewer backend must have bounded execution");
    constexpr uint32_t kRequiredCapabilities = 1u | 2u | 16u;
    require((descriptor.value("capabilities", 0u) & kRequiredCapabilities) == kRequiredCapabilities,
        "backend is missing prepare/evaluate/anisotropic-frame capabilities required by the viewer");
    const auto& compiler = manifest.at("compiler");
    require(compiler.value("runtime_implementation", "") == "slang", "viewer does not load Python inference");
    require(!compiler.value("compile_mode", "").empty(), "bundle compiler must declare its compile mode");
    const auto& runtime = manifest.at("runtime");
    require(runtime.value("platform", "") == "windows-x86_64", "bundle targets another platform");
    require(runtime.value("graphics_api", "") == "d3d12", "bundle targets another graphics API");
    require(runtime.value("shader_model", "") == "6.5", "bundle targets another shader model");
    require(runtime.value("slang_version", "") == "2024.1.34", "bundle targets another Slang version");

    const auto& files = manifest.at("files");
    const auto& hashes = manifest.at("content_hashes");
    require(files.is_object() && hashes.is_object(), "bundle files/content_hashes must be objects");
    std::set<std::string> fileUris;
    for (auto iterator = files.begin(); iterator != files.end(); ++iterator)
        fileUris.insert(iterator.value().get<std::string>());
    require(fileUris.size() == files.size(), "bundle logical files must have unique URIs");
    std::set<std::string> hashUris;
    for (auto iterator = hashes.begin(); iterator != hashes.end(); ++iterator) hashUris.insert(iterator.key());
    require(fileUris == hashUris, "content_hashes must cover bundle files exactly");
    for (const auto& uri : fileUris)
    {
        const auto path = safeRelativePath(root, uri);
        require(std::filesystem::is_regular_file(path), "bundle file is missing: " + uri);
        require(sha256FileHex(path) == hashes.at(uri).get<std::string>(), "content hash mismatch: " + uri);
    }
    const auto logicalPath = [&](const char* name) {
        require(files.contains(name), std::string("bundle is missing logical file: ") + name);
        return safeRelativePath(root, files.at(name).get<std::string>());
    };

    const auto& specialization = runtime.at("shader_specialization");
    const std::string shaderModule = specialization.at("module").get<std::string>();
    const auto runtimeBackend = safeRelativePath(runtimeShaderDirectory, shaderModule);
    require(std::filesystem::is_regular_file(runtimeBackend), "viewer runtime shader module is missing: " + shaderModule);
    require(sha256FileHex(logicalPath("backend_shader")) == sha256FileHex(runtimeBackend),
        "bundle backend differs from this viewer build; rebuild viewer for that shader variant");
    require(specialization.value("shared_weight_storage", "") == "float16-little-endian",
        "unsupported shared weight storage");

    const uint32_t compiledMaterialStride = specialization.at("compiled_material_stride").get<uint32_t>();
    const uint32_t stateStride = specialization.at("packed_state_stride").get<uint32_t>();
    const uint32_t compiledMaterialIndex = specialization.at("compiled_material_index").get<uint32_t>();
    require(compiledMaterialStride > 0u && compiledMaterialStride % 16u == 0u,
        "compiled material stride must be positive and 16-byte aligned");
    require(stateStride == descriptor.at("state_stride").get<uint32_t>(),
        "shader specialization state stride disagrees with backend descriptor");
    require(compiledMaterialStride == descriptor.at("cost_model").at("compiled_material_bytes").get<uint32_t>(),
        "shader specialization material stride disagrees with backend descriptor");

    std::map<std::string, std::string> shaderDefines;
    const auto& defines = specialization.at("defines");
    require(defines.is_object() && !defines.empty(), "shader specialization defines must be a nonempty object");
    for (auto iterator = defines.begin(); iterator != defines.end(); ++iterator)
    {
        const std::string name = iterator.key();
        const std::string value = iterator.value().get<std::string>();
        require(validDefineName(name) && validDefineValue(value), "unsafe shader specialization define");
        shaderDefines.emplace(name, value);
    }

    const auto weightBytes = readBytes(logicalPath("shared_weights"));
    require(!weightBytes.empty() && weightBytes.size() % sizeof(uint32_t) == 0u,
        "packed shared weights must contain complete uint words");
    std::vector<uint32_t> sharedWeightWords(weightBytes.size() / sizeof(uint32_t));
    std::memcpy(sharedWeightWords.data(), weightBytes.data(), weightBytes.size());
    auto compiledMaterials = readBytes(logicalPath("compiled_materials"));
    require(!compiledMaterials.empty() && compiledMaterials.size() % compiledMaterialStride == 0u,
        "compiled material table length disagrees with its stride");
    const size_t materialCount64 = compiledMaterials.size() / compiledMaterialStride;
    require(materialCount64 <= std::numeric_limits<uint32_t>::max(), "compiled material table is too large");
    const uint32_t compiledMaterialCount = static_cast<uint32_t>(materialCount64);
    require(compiledMaterialIndex < compiledMaterialCount, "compiled material index is outside its table");

    const json parity = readJson(logicalPath("parity"));
    require(parity.value("format_name", "") == "ncls.backend-parity-probe", "unsupported parity probe");
    require(parity.value("format_version", 0u) == 1u, "unsupported parity probe version");
    require(parity.value("architecture_id", "") == compiler.value("architecture_id", ""),
        "parity architecture mismatch");
    require(parity.value("compiled_set_id", "") == compiler.value("compiled_set_id", ""),
        "parity compiled set mismatch");
    require(parity.value("compiled_state_id", "") == compiler.value("compiled_state_id", ""),
        "parity compiled state mismatch");
    ParityProbe probe{};
    const auto& view = parity.at("view_direction_local");
    require(view.is_array() && view.size() == 3, "parity view direction must be float3");
    for (size_t index = 0; index < 3; ++index) probe.view[index] = view[index].get<float>();
    const auto& lights = parity.at("light_directions_local");
    const auto& expected = parity.at("expected_response_cos");
    require(lights.is_array() && expected.is_array() && lights.size() == expected.size() && !lights.empty(),
        "parity light/expected arrays disagree");
    for (size_t item = 0; item < lights.size(); ++item)
    {
        require(lights[item].size() == 3 && expected[item].size() == 3, "parity entries must be float3");
        probe.lights.push_back({lights[item][0].get<float>(), lights[item][1].get<float>(), lights[item][2].get<float>()});
        probe.expectedResponseCos.push_back({expected[item][0].get<float>(), expected[item][1].get<float>(), expected[item][2].get<float>()});
    }
    probe.relativeTolerance = parity.at("tolerance").at("rtol").get<float>();
    probe.absoluteTolerance = parity.at("tolerance").at("atol").get<float>();

    ViewerMethod method{};
    method.root = root;
    method.methodId = manifest.at("method_id").get<std::string>();
    method.displayName = manifest.at("display_name").get<std::string>();
    method.sourceGitCommit = manifest.at("source_git_commit").get<std::string>();
    method.backendId = manifest.at("backend_id").get<std::string>();
    method.backendVersion = manifest.at("backend_version").get<uint32_t>();
    method.runtimeClass = runtimeClass;
    method.architectureId = compiler.at("architecture_id").get<std::string>();
    method.compiledStateId = compiler.at("compiled_state_id").get<std::string>();
    method.compiledMaterialIrSha256 = compiler.at("compiled_material_ir_sha256").get<std::string>();
    method.previewMaterial = logicalPath("preview_material");
    for (const auto& value : manifest.at("supported_ir_ids")) method.supportedIrIds.push_back(value.get<std::string>());
    method.shaderModule = shaderModule;
    method.shaderDefines = std::move(shaderDefines);
    method.parameterCount = static_cast<uint32_t>(weightBytes.size() / sizeof(uint16_t));
    method.capabilities = descriptor.at("capabilities").get<uint32_t>();
    method.stateBytesPerPixel = stateStride;
    method.compiledMaterialBytes = compiledMaterialStride;
    method.compiledMaterialCount = compiledMaterialCount;
    method.compiledMaterialIndex = compiledMaterialIndex;
    method.environmentQueryBudget = runtime.value("environment_query_budget", 1u);
    method.rectangleQueryBudget = runtime.value("rectangle_query_budget", 1u);
    method.sharedWeightWords = std::move(sharedWeightWords);
    method.compiledMaterials = std::move(compiledMaterials);
    method.parity = std::move(probe);
    require(method.methodId.size() == 64, "method_id is not SHA-256 sized");
    require(method.compiledStateId.size() == 64, "compiled state id is not SHA-256 sized");
    require(method.compiledMaterialIrSha256.size() == 64, "compiled material IR hash is not SHA-256 sized");
    require(method.environmentQueryBudget >= 1u && method.environmentQueryBudget <= 8u,
        "environment query budget is outside viewer bounds");
    require(method.rectangleQueryBudget >= 1u && method.rectangleQueryBudget <= 8u,
        "rectangle query budget is outside viewer bounds");
    return method;
}
} // namespace

BundleScanResult scanMethodBundles(const std::filesystem::path& root, const std::filesystem::path& runtimeShaderDirectory)
{
    BundleScanResult result;
    if (!std::filesystem::is_directory(root))
    {
        result.failures.push_back({root, "bundle root does not exist"});
        return result;
    }
    std::vector<std::filesystem::path> candidates;
    if (std::filesystem::is_regular_file(root / "manifest.json")) candidates.push_back(root);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(
             root, std::filesystem::directory_options::skip_permission_denied))
    {
        if (entry.is_regular_file() && entry.path().filename() == "manifest.json") candidates.push_back(entry.path().parent_path());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (const auto& candidate : candidates)
    {
        try
        {
            result.methods.push_back(loadBundle(candidate, runtimeShaderDirectory));
        }
        catch (const std::exception& error)
        {
            result.failures.push_back({candidate, error.what()});
        }
    }
    std::sort(result.methods.begin(), result.methods.end(), [](const ViewerMethod& left, const ViewerMethod& right) {
        return left.displayName < right.displayName;
    });
    return result;
}
} // namespace ncls
