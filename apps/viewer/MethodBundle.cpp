#include "MethodBundle.h"

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

std::filesystem::path safeBundlePath(const std::filesystem::path& root, const std::string& uri)
{
    require(!uri.empty() && uri.find('\\') == std::string::npos, "bundle URI must be POSIX-relative");
    const std::filesystem::path relative(uri);
    require(!relative.is_absolute(), "bundle URI must be relative");
    for (const auto& part : relative) require(part != "..", "bundle URI cannot contain '..'");
    const auto canonicalRoot = std::filesystem::weakly_canonical(root);
    const auto target = std::filesystem::weakly_canonical(root / relative);
    auto rootIterator = canonicalRoot.begin();
    auto targetIterator = target.begin();
    for (; rootIterator != canonicalRoot.end(); ++rootIterator, ++targetIterator)
        require(targetIterator != target.end() && *rootIterator == *targetIterator, "bundle URI escapes its root");
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

ViewerMethod loadBundle(
    const std::filesystem::path& root,
    const std::filesystem::path& runtimeShaderDirectory
)
{
    const json manifest = readJson(root / "manifest.json");
    require(manifest.value("format_name", "") == "ncls.method-bundle", "unsupported MethodBundle format_name");
    require(manifest.value("format_version", 0u) == 1u, "unsupported MethodBundle format_version");
    require(manifest.value("runtime_class", "") == "realtime", "only realtime bundles enter the main viewer");
    require(manifest.value("scattering_contract_version", 0u) == 1u, "unsupported scattering contract");
    require(supportsIr(manifest.at("supported_ir_ids"), "ncls.layer-stack-ir@1"), "bundle does not support LayerStackIR v1");
    require(manifest.value("backend_id", "") == "legacy-ltc-k2", "current viewer build does not contain this backend variant");

    const auto& descriptor = manifest.at("backend_descriptor");
    require(descriptor.value("bounded_execution", false), "realtime backend must have bounded execution");
    constexpr uint32_t kRequiredCapabilities = 1u | 2u | 16u;
    require((descriptor.value("capabilities", 0u) & kRequiredCapabilities) == kRequiredCapabilities,
        "backend is missing prepare/evaluate/anisotropic-frame capabilities required by the viewer");
    require(descriptor.at("shader_entry_points").value("prepare", "") == "LegacyLtcK2P1Backend.prepare",
        "bundle prepare entry does not match the compiled P1 runtime");
    const auto& compiler = manifest.at("compiler");
    require(compiler.value("kind", "") == "parameter-network", "unsupported compiler kind");
    require(compiler.value("runtime_implementation", "") == "slang", "viewer does not load Python inference");
    require(compiler.value("architecture_id", "") == "legacy-ltc-k2-p1@2", "unsupported P1 architecture version");
    const auto& runtime = manifest.at("runtime");
    require(runtime.value("platform", "") == "windows-x86_64", "bundle targets another platform");
    require(runtime.value("graphics_api", "") == "d3d12", "bundle targets another graphics API");

    const auto& files = manifest.at("files");
    const auto& hashes = manifest.at("content_hashes");
    require(files.is_object() && hashes.is_object(), "bundle files/content_hashes must be objects");
    std::set<std::string> fileUris;
    for (auto iterator = files.begin(); iterator != files.end(); ++iterator)
        fileUris.insert(iterator.value().get<std::string>());
    std::set<std::string> hashUris;
    for (auto iterator = hashes.begin(); iterator != hashes.end(); ++iterator) hashUris.insert(iterator.key());
    require(fileUris == hashUris, "content_hashes must cover bundle files exactly");
    for (const auto& uri : fileUris)
    {
        const auto path = safeBundlePath(root, uri);
        require(std::filesystem::is_regular_file(path), "bundle file is missing: " + uri);
        require(sha256FileHex(path) == hashes.at(uri).get<std::string>(), "content hash mismatch: " + uri);
    }

    const auto logicalPath = [&](const char* name) {
        require(files.contains(name), std::string("bundle is missing logical file: ") + name);
        return safeBundlePath(root, files.at(name).get<std::string>());
    };
    const auto compilerShader = logicalPath("compiler_shader");
    const auto backendShader = logicalPath("backend_shader");
    const auto runtimeCompiler = runtimeShaderDirectory / "ncls/backends/legacy_ltc_k2/p1_compiler.slang";
    const auto runtimeBackend = runtimeShaderDirectory / "ncls/backends/legacy_ltc_k2/legacy_ltc_k2.slang";
    require(std::filesystem::is_regular_file(runtimeCompiler) && std::filesystem::is_regular_file(runtimeBackend),
        "viewer runtime shader copy is incomplete");
    require(sha256FileHex(compilerShader) == sha256FileHex(runtimeCompiler),
        "bundle P1 compiler differs from this viewer build; rebuild viewer for that shader variant");
    require(sha256FileHex(backendShader) == sha256FileHex(runtimeBackend),
        "bundle backend differs from this viewer build; rebuild viewer for that shader variant");

    const json layout = readJson(logicalPath("weight_layout"));
    require(layout.value("format_name", "") == "ncls.legacy-ltc-k2-p1-weights", "unsupported weight layout");
    require(layout.value("format_version", 0u) == 1u, "unsupported weight layout version");
    require(layout.value("dtype", "") == "float32-little-endian", "unsupported inference precision/layout");
    const uint32_t width = layout.at("width").get<uint32_t>();
    const uint32_t parameterCount = layout.at("total_floats").get<uint32_t>();
    require(width >= 8 && width <= 64 && width % 2 == 0 && layout.value("type_width", 0u) == 8u,
        "P1 width/type_width is outside the Slang runtime bounds");
    const auto weightBytes = readBytes(logicalPath("weights"));
    require(weightBytes.size() == static_cast<size_t>(parameterCount) * sizeof(float), "weight binary length disagrees with layout");
    std::vector<float> weights(parameterCount);
    std::memcpy(weights.data(), weightBytes.data(), weightBytes.size());

    const json parity = readJson(logicalPath("parity"));
    require(parity.value("format_name", "") == "ncls.backend-parity-probe", "unsupported parity probe");
    require(parity.value("architecture_id", "") == compiler.value("architecture_id", ""), "parity architecture mismatch");
    require(parity.value("weight_width", 0u) == width, "parity weight width mismatch");
    const auto parityBytes = readBytes(logicalPath("parity_material"));
    require(parityBytes.size() == 752, "parity material does not contain one LayerStackIR v1 record");
    ParityProbe probe{};
    std::memcpy(probe.material.data(), parityBytes.data(), parityBytes.size());
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
    probe.relativeTolerance = parity.at("tolerance").value("rtol", 4e-5f);
    probe.absoluteTolerance = parity.at("tolerance").value("atol", 4e-6f);

    ViewerMethod method{};
    method.root = root;
    method.methodId = manifest.at("method_id").get<std::string>();
    method.displayName = manifest.at("display_name").get<std::string>();
    method.sourceGitCommit = manifest.at("source_git_commit").get<std::string>();
    method.backendId = manifest.at("backend_id").get<std::string>();
    method.backendVersion = manifest.at("backend_version").get<uint32_t>();
    method.architectureId = compiler.at("architecture_id").get<std::string>();
    for (const auto& value : manifest.at("supported_ir_ids"))
        method.supportedIrIds.push_back(value.get<std::string>());
    method.width = width;
    method.parameterCount = parameterCount;
    method.stateBytesPerPixel = descriptor.at("state_stride").get<uint32_t>();
    method.compiledMaterialBytes = descriptor.at("cost_model").at("compiled_material_bytes").get<uint32_t>();
    method.weights = std::move(weights);
    method.parity = std::move(probe);
    require(method.methodId.size() == 64, "method_id is not SHA-256 sized");
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
