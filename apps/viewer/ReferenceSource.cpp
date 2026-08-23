#include "ReferenceSource.h"

#include "Hash.h"

#include <nlohmann/json.hpp>
#include <pugixml.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace ncls
{
namespace
{
constexpr int32_t kMerlThetaHalf = 90;
constexpr int32_t kMerlThetaDifference = 90;
constexpr int32_t kMerlPhiDifference = 180;
constexpr size_t kMerlSampleCount = size_t(kMerlThetaHalf) * kMerlThetaDifference * kMerlPhiDifference;
constexpr double kMerlScale[3] = {1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0};
using json = nlohmann::json;

constexpr uint32_t kMxBase = 0;
constexpr uint32_t kMxBaseColor = 1;
constexpr uint32_t kMxDiffuseRoughness = 4;
constexpr uint32_t kMxMetalness = 5;
constexpr uint32_t kMxHasBaseColorTexture = 6;
constexpr uint32_t kMxHasMetalnessTexture = 7;
constexpr uint32_t kMxSpecular = 8;
constexpr uint32_t kMxSpecularColor = 9;
constexpr uint32_t kMxSpecularRoughness = 12;
constexpr uint32_t kMxHasRoughnessTexture = 13;
constexpr uint32_t kMxSpecularIor = 14;
constexpr uint32_t kMxSpecularAnisotropy = 15;
constexpr uint32_t kMxSpecularRotation = 16;
constexpr uint32_t kMxNormalScale = 17;
constexpr uint32_t kMxHasNormalTexture = 18;
constexpr uint32_t kMxEmission = 19;
constexpr uint32_t kMxEmissionColor = 20;
constexpr uint32_t kMxOpacity = 23;

std::array<float, 24> defaultMaterialXInputs()
{
    std::array<float, 24> result{};
    result[kMxBase] = 1.f;
    result[kMxBaseColor + 0] = .8f;
    result[kMxBaseColor + 1] = .8f;
    result[kMxBaseColor + 2] = .8f;
    result[kMxSpecular] = 1.f;
    result[kMxSpecularColor + 0] = 1.f;
    result[kMxSpecularColor + 1] = 1.f;
    result[kMxSpecularColor + 2] = 1.f;
    result[kMxSpecularRoughness] = .2f;
    result[kMxSpecularIor] = 1.5f;
    result[kMxNormalScale] = 1.f;
    result[kMxEmissionColor + 0] = 1.f;
    result[kMxEmissionColor + 1] = 1.f;
    result[kMxEmissionColor + 2] = 1.f;
    result[kMxOpacity] = 1.f;
    return result;
}

pugi::xml_node requireNamedChild(pugi::xml_node parent, const char* category, const std::string& name)
{
    const auto result = parent.find_child_by_attribute(category, "name", name.c_str());
    if (!result) throw std::runtime_error("MaterialX missing " + std::string(category) + " node: " + name);
    return result;
}

pugi::xml_node findInput(pugi::xml_node parent, const char* name)
{
    return parent.find_child_by_attribute("input", "name", name);
}

float parseFloat(const std::string& value, const std::string& label)
{
    size_t consumed = 0;
    const float result = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result))
        throw std::runtime_error("MaterialX input is not a finite float: " + label);
    return result;
}

std::array<float, 3> parseFloat3(std::string value, const std::string& label)
{
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream stream(value);
    std::array<float, 3> result{};
    std::string trailing;
    if (!(stream >> result[0] >> result[1] >> result[2]) || (stream >> trailing))
        throw std::runtime_error("MaterialX input is not a float3: " + label);
    for (float component : result)
        if (!std::isfinite(component)) throw std::runtime_error("MaterialX input is non-finite: " + label);
    return result;
}

float inputFloat(pugi::xml_node node, const char* name, float fallback)
{
    const auto input = findInput(node, name);
    if (!input) return fallback;
    if (!input.attribute("value") || input.attribute("nodegraph") || input.attribute("nodename"))
        throw std::runtime_error("MaterialX Falcor subset requires a constant float input: " + std::string(name));
    return parseFloat(input.attribute("value").value(), name);
}

std::array<float, 3> inputFloat3(
    pugi::xml_node node,
    const char* name,
    const std::array<float, 3>& fallback)
{
    const auto input = findInput(node, name);
    if (!input) return fallback;
    if (!input.attribute("value") || input.attribute("nodegraph") || input.attribute("nodename"))
        throw std::runtime_error("MaterialX Falcor subset requires a constant float3 input: " + std::string(name));
    return parseFloat3(input.attribute("value").value(), name);
}

bool isZero(float value) { return std::abs(value) <= 1e-8f; }

std::filesystem::path resolveMaterialXFile(
    const std::filesystem::path& documentPath,
    const std::string& authoredPath)
{
    if (authoredPath.empty()) throw std::runtime_error("MaterialX image node has an empty filename");
    const auto root = std::filesystem::weakly_canonical(documentPath.parent_path());
    const auto result = std::filesystem::weakly_canonical(root / std::filesystem::path(authoredPath));
    const auto relative = std::filesystem::relative(result, root);
    if (relative.empty() || relative.is_absolute())
        throw std::runtime_error("MaterialX texture path escapes the document directory: " + authoredPath);
    for (const auto& component : relative)
        if (component == "..") throw std::runtime_error("MaterialX texture path escapes the document directory: " + authoredPath);
    if (!std::filesystem::is_regular_file(result))
        throw std::runtime_error("MaterialX texture is missing: " + result.string());
    return result;
}

pugi::xml_node resolveGraphOutput(
    pugi::xml_node root,
    pugi::xml_node connectedInput,
    const char* expectedCategory)
{
    if (!connectedInput.attribute("nodegraph") || !connectedInput.attribute("output"))
        throw std::runtime_error("MaterialX input does not use an explicit nodegraph output: "
            + std::string(connectedInput.attribute("name").value()));
    const auto graph = requireNamedChild(root, "nodegraph", connectedInput.attribute("nodegraph").value());
    const auto output = requireNamedChild(graph, "output", connectedInput.attribute("output").value());
    if (!output.attribute("nodename")) throw std::runtime_error("MaterialX graph output has no nodename");
    return requireNamedChild(graph, expectedCategory, output.attribute("nodename").value());
}

std::filesystem::path imagePath(
    const std::filesystem::path& documentPath,
    pugi::xml_node image,
    const char* expectedType,
    const char* expectedColorSpace = nullptr)
{
    if (std::string(image.attribute("type").value()) != expectedType)
        throw std::runtime_error("MaterialX image node has the wrong type: " + std::string(image.attribute("name").value()));
    const auto file = findInput(image, "file");
    if (!file || std::string(file.attribute("type").value()) != "filename" || !file.attribute("value"))
        throw std::runtime_error("MaterialX image node has no filename input: " + std::string(image.attribute("name").value()));
    const std::string colorSpace = file.attribute("colorspace").value();
    if (expectedColorSpace && colorSpace != expectedColorSpace)
        throw std::runtime_error("MaterialX image colorspace mismatch: " + std::string(image.attribute("name").value()));
    if (!expectedColorSpace && !colorSpace.empty())
        throw std::runtime_error("MaterialX raw data image has an unexpected colorspace: " + std::string(image.attribute("name").value()));
    const auto texcoord = findInput(image, "texcoord");
    if (!texcoord || std::string(texcoord.attribute("type").value()) != "vector2" || !texcoord.attribute("nodename"))
        throw std::runtime_error("MaterialX image must use an explicit texcoord node: " + std::string(image.attribute("name").value()));
    return resolveMaterialXFile(documentPath, file.attribute("value").value());
}

void requireZeroInput(pugi::xml_node surface, const char* name)
{
    if (!isZero(inputFloat(surface, name, 0.f)))
        throw std::runtime_error("MaterialX Falcor surface-response v1 does not support nonzero input: " + std::string(name));
}

ReferenceSource loadMaterialX(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MaterialX source material: " + path.string());
    pugi::xml_document document;
    const auto parsed = document.load(stream, pugi::parse_default, pugi::encoding_utf8);
    if (!parsed) throw std::runtime_error("invalid MaterialX XML: " + std::string(parsed.description()));
    const auto root = document.child("materialx");
    if (!root || std::string(root.attribute("version").value()) != "1.38")
        throw std::runtime_error("Falcor MaterialX subset requires a 1.38 source document");
    const auto material = root.child("surfacematerial");
    if (!material) throw std::runtime_error("MaterialX document has no surfacematerial");
    const auto shaderInput = findInput(material, "surfaceshader");
    if (!shaderInput || !shaderInput.attribute("nodename"))
        throw std::runtime_error("MaterialX surfacematerial has no surfaceshader connection");
    const auto surface = requireNamedChild(root, "standard_surface", shaderInput.attribute("nodename").value());

    ReferenceSource source;
    source.family = ReferenceFamily::MaterialX;
    source.materialXInputs = defaultMaterialXInputs();
    auto& inputs = source.materialXInputs;
    inputs[kMxBase] = inputFloat(surface, "base", inputs[kMxBase]);
    inputs[kMxDiffuseRoughness] = inputFloat(surface, "diffuse_roughness", inputs[kMxDiffuseRoughness]);
    inputs[kMxSpecular] = inputFloat(surface, "specular", inputs[kMxSpecular]);
    const auto specularColor = inputFloat3(surface, "specular_color", {1.f, 1.f, 1.f});
    std::copy(specularColor.begin(), specularColor.end(), inputs.begin() + kMxSpecularColor);
    inputs[kMxSpecularIor] = inputFloat(surface, "specular_IOR", inputs[kMxSpecularIor]);
    inputs[kMxSpecularAnisotropy] = inputFloat(surface, "specular_anisotropy", 0.f);
    inputs[kMxSpecularRotation] = inputFloat(surface, "specular_rotation", 0.f);
    inputs[kMxEmission] = inputFloat(surface, "emission", 0.f);
    const auto emissionColor = inputFloat3(surface, "emission_color", {1.f, 1.f, 1.f});
    std::copy(emissionColor.begin(), emissionColor.end(), inputs.begin() + kMxEmissionColor);

    const auto baseColorInput = findInput(surface, "base_color");
    if (!baseColorInput) throw std::runtime_error("MaterialX standard_surface has no base_color input");
    if (baseColorInput.attribute("value"))
    {
        const auto value = parseFloat3(baseColorInput.attribute("value").value(), "base_color");
        std::copy(value.begin(), value.end(), inputs.begin() + kMxBaseColor);
    }
    else
    {
        const auto image = resolveGraphOutput(root, baseColorInput, "image");
        source.materialXBaseColorTexture = imagePath(path, image, "color3", "srgb_texture");
        inputs[kMxHasBaseColorTexture] = 1.f;
    }

    const auto roughnessInput = findInput(surface, "specular_roughness");
    if (!roughnessInput) throw std::runtime_error("MaterialX standard_surface has no specular_roughness input");
    if (roughnessInput.attribute("value"))
        inputs[kMxSpecularRoughness] = parseFloat(roughnessInput.attribute("value").value(), "specular_roughness");
    else
    {
        const auto image = resolveGraphOutput(root, roughnessInput, "image");
        source.materialXRoughnessTexture = imagePath(path, image, "float");
        inputs[kMxHasRoughnessTexture] = 1.f;
    }

    const auto metalnessInput = findInput(surface, "metalness");
    if (!metalnessInput) throw std::runtime_error("MaterialX standard_surface has no metalness input");
    if (metalnessInput.attribute("value"))
        inputs[kMxMetalness] = parseFloat(metalnessInput.attribute("value").value(), "metalness");
    else
    {
        const auto image = resolveGraphOutput(root, metalnessInput, "image");
        source.materialXMetalnessTexture = imagePath(path, image, "float");
        inputs[kMxHasMetalnessTexture] = 1.f;
    }

    const auto normalInput = findInput(surface, "normal");
    if (normalInput && (normalInput.attribute("nodegraph") || normalInput.attribute("output")))
    {
        const auto normalMap = resolveGraphOutput(root, normalInput, "normalmap");
        const auto normalMapInput = findInput(normalMap, "in");
        if (!normalMapInput || !normalMapInput.attribute("nodename"))
            throw std::runtime_error("MaterialX normalmap has no image connection");
        const auto graph = requireNamedChild(root, "nodegraph", normalInput.attribute("nodegraph").value());
        const auto image = requireNamedChild(graph, "image", normalMapInput.attribute("nodename").value());
        source.materialXNormalTexture = imagePath(path, image, "vector3");
        inputs[kMxNormalScale] = inputFloat(normalMap, "scale", 1.f);
        inputs[kMxHasNormalTexture] = 1.f;
    }

    for (const char* name : {
             "transmission", "transmission_scatter_anisotropy", "transmission_dispersion",
             "transmission_extra_roughness", "subsurface", "subsurface_anisotropy", "sheen",
             "coat", "thin_film_thickness"})
        requireZeroInput(surface, name);
    const auto thinWalled = findInput(surface, "thin_walled");
    if (thinWalled && std::string(thinWalled.attribute("value").value()) != "false")
        throw std::runtime_error("MaterialX Falcor surface-response v1 requires thin_walled=false");

    const auto displacementInput = findInput(material, "displacementshader");
    if (displacementInput && displacementInput.attribute("nodename"))
    {
        const auto displacement = requireNamedChild(root, "displacement", displacementInput.attribute("nodename").value());
        const auto value = findInput(displacement, "displacement");
        if (!value) throw std::runtime_error("MaterialX displacement shader has no displacement input");
        const auto image = resolveGraphOutput(root, value, "image");
        source.materialXDisplacementTexture = imagePath(path, image, "float");
    }

    if (!(inputs[kMxSpecularIor] > 0.f)) throw std::runtime_error("MaterialX specular_IOR must be positive");
    for (float value : inputs)
        if (!std::isfinite(value)) throw std::runtime_error("MaterialX source material contains non-finite input");

    source.sourcePath = std::filesystem::absolute(path);
    source.displayName = material.attribute("name").value();
    std::string identity = sha256FileHex(path);
    for (const auto& texture : {
             source.materialXBaseColorTexture,
             source.materialXRoughnessTexture,
             source.materialXMetalnessTexture,
             source.materialXNormalTexture,
             source.materialXDisplacementTexture})
        if (!texture.empty()) identity += "\n" + texture.filename().string() + ":" + sha256FileHex(texture);
    source.sourceSha256 = sha256Hex(identity.data(), identity.size());
    return source;
}

struct OpenPbrParameterLayout
{
    uint32_t offset;
    uint32_t width;
};

const std::unordered_map<std::string, OpenPbrParameterLayout> kOpenPbrParameterLayout = {
    {"base_weight", {0, 1}}, {"base_color", {1, 3}}, {"base_diffuse_roughness", {4, 1}}, {"base_metalness", {5, 1}},
    {"subsurface_weight", {6, 1}}, {"subsurface_color", {7, 3}}, {"subsurface_radius", {10, 1}},
    {"subsurface_radius_scale", {11, 3}}, {"subsurface_scatter_anisotropy", {14, 1}},
    {"specular_weight", {15, 1}}, {"specular_color", {16, 3}}, {"specular_roughness", {19, 1}},
    {"specular_roughness_anisotropy", {20, 1}}, {"specular_ior", {21, 1}}, {"specular_anisotropy_rotation_cos_sin", {22, 2}},
    {"coat_weight", {24, 1}}, {"coat_color", {25, 3}}, {"coat_roughness", {28, 1}},
    {"coat_roughness_anisotropy", {29, 1}}, {"coat_ior", {30, 1}}, {"coat_darkening", {31, 1}},
    {"coat_anisotropy_rotation_cos_sin", {32, 2}}, {"fuzz_weight", {34, 1}}, {"fuzz_color", {35, 3}},
    {"fuzz_roughness", {38, 1}}, {"transmission_weight", {39, 1}}, {"transmission_color", {40, 3}},
    {"transmission_depth", {43, 1}}, {"transmission_scatter", {44, 3}}, {"transmission_scatter_anisotropy", {47, 1}},
    {"transmission_dispersion_scale", {48, 1}}, {"transmission_dispersion_abbe_number", {49, 1}},
    {"thin_film_weight", {50, 1}}, {"thin_film_thickness", {51, 1}}, {"thin_film_ior", {52, 1}},
    {"emission_luminance", {53, 1}}, {"emission_color", {54, 3}}, {"geometry_opacity", {57, 1}},
    {"geometry_thin_walled", {58, 1}},
};

std::array<float, 77> defaultOpenPbrInputs()
{
    return {
        1.f, .8f, .8f, .8f, 0.f, 0.f,
        0.f, .8f, .8f, .8f, 1.f, 1.f, .5f, .25f, 0.f,
        1.f, 1.f, 1.f, 1.f, .3f, 0.f, 1.5f, 1.f, 0.f,
        0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 1.6f, 1.f, 1.f, 0.f,
        0.f, 1.f, 1.f, 1.f, .5f,
        0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 20.f,
        0.f, .5f, 1.4f,
        0.f, 1.f, 1.f, 1.f, 1.f, 0.f,
        1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f,
        1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f,
    };
}

ReferenceSource loadOpenPbr(const std::filesystem::path& path, const json& document)
{
    if (document.value("schema_version", 0u) != 1u)
        throw std::runtime_error("unsupported OpenPBR source material schema version");
    ReferenceSource source;
    source.family = ReferenceFamily::OpenPbr;
    source.openPbrInputs = defaultOpenPbrInputs();
    const auto& parameters = document.at("parameters");
    for (const auto& [name, layout] : kOpenPbrParameterLayout)
    {
        if (!parameters.contains(name)) continue;
        const auto& binding = parameters.at(name);
        if (binding.value("source", "") != "constant")
            throw std::runtime_error("Falcor OpenPBR v1 runtime requires resolved constant binding: " + name);
        const auto& value = binding.at("value");
        if (layout.width == 1)
        {
            source.openPbrInputs[layout.offset] = value.is_boolean() ? float(value.get<bool>()) : value.get<float>();
        }
        else
        {
            if (!value.is_array() || value.size() != layout.width)
                throw std::runtime_error("OpenPBR parameter has wrong vector width: " + name);
            for (uint32_t component = 0; component < layout.width; ++component)
                source.openPbrInputs[layout.offset + component] = value[component].get<float>();
        }
    }
    for (float value : source.openPbrInputs)
        if (!std::isfinite(value)) throw std::runtime_error("OpenPBR source material contains non-finite input");
    const std::string colorSpace = document.value("color_space", "");
    if (colorSpace == "acescg") source.openPbrColorSpace = 1;
    else if (colorSpace == "linear-srgb" || colorSpace == "lin_rec709") source.openPbrColorSpace = 0;
    else throw std::runtime_error("Falcor OpenPBR runtime requires acescg or linear-srgb: " + colorSpace);
    source.sourcePath = std::filesystem::absolute(path);
    source.displayName = document.value("material_id", path.stem().string());
    source.sourceSha256 = sha256FileHex(path);
    return source;
}

ReferenceSource loadMerl(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MERL table: " + path.string());
    int32_t dimensions[3]{};
    stream.read(reinterpret_cast<char*>(dimensions), sizeof(dimensions));
    if (!stream || dimensions[0] != kMerlThetaHalf || dimensions[1] != kMerlThetaDifference
        || dimensions[2] != kMerlPhiDifference)
        throw std::runtime_error("MERL table has unsupported dimensions: " + path.string());
    std::vector<double> planar(3 * kMerlSampleCount);
    stream.read(reinterpret_cast<char*>(planar.data()), static_cast<std::streamsize>(planar.size() * sizeof(double)));
    if (!stream) throw std::runtime_error("MERL table is truncated: " + path.string());
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("MERL table contains trailing data: " + path.string());

    ReferenceSource source;
    source.family = ReferenceFamily::Merl;
    source.merlBrdf.resize(kMerlSampleCount);
    for (size_t index = 0; index < kMerlSampleCount; ++index)
    {
        auto& value = source.merlBrdf[index];
        for (size_t channel = 0; channel < 3; ++channel)
        {
            const double scaled = planar[channel * kMerlSampleCount + index] * kMerlScale[channel];
            if (!std::isfinite(scaled) || std::abs(scaled) > std::numeric_limits<float>::max())
                throw std::runtime_error("MERL table contains a non-finite sample: " + path.string());
            // Negative measurements are preserved as source data. The display transform may clamp them,
            // but the linear reference capture and parity path must retain the original table semantics.
            value[channel] = static_cast<float>(scaled);
        }
    }
    source.sourcePath = std::filesystem::absolute(path);
    source.displayName = path.stem().string();
    source.sourceSha256 = sha256FileHex(path);
    return source;
}
} // namespace

const char* ReferenceSource::familyId() const
{
    switch (family)
    {
    case ReferenceFamily::LayerStack: return "ncls.layer-stack@1";
    case ReferenceFamily::Merl: return "merl.measured-brdf@1";
    case ReferenceFamily::OpenPbr: return "openpbr.surface@1.1.1";
    case ReferenceFamily::MaterialX: return "materialx.textured-surface@1";
    }
    return "unknown";
}

ReferenceSource makeDefaultReferenceSource()
{
    ReferenceSource source;
    source.sourceSha256 = layerStackHash(source.layerStack);
    return source;
}

ReferenceSource loadReferenceSource(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    for (auto& character : extension) character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
    if (extension == ".binary") return loadMerl(path);
    if (extension == ".mtlx") return loadMaterialX(path);
    if (extension == ".json")
    {
        std::ifstream stream(path);
        if (!stream) throw std::runtime_error("cannot open source material: " + path.string());
        const json document = json::parse(stream);
        if (document.value("schema_name", "") == "ncls.openpbr-material") return loadOpenPbr(path, document);
        ReferenceSource source;
        source.family = ReferenceFamily::LayerStack;
        source.layerStack = loadMaterialProgram(path, &source.displayName);
        source.sourcePath = std::filesystem::absolute(path);
        source.sourceSha256 = sha256FileHex(path);
        return source;
    }
    throw std::runtime_error("unsupported source material extension: " + extension);
}
} // namespace ncls
