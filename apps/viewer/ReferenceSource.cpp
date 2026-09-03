#include "ReferenceSource.h"

#include "Hash.h"
#include "ScatteringPackage.h"

#include <nlohmann/json.hpp>
#include <pugixml.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <regex>
#include <set>
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
    const auto opacity = findInput(surface, "opacity");
    if (opacity)
    {
        if (!opacity.attribute("value") || opacity.attribute("nodegraph") || opacity.attribute("nodename"))
            throw std::runtime_error("MaterialX Falcor subset requires constant opacity");
        if (std::string(opacity.attribute("type").value()) == "color3")
        {
            const auto value = parseFloat3(opacity.attribute("value").value(), "opacity");
            if (std::max(value[0], std::max(value[1], value[2]))
                - std::min(value[0], std::min(value[1], value[2])) > 1e-8f)
                throw std::runtime_error("MaterialX Falcor subset requires achromatic opacity");
            inputs[kMxOpacity] = value[0];
        }
        else inputs[kMxOpacity] = parseFloat(opacity.attribute("value").value(), "opacity");
    }

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

struct MaterialXParameterLayout
{
    const char* name;
    uint32_t offset;
    uint32_t width;
    int32_t textureFlagOffset;
};

constexpr std::array<MaterialXParameterLayout, 14> kMaterialXParameterLayout = {{
    {"base", kMxBase, 1, -1},
    {"base_color", kMxBaseColor, 3, int32_t(kMxHasBaseColorTexture)},
    {"diffuse_roughness", kMxDiffuseRoughness, 1, -1},
    {"metalness", kMxMetalness, 1, int32_t(kMxHasMetalnessTexture)},
    {"specular", kMxSpecular, 1, -1},
    {"specular_color", kMxSpecularColor, 3, -1},
    {"specular_roughness", kMxSpecularRoughness, 1, int32_t(kMxHasRoughnessTexture)},
    {"specular_IOR", kMxSpecularIor, 1, -1},
    {"specular_anisotropy", kMxSpecularAnisotropy, 1, -1},
    {"specular_rotation", kMxSpecularRotation, 1, -1},
    {"normal_scale", kMxNormalScale, 1, int32_t(kMxHasNormalTexture)},
    {"emission", kMxEmission, 1, -1},
    {"emission_color", kMxEmissionColor, 3, -1},
    {"opacity", kMxOpacity, 1, -1},
}};

std::array<float, 3> constantFloat3Binding(
    const json& parameters,
    const char* name,
    const std::array<float, 3>& fallback)
{
    if (!parameters.contains(name)) return fallback;
    const auto& binding = parameters.at(name);
    if (binding.value("source", "") == "geometry") return fallback;
    if (binding.value("source", "") != "constant")
        throw std::runtime_error("Falcor OpenPBR v1 runtime requires resolved constant binding: " + std::string(name));
    const auto& value = binding.at("value");
    if (!value.is_array() || value.size() != 3u)
        throw std::runtime_error("OpenPBR parameter has wrong vector width: " + std::string(name));
    std::array<float, 3> result{};
    for (uint32_t component = 0u; component < 3u; ++component)
    {
        result[component] = value[component].get<float>();
        if (!std::isfinite(result[component]))
            throw std::runtime_error("OpenPBR parameter is non-finite: " + std::string(name));
    }
    return result;
}

void writeOpenPbrBasis(
    std::array<float, 77>& values,
    uint32_t offset,
    const std::array<float, 3>& normalValue,
    const std::array<float, 3>& tangentValue)
{
    auto dot3 = [](const std::array<float, 3>& left, const std::array<float, 3>& right) {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    };
    auto normalize = [&](const char* label, std::array<float, 3> value) {
        const float lengthSquared = dot3(value, value);
        if (!(lengthSquared > 1e-12f) || !std::isfinite(lengthSquared))
            throw std::runtime_error(std::string("OpenPBR ") + label + " must be finite and nonzero");
        const float inverseLength = 1.f / std::sqrt(lengthSquared);
        for (float& component : value) component *= inverseLength;
        return value;
    };
    const auto normal = normalize("geometry normal", normalValue);
    std::array<float, 3> tangent = tangentValue;
    const float tangentNormal = dot3(tangent, normal);
    for (uint32_t component = 0u; component < 3u; ++component)
        tangent[component] -= tangentNormal * normal[component];
    tangent = normalize("geometry tangent", tangent);
    const auto bitangent = normalize("geometry bitangent", {
        normal[1] * tangent[2] - normal[2] * tangent[1],
        normal[2] * tangent[0] - normal[0] * tangent[2],
        normal[0] * tangent[1] - normal[1] * tangent[0],
    });
    std::copy(tangent.begin(), tangent.end(), values.begin() + offset);
    std::copy(bitangent.begin(), bitangent.end(), values.begin() + offset + 3u);
    std::copy(normal.begin(), normal.end(), values.begin() + offset + 6u);
}

json openPbrParameterValues(const ReferenceSource& source)
{
    json result = json::object();
    for (const auto& [name, layout] : kOpenPbrParameterLayout)
    {
        if (name == "geometry_thin_walled") result[name] = source.openPbrInputs[layout.offset] != 0.f;
        else if (layout.width == 1u) result[name] = source.openPbrInputs[layout.offset];
        else
        {
            json value = json::array();
            for (uint32_t component = 0u; component < layout.width; ++component)
                value.push_back(source.openPbrInputs[layout.offset + component]);
            result[name] = std::move(value);
        }
    }
    result["geometry_tangent"] = {
        source.openPbrInputs[59], source.openPbrInputs[60], source.openPbrInputs[61]};
    result["geometry_normal"] = {
        source.openPbrInputs[65], source.openPbrInputs[66], source.openPbrInputs[67]};
    result["geometry_coat_tangent"] = {
        source.openPbrInputs[68], source.openPbrInputs[69], source.openPbrInputs[70]};
    result["geometry_coat_normal"] = {
        source.openPbrInputs[74], source.openPbrInputs[75], source.openPbrInputs[76]};
    return result;
}

void applyOpenPbrParameterValues(ReferenceSource& source, const json& parameters)
{
    for (const auto& [name, layout] : kOpenPbrParameterLayout)
    {
        if (!parameters.contains(name)) continue;
        const auto& value = parameters.at(name);
        if (layout.width == 1u)
            source.openPbrInputs[layout.offset] = value.is_boolean() ? float(value.get<bool>()) : value.get<float>();
        else
        {
            if (!value.is_array() || value.size() != layout.width)
                throw std::runtime_error("OpenPBR scene parameter has wrong vector width: " + name);
            for (uint32_t component = 0u; component < layout.width; ++component)
                source.openPbrInputs[layout.offset + component] = value[component].get<float>();
        }
    }
    const auto vector3 = [&](const char* name, const std::array<float, 3>& fallback) {
        if (!parameters.contains(name)) return fallback;
        const auto& value = parameters.at(name);
        if (!value.is_array() || value.size() != 3u)
            throw std::runtime_error("OpenPBR scene parameter has wrong vector width: " + std::string(name));
        return std::array<float, 3>{value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
    };
    writeOpenPbrBasis(
        source.openPbrInputs, 59u,
        vector3("geometry_normal", {0.f, 0.f, 1.f}),
        vector3("geometry_tangent", {1.f, 0.f, 0.f}));
    writeOpenPbrBasis(
        source.openPbrInputs, 68u,
        vector3("geometry_coat_normal", {0.f, 0.f, 1.f}),
        vector3("geometry_coat_tangent", {1.f, 0.f, 0.f}));
    for (float value : source.openPbrInputs)
        if (!std::isfinite(value)) throw std::runtime_error("OpenPBR scene state contains non-finite input");
}

json materialXParameterValues(const ReferenceSource& source)
{
    json result = json::object();
    for (const auto& layout : kMaterialXParameterLayout)
    {
        if (layout.textureFlagOffset >= 0
            && source.materialXInputs[static_cast<uint32_t>(layout.textureFlagOffset)] != 0.f
            && std::string(layout.name) != "normal_scale")
            continue;
        if (layout.width == 1u) result[layout.name] = source.materialXInputs[layout.offset];
        else result[layout.name] = {
            source.materialXInputs[layout.offset],
            source.materialXInputs[layout.offset + 1u],
            source.materialXInputs[layout.offset + 2u]};
    }
    return result;
}

void applyMaterialXParameterValues(ReferenceSource& source, const json& parameters)
{
    for (const auto& layout : kMaterialXParameterLayout)
    {
        if (!parameters.contains(layout.name)) continue;
        if (layout.textureFlagOffset >= 0
            && source.materialXInputs[static_cast<uint32_t>(layout.textureFlagOffset)] != 0.f
            && std::string(layout.name) != "normal_scale")
            throw std::runtime_error("MaterialX scene state cannot override texture-driven input: " + std::string(layout.name));
        const auto& value = parameters.at(layout.name);
        if (layout.width == 1u) source.materialXInputs[layout.offset] = value.get<float>();
        else
        {
            if (!value.is_array() || value.size() != layout.width)
                throw std::runtime_error("MaterialX scene parameter has wrong vector width: " + std::string(layout.name));
            for (uint32_t component = 0u; component < layout.width; ++component)
                source.materialXInputs[layout.offset + component] = value[component].get<float>();
        }
    }
    for (float value : source.materialXInputs)
        if (!std::isfinite(value)) throw std::runtime_error("MaterialX scene state contains non-finite input");
    if (!(source.materialXInputs[kMxSpecularIor] > 0.f))
        throw std::runtime_error("MaterialX specular_IOR must be positive");
}

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
    writeOpenPbrBasis(
        source.openPbrInputs,
        59u,
        constantFloat3Binding(parameters, "geometry_normal", {0.f, 0.f, 1.f}),
        constantFloat3Binding(parameters, "geometry_tangent", {1.f, 0.f, 0.f}));
    writeOpenPbrBasis(
        source.openPbrInputs,
        68u,
        constantFloat3Binding(parameters, "geometry_coat_normal", {0.f, 0.f, 1.f}),
        constantFloat3Binding(parameters, "geometry_coat_tangent", {1.f, 0.f, 0.f}));
    for (float value : source.openPbrInputs)
        if (!std::isfinite(value)) throw std::runtime_error("OpenPBR source material contains non-finite input");
    const std::string colorSpace = document.value("color_space", "");
    if (colorSpace == "acescg") source.openPbrColorSpace = 1;
    else if (colorSpace == "linear-srgb" || colorSpace == "lin_rec709") source.openPbrColorSpace = 0;
    else throw std::runtime_error("Falcor OpenPBR runtime requires acescg or linear-srgb: " + colorSpace);
    const std::string sourceUri = document.at("source_document").get<std::string>();
    if (sourceUri.empty()) throw std::runtime_error("OpenPBR adapter has no native source_document");
    const std::filesystem::path authoredSource(sourceUri);
    const auto nativeSource = std::filesystem::absolute(
        authoredSource.is_absolute() ? authoredSource : path.parent_path() / authoredSource).lexically_normal();
    if (!std::filesystem::is_regular_file(nativeSource))
        throw std::runtime_error("OpenPBR native source document is missing: " + nativeSource.string());
    const std::string declaredHash = document.at("metadata").at("source_sha256").get<std::string>();
    const std::string actualHash = sha256FileHex(nativeSource);
    if (actualHash != declaredHash)
        throw std::runtime_error("OpenPBR native source asset SHA-256 mismatch: " + nativeSource.string());
    source.sourcePath = nativeSource;
    source.displayName = document.value("material_id", path.stem().string());
    source.sourceSha256 = actualHash;
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
    case ReferenceFamily::MaterialX: return "materialx.document@1.39.4";
    case ReferenceFamily::Mdl: return "mdl.program@1";
    }
    return "unknown";
}

ReferenceSource makeDefaultReferenceSource()
{
    return makeDefaultReferenceSource(ReferenceFamily::LayerStack);
}

ReferenceSource makeDefaultReferenceSource(ReferenceFamily family)
{
    ReferenceSource source;
    source.family = family;
    if (family == ReferenceFamily::LayerStack)
    {
        source.displayName = "Default layered material";
        source.sourceSha256 = layerStackHash(source.layerStack);
    }
    else if (family == ReferenceFamily::OpenPbr)
    {
        source.openPbrInputs = defaultOpenPbrInputs();
        source.openPbrColorSpace = 0u;
        source.displayName = "Default OpenPBR material";
    }
    else
    {
        throw std::runtime_error("resource-backed reference family requires a source file");
    }
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
        const std::string schemaName = document.value("schema_name", "");
        if (schemaName == "ncls.mdl-viewer-catalog"
            || schemaName == "ncls.viewer-material-catalog")
        {
            ReferenceSource source;
            source.family = ReferenceFamily::Mdl;
            source.mdlCatalog = std::make_shared<const MdlViewerCatalog>(loadMdlViewerCatalog(path));
            const auto found = std::find_if(
                source.mdlCatalog->entries.begin(), source.mdlCatalog->entries.end(),
                [&](const MdlCatalogEntry& entry) {
                    return (source.mdlCatalog->linked() ? entry.exportId : entry.assetId)
                        == source.mdlCatalog->defaultAssetId;
                });
            if (found == source.mdlCatalog->entries.end())
                throw std::runtime_error("MDL viewer catalog default asset is missing");
            source.mdlCatalogIndex = static_cast<uint32_t>(found - source.mdlCatalog->entries.begin());
            return selectMdlCatalogEntry(source, source.mdlCatalogIndex);
        }
        ReferenceSource source;
        source.family = ReferenceFamily::LayerStack;
        source.layerStack = loadMaterialProgram(path, &source.displayName);
        source.sourcePath = std::filesystem::absolute(path);
        source.sourceSha256 = sha256FileHex(path);
        return source;
    }
    throw std::runtime_error("unsupported source material extension: " + extension);
}

ReferenceSource selectMdlCatalogEntry(const ReferenceSource& source, uint32_t index)
{
    if (source.family != ReferenceFamily::Mdl || !source.mdlCatalog
        || index >= source.mdlCatalog->entries.size())
        throw std::runtime_error("MDL catalog selection is out of range");
    ReferenceSource result = source;
    const auto& entry = result.mdlCatalog->entries[index];
    result.mdlArtifact = loadMdlCompiledArtifact(entry);
    result.mdlAuthoredArgumentBlock = result.mdlArtifact->argumentBlock;
    result.mdlParameterView = entry.parameterView;
    result.mdlCatalogIndex = index;
    result.sourcePath = result.mdlCatalog->sourcePath;
    result.sourceSha256 = entry.sourceSnapshotId;
    result.displayName = entry.displayName;
    result.mdlEdited = false;
    result.mdlEditStateSha256.clear();
    if (entry.linked())
        result = applyMdlCatalogParameterView(result, result.mdlParameterView);
    return result;
}

namespace
{
json mdlParameterValues(const json& parameterView)
{
    json values = json::object();
    std::function<void(const json&)> collect = [&](const json& node) {
        if (node.value("editable", false))
            values[node.at("path").get<std::string>()] = node.at("value");
        for (const auto& child : node.at("children")) collect(child);
    };
    collect(parameterView.at("root"));
    return values;
}

void setMdlParameterValues(json& parameterView, const json& values)
{
    if (!values.is_object()) throw std::runtime_error("MDL viewer parameter values must be an object");
    std::set<std::string> remaining;
    for (auto item = values.begin(); item != values.end(); ++item) remaining.insert(item.key());
    size_t expected = 0u;
    std::function<void(json&)> apply = [&](json& node) {
        if (node.value("editable", false))
        {
            ++expected;
            const std::string path = node.at("path").get<std::string>();
            if (values.contains(path))
            {
                node["value"] = values.at(path);
                remaining.erase(path);
            }
        }
        for (auto& child : node["children"]) apply(child);
    };
    apply(parameterView["root"]);
    if (!remaining.empty() || values.size() != expected)
        throw std::runtime_error("MDL viewer state must contain exactly every editable parameter path");
    validateViewerTypedParameterView(parameterView);
}
} // namespace

ReferenceSource applyMdlCatalogParameterView(
    const ReferenceSource& source,
    const json& parameterView)
{
    if (source.family != ReferenceFamily::Mdl || !source.mdlCatalog || !source.mdlArtifact
        || source.mdlCatalogIndex >= source.mdlCatalog->entries.size())
        throw std::runtime_error("MDL typed edit requires a selected catalog entry");
    const auto& entry = source.mdlCatalog->entries[source.mdlCatalogIndex];
    if (!entry.linked()) throw std::runtime_error("legacy MDL catalog entries are not typed-editable");
    validateViewerTypedParameterView(parameterView);
    if (parameterView.at("snapshot_id") != entry.sourceSnapshotId)
        throw std::runtime_error("MDL typed edit parameter view has a stale base snapshot");

    auto artifact = std::make_shared<MdlCompiledArtifact>(*source.mdlArtifact);
    artifact->argumentBlock = source.mdlAuthoredArgumentBlock;
    if (artifact->argumentBlock.empty())
        throw std::runtime_error("MDL typed edit requires an argument block");
    auto writeBytes = [&](uint32_t offset, const void* data, size_t size) {
        if (size == 0u || size_t(offset) + size > artifact->argumentBlock.size())
            throw std::runtime_error("MDL typed edit write is outside the argument block");
        std::memcpy(artifact->argumentBlock.data() + offset, data, size);
    };
    std::function<void(const json&)> patch = [&](const json& node) {
        if (node.value("editable", false))
        {
            const auto& write = node.at("metadata").at("reference_write");
            const uint32_t offset = write.at("offset").get<uint32_t>();
            const uint32_t size = write.at("size").get<uint32_t>();
            const std::string mdlType = write.at("mdl_type").get<std::string>();
            const auto& value = node.at("value");
            const auto requireInHardRange = [&](double item) {
                if (node.contains("minimum") && item < node.at("minimum").get<double>())
                    throw std::runtime_error("MDL typed edit is below the hard minimum");
                if (node.contains("maximum") && item > node.at("maximum").get<double>())
                    throw std::runtime_error("MDL typed edit is above the hard maximum");
            };
            if (mdlType == "bool")
            {
                if (size != 1u || !value.is_boolean()) throw std::runtime_error("invalid MDL bool write");
                const uint8_t converted = value.get<bool>() ? 1u : 0u;
                writeBytes(offset, &converted, sizeof(converted));
            }
            else if (mdlType == "int")
            {
                if (size != 4u || (!value.is_number_integer() && !value.is_number_unsigned()))
                    throw std::runtime_error("invalid MDL int write");
                const int64_t wide = value.get<int64_t>();
                if (wide < std::numeric_limits<int32_t>::min()
                    || wide > std::numeric_limits<int32_t>::max())
                    throw std::runtime_error("MDL int edit is outside int32 range");
                requireInHardRange(static_cast<double>(wide));
                const int32_t converted = static_cast<int32_t>(wide);
                writeBytes(offset, &converted, sizeof(converted));
            }
            else if (mdlType == "enum")
            {
                if (size != 4u || !value.is_string() || !write.contains("choices")
                    || !write.at("choices").contains(value.get<std::string>()))
                    throw std::runtime_error("invalid MDL enum write");
                const int32_t converted = write.at("choices").at(value.get<std::string>()).get<int32_t>();
                writeBytes(offset, &converted, sizeof(converted));
            }
            else
            {
                const size_t components = mdlType == "float" ? 1u
                    : mdlType == "float2" ? 2u : mdlType == "color" ? 3u : 0u;
                if (components == 0u || size != components * sizeof(float))
                    throw std::runtime_error("invalid MDL floating-point write descriptor");
                std::array<float, 3> converted{};
                if (components == 1u)
                {
                    if (!value.is_number()) throw std::runtime_error("invalid MDL float write");
                    converted[0] = value.get<float>();
                }
                else
                {
                    if (!value.is_array() || value.size() != components)
                        throw std::runtime_error("invalid MDL vector write");
                    for (size_t component = 0u; component < components; ++component)
                        converted[component] = value.at(component).get<float>();
                }
                if (!std::all_of(converted.begin(), converted.begin() + components,
                        [](float item) { return std::isfinite(item); }))
                    throw std::runtime_error("MDL typed edit contains a non-finite value");
                for (size_t component = 0u; component < components; ++component)
                    requireInHardRange(converted[component]);
                writeBytes(offset, converted.data(), size);
            }
        }
        for (const auto& child : node.at("children")) patch(child);
    };
    patch(parameterView.at("root"));

    const json values = mdlParameterValues(parameterView);
    const json identity = {
        {"schema", "ncls.viewer-material-state@1"},
        {"catalog_id", source.mdlCatalog->catalogId},
        {"registry_identity", source.mdlCatalog->registryIdentity},
        {"export_id", entry.exportId},
        {"base_source_snapshot_id", entry.sourceSnapshotId},
        {"values", values},
    };
    const std::string canonical = identity.dump();
    ReferenceSource result = source;
    result.mdlArtifact = std::move(artifact);
    result.mdlParameterView = parameterView;
    result.mdlEditStateSha256 = sha256Hex(canonical.data(), canonical.size());
    result.mdlEdited = values != mdlParameterValues(entry.parameterView);
    return result;
}

namespace
{
std::string portableSourceUri(
    const std::filesystem::path& path,
    const std::filesystem::path& manifestDirectory)
{
    if (path.empty()) return {};
    const auto absolutePath = std::filesystem::absolute(path).lexically_normal();
    std::error_code error;
    const auto relative = std::filesystem::relative(
        absolutePath,
        std::filesystem::absolute(manifestDirectory).lexically_normal(),
        error);
    return error ? absolutePath.generic_string() : relative.generic_string();
}

std::filesystem::path resolveSourceUri(
    const std::string& uri,
    const std::filesystem::path& manifestDirectory)
{
    if (uri.empty()) return {};
    const std::filesystem::path path(uri);
    return std::filesystem::absolute(
        path.is_absolute() ? path : manifestDirectory / path).lexically_normal();
}

void requireSha256(const std::string& value, const char* label)
{
    static const std::regex pattern("^[0-9a-f]{64}$");
    if (!std::regex_match(value, pattern))
        throw std::runtime_error(std::string(label) + " must be a lowercase SHA-256 digest");
}

json referenceSourceStatePayload(const ReferenceSource& source)
{
    json payload = {{"family_id", source.familyId()}};
    switch (source.family)
    {
    case ReferenceFamily::LayerStack:
        payload["material_program"] = makeMaterialProgramDocument(source.layerStack, source.displayName);
        break;
    case ReferenceFamily::OpenPbr:
        payload["color_space"] = source.openPbrColorSpace == 0u ? "linear-srgb" : "acescg";
        payload["parameters"] = openPbrParameterValues(source);
        break;
    case ReferenceFamily::MaterialX:
        payload["source_sha256"] = source.sourceSha256;
        payload["parameters"] = materialXParameterValues(source);
        break;
    case ReferenceFamily::Merl:
        payload["source_sha256"] = source.sourceSha256;
        break;
    case ReferenceFamily::Mdl:
        if (!source.mdlCatalog || !source.mdlArtifact
            || source.mdlCatalogIndex >= source.mdlCatalog->entries.size())
            throw std::runtime_error("MDL source has no validated compiled artifact");
        payload["asset_id"] = source.mdlCatalog->entries[source.mdlCatalogIndex].assetId;
        payload["source_snapshot_id"] = source.sourceSha256;
        payload["compiled_artifact_sha256"] = source.mdlArtifact->artifactSha256;
        payload["texture_filtering"] = "explicit-lod0";
        if (source.mdlCatalog->linked())
        {
            payload["viewer_material_state"] = {
                {"catalog_id", source.mdlCatalog->catalogId},
                {"registry_identity", source.mdlCatalog->registryIdentity},
                {"export_id", source.mdlCatalog->entries[source.mdlCatalogIndex].exportId},
                {"base_source_snapshot_id", source.sourceSha256},
                {"values", mdlParameterValues(source.mdlParameterView)},
                {"state_sha256", source.mdlEditStateSha256},
                {"status", source.mdlEdited ? "edited-preview" : "authored"},
            };
        }
        break;
    }
    return payload;
}
} // namespace

std::string referenceSourceStateHash(const ReferenceSource& source)
{
    const std::string canonical = referenceSourceStatePayload(source).dump();
    return sha256Hex(canonical.data(), canonical.size());
}

json serializeReferenceSourceState(
    const ReferenceSource& source,
    const std::filesystem::path& manifestDirectory)
{
    json result = referenceSourceStatePayload(source);
    result["display_name"] = source.displayName;
    result["source_uri"] = portableSourceUri(source.sourcePath, manifestDirectory);
    result["source_asset_sha256"] = source.sourceSha256;
    result["state_sha256"] = referenceSourceStateHash(source);
    return result;
}

ReferenceSource deserializeReferenceSourceState(
    const json& document,
    const std::filesystem::path& manifestDirectory)
{
    if (!document.is_object()) throw std::runtime_error("viewer scene source binding must be an object");
    const std::string familyId = document.at("family_id").get<std::string>();
    const auto sourcePath = resolveSourceUri(document.value("source_uri", std::string()), manifestDirectory);
    const std::string expectedSourceHash = document.at("source_asset_sha256").get<std::string>();
    const std::string expectedStateHash = document.at("state_sha256").get<std::string>();
    requireSha256(expectedSourceHash, "source_asset_sha256");
    requireSha256(expectedStateHash, "state_sha256");
    ReferenceSource source;
    if (familyId == "ncls.layer-stack@1")
    {
        source = makeDefaultReferenceSource(ReferenceFamily::LayerStack);
        source.layerStack = loadMaterialProgramDocument(
            document.at("material_program"), &source.displayName, "Embedded LayerStack material");
        source.sourcePath.clear();
        source.sourceSha256 = layerStackHash(source.layerStack);
        if (source.sourceSha256 != expectedSourceHash)
            throw std::runtime_error("viewer scene LayerStack source asset SHA-256 mismatch");
    }
    else if (familyId == "openpbr.surface@1.1.1")
    {
        if (sourcePath.empty() || !std::filesystem::is_regular_file(sourcePath))
            throw std::runtime_error("viewer scene OpenPBR native source is missing: " + sourcePath.string());
        if (sha256FileHex(sourcePath) != expectedSourceHash)
            throw std::runtime_error("viewer scene OpenPBR source asset SHA-256 mismatch: " + sourcePath.string());
        source = makeDefaultReferenceSource(ReferenceFamily::OpenPbr);
        source.sourcePath = sourcePath;
        source.sourceSha256 = expectedSourceHash;
        const std::string colorSpace = document.at("color_space").get<std::string>();
        if (colorSpace == "linear-srgb") source.openPbrColorSpace = 0u;
        else if (colorSpace == "acescg") source.openPbrColorSpace = 1u;
        else throw std::runtime_error("viewer scene OpenPBR color space is unsupported: " + colorSpace);
        applyOpenPbrParameterValues(source, document.at("parameters"));
    }
    else if (familyId == "materialx.document@1.39.4" || familyId == "merl.measured-brdf@1")
    {
        if (sourcePath.empty() || !std::filesystem::is_regular_file(sourcePath))
            throw std::runtime_error("viewer scene resource-backed source is missing: " + sourcePath.string());
        source = loadReferenceSource(sourcePath);
        const ReferenceFamily expectedFamily = familyId == "materialx.document@1.39.4"
            ? ReferenceFamily::MaterialX : ReferenceFamily::Merl;
        if (source.family != expectedFamily)
            throw std::runtime_error("viewer scene source URI resolved to another family: " + sourcePath.string());
        if (source.sourceSha256 != expectedSourceHash)
            throw std::runtime_error("viewer scene source asset SHA-256 mismatch: " + sourcePath.string());
        if (source.family == ReferenceFamily::MaterialX)
            applyMaterialXParameterValues(source, document.at("parameters"));
    }
    else if (familyId == "mdl.program@1")
    {
        if (sourcePath.empty() || !std::filesystem::is_regular_file(sourcePath))
            throw std::runtime_error("viewer scene MDL catalog is missing: " + sourcePath.string());
        source = loadReferenceSource(sourcePath);
        const std::string assetId = document.at("asset_id").get<std::string>();
        if (!source.mdlCatalog)
            throw std::runtime_error("viewer scene MDL source has no catalog");
        const auto found = std::find_if(
            source.mdlCatalog->entries.begin(), source.mdlCatalog->entries.end(),
            [&](const MdlCatalogEntry& entry) { return entry.assetId == assetId; });
        if (found == source.mdlCatalog->entries.end())
            throw std::runtime_error("viewer scene MDL asset is not present in the catalog: " + assetId);
        source = selectMdlCatalogEntry(
            source, static_cast<uint32_t>(found - source.mdlCatalog->entries.begin()));
        if (document.contains("viewer_material_state"))
        {
            if (!source.mdlCatalog->linked())
                throw std::runtime_error("viewer scene linked MDL state requires a ViewerMaterialCatalog");
            const auto& state = document.at("viewer_material_state");
            if (state.at("catalog_id") != source.mdlCatalog->catalogId
                || state.at("registry_identity") != source.mdlCatalog->registryIdentity
                || state.at("export_id") != source.mdlCatalog->entries[source.mdlCatalogIndex].exportId
                || state.at("base_source_snapshot_id") != source.sourceSha256)
                throw std::runtime_error("viewer scene linked MDL identity mismatch");
            auto view = source.mdlParameterView;
            setMdlParameterValues(view, state.at("values"));
            source = applyMdlCatalogParameterView(source, view);
            if (state.at("state_sha256") != source.mdlEditStateSha256
                || state.at("status") != (source.mdlEdited ? "edited-preview" : "authored"))
                throw std::runtime_error("viewer scene linked MDL edit state mismatch");
        }
        if (source.sourceSha256 != expectedSourceHash)
            throw std::runtime_error("viewer scene MDL source snapshot mismatch");
        if (!source.mdlArtifact
            || source.mdlArtifact->artifactSha256 != document.at("compiled_artifact_sha256").get<std::string>())
            throw std::runtime_error("viewer scene MDL compiled artifact mismatch");
        if (document.value("texture_filtering", "") != "explicit-lod0")
            throw std::runtime_error("viewer scene MDL filtering capability is unsupported");
    }
    else throw std::runtime_error("viewer scene contains unsupported source family: " + familyId);

    source.displayName = document.value("display_name", source.displayName);
    if (referenceSourceStateHash(source) != expectedStateHash)
        throw std::runtime_error("viewer scene source-material state SHA-256 mismatch: " + familyId);
    return source;
}

void saveOpenPbrReferenceSource(const std::filesystem::path& path, const ReferenceSource& source)
{
    if (source.family != ReferenceFamily::OpenPbr)
        throw std::runtime_error("cannot save a non-OpenPBR source as OpenPBR JSON");
    const json values = openPbrParameterValues(source);
    json bindings = json::object();
    std::vector<std::string> authored;
    for (auto iterator = values.begin(); iterator != values.end(); ++iterator)
    {
        bindings[iterator.key()] = {{"source", "constant"}, {"value", iterator.value()}};
        authored.push_back(iterator.key());
    }
    const json document = {
        {"schema_name", "ncls.openpbr-material"},
        {"schema_version", 1},
        {"material_id", source.displayName},
        {"color_space", source.openPbrColorSpace == 0u ? "linear-srgb" : "acescg"},
        {"source_document", portableSourceUri(source.sourcePath, path.parent_path())},
        {"authored_parameters", authored},
        {"parameters", bindings},
        {"metadata", {{"source_sha256", source.sourceSha256}}},
    };
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    std::error_code error;
    std::filesystem::remove(temporary, error);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot write OpenPBR source material: " + path.string());
        stream << document.dump(2) << '\n';
    }
    std::filesystem::remove(path, error);
    std::filesystem::rename(temporary, path);
}
} // namespace ncls
