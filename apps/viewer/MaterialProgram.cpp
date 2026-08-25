#include "MaterialProgram.h"

#include "Hash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace ncls
{
namespace
{
using json = nlohmann::json;

const json& nodeById(const std::unordered_map<std::string, const json*>& nodes, const json& connection)
{
    const std::string id = connection.at("node").get<std::string>();
    const auto found = nodes.find(id);
    if (found == nodes.end()) throw std::runtime_error("MaterialProgram connection references missing node: " + id);
    return *found->second;
}

std::string operationKey(const json& node)
{
    const auto& operation = node.at("operation");
    return operation.at("namespace").get<std::string>() + "." + operation.at("name").get<std::string>() + "@"
        + std::to_string(operation.at("version").get<uint32_t>());
}

const json& constant(const json& node, const char* name)
{
    const auto& source = node.at("parameters").at(name);
    if (source.at("source") != "constant")
        throw std::runtime_error("viewer v1 only accepts constant MaterialProgram parameters");
    return source.at("value");
}

float scalar(const json& node, const char* name)
{
    return constant(node, name).get<float>();
}

std::array<float, 3> color(const json& node, const char* name)
{
    const auto& value = constant(node, name);
    if (!value.is_array() || value.size() != 3) throw std::runtime_error(std::string(name) + " must be RGB");
    return {value[0].get<float>(), value[1].get<float>(), value[2].get<float>()};
}

LayerInterface parseInterface(const json& node)
{
    LayerInterface result{};
    const std::string operation = operationKey(node);
    if (operation == "ncls.interface.rough_dielectric@1")
    {
        result.kind = static_cast<uint32_t>(InterfaceKind::RoughDielectric);
        result.alphaX = scalar(node, "alpha_x");
        result.alphaY = scalar(node, "alpha_y");
        result.relativeIor = scalar(node, "relative_ior");
        result.tangentRotation = scalar(node, "tangent_rotation");
    }
    else if (operation == "ncls.interface.rough_conductor@1")
    {
        result.kind = static_cast<uint32_t>(InterfaceKind::RoughConductor);
        result.alphaX = scalar(node, "alpha_x");
        result.alphaY = scalar(node, "alpha_y");
        const auto eta = color(node, "eta");
        const auto k = color(node, "k");
        result.etaR = eta[0]; result.etaG = eta[1]; result.etaB = eta[2];
        result.kR = k[0]; result.kG = k[1]; result.kB = k[2];
        result.tangentRotation = scalar(node, "tangent_rotation");
    }
    else if (operation == "ncls.interface.diffuse@1")
    {
        result.kind = static_cast<uint32_t>(InterfaceKind::Diffuse);
        const auto value = color(node, "color");
        result.colorR = value[0]; result.colorG = value[1]; result.colorB = value[2];
    }
    else if (operation == "ncls.interface.sheen@1")
    {
        result.kind = static_cast<uint32_t>(InterfaceKind::Sheen);
        result.alphaX = result.alphaY = scalar(node, "roughness");
        const auto value = color(node, "color");
        result.colorR = value[0]; result.colorG = value[1]; result.colorB = value[2];
    }
    else
    {
        throw std::runtime_error("unsupported MaterialProgram interface operation: " + operation);
    }
    return result;
}

HomogeneousMedium parseMedium(const json& node)
{
    if (operationKey(node) != "ncls.medium.homogeneous@1")
        throw std::runtime_error("viewer v1 requires ncls.medium.homogeneous@1");
    const auto sigmaA = color(node, "sigma_a");
    const auto sigmaS = color(node, "sigma_s");
    HomogeneousMedium result{};
    result.sigmaAR = sigmaA[0]; result.sigmaAG = sigmaA[1]; result.sigmaAB = sigmaA[2];
    result.sigmaSR = sigmaS[0]; result.sigmaSG = sigmaS[1]; result.sigmaSB = sigmaS[2];
    result.g = scalar(node, "g");
    result.thickness = scalar(node, "thickness");
    return result;
}

json operation(const char* nameSpace, const char* name)
{
    return {{"namespace", nameSpace}, {"name", name}, {"version", 1}};
}

json parameter(const char* type, const json& value)
{
    return {{"source", "constant"}, {"type", type}, {"value", value}};
}

json connection(const std::string& node, const char* port)
{
    return {{"node", node}, {"port", port}};
}

json serializeInterface(uint32_t index, const LayerInterface& value)
{
    const std::string id = "interface-" + std::to_string(index);
    json parameters = json::object();
    json operationValue;
    switch (static_cast<InterfaceKind>(value.kind))
    {
    case InterfaceKind::RoughDielectric:
        operationValue = operation("ncls.interface", "rough_dielectric");
        parameters = {
            {"alpha_x", parameter("Float", value.alphaX)},
            {"alpha_y", parameter("Float", value.alphaY)},
            {"relative_ior", parameter("Float", value.relativeIor)},
            {"tangent_rotation", parameter("Float", value.tangentRotation)},
        };
        break;
    case InterfaceKind::RoughConductor:
        operationValue = operation("ncls.interface", "rough_conductor");
        parameters = {
            {"alpha_x", parameter("Float", value.alphaX)},
            {"alpha_y", parameter("Float", value.alphaY)},
            {"eta", parameter("Color3", {value.etaR, value.etaG, value.etaB})},
            {"k", parameter("Color3", {value.kR, value.kG, value.kB})},
            {"tangent_rotation", parameter("Float", value.tangentRotation)},
        };
        break;
    case InterfaceKind::Diffuse:
        operationValue = operation("ncls.interface", "diffuse");
        parameters = {{"color", parameter("Color3", {value.colorR, value.colorG, value.colorB})}};
        break;
    case InterfaceKind::Sheen:
        operationValue = operation("ncls.interface", "sheen");
        parameters = {
            {"color", parameter("Color3", {value.colorR, value.colorG, value.colorB})},
            {"roughness", parameter("Float", value.alphaX)},
        };
        break;
    }
    return {{"id", id}, {"operation", operationValue}, {"inputs", json::object()}, {"parameters", parameters}};
}

json serializeMedium(uint32_t index, const HomogeneousMedium& value)
{
    return {
        {"id", "medium-" + std::to_string(index)},
        {"operation", operation("ncls.medium", "homogeneous")},
        {"inputs", json::object()},
        {"parameters", {
            {"sigma_a", parameter("Color3", {value.sigmaAR, value.sigmaAG, value.sigmaAB})},
            {"sigma_s", parameter("Color3", {value.sigmaSR, value.sigmaSG, value.sigmaSB})},
            {"g", parameter("Float", value.g)},
            {"thickness", parameter("Float", value.thickness)},
        }},
    };
}

bool finite(float value) { return std::isfinite(value); }
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
} // namespace

LayerStackIR makeDefaultMaterial()
{
    LayerStackIR stack{};
    stack.interfaceCount = 3;
    stack.mediumCount = 2;
    stack.interfaces[0].kind = static_cast<uint32_t>(InterfaceKind::RoughDielectric);
    stack.interfaces[0].alphaX = 0.08f;
    stack.interfaces[0].alphaY = 0.18f;
    stack.interfaces[0].relativeIor = 1.48f;
    stack.interfaces[0].tangentRotation = 0.25f;
    stack.interfaces[1].kind = static_cast<uint32_t>(InterfaceKind::RoughDielectric);
    stack.interfaces[1].alphaX = 0.24f;
    stack.interfaces[1].alphaY = 0.11f;
    stack.interfaces[1].relativeIor = 1.32f;
    stack.interfaces[1].tangentRotation = -0.35f;
    stack.interfaces[2].kind = static_cast<uint32_t>(InterfaceKind::RoughConductor);
    stack.interfaces[2].alphaX = 0.32f;
    stack.interfaces[2].alphaY = 0.12f;
    stack.interfaces[2].etaR = 0.18f; stack.interfaces[2].etaG = 0.48f; stack.interfaces[2].etaB = 1.25f;
    stack.interfaces[2].kR = 3.85f; stack.interfaces[2].kG = 2.35f; stack.interfaces[2].kB = 1.85f;
    stack.interfaces[2].tangentRotation = 0.55f;
    stack.media[0] = {0.03f, 0.08f, 0.12f, 0.15f, 0.10f, 0.06f, 0.25f, 0.42f};
    stack.media[1] = {0.02f, 0.04f, 0.06f, 0.0f, 0.0f, 0.0f, 0.0f, 0.22f};
    validateLayerStack(stack);
    return stack;
}

LayerStackIR loadMaterialProgramDocument(
    const json& document,
    std::string* displayName,
    const std::string& fallbackDisplayName)
{
    require(document.value("schema_name", "") == "ncls.material-program", "unsupported MaterialProgram schema_name");
    require(document.value("schema_version", 0u) == 1u, "unsupported MaterialProgram schema_version");
    require(document.value("color_model", "") == "linear-srgb", "viewer requires linear-srgb MaterialProgram");
    if (displayName)
    {
        *displayName = document.value("metadata", json::object()).value("display_name", fallbackDisplayName);
        if (displayName->empty()) *displayName = fallbackDisplayName;
    }

    std::unordered_map<std::string, const json*> nodes;
    for (const auto& node : document.at("nodes"))
    {
        const std::string id = node.at("id").get<std::string>();
        if (!nodes.emplace(id, &node).second) throw std::runtime_error("duplicate MaterialProgram node: " + id);
    }
    const auto& surfaceConnection = document.at("outputs").at("surface");
    const auto& surface = nodeById(nodes, surfaceConnection);
    require(operationKey(surface) == "ncls.composition.layer_stack@1", "surface output is not layer_stack@1");
    require(surfaceConnection.at("port") == "surface", "surface connection uses wrong port");
    const auto& interfaces = surface.at("inputs").at("interfaces");
    const auto& media = surface.at("inputs").at("media");
    require(interfaces.is_array() && media.is_array(), "layer_stack interfaces/media must be arrays");
    require(interfaces.size() >= 1 && interfaces.size() <= kMaximumInterfaces, "interface count must lie in [1,8]");
    require(media.size() + 1 == interfaces.size(), "N-interface stack must contain N-1 media");

    LayerStackIR result{};
    result.interfaceCount = static_cast<uint32_t>(interfaces.size());
    result.mediumCount = static_cast<uint32_t>(media.size());
    for (uint32_t index = 0; index < result.interfaceCount; ++index)
        result.interfaces[index] = parseInterface(nodeById(nodes, interfaces[index]));
    for (uint32_t index = 0; index < result.mediumCount; ++index)
        result.media[index] = parseMedium(nodeById(nodes, media[index]));
    validateLayerStack(result);
    return result;
}

LayerStackIR loadMaterialProgram(const std::filesystem::path& path, std::string* displayName)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open MaterialProgram: " + path.string());
    return loadMaterialProgramDocument(json::parse(stream), displayName, path.stem().string());
}

json makeMaterialProgramDocument(const LayerStackIR& stack, const std::string& displayName)
{
    validateLayerStack(stack);
    json nodes = json::array();
    json interfaceConnections = json::array();
    json mediumConnections = json::array();
    for (uint32_t index = 0; index < stack.interfaceCount; ++index)
    {
        nodes.push_back(serializeInterface(index, stack.interfaces[index]));
        interfaceConnections.push_back(connection("interface-" + std::to_string(index), "interface"));
    }
    for (uint32_t index = 0; index < stack.mediumCount; ++index)
    {
        nodes.push_back(serializeMedium(index, stack.media[index]));
        mediumConnections.push_back(connection("medium-" + std::to_string(index), "medium"));
    }
    nodes.push_back({
        {"id", "surface"},
        {"operation", operation("ncls.composition", "layer_stack")},
        {"inputs", {{"interfaces", interfaceConnections}, {"media", mediumConnections}}},
        {"parameters", json::object()},
    });
    json outputs = {
        {"surface", connection("surface", "surface")},
        {"interior_medium", nullptr}, {"exterior_medium", nullptr}, {"emission", nullptr},
        {"opacity", nullptr}, {"displacement", nullptr},
    };
    return {
        {"schema_name", "ncls.material-program"},
        {"schema_version", 1},
        {"color_model", "linear-srgb"},
        {"nodes", nodes},
        {"resources", json::array()},
        {"outputs", outputs},
        {"metadata", {{"display_name", displayName}}},
    };
}

void saveMaterialProgram(const std::filesystem::path& path, const LayerStackIR& stack, const std::string& displayName)
{
    const json document = makeMaterialProgramDocument(stack, displayName);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    const auto temporary = path.string() + ".tmp";
    std::error_code removeError;
    std::filesystem::remove(temporary, removeError);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot write MaterialProgram: " + path.string());
        stream << document.dump(2) << '\n';
    }
    std::filesystem::remove(path, removeError);
    std::filesystem::rename(temporary, path);
}

void validateLayerStack(const LayerStackIR& stack)
{
    require(stack.magic == kLayerStackMagic && stack.abiVersion == kLayerStackVersion, "invalid LayerStackIR header");
    require(stack.interfaceCount >= 1 && stack.interfaceCount <= kMaximumInterfaces, "invalid LayerStackIR interface count");
    require(stack.mediumCount + 1 == stack.interfaceCount, "invalid LayerStackIR medium count");
    for (uint32_t index = 0; index < stack.interfaceCount; ++index)
    {
        const auto& value = stack.interfaces[index];
        require(value.flags == 0 && value.reserved == 0.f, "LayerStackIR reserved interface fields must be zero");
        require(value.kind <= static_cast<uint32_t>(InterfaceKind::Sheen), "unknown interface kind");
        const auto kind = static_cast<InterfaceKind>(value.kind);
        if (index + 1 < stack.interfaceCount) require(kind == InterfaceKind::RoughDielectric, "only rough dielectric coats may precede the base");
        else require(kind != InterfaceKind::RoughDielectric, "LayerStackIR v1 base must be opaque");
        if (kind == InterfaceKind::RoughDielectric || kind == InterfaceKind::RoughConductor || kind == InterfaceKind::Sheen)
            require(finite(value.alphaX) && finite(value.alphaY) && value.alphaX >= 0.f && value.alphaX <= 1.f
                && value.alphaY >= 0.f && value.alphaY <= 1.f, "roughness must lie in [0,1]");
        if (kind == InterfaceKind::RoughDielectric) require(finite(value.relativeIor) && value.relativeIor > 0.f, "relative IOR must be positive");
        for (float channel : {value.etaR, value.etaG, value.etaB, value.kR, value.kG, value.kB, value.colorR, value.colorG, value.colorB})
            require(finite(channel) && channel >= 0.f, "interface optical values must be finite and nonnegative");
        require(finite(value.tangentRotation), "tangent rotation must be finite");
    }
    for (uint32_t index = 0; index < stack.mediumCount; ++index)
    {
        const auto& value = stack.media[index];
        for (float channel : {value.sigmaAR, value.sigmaAG, value.sigmaAB, value.sigmaSR, value.sigmaSG, value.sigmaSB})
            require(finite(channel) && channel >= 0.f, "medium coefficients must be finite and nonnegative");
        require(finite(value.g) && value.g > -1.f && value.g < 1.f, "phase g must lie in (-1,1)");
        require(finite(value.thickness) && value.thickness >= 0.f, "medium thickness must be nonnegative");
        const float sigmaT[3] = {value.sigmaAR + value.sigmaSR, value.sigmaAG + value.sigmaSG, value.sigmaAB + value.sigmaSB};
        if (value.sigmaSR > 0.f || value.sigmaSG > 0.f || value.sigmaSB > 0.f)
        {
            const float maximum = std::max({sigmaT[0], sigmaT[1], sigmaT[2]});
            const float minimum = std::min({sigmaT[0], sigmaT[1], sigmaT[2]});
            require(maximum - minimum <= std::max(1e-5f, maximum * 1e-4f),
                "homogeneous@1 requires achromatic total extinction when volume scattering is enabled");
        }
    }
}

std::string layerStackHash(const LayerStackIR& stack)
{
    validateLayerStack(stack);
    return sha256Hex(&stack, sizeof(stack));
}

bool addDielectricCoat(LayerStackIR& stack)
{
    if (stack.interfaceCount >= kMaximumInterfaces) return false;
    const uint32_t base = stack.interfaceCount - 1;
    stack.interfaces[base + 1] = stack.interfaces[base];
    LayerInterface coat{};
    coat.kind = static_cast<uint32_t>(InterfaceKind::RoughDielectric);
    coat.alphaX = 0.18f; coat.alphaY = 0.18f; coat.relativeIor = 1.45f;
    stack.interfaces[base] = coat;
    stack.media[stack.mediumCount] = HomogeneousMedium{};
    stack.media[stack.mediumCount].thickness = 1.f;
    ++stack.interfaceCount;
    ++stack.mediumCount;
    return true;
}

bool removeCoat(LayerStackIR& stack, uint32_t index)
{
    if (stack.interfaceCount <= 1 || index + 1 >= stack.interfaceCount) return false;
    for (uint32_t item = index; item + 1 < stack.interfaceCount; ++item) stack.interfaces[item] = stack.interfaces[item + 1];
    const uint32_t mediumToRemove = std::min(index, stack.mediumCount - 1);
    for (uint32_t item = mediumToRemove; item + 1 < stack.mediumCount; ++item) stack.media[item] = stack.media[item + 1];
    stack.interfaces[stack.interfaceCount - 1] = {};
    stack.media[stack.mediumCount - 1] = {};
    --stack.interfaceCount;
    --stack.mediumCount;
    return true;
}

bool moveCoat(LayerStackIR& stack, uint32_t index, int direction)
{
    if (index + 1 >= stack.interfaceCount) return false;
    const int target = static_cast<int>(index) + direction;
    if (target < 0 || target + 1 >= static_cast<int>(stack.interfaceCount)) return false;
    std::swap(stack.interfaces[index], stack.interfaces[target]);
    std::swap(stack.media[index], stack.media[target]);
    return true;
}
} // namespace ncls
