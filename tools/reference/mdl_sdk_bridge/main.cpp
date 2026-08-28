// SPDX-FileCopyrightText: Copyright (c) 2026 NeuralShading contributors
// SPDX-License-Identifier: BSD-3-Clause

#include <mi/mdl_sdk.h>
#include <mi/neuraylib/definition_wrapper.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#include <stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <Windows.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ncls
{
namespace fs = std::filesystem;

struct Options
{
    std::string command;
    fs::path sdkRoot;
    fs::path moduleRoot;
    fs::path outputDirectory;
    std::string module;
    std::string material;
    std::map<std::string, std::string> arguments;
    bool skipTexturePayloads = false;
    fs::path nativeQueries;
    fs::path nativeOutput;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

std::string jsonEscape(std::string_view value)
{
    std::ostringstream stream;
    for (const unsigned char c : value)
    {
        switch (c)
        {
        case '\"': stream << "\\\""; break;
        case '\\': stream << "\\\\"; break;
        case '\b': stream << "\\b"; break;
        case '\f': stream << "\\f"; break;
        case '\n': stream << "\\n"; break;
        case '\r': stream << "\\r"; break;
        case '\t': stream << "\\t"; break;
        default:
            if (c < 0x20)
                stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c);
            else
                stream << static_cast<char>(c);
        }
    }
    return stream.str();
}

std::string quote(std::string_view value)
{
    return "\"" + jsonEscape(value) + "\"";
}

std::string uuidHex(const mi::base::Uuid& value)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0')
           << std::setw(8) << value.m_id1
           << std::setw(8) << value.m_id2
           << std::setw(8) << value.m_id3
           << std::setw(8) << value.m_id4;
    return stream.str();
}

void writeText(const fs::path& path, std::string_view content)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        fail("unable to create file: " + path.string());
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!stream)
        fail("unable to write file: " + path.string());
}

void writeBinary(const fs::path& path, const void* data, size_t size)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream)
        fail("unable to create file: " + path.string());
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream)
        fail("unable to write file: " + path.string());
}

struct DecodedJpeg
{
    std::vector<std::uint8_t> pixels;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
};

DecodedJpeg decodeJpeg(const fs::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        fail("unable to open JPEG texture: " + path.string());
    const std::streamsize fileSize = stream.tellg();
    if (fileSize <= 0 || fileSize > std::numeric_limits<int>::max())
        fail("JPEG texture size is unsupported: " + path.string());
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> source(static_cast<size_t>(fileSize));
    stream.read(reinterpret_cast<char*>(source.data()), fileSize);
    if (!stream)
        fail("unable to read JPEG texture: " + path.string());

    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info_from_memory(source.data(), static_cast<int>(source.size()), &width, &height, &channels))
        fail("stb_image could not inspect JPEG texture: " + path.string());
    stbi_uc* pixels = stbi_load_from_memory(
        source.data(),
        static_cast<int>(source.size()),
        &width,
        &height,
        &channels,
        channels
    );
    if (!pixels)
        fail("stb_image could not decode JPEG texture: " + path.string());
    DecodedJpeg result;
    result.width = static_cast<std::uint32_t>(width);
    result.height = static_cast<std::uint32_t>(height);
    result.channels = static_cast<std::uint32_t>(channels);
    if (result.channels != 1 && result.channels != 3)
        fail("JPEG texture must decode to one or three channels: " + path.string());
    const size_t size = static_cast<size_t>(result.width) * result.height * result.channels;
    result.pixels.assign(pixels, pixels + size);
    stbi_image_free(pixels);
    return result;
}

std::pair<std::string, std::string> splitMaterialName(const std::string& qualifiedName)
{
    const size_t leftParen = qualifiedName.rfind('(');
    const size_t searchEnd = leftParen == std::string::npos ? qualifiedName.size() : leftParen;
    const size_t separator = qualifiedName.rfind("::", searchEnd);
    if (qualifiedName.size() < 4 || qualifiedName.rfind("::", 0) != 0 || separator == std::string::npos || separator < 2)
        fail("material must be an absolute qualified MDL name: " + qualifiedName);
    return {qualifiedName.substr(0, separator), qualifiedName.substr(separator + 2)};
}

std::vector<std::string> splitValues(const std::string& text)
{
    std::vector<std::string> values;
    size_t begin = 0;
    while (begin <= text.size())
    {
        const size_t end = text.find(',', begin);
        values.emplace_back(text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return values;
}

mi::neuraylib::tct_float3 normalized(float x, float y, float z)
{
    const float lengthSquared = x * x + y * y + z * z;
    if (!(lengthSquared > 0.0f) || !std::isfinite(lengthSquared))
        fail("native query direction must be finite and nonzero");
    const float inverseLength = 1.0f / std::sqrt(lengthSquared);
    return {x * inverseLength, y * inverseLength, z * inverseLength};
}

std::uint32_t readU32(std::istream& stream)
{
    std::uint32_t value = 0;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream)
        fail("native query packet is truncated");
    return value;
}

float readFloat(std::istream& stream)
{
    float value = 0.0f;
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream || !std::isfinite(value))
        fail("native query packet contains a truncated or non-finite float");
    return value;
}

void writeU32(std::ostream& stream, std::uint32_t value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeFloat(std::ostream& stream, float value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

double parseFinite(const std::string& text)
{
    size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value))
        fail("argument is not a finite number: " + text);
    return value;
}

std::string typeName(const mi::neuraylib::IType* input)
{
    mi::base::Handle<const mi::neuraylib::IType> type(input->skip_all_type_aliases());
    using Type = mi::neuraylib::IType;
    switch (type->get_kind())
    {
    case Type::TK_BOOL: return "bool";
    case Type::TK_INT: return "int";
    case Type::TK_ENUM: return "enum";
    case Type::TK_FLOAT: return "float";
    case Type::TK_DOUBLE: return "double";
    case Type::TK_STRING: return "string";
    case Type::TK_COLOR: return "color";
    case Type::TK_VECTOR:
    {
        mi::base::Handle<const mi::neuraylib::IType_vector> vector(type->get_interface<mi::neuraylib::IType_vector>());
        mi::base::Handle<const mi::neuraylib::IType_atomic> component(vector->get_element_type());
        std::string prefix;
        switch (component->get_kind())
        {
        case Type::TK_FLOAT: prefix = "float"; break;
        case Type::TK_DOUBLE: prefix = "double"; break;
        case Type::TK_INT: prefix = "int"; break;
        case Type::TK_BOOL: prefix = "bool"; break;
        default: prefix = "vector"; break;
        }
        return prefix + std::to_string(vector->get_size());
    }
    case Type::TK_MATRIX: return "matrix";
    case Type::TK_ARRAY: return "array";
    case Type::TK_STRUCT: return "struct";
    case Type::TK_TEXTURE:
    {
        mi::base::Handle<const mi::neuraylib::IType_texture> texture(type->get_interface<mi::neuraylib::IType_texture>());
        using Shape = mi::neuraylib::IType_texture::Shape;
        switch (texture->get_shape())
        {
        case Shape::TS_2D: return "texture_2d";
        case Shape::TS_3D: return "texture_3d";
        case Shape::TS_CUBE: return "texture_cube";
        case Shape::TS_PTEX: return "texture_ptex";
        case Shape::TS_BSDF_DATA: return "texture_bsdf_data";
        default: return "texture";
        }
    }
    case Type::TK_LIGHT_PROFILE: return "light_profile";
    case Type::TK_BSDF_MEASUREMENT: return "bsdf_measurement";
    default: return "unsupported";
    }
}

std::string valueJson(const mi::neuraylib::IValue* value)
{
    using Value = mi::neuraylib::IValue;
    std::ostringstream stream;
    stream << std::setprecision(17);
    switch (value->get_kind())
    {
    case Value::VK_BOOL:
    {
        mi::base::Handle<const mi::neuraylib::IValue_bool> item(value->get_interface<mi::neuraylib::IValue_bool>());
        return item->get_value() ? "true" : "false";
    }
    case Value::VK_INT:
    {
        mi::base::Handle<const mi::neuraylib::IValue_int> item(value->get_interface<mi::neuraylib::IValue_int>());
        return std::to_string(item->get_value());
    }
    case Value::VK_ENUM:
    {
        mi::base::Handle<const mi::neuraylib::IValue_enum> item(value->get_interface<mi::neuraylib::IValue_enum>());
        return std::string("{\"name\":") + quote(item->get_name()) + ",\"value\":" + std::to_string(item->get_value()) + "}";
    }
    case Value::VK_FLOAT:
    {
        mi::base::Handle<const mi::neuraylib::IValue_float> item(value->get_interface<mi::neuraylib::IValue_float>());
        stream << item->get_value();
        return stream.str();
    }
    case Value::VK_DOUBLE:
    {
        mi::base::Handle<const mi::neuraylib::IValue_double> item(value->get_interface<mi::neuraylib::IValue_double>());
        stream << item->get_value();
        return stream.str();
    }
    case Value::VK_STRING:
    {
        mi::base::Handle<const mi::neuraylib::IValue_string> item(value->get_interface<mi::neuraylib::IValue_string>());
        return quote(item->get_value());
    }
    case Value::VK_VECTOR:
    case Value::VK_COLOR:
    case Value::VK_ARRAY:
    case Value::VK_MATRIX:
    case Value::VK_STRUCT:
    {
        mi::base::Handle<const mi::neuraylib::IValue_compound> item(value->get_interface<mi::neuraylib::IValue_compound>());
        stream << '[';
        for (mi::Size index = 0; index < item->get_size(); ++index)
        {
            if (index)
                stream << ',';
            mi::base::Handle<const mi::neuraylib::IValue> component(item->get_value(index));
            stream << valueJson(component.get());
        }
        stream << ']';
        return stream.str();
    }
    case Value::VK_TEXTURE:
    case Value::VK_LIGHT_PROFILE:
    case Value::VK_BSDF_MEASUREMENT:
    {
        mi::base::Handle<const mi::neuraylib::IValue_resource> item(value->get_interface<mi::neuraylib::IValue_resource>());
        const char* path = item->get_value();
        return path ? quote(path) : "null";
    }
    default: return "null";
    }
}

bool editableType(const mi::neuraylib::IType* type)
{
    const std::string name = typeName(type);
    return name == "bool" || name == "int" || name == "enum" || name == "float" || name == "double"
        || name == "color" || name == "float2" || name == "float3" || name == "float4"
        || name == "texture_2d";
}

std::optional<std::pair<std::string, std::string>> annotationRange(
    const mi::neuraylib::IAnnotation_block* block,
    std::string_view annotationPrefix)
{
    if (!block)
        return std::nullopt;
    for (mi::Size index = 0; index < block->get_size(); ++index)
    {
        mi::base::Handle<const mi::neuraylib::IAnnotation> annotation(block->get_annotation(index));
        if (!annotation || !std::string_view(annotation->get_name()).starts_with(annotationPrefix))
            continue;
        mi::base::Handle<const mi::neuraylib::IExpression_list> arguments(annotation->get_arguments());
        if (!arguments || arguments->get_size() != 2)
            fail("MDL range annotation must have two arguments");
        std::array<std::string, 2> result;
        for (mi::Size component = 0; component < 2; ++component)
        {
            mi::base::Handle<const mi::neuraylib::IExpression_constant> expression(
                arguments->get_expression<mi::neuraylib::IExpression_constant>(component)
            );
            if (!expression)
                fail("MDL range annotation arguments must be constant");
            mi::base::Handle<const mi::neuraylib::IValue> value(expression->get_value());
            result[component] = valueJson(value.get());
        }
        return std::make_pair(result[0], result[1]);
    }
    return std::nullopt;
}

std::string enumChoicesJson(const mi::neuraylib::IType* input)
{
    mi::base::Handle<const mi::neuraylib::IType> type(input->skip_all_type_aliases());
    if (type->get_kind() != mi::neuraylib::IType::TK_ENUM)
        return "[]";
    mi::base::Handle<const mi::neuraylib::IType_enum> enumeration(
        type->get_interface<mi::neuraylib::IType_enum>()
    );
    std::ostringstream stream;
    stream << '[';
    for (mi::Size index = 0; index < enumeration->get_size(); ++index)
    {
        if (index)
            stream << ',';
        stream << "{\"name\":" << quote(enumeration->get_value_name(index))
               << ",\"value\":" << enumeration->get_value_code(index) << '}';
    }
    stream << ']';
    return stream.str();
}

void setCompound(mi::neuraylib::IValue_compound* compound, const std::vector<std::string>& values)
{
    if (compound->get_size() != values.size())
        fail("compound argument has the wrong number of components");
    for (mi::Size index = 0; index < compound->get_size(); ++index)
    {
        mi::base::Handle<mi::neuraylib::IValue> component(compound->get_value(index));
        if (component->get_kind() == mi::neuraylib::IValue::VK_FLOAT)
        {
            mi::base::Handle<mi::neuraylib::IValue_float> scalar(component->get_interface<mi::neuraylib::IValue_float>());
            scalar->set_value(static_cast<mi::Float32>(parseFinite(values[index])));
        }
        else if (component->get_kind() == mi::neuraylib::IValue::VK_DOUBLE)
        {
            mi::base::Handle<mi::neuraylib::IValue_double> scalar(component->get_interface<mi::neuraylib::IValue_double>());
            scalar->set_value(parseFinite(values[index]));
        }
        else
            fail("only floating-point compound arguments are editable in V1");
    }
}

void applyArgument(
    mi::neuraylib::IFunction_call* call,
    mi::neuraylib::IMdl_factory* mdlFactory,
    mi::neuraylib::ITransaction* transaction,
    const std::string& name,
    const std::string& text
)
{
    mi::base::Handle<const mi::neuraylib::IType_list> types(call->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IType> type(types->get_type(name.c_str()));
    if (!type)
        fail("unknown material argument: " + name);
    mi::base::Handle<mi::neuraylib::IValue_factory> values(mdlFactory->create_value_factory(transaction));
    mi::base::Handle<mi::neuraylib::IExpression_factory> expressions(mdlFactory->create_expression_factory(transaction));
    mi::base::Handle<mi::neuraylib::IValue> value(values->create(type.get()));
    if (!value)
        fail("unable to create value for material argument: " + name);

    using Value = mi::neuraylib::IValue;
    switch (value->get_kind())
    {
    case Value::VK_BOOL:
    {
        mi::base::Handle<mi::neuraylib::IValue_bool> item(value->get_interface<mi::neuraylib::IValue_bool>());
        if (text == "true" || text == "1") item->set_value(true);
        else if (text == "false" || text == "0") item->set_value(false);
        else fail("bool argument must be true/false: " + name);
        break;
    }
    case Value::VK_INT:
    {
        mi::base::Handle<mi::neuraylib::IValue_int> item(value->get_interface<mi::neuraylib::IValue_int>());
        item->set_value(static_cast<mi::Sint32>(std::stol(text)));
        break;
    }
    case Value::VK_ENUM:
    {
        mi::base::Handle<mi::neuraylib::IValue_enum> item(value->get_interface<mi::neuraylib::IValue_enum>());
        if (item->set_name(text.c_str()) != 0)
        {
            const mi::Sint32 numeric = static_cast<mi::Sint32>(std::stol(text));
            if (item->set_value(numeric) != 0)
                fail("invalid enum argument: " + name);
        }
        break;
    }
    case Value::VK_FLOAT:
    {
        mi::base::Handle<mi::neuraylib::IValue_float> item(value->get_interface<mi::neuraylib::IValue_float>());
        item->set_value(static_cast<mi::Float32>(parseFinite(text)));
        break;
    }
    case Value::VK_DOUBLE:
    {
        mi::base::Handle<mi::neuraylib::IValue_double> item(value->get_interface<mi::neuraylib::IValue_double>());
        item->set_value(parseFinite(text));
        break;
    }
    case Value::VK_VECTOR:
    case Value::VK_COLOR:
    {
        mi::base::Handle<mi::neuraylib::IValue_compound> item(value->get_interface<mi::neuraylib::IValue_compound>());
        setCompound(item.get(), splitValues(text));
        break;
    }
    case Value::VK_TEXTURE:
    {
        mi::base::Handle<const mi::neuraylib::IType> unaliased(type->skip_all_type_aliases());
        mi::base::Handle<const mi::neuraylib::IType_texture> textureType(
            unaliased->get_interface<mi::neuraylib::IType_texture>()
        );
        if (!textureType || textureType->get_shape() != mi::neuraylib::IType_texture::TS_2D)
            fail("only texture_2d resource arguments are editable in V1: " + name);
        const size_t separator = text.rfind('|');
        if (separator == std::string::npos || separator == 0 || separator + 1 >= text.size())
            fail("texture_2d argument must be /pack/path|gamma: " + name);
        const std::string path = text.substr(0, separator);
        const mi::Float32 gamma = static_cast<mi::Float32>(parseFinite(text.substr(separator + 1)));
        if (path.front() != '/' || gamma <= 0.0f)
            fail("texture_2d argument requires an absolute MDL path and positive gamma: " + name);
        mi::base::Handle<mi::neuraylib::IValue_texture> texture(
            mdlFactory->create_texture(
                transaction,
                path.c_str(),
                mi::neuraylib::IType_texture::TS_2D,
                gamma,
                nullptr,
                true,
                nullptr
            )
        );
        if (!texture)
            fail("unable to resolve texture_2d argument: " + name + " (" + path + ")");
        value = texture;
        break;
    }
    default: fail("argument type is read-only in V1: " + name + " (" + typeName(type.get()) + ")");
    }

    mi::base::Handle<mi::neuraylib::IExpression_constant> expression(expressions->create_constant(value.get()));
    const mi::Sint32 result = call->set_argument(name.c_str(), expression.get());
    if (result != 0)
        fail("MDL SDK rejected argument " + name + " with code " + std::to_string(result));
}

std::string diagnostics(mi::neuraylib::IMdl_execution_context* context)
{
    std::ostringstream stream;
    for (mi::Size index = 0; index < context->get_messages_count(); ++index)
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(index));
        stream << message->get_string() << '\n';
    }
    return stream.str();
}

void requireNoErrors(mi::neuraylib::IMdl_execution_context* context, const std::string& operation)
{
    if (context->get_error_messages_count() != 0)
        fail(operation + ":\n" + diagnostics(context));
}

bool constantComponentsEqual(const mi::neuraylib::IValue* value, double expected)
{
    using Value = mi::neuraylib::IValue;
    switch (value->get_kind())
    {
    case Value::VK_FLOAT:
    {
        mi::base::Handle<const mi::neuraylib::IValue_float> item(
            value->get_interface<mi::neuraylib::IValue_float>()
        );
        return item && static_cast<double>(item->get_value()) == expected;
    }
    case Value::VK_DOUBLE:
    {
        mi::base::Handle<const mi::neuraylib::IValue_double> item(
            value->get_interface<mi::neuraylib::IValue_double>()
        );
        return item && item->get_value() == expected;
    }
    case Value::VK_COLOR:
    case Value::VK_VECTOR:
    {
        mi::base::Handle<const mi::neuraylib::IValue_compound> item(
            value->get_interface<mi::neuraylib::IValue_compound>()
        );
        if (!item)
            return false;
        for (mi::Size index = 0; index < item->get_size(); ++index)
        {
            mi::base::Handle<const mi::neuraylib::IValue> component(item->get_value(index));
            if (!component || !constantComponentsEqual(component.get(), expected))
                return false;
        }
        return true;
    }
    default: return false;
    }
}

mi::base::Handle<const mi::neuraylib::IValue> constantSubExpression(
    const mi::neuraylib::ICompiled_material* compiled,
    const char* path)
{
    mi::base::Handle<const mi::neuraylib::IExpression> expression(
        compiled->lookup_sub_expression(path)
    );
    if (!expression || expression->get_kind() != mi::neuraylib::IExpression::EK_CONSTANT)
        return {};
    mi::base::Handle<const mi::neuraylib::IExpression_constant> constant(
        expression->get_interface<mi::neuraylib::IExpression_constant>()
    );
    return constant
        ? mi::base::Handle<const mi::neuraylib::IValue>(constant->get_value())
        : mi::base::Handle<const mi::neuraylib::IValue>();
}

void requireInvalidDf(
    const mi::neuraylib::ICompiled_material* compiled,
    const char* path,
    const char* capability)
{
    const auto value = constantSubExpression(compiled, path);
    if (!value || value->get_kind() != mi::neuraylib::IValue::VK_INVALID_DF)
        fail(std::string("MDL V1 does not support ") + capability + " at " + path);
}

void requireConstantComponents(
    const mi::neuraylib::ICompiled_material* compiled,
    const char* path,
    double expected,
    const char* capability)
{
    const auto value = constantSubExpression(compiled, path);
    if (!value || !constantComponentsEqual(value.get(), expected))
        fail(std::string("MDL V1 does not support ") + capability + " at " + path);
}

void requireSurfaceEvaluateOnly(const mi::neuraylib::ICompiled_material* compiled)
{
    requireInvalidDf(compiled, "surface.emission.emission", "surface emission");
    requireConstantComponents(compiled, "surface.emission.intensity", 0.0, "surface emission");
    requireInvalidDf(compiled, "volume.scattering", "volume scattering");
    requireConstantComponents(
        compiled, "volume.absorption_coefficient", 0.0, "volume absorption"
    );
    requireConstantComponents(
        compiled, "volume.scattering_coefficient", 0.0, "volume scattering"
    );
    requireConstantComponents(compiled, "volume.emission_intensity", 0.0, "volume emission");
    requireConstantComponents(compiled, "geometry.displacement", 0.0, "displacement");
}

bool hasNonOpaqueCutout(const mi::neuraylib::ICompiled_material* compiled)
{
    const auto value = constantSubExpression(compiled, "geometry.cutout_opacity");
    return !value || !constantComponentsEqual(value.get(), 1.0);
}

size_t decodedBytesPerPixel(std::string_view pixelType)
{
    if (pixelType == "Sint8") return 1;
    if (pixelType == "Rgb") return 3;
    if (pixelType == "Rgba") return 4;
    if (pixelType == "Rgb_16") return 6;
    if (pixelType == "Rgba_16") return 8;
    if (pixelType == "Float32") return 4;
    if (pixelType == "Float32<2>") return 8;
    if (pixelType == "Float32<3>" || pixelType == "Rgb_fp") return 12;
    if (pixelType == "Float32<4>" || pixelType == "Color") return 16;
    return 0;
}

class MdlRuntime
{
public:
    explicit MdlRuntime(const fs::path& sdkRoot)
    {
        const fs::path library = sdkRoot / "bin" / "libmdl_sdk.dll";
        m_module = LoadLibraryW(library.c_str());
        if (!m_module)
            fail("unable to load MDL SDK: " + library.string());
        void* symbol = reinterpret_cast<void*>(GetProcAddress(m_module, "mi_factory"));
        if (!symbol)
            fail("MDL SDK does not export mi_factory");
        m_neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
        if (!m_neuray)
            fail("MDL SDK headers do not match the loaded binary");

        mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugins(
            m_neuray->get_api_component<mi::neuraylib::IPlugin_configuration>()
        );
        for (const char* plugin : {"nv_openimageio.dll", "dds.dll"})
        {
            const fs::path path = sdkRoot / "bin" / plugin;
            if (plugins->load_plugin_library(path.string().c_str()) != 0)
                fail("unable to load MDL resource plugin: " + path.string());
        }
    }

    ~MdlRuntime()
    {
        if (m_started && m_neuray)
            m_neuray->shutdown(true);
        m_neuray = nullptr;
        if (m_module)
            FreeLibrary(m_module);
    }

    mi::neuraylib::INeuray* get() const { return m_neuray.get(); }

    void start(const fs::path& sdkRoot, const fs::path& moduleRoot)
    {
        mi::base::Handle<mi::neuraylib::IMdl_configuration> configuration(
            m_neuray->get_api_component<mi::neuraylib::IMdl_configuration>()
        );
        for (const fs::path& path : {moduleRoot, sdkRoot / "examples" / "mdl"})
        {
            if (!fs::is_directory(path))
                fail("MDL search root is missing: " + path.string());
            if (configuration->add_mdl_path(path.string().c_str()) != 0
                || configuration->add_resource_path(path.string().c_str()) != 0)
                fail("unable to add MDL search root: " + path.string());
        }
        const mi::Sint32 result = m_neuray->start(true);
        if (result != 0)
            fail("unable to start MDL SDK: " + std::to_string(result));
        m_started = true;
    }

private:
    HMODULE m_module = nullptr;
    mi::base::Handle<mi::neuraylib::INeuray> m_neuray;
    bool m_started = false;
};

void discover(const Options& options)
{
    if (!fs::is_directory(options.sdkRoot) || !fs::is_directory(options.moduleRoot))
        fail("SDK root and module root must be existing directories");
    if (fs::exists(options.outputDirectory))
    {
        if (!fs::is_directory(options.outputDirectory) || !fs::is_empty(options.outputDirectory))
            fail("output directory must be absent or empty: " + options.outputDirectory.string());
    }
    fs::create_directories(options.outputDirectory);

    MdlRuntime runtime(options.sdkRoot);
    runtime.start(options.sdkRoot, options.moduleRoot);
    mi::neuraylib::INeuray* neuray = runtime.get();
    mi::base::Handle<mi::neuraylib::IDatabase> database(
        neuray->get_api_component<mi::neuraylib::IDatabase>()
    );
    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());
    {
        mi::base::Handle<mi::neuraylib::IMdl_impexp_api> impexp(
            neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>()
        );
        mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory(
            neuray->get_api_component<mi::neuraylib::IMdl_factory>()
        );
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(
            mdlFactory->create_execution_context()
        );
        impexp->load_module(transaction.get(), options.module.c_str(), context.get());
        requireNoErrors(context.get(), "unable to load module " + options.module);
        mi::base::Handle<const mi::IString> moduleDbName(
            mdlFactory->get_db_module_name(options.module.c_str())
        );
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str())
        );
        if (!module)
            fail("unable to access loaded module: " + options.module);

        std::vector<std::string> materials;
        for (mi::Size index = 0; index < module->get_material_count(); ++index)
        {
            const char* definitionDbName = module->get_material(index);
            if (!definitionDbName)
                fail("MDL module returned a removed material definition");
            mi::base::Handle<const mi::neuraylib::IFunction_definition> definition(
                transaction->access<mi::neuraylib::IFunction_definition>(definitionDbName)
            );
            if (!definition || !definition->is_material())
                fail("MDL module discovery returned a non-material definition");
            materials.emplace_back(definition->get_mdl_name());
        }
        std::sort(materials.begin(), materials.end());
        if (std::adjacent_find(materials.begin(), materials.end()) != materials.end())
            fail("MDL module discovery returned duplicate exact material definitions");

        std::ostringstream document;
        document << "{\n"
                 << "  \"schema\": \"ncls.mdl-module-discovery@1\",\n"
                 << "  \"mdl_sdk\": \"2025.0.0-387700.1252\",\n"
                 << "  \"module\": " << quote(module->get_mdl_name()) << ",\n"
                 << "  \"materials\": [";
        for (size_t index = 0; index < materials.size(); ++index)
        {
            if (index)
                document << ',';
            document << "\n    " << quote(materials[index]);
        }
        if (!materials.empty())
            document << '\n' << "  ";
        document << "],\n"
                 << "  \"diagnostics\": " << quote(diagnostics(context.get())) << "\n"
                 << "}\n";
        writeText(options.outputDirectory / "discovery.json", document.str());
    }
    transaction->abort();
}

std::string gammaName(mi::neuraylib::ITarget_code::Gamma_mode gamma)
{
    using Mode = mi::neuraylib::ITarget_code::Gamma_mode;
    switch (gamma)
    {
    case Mode::GM_GAMMA_DEFAULT: return "default";
    case Mode::GM_GAMMA_LINEAR: return "linear";
    case Mode::GM_GAMMA_SRGB: return "srgb";
    default: return "unknown";
    }
}

std::string textureShapeName(mi::neuraylib::ITarget_code::Texture_shape shape)
{
    using Shape = mi::neuraylib::ITarget_code::Texture_shape;
    switch (shape)
    {
    case Shape::Texture_shape_2d: return "2d";
    case Shape::Texture_shape_3d: return "3d";
    case Shape::Texture_shape_cube: return "cube";
    case Shape::Texture_shape_bsdf_data: return "bsdf_data";
    default: return "unsupported";
    }
}

struct SourceModule
{
    std::string name;
    std::string path;
};

std::vector<SourceModule> collectSourceModules(
    mi::neuraylib::ITransaction* transaction,
    const std::string& rootDatabaseName,
    const fs::path& moduleRoot)
{
    const fs::path canonicalRoot = fs::canonical(moduleRoot);
    std::vector<std::string> pending = {rootDatabaseName};
    std::set<std::string> visited;
    std::vector<SourceModule> result;
    while (!pending.empty())
    {
        const std::string databaseName = pending.back();
        pending.pop_back();
        if (!visited.insert(databaseName).second)
            continue;
        mi::base::Handle<const mi::neuraylib::IModule> dependency(
            transaction->access<mi::neuraylib::IModule>(databaseName.c_str())
        );
        if (!dependency)
            fail("unable to access imported MDL module: " + databaseName);
        if (const char* filename = dependency->get_filename())
        {
            const fs::path canonicalFile = fs::canonical(filename);
            std::error_code error;
            const fs::path relative = fs::relative(canonicalFile, canonicalRoot, error);
            const bool contained = !error && !relative.empty()
                && *relative.begin() != fs::path("..");
            if (contained)
                result.push_back({dependency->get_mdl_name(), relative.generic_string()});
            else if (databaseName == rootDatabaseName)
                fail("root MDL module escapes its configured module root: " + canonicalFile.string());
        }
        for (mi::Size index = 0; index < dependency->get_import_count(); ++index)
            pending.emplace_back(dependency->get_import(index));
    }
    std::sort(result.begin(), result.end(), [](const SourceModule& a, const SourceModule& b)
    {
        return a.path < b.path;
    });
    return result;
}

void evaluateNative(
    const Options& options,
    mi::neuraylib::ITransaction* transaction,
    mi::neuraylib::IMdl_backend_api* backendApi,
    mi::neuraylib::IMdl_execution_context* context,
    const mi::neuraylib::ICompiled_material* compiled)
{
    if (options.nativeQueries.empty() && options.nativeOutput.empty())
        return;
    if (options.nativeQueries.empty() || options.nativeOutput.empty())
        fail("native evaluation requires both --native-queries and --native-output");
    if (!fs::is_regular_file(options.nativeQueries) || fs::exists(options.nativeOutput))
        fail("native query input must exist and native output must be absent");

    mi::base::Handle<mi::neuraylib::IMdl_backend> backend(
        backendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE)
    );
    if (!backend)
        fail("locked MDL SDK has no native backend");
    for (const auto& [name, value] : std::vector<std::pair<const char*, const char*>>{
             {"fast_math", "on"},
             {"opt_level", "2"},
             {"enable_auxiliary", "on"},
             {"df_handle_slot_mode", "none"},
             {"texture_runtime_with_derivs", "off"},
             {"num_texture_results", "16"},
             {"num_texture_spaces", "4"},
         })
    {
        if (backend->set_option(name, value) != 0)
            fail(std::string("MDL native backend rejected option ") + name + "=" + value);
    }
    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(
        backend->create_link_unit(transaction, context)
    );
    requireNoErrors(context, "unable to create native link unit");
    using Descriptor = mi::neuraylib::Target_function_description;
    std::array<Descriptor, 3> descriptors = {
        Descriptor("init", "native_init"),
        Descriptor("ior", "native_ior"),
        Descriptor("surface.scattering", "native_surface_scattering"),
    };
    linkUnit->add_material(compiled, descriptors.data(), descriptors.size(), context);
    requireNoErrors(context, "unable to add native material target functions");
    mi::base::Handle<const mi::neuraylib::ITarget_code> target(
        backend->translate_link_unit(linkUnit.get(), context)
    );
    requireNoErrors(context, "unable to translate native link unit");
    if (!target)
        fail("MDL SDK returned no native target code");
    const mi::Size argumentBlockIndex = descriptors[2].argument_block_index;
    mi::base::Handle<const mi::neuraylib::ITarget_argument_block> argumentBlock;
    if (argumentBlockIndex != ~mi::Size(0))
        argumentBlock = target->get_argument_block(argumentBlockIndex);

    std::ifstream input(options.nativeQueries, std::ios::binary);
    if (!input)
        fail("unable to open native query packet");
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    if (!input || magic != std::array<char, 8>{'N', 'C', 'L', 'S', 'M', 'Q', '1', '\0'})
        fail("unsupported native query packet schema");
    const std::uint32_t count = readU32(input);
    const std::uint32_t stride = readU32(input);
    if (count == 0 || count > 1'000'000 || stride != 11 * sizeof(float))
        fail("invalid native query count or record stride");

    fs::create_directories(options.nativeOutput.parent_path());
    std::ofstream output(options.nativeOutput, std::ios::binary);
    if (!output)
        fail("unable to create native result packet");
    const std::array<char, 8> resultMagic = {'N', 'C', 'L', 'S', 'M', 'R', '1', '\0'};
    output.write(resultMagic.data(), resultMagic.size());
    writeU32(output, count);
    writeU32(output, 4 * sizeof(float));

    const mi::neuraylib::tct_float3 tangentU[4] = {
        {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
    };
    const mi::neuraylib::tct_float3 tangentV[4] = {
        {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
    };
    const mi::neuraylib::tct_float4 identity[3] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
    };
    for (std::uint32_t queryIndex = 0; queryIndex < count; ++queryIndex)
    {
        const float woX = readFloat(input);
        const float woY = readFloat(input);
        const float woZ = readFloat(input);
        const float wiX = readFloat(input);
        const float wiY = readFloat(input);
        const float wiZ = readFloat(input);
        const mi::neuraylib::tct_float3 wo = normalized(woX, woY, woZ);
        const mi::neuraylib::tct_float3 wi = normalized(wiX, wiY, wiZ);
        const mi::neuraylib::tct_float3 position = {readFloat(input), readFloat(input), readFloat(input)};
        const float u = readFloat(input);
        const float v = readFloat(input);
        mi::neuraylib::tct_float3 textureCoordinates[4] = {
            {u, v, 0.0f}, {0.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
        };
        mi::neuraylib::tct_float4 textureResults[16] = {};
        mi::neuraylib::Shading_state_material state{};
        state.normal = {0.0f, 0.0f, 1.0f};
        state.geom_normal = state.normal;
        state.position = position;
        state.animation_time = 0.0f;
        state.text_coords = textureCoordinates;
        state.tangent_u = tangentU;
        state.tangent_v = tangentV;
        state.text_results = textureResults;
        state.ro_data_segment = nullptr;
        state.world_to_object = identity;
        state.object_to_world = identity;
        state.object_id = 0;
        state.meters_per_scene_unit = 1.0f;
        if (target->execute_init(
                descriptors[0].function_index, state, nullptr, argumentBlock.get()) != 0)
            fail("MDL native init execution failed");

        mi::neuraylib::tct_float3 materialIor{};
        if (target->execute(
                descriptors[1].function_index,
                state,
                nullptr,
                argumentBlock.get(),
                &materialIor) != 0)
            fail("MDL native IOR execution failed");
        if (!std::isfinite(materialIor.x) || !std::isfinite(materialIor.y)
            || !std::isfinite(materialIor.z) || materialIor.x <= 0.0f
            || materialIor.y <= 0.0f || materialIor.z <= 0.0f)
            fail("MDL native IOR callable returned an invalid value");
        mi::neuraylib::Bsdf_evaluate_data<mi::neuraylib::DF_HSM_NONE> evaluation{};
        evaluation.ior1 = {1.0f, 1.0f, 1.0f};
        // Native libbsdf's documented convention requests the material IOR
        // with this sentinel.  The standalone ior callable above remains a
        // protocol check, while the DF receives the ABI-native representation.
        evaluation.ior2 = {
            MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR,
            MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR,
            MI_NEURAYLIB_BSDF_USE_MATERIAL_IOR,
        };
        evaluation.k1 = wo;
        evaluation.k2 = wi;
        evaluation.flags = mi::neuraylib::DF_FLAGS_ALLOW_REFLECT_AND_TRANSMIT;
        if (target->execute_bsdf_evaluate(
                descriptors[2].function_index + 1,
                &evaluation,
                state,
                nullptr,
                argumentBlock.get()) != 0)
            fail("MDL native BSDF evaluation failed");
        mi::neuraylib::Bsdf_pdf_data pdf{};
        pdf.ior1 = evaluation.ior1;
        pdf.ior2 = evaluation.ior2;
        pdf.k1 = wo;
        pdf.k2 = wi;
        if (target->execute_bsdf_pdf(
                descriptors[2].function_index + 2,
                &pdf,
                state,
                nullptr,
                argumentBlock.get()) != 0)
            fail("MDL native BSDF PDF execution failed");
        writeFloat(output, evaluation.bsdf_diffuse.x + evaluation.bsdf_glossy.x);
        writeFloat(output, evaluation.bsdf_diffuse.y + evaluation.bsdf_glossy.y);
        writeFloat(output, evaluation.bsdf_diffuse.z + evaluation.bsdf_glossy.z);
        writeFloat(output, pdf.pdf);
    }
    if (input.peek() != std::char_traits<char>::eof() || !output)
        fail("native query packet has trailing bytes or result write failed");
}

void compile(const Options& options)
{
    if (!fs::is_directory(options.sdkRoot) || !fs::is_directory(options.moduleRoot))
        fail("SDK root and module root must be existing directories");
    if (fs::exists(options.outputDirectory))
    {
        if (!fs::is_directory(options.outputDirectory) || !fs::is_empty(options.outputDirectory))
            fail("output directory must be absent or empty: " + options.outputDirectory.string());
    }
    fs::create_directories(options.outputDirectory);

    MdlRuntime runtime(options.sdkRoot);
    runtime.start(options.sdkRoot, options.moduleRoot);
    mi::neuraylib::INeuray* neuray = runtime.get();
    mi::base::Handle<mi::neuraylib::IDatabase> database(neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());
    {
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> impexp(neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>());
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdlFactory(neuray->get_api_component<mi::neuraylib::IMdl_factory>());
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(mdlFactory->create_execution_context());

    const auto [moduleName, materialName] = splitMaterialName(options.material);
    impexp->load_module(transaction.get(), moduleName.c_str(), context.get());
    requireNoErrors(context.get(), "unable to load module " + moduleName);
    mi::base::Handle<const mi::IString> moduleDbName(mdlFactory->get_db_module_name(moduleName.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        transaction->access<mi::neuraylib::IModule>(moduleDbName->get_c_str())
    );
    if (!module)
        fail("unable to access loaded module: " + moduleName);

    std::string definitionDbName = std::string(moduleDbName->get_c_str()) + "::" + materialName;
    if (definitionDbName.back() != ')')
    {
        mi::base::Handle<const mi::IArray> overloads(module->get_function_overloads(definitionDbName.c_str()));
        if (!overloads || overloads->get_length() != 1)
            fail("material name is missing an unambiguous signature: " + options.material);
        mi::base::Handle<const mi::IString> overload(overloads->get_element<mi::IString>(0));
        definitionDbName = overload->get_c_str();
    }
    mi::base::Handle<const mi::neuraylib::IFunction_definition> definition(
        transaction->access<mi::neuraylib::IFunction_definition>(definitionDbName.c_str())
    );
    if (!definition || !definition->is_material())
        fail("selected definition is not a material: " + options.material);

    mi::neuraylib::Definition_wrapper wrapper(transaction.get(), definitionDbName.c_str(), mdlFactory.get());
    mi::Sint32 createResult = -1;
    mi::base::Handle<mi::neuraylib::IFunction_call> call(
        wrapper.create_instance<mi::neuraylib::IFunction_call>(nullptr, &createResult)
    );
    if (!call || createResult != 0)
        fail("unable to instantiate material: " + std::to_string(createResult));
    for (const auto& [name, value] : options.arguments)
        applyArgument(call.get(), mdlFactory.get(), transaction.get(), name, value);

    mi::base::Handle<mi::neuraylib::IMaterial_instance> material(call->get_interface<mi::neuraylib::IMaterial_instance>());
    context->set_option("fold_ternary_on_df", false);
    context->set_option("fold_all_bool_parameters", false);
    context->set_option("fold_all_enum_parameters", false);
    context->set_option("ignore_noinline", true);
    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled(
        material->create_compiled_material(mi::neuraylib::IMaterial_instance::CLASS_COMPILATION, context.get())
    );
    requireNoErrors(context.get(), "unable to class-compile material");
    if (!compiled)
        fail("MDL SDK returned no compiled material");
    requireSurfaceEvaluateOnly(compiled.get());
    const bool nonOpaqueCutout = hasNonOpaqueCutout(compiled.get());

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>()
    );
    mi::base::Handle<mi::neuraylib::IMdl_backend> backend(
        backendApi->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL)
    );
    // falcor2 requests this option without treating rejection as fatal.  The
    // pinned MDL 2025 HLSL backend rejects it, so the effective compiler state
    // is the backend default in both integrations.
    backend->set_option("internal_space", "coordinate_world");
    for (const auto& [name, value] : std::vector<std::pair<const char*, const char*>>{
             {"compile_constants", "on"},
             {"fast_math", "on"},
             {"opt_level", "2"},
             {"enable_auxiliary", "on"},
             {"use_renderer_adapt_normal", "on"},
             {"df_handle_slot_mode", "none"},
             {"texture_runtime_with_derivs", "off"},
             {"num_texture_results", "16"},
             {"num_texture_spaces", "4"},
             {"jit_warn_spectrum_conversion", "off"},
         })
    {
        if (backend->set_option(name, value) != 0)
            fail(std::string("MDL HLSL backend rejected option ") + name + "=" + value);
    }

    mi::base::Handle<mi::neuraylib::ILink_unit> linkUnit(backend->create_link_unit(transaction.get(), context.get()));
    requireNoErrors(context.get(), "unable to create HLSL link unit");
    using Descriptor = mi::neuraylib::Target_function_description;
    std::vector<Descriptor> descriptors;
    descriptors.emplace_back("init", "init");
    descriptors.emplace_back("ior", "ior");
    descriptors.emplace_back("thin_walled", "thin_walled");
    descriptors.emplace_back("surface.scattering", "surface_scattering");
    descriptors.emplace_back("geometry.normal", "geometry_normal");
    descriptors.emplace_back("geometry.cutout_opacity", "geometry_cutout_opacity");
    linkUnit->add_material(compiled.get(), descriptors.data(), descriptors.size(), context.get());
    requireNoErrors(context.get(), "unable to add material target functions");
    mi::base::Handle<const mi::neuraylib::ITarget_code> target(
        backend->translate_link_unit(linkUnit.get(), context.get())
    );
    requireNoErrors(context.get(), "unable to translate MDL HLSL link unit");
    if (!target)
        fail("MDL SDK returned no HLSL target code");

    writeText(options.outputDirectory / "generated.hlsl", target->get_code());
    fs::create_directories(options.outputDirectory / "ro-data");
    fs::create_directories(options.outputDirectory / "bsdf-data");
    fs::create_directories(options.outputDirectory / "texture-data");

    std::ostringstream manifest;
    const std::vector<SourceModule> sourceModules = collectSourceModules(
        transaction.get(), moduleDbName->get_c_str(), options.moduleRoot
    );
    manifest << "{\n"
             << "  \"schema\": \"ncls.mdl-compiled-artifact@1\",\n"
             << "  \"mdl_sdk\": \"2025.0.0-387700.1252\",\n"
             << "  \"module\": " << quote(moduleName) << ",\n"
             << "  \"material\": " << quote(definition->get_mdl_name()) << ",\n"
             << "  \"code\": \"generated.hlsl\",\n"
             << "  \"texture_payloads\": "
             << quote(options.skipTexturePayloads ? "metadata-only" : "decoded") << ",\n"
             << "  \"capability_audit\": {\"surface_bsdf_evaluate\":true,"
                "\"emission\":false,\"volume\":false,\"displacement\":false,"
                "\"cutout_opacity\":" << (nonOpaqueCutout ? "true" : "false") << "},\n"
             << "  \"compiled_material_hash\": " << quote(uuidHex(compiled->get_hash())) << ",\n"
             << "  \"sub_expression_hashes\": {"
                "\"surface.scattering\":"
             << quote(uuidHex(compiled->get_sub_expression_hash("surface.scattering"))) << ','
             << "\"geometry.normal\":"
             << quote(uuidHex(compiled->get_sub_expression_hash("geometry.normal"))) << ','
             << "\"geometry.cutout_opacity\":"
             << quote(uuidHex(compiled->get_sub_expression_hash("geometry.cutout_opacity"))) << "},\n"
             << "  \"source_modules\": [";
    for (size_t index = 0; index < sourceModules.size(); ++index)
    {
        if (index)
            manifest << ',';
        manifest << "{\"name\":" << quote(sourceModules[index].name)
                 << ",\"path\":" << quote(sourceModules[index].path) << '}';
    }
    manifest << "],\n"
             << "  \"df_handle_count\": " << target->get_callable_function_df_handle_count(descriptors[3].function_index) << ",\n";

    const mi::Size argumentBlockCount = target->get_argument_block_count();
    if (argumentBlockCount > 1)
        fail("V1 requires at most one MDL argument block");
    manifest << "  \"argument_block\": ";
    if (argumentBlockCount == 1)
    {
        mi::base::Handle<const mi::neuraylib::ITarget_argument_block> block(target->get_argument_block(0));
        writeBinary(options.outputDirectory / "argument-block.bin", block->get_data(), block->get_size());
        manifest << "{\"path\":\"argument-block.bin\",\"size\":" << block->get_size() << "},\n";
    }
    else
        manifest << "null,\n";

    manifest << "  \"parameters\": [";
    mi::base::Handle<const mi::neuraylib::IAnnotation_list> parameterAnnotations(
        definition->get_parameter_annotations()
    );
    mi::base::Handle<const mi::neuraylib::ITarget_value_layout> layout;
    const mi::Size argumentBlockIndex = descriptors[3].argument_block_index;
    if (argumentBlockIndex != ~mi::Size(0))
        layout = target->get_argument_block_layout(argumentBlockIndex);
    for (mi::Size index = 0; index < compiled->get_parameter_count(); ++index)
    {
        if (index)
            manifest << ',';
        mi::base::Handle<const mi::neuraylib::IValue> value(compiled->get_argument(index));
        mi::base::Handle<const mi::neuraylib::IType> type(value->get_type());
        mi::Size offset = ~mi::Size(0);
        mi::Size size = 0;
        if (layout)
        {
            mi::neuraylib::Target_value_layout_state state(layout->get_nested_state(index));
            mi::neuraylib::IValue::Kind kind;
            offset = layout->get_layout(kind, size, state);
        }
        manifest << "\n    {\"name\":" << quote(compiled->get_parameter_name(index))
                 << ",\"type\":" << quote(typeName(type.get()))
                 << ",\"editable\":" << (editableType(type.get()) ? "true" : "false")
                 << ",\"value\":" << valueJson(value.get());
        mi::base::Handle<const mi::neuraylib::IAnnotation_block> annotations;
        if (parameterAnnotations)
            annotations = parameterAnnotations->get_annotation_block(compiled->get_parameter_name(index));
        if (const auto range = annotationRange(annotations.get(), "::anno::hard_range("))
            manifest << ",\"minimum\":" << range->first << ",\"maximum\":" << range->second;
        if (const auto range = annotationRange(annotations.get(), "::anno::soft_range("))
            manifest << ",\"soft_minimum\":" << range->first << ",\"soft_maximum\":" << range->second;
        if (type->skip_all_type_aliases()->get_kind() == mi::neuraylib::IType::TK_ENUM)
            manifest << ",\"choices\":" << enumChoicesJson(type.get());
        if (offset != ~mi::Size(0))
            manifest << ",\"offset\":" << offset << ",\"size\":" << size;
        manifest << '}';
    }
    if (compiled->get_parameter_count())
        manifest << '\n' << "  ";
    manifest << "],\n";

    manifest << "  \"ro_data\": [";
    for (mi::Size index = 0; index < target->get_ro_data_segment_count(); ++index)
    {
        if (index)
            manifest << ',';
        const std::string relative = "ro-data/segment-" + std::to_string(index) + ".bin";
        writeBinary(
            options.outputDirectory / relative,
            target->get_ro_data_segment_data(index),
            target->get_ro_data_segment_size(index)
        );
        manifest << "{\"name\":" << quote(target->get_ro_data_segment_name(index))
                 << ",\"path\":" << quote(relative)
                 << ",\"size\":" << target->get_ro_data_segment_size(index) << '}';
    }
    manifest << "],\n";

    if (target->get_bsdf_measurement_count() > 1)
        fail("V1 does not support measured BSDF resources");
    if (target->get_light_profile_count() > 1)
        fail("V1 does not support light profile resources");
    if (target->get_texture_count() > 17)
        fail("V1 supports at most 16 MDL texture resources");
    manifest << "  \"textures\": [";
    bool firstTexture = true;
    for (mi::Size index = 1; index < target->get_texture_count(); ++index)
    {
        const auto shape = target->get_texture_shape(index);
        const std::string shapeName = textureShapeName(shape);
        if (shapeName != "2d" && shapeName != "bsdf_data")
            fail("V1 does not support MDL texture shape: " + shapeName);
        mi::base::Handle<const mi::neuraylib::ITexture> texture(
            transaction->access<mi::neuraylib::ITexture>(target->get_texture(index))
        );
        mi::base::Handle<const mi::neuraylib::IImage> image(
            transaction->access<mi::neuraylib::IImage>(texture->get_image())
        );
        if (!image || image->get_length() != 1 || image->is_uvtile())
            fail("V1 requires a single-frame non-UDIM texture");
        mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
        std::string relative;
        std::string filePath;
        std::string pixelType = canvas->get_type();
        std::string dataOrigin;
        if (shapeName == "bsdf_data")
        {
            relative = "bsdf-data/texture-" + std::to_string(index) + ".bin";
            mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0));
            const size_t size = canvas->get_resolution_x() * canvas->get_resolution_y()
                * canvas->get_layers_size() * sizeof(float);
            writeBinary(options.outputDirectory / relative, tile->get_data(), size);
            dataOrigin = "lower_left";
        }
        else
        {
            if (const char* filename = image->get_filename(0, 0))
                filePath = filename;
            relative = "texture-data/texture-" + std::to_string(index) + ".bin";
            if (options.skipTexturePayloads)
            {
                relative.clear();
                dataOrigin = "unavailable";
            }
            std::string extension = fs::path(filePath).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value)
            {
                return static_cast<char>(std::tolower(value));
            });
            if (!options.skipTexturePayloads && (extension == ".jpg" || extension == ".jpeg"))
            {
                const DecodedJpeg decoded = decodeJpeg(filePath);
                if (decoded.width != canvas->get_resolution_x()
                    || decoded.height != canvas->get_resolution_y())
                    fail("libjpeg-turbo dimensions differ from the MDL canvas");
                pixelType = decoded.channels == 1 ? "Sint8" : "Rgb";
                writeBinary(
                    options.outputDirectory / relative,
                    decoded.pixels.data(),
                    decoded.pixels.size()
                );
                dataOrigin = "top_left";
            }
            else if (!options.skipTexturePayloads)
            {
                const size_t bytesPerPixel = decodedBytesPerPixel(pixelType);
                if (!bytesPerPixel)
                    fail("V1 does not support decoded 2D texture pixel type: " + pixelType);
                mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0));
                const size_t size = canvas->get_resolution_x() * canvas->get_resolution_y()
                    * bytesPerPixel;
                writeBinary(options.outputDirectory / relative, tile->get_data(), size);
                dataOrigin = "lower_left";
            }
        }
        if (!firstTexture)
            manifest << ',';
        firstTexture = false;
        manifest << "{\"index\":" << index
                 << ",\"name\":" << quote(target->get_texture(index))
                 << ",\"shape\":" << quote(shapeName)
                 << ",\"gamma\":" << quote(gammaName(target->get_texture_gamma(index)))
                 << ",\"effective_gamma\":" << texture->get_effective_gamma(0, 0)
                 << ",\"pixel_type\":" << quote(pixelType)
                 << ",\"width\":" << canvas->get_resolution_x()
                 << ",\"height\":" << canvas->get_resolution_y()
                 << ",\"depth\":" << canvas->get_layers_size()
                 << ",\"path\":" << quote(filePath)
                 << ",\"data\":" << (relative.empty() ? "null" : quote(relative))
                 << ",\"data_origin\":" << quote(dataOrigin) << '}';
    }
    manifest << "],\n";
    evaluateNative(options, transaction.get(), backendApi.get(), context.get(), compiled.get());
    manifest
             << "  \"diagnostics\": " << quote(diagnostics(context.get())) << "\n"
             << "}\n";
    writeText(options.outputDirectory / "manifest.json", manifest.str());
    }
    // The bridge exports every artifact before leaving the transaction and
    // does not persist MDL DB elements across invocations.  Aborting is the
    // correct read/compile-only teardown and avoids committing transient
    // resource handles into a process that immediately shuts down.
    transaction->abort();
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string
        {
            if (++index >= argc)
                fail("missing value after " + argument);
            return argv[index];
        };
        if (argument == "compile" || argument == "native-evaluate" || argument == "discover")
        {
            if (!options.command.empty())
                fail("multiple bridge commands were provided");
            options.command = argument;
        }
        else if (argument == "--sdk-root") options.sdkRoot = fs::absolute(next());
        else if (argument == "--module-root") options.moduleRoot = fs::absolute(next());
        else if (argument == "--module") options.module = next();
        else if (argument == "--material") options.material = next();
        else if (argument == "--output-dir") options.outputDirectory = fs::absolute(next());
        else if (argument == "--native-queries") options.nativeQueries = fs::absolute(next());
        else if (argument == "--skip-texture-payloads") options.skipTexturePayloads = true;
        else if (argument == "--native-output") options.nativeOutput = fs::absolute(next());
        else if (argument == "--argument")
        {
            const std::string assignment = next();
            const size_t equals = assignment.find('=');
            if (equals == std::string::npos || equals == 0)
                fail("--argument expects name=value");
            options.arguments.emplace(assignment.substr(0, equals), assignment.substr(equals + 1));
        }
        else fail("unknown argument: " + argument);
    }
    const bool commonMissing = options.command.empty() || options.sdkRoot.empty()
        || options.moduleRoot.empty() || options.outputDirectory.empty();
    const bool discoverInvalid = options.command == "discover"
        && (options.module.empty() || !options.material.empty() || !options.arguments.empty()
            || options.skipTexturePayloads);
    const bool compileInvalid = options.command != "discover"
        && (options.material.empty() || !options.module.empty()
            || (options.command == "native-evaluate" && options.skipTexturePayloads));
    if (commonMissing || discoverInvalid || compileInvalid)
        fail("usage: ncls_mdl_sdk_bridge discover --sdk-root PATH --module-root PATH --module ::module --output-dir PATH | ncls_mdl_sdk_bridge compile --sdk-root PATH --module-root PATH --material ::module::material --output-dir PATH [--argument name=value] [--skip-texture-payloads] | ncls_mdl_sdk_bridge native-evaluate --sdk-root PATH --module-root PATH --material ::module::material --output-dir PATH [--argument name=value] --native-queries FILE --native-output FILE");
    return options;
}

} // namespace ncls

int main(int argc, char** argv)
{
    try
    {
        const ncls::Options options = ncls::parseOptions(argc, argv);
        if (options.command == "discover")
            ncls::discover(options);
        else
            ncls::compile(options);
        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "ncls_mdl_sdk_bridge: " << exception.what() << '\n';
        return 1;
    }
}
