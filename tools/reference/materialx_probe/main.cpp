#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXRender/LightHandler.h>
#include <MaterialXRender/OiioImageLoader.h>
#include <MaterialXRenderGlsl/GlslRenderer.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = MaterialX;

namespace
{

struct Arguments
{
    std::filesystem::path material;
    std::filesystem::path materialXRoot;
    std::filesystem::path sphere;
    std::filesystem::path output;
    unsigned int size = 240;
    mx::Vector3 lightDirection = mx::Vector3(0.36514837f, 0.54772256f, 0.73029673f);
    mx::Color3 lightColor = mx::Color3(1.0f);
    float lightIntensity = 1.0f;
    std::filesystem::path dumpShader;
};

float parseFloat(const char* value, const char* name)
{
    try
    {
        size_t consumed = 0;
        const float result = std::stof(value, &consumed);
        if (consumed != std::string(value).size() || !std::isfinite(result)) throw std::invalid_argument("invalid");
        return result;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
}

Arguments parseArguments(int argc, char** argv)
{
    if (argc < 5)
    {
        throw std::runtime_error(
            "usage: ncls_materialx_probe MATERIAL.mtlx MATERIALX_ROOT SPHERE.obj OUTPUT.exr "
            "[--size N] [--light-direction X Y Z] [--light-color R G B] [--light-intensity V] "
            "[--dump-shader FILE]");
    }
    Arguments arguments;
    arguments.material = std::filesystem::absolute(argv[1]);
    arguments.materialXRoot = std::filesystem::absolute(argv[2]);
    arguments.sphere = std::filesystem::absolute(argv[3]);
    arguments.output = std::filesystem::absolute(argv[4]);
    for (int index = 5; index < argc; ++index)
    {
        const std::string option = argv[index];
        const auto require = [&](int count) {
            if (index + count >= argc) throw std::runtime_error(option + " requires more values");
        };
        if (option == "--size")
        {
            require(1);
            arguments.size = static_cast<unsigned int>(std::stoul(argv[++index]));
            if (arguments.size == 0) throw std::runtime_error("--size must be positive");
        }
        else if (option == "--light-direction")
        {
            require(3);
            const float x = parseFloat(argv[++index], "light direction X");
            const float y = parseFloat(argv[++index], "light direction Y");
            const float z = parseFloat(argv[++index], "light direction Z");
            arguments.lightDirection = mx::Vector3(x, y, z);
        }
        else if (option == "--light-color")
        {
            require(3);
            const float r = parseFloat(argv[++index], "light color R");
            const float g = parseFloat(argv[++index], "light color G");
            const float b = parseFloat(argv[++index], "light color B");
            arguments.lightColor = mx::Color3(r, g, b);
        }
        else if (option == "--light-intensity")
        {
            require(1);
            arguments.lightIntensity = parseFloat(argv[++index], "light intensity");
        }
        else if (option == "--dump-shader")
        {
            require(1);
            arguments.dumpShader = std::filesystem::absolute(argv[++index]);
        }
        else
        {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    const float length = arguments.lightDirection.getMagnitude();
    if (!(length > 0.0f)) throw std::runtime_error("light direction must be nonzero");
    arguments.lightDirection /= length;
    for (const auto& path : {arguments.material, arguments.materialXRoot, arguments.sphere})
    {
        if (!std::filesystem::exists(path)) throw std::runtime_error("missing input: " + path.string());
    }
    std::filesystem::create_directories(arguments.output.parent_path());
    return arguments;
}

mx::DocumentPtr loadStandardLibraries(const std::filesystem::path& materialXRoot, mx::FileSearchPath& searchPath)
{
    searchPath = mx::FileSearchPath(materialXRoot.string());
    mx::DocumentPtr libraries = mx::createDocument();
    if (mx::loadLibraries({"libraries"}, searchPath, libraries).empty())
        throw std::runtime_error("MaterialX standard libraries were not found under " + materialXRoot.string());
    return libraries;
}

mx::ShaderPtr generateShader(
    mx::DocumentPtr document,
    mx::DocumentPtr libraries,
    mx::FileSearchPath searchPath,
    mx::LightHandlerPtr lightHandler)
{
    mx::ShaderGeneratorPtr generator = mx::GlslShaderGenerator::create();
    mx::DefaultColorManagementSystemPtr colorManagement =
        mx::DefaultColorManagementSystem::create(generator->getTarget());
    colorManagement->loadLibrary(libraries);
    generator->setColorManagementSystem(colorManagement);

    mx::GenContext context(generator);
    context.registerSourceCodeSearchPath(searchPath);
    context.getOptions().targetColorSpaceOverride = "lin_rec709";
    context.getOptions().shaderInterfaceType = mx::SHADER_INTERFACE_COMPLETE;
    context.getOptions().hwMaxActiveLightSources = 1;
    generator->registerShaderMetadata(libraries, context);

    std::vector<mx::NodePtr> lights;
    lightHandler->findLights(document, lights);
    if (lights.size() != 1) throw std::runtime_error("native probe expected exactly one directional light");
    lightHandler->registerLights(document, lights, context);
    lightHandler->setLightSources(lights);

    const std::vector<mx::TypedElementPtr> renderables = mx::findRenderableElements(document);
    if (renderables.size() != 1)
        throw std::runtime_error("native probe expected exactly one renderable MaterialX element");
    const std::string name = mx::createValidName(renderables.front()->getName());
    mx::ShaderPtr shader = generator->generate(name, renderables.front(), context);
    if (!shader) throw std::runtime_error("MaterialX GLSL generation returned no shader");
    return shader;
}

int run(const Arguments& arguments)
{
    if (mx::getVersionString() != "1.39.4")
        throw std::runtime_error("MaterialX version mismatch: expected 1.39.4, got " + mx::getVersionString());

    mx::FileSearchPath librarySearchPath;
    mx::DocumentPtr libraries = loadStandardLibraries(arguments.materialXRoot, librarySearchPath);
    mx::FileSearchPath documentSearchPath(arguments.material.parent_path().string());
    documentSearchPath.append(librarySearchPath);

    mx::DocumentPtr document = mx::createDocument();
    mx::readFromXmlFile(document, arguments.material.string(), documentSearchPath);
    document->setDataLibrary(libraries);

    mx::NodePtr light = document->addNode("directional_light", "ncls_parity_light", mx::LIGHT_SHADER_TYPE_STRING);
    light->setInputValue("direction", -arguments.lightDirection);
    light->setInputValue("color", arguments.lightColor);
    light->setInputValue("intensity", arguments.lightIntensity);

    std::string validationMessage;
    if (!document->validate(&validationMessage))
        throw std::runtime_error("invalid MaterialX document: " + validationMessage);

    mx::GlslRendererPtr renderer = mx::GlslRenderer::create(
        arguments.size, arguments.size, mx::Image::BaseType::FLOAT);
    renderer->initialize();
    renderer->setScreenColor(mx::Color3(0.0f));

    mx::OiioImageLoaderPtr oiioLoader = mx::OiioImageLoader::create();
    mx::ImageHandlerPtr imageHandler = renderer->createImageHandler(oiioLoader);
    imageHandler->setSearchPath(documentSearchPath);
    renderer->setImageHandler(imageHandler);

    mx::GeometryHandlerPtr geometryHandler = renderer->getGeometryHandler();
    geometryHandler->clearGeometry();
    if (!geometryHandler->loadGeometry(arguments.sphere.string(), false))
        throw std::runtime_error("failed to load sphere geometry: " + arguments.sphere.string());

    mx::LightHandlerPtr lightHandler = mx::LightHandler::create();
    lightHandler->setDirectLighting(true);
    lightHandler->setIndirectLighting(false);
    lightHandler->setRefractionTwoSided(true);
    mx::ShaderPtr shader = generateShader(document, libraries, documentSearchPath, lightHandler);
    if (!arguments.dumpShader.empty())
    {
        std::filesystem::create_directories(arguments.dumpShader.parent_path());
        std::ofstream stream(arguments.dumpShader, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("failed to open shader dump: " + arguments.dumpShader.string());
        stream << shader->getSourceCode(mx::Stage::VERTEX) << "\n\n"
               << shader->getSourceCode(mx::Stage::PIXEL);
    }
    renderer->setLightHandler(lightHandler);
    renderer->createProgram(shader);
    renderer->validateInputs();
    renderer->render();

    mx::ImagePtr image = renderer->captureImage();
    if (!image || !oiioLoader->saveImage(arguments.output.string(), image, true))
        throw std::runtime_error("failed to save native float EXR: " + arguments.output.string());

    std::cout << "MaterialX " << mx::getVersionString() << " native float render: "
              << arguments.output.string() << std::endl;
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        return run(parseArguments(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::cerr << "ncls_materialx_probe: " << error.what() << std::endl;
        return 1;
    }
}
