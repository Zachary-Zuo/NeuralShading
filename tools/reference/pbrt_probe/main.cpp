#include <pbrt/bxdfs.h>
#include <pbrt/options.h>
#include <pbrt/pbrt.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

using namespace pbrt;

namespace {

struct ProbeArguments {
    std::string material = "diffuse";
    int samples = 262144;
    Float viewTheta = 20.0f;
    Float viewPhi = 0.0f;
    int maxDepth = 32;
    Float opticalThickness = 1e-6f;
    int seed = 1;
    Float mediumAlbedo = 0.0f;
    Float g = 0.0f;
    Float coatAlphaX = 0.12f;
    Float coatAlphaY = 0.12f;
    Float coatIor = 1.5f;
    Float baseAlphaX = 0.2f;
    Float baseAlphaY = 0.2f;
    Float tangentRotation = 0.0f;
    std::array<Float, 3> eta = {0.2f, 0.9f, 1.1f};
    std::array<Float, 3> k = {3.9f, 2.5f, 2.1f};
};

Float argument(int argc, char **argv, int index, Float fallback) {
    return argc > index ? std::atof(argv[index]) : fallback;
}

int integerArgument(int argc, char **argv, int index, int fallback) {
    return argc > index ? std::atoi(argv[index]) : fallback;
}

ProbeArguments parseArguments(int argc, char **argv) {
    ProbeArguments result;
    if (argc > 1)
        result.material = argv[1];
    result.samples = integerArgument(argc, argv, 2, result.samples);
    result.viewTheta = argument(argc, argv, 3, result.viewTheta);
    result.viewPhi = argument(argc, argv, 4, result.viewPhi);
    result.maxDepth = integerArgument(argc, argv, 5, result.maxDepth);
    result.opticalThickness = argument(argc, argv, 6, result.opticalThickness);
    result.seed = integerArgument(argc, argv, 7, result.seed);
    result.mediumAlbedo = argument(argc, argv, 8, result.mediumAlbedo);
    result.g = argument(argc, argv, 9, result.g);
    result.coatAlphaX = argument(argc, argv, 10, result.coatAlphaX);
    result.coatAlphaY = argument(argc, argv, 11, result.coatAlphaY);
    result.coatIor = argument(argc, argv, 12, result.coatIor);
    result.baseAlphaX = argument(argc, argv, 13, result.baseAlphaX);
    result.baseAlphaY = argument(argc, argv, 14, result.baseAlphaY);
    result.tangentRotation = argument(argc, argv, 15, result.tangentRotation);
    result.eta[0] = argument(argc, argv, 16, result.eta[0]);
    result.eta[1] = argument(argc, argv, 17, result.eta[1]);
    result.eta[2] = argument(argc, argv, 18, result.eta[2]);
    result.k[0] = argument(argc, argv, 19, result.k[0]);
    result.k[1] = argument(argc, argv, 20, result.k[1]);
    result.k[2] = argument(argc, argv, 21, result.k[2]);
    return result;
}

Vector3f direction(Float thetaDegrees, Float phiDegrees) {
    const Float theta = thetaDegrees * Pi / 180.0f;
    const Float phi = phiDegrees * Pi / 180.0f;
    const Float sinTheta = std::sin(theta);
    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), std::cos(theta));
}

Vector3f toBaseFrame(Vector3f w, Float rotation) {
    const Float c = std::cos(rotation);
    const Float s = std::sin(rotation);
    return Vector3f(c * w.x + s * w.y, -s * w.x + c * w.y, w.z);
}

SampledSpectrum spectrum(const std::array<Float, 3> &value) {
    const Float samples[NSpectrumSamples] = {value[0], value[1], value[2], value[2]};
    return SampledSpectrum(pstd::span<const Float>(samples, NSpectrumSamples));
}

void printResponse(Float theta, Float phi, const SampledSpectrum &value, Float cosine) {
    std::cout << "theta=" << std::setw(6) << theta << " phi=" << std::setw(6) << phi
              << " response=" << value[0] * cosine << "," << value[1] * cosine << ","
              << value[2] * cosine << "\n";
}

}  // namespace

int main(int argc, char **argv) {
    const ProbeArguments args = parseArguments(argc, argv);
    if (args.material != "diffuse" && args.material != "conductor") {
        std::cerr << "material must be 'diffuse' or 'conductor'\n";
        return 2;
    }

    PBRTOptions options;
    options.quiet = true;
    options.seed = args.seed;
    InitPBRT(options);

    const TrowbridgeReitzDistribution coatDistribution(args.coatAlphaX, args.coatAlphaY);
    const DielectricBxDF coat(args.coatIor, coatDistribution);
    const TrowbridgeReitzDistribution conductorDistribution(args.baseAlphaX, args.baseAlphaY);
    const ConductorBxDF conductor(conductorDistribution, spectrum(args.eta), spectrum(args.k));
    const CoatedDiffuseBxDF coatedDiffuse(
        coat, DiffuseBxDF(SampledSpectrum(0.5f)), args.opticalThickness,
        SampledSpectrum(args.mediumAlbedo), args.g, args.maxDepth, args.samples);
    const CoatedConductorBxDF coatedConductor(
        coat, conductor, args.opticalThickness, SampledSpectrum(args.mediumAlbedo), args.g,
        args.maxDepth, args.samples);

    // Multiple azimuths make anisotropy and tangent rotation observable.
    constexpr Float lightDirections[][2] = {{-60.0f, 0.0f},
                                             {-35.0f, 45.0f},
                                             {-20.0f, 90.0f},
                                             {0.0f, 0.0f},
                                             {20.0f, 135.0f},
                                             {35.0f, 45.0f},
                                             {50.0f, 90.0f},
                                             {60.0f, 180.0f},
                                             {70.0f, 270.0f}};
    const Vector3f viewWorld = direction(args.viewTheta, args.viewPhi);
    const Vector3f view = args.material == "conductor"
        ? toBaseFrame(viewWorld, args.tangentRotation)
        : viewWorld;

    std::cout << std::setprecision(9);
    for (const auto &angles : lightDirections) {
        const Vector3f lightWorld = direction(angles[0], angles[1]);
        const Vector3f light = args.material == "conductor"
            ? toBaseFrame(lightWorld, args.tangentRotation)
            : lightWorld;
        const SampledSpectrum value = args.material == "conductor"
            ? coatedConductor.f(view, light, TransportMode::Radiance)
            : coatedDiffuse.f(view, light, TransportMode::Radiance);
        printResponse(angles[0], angles[1], value, std::abs(light.z));
    }

    CleanupPBRT();
    return 0;
}
