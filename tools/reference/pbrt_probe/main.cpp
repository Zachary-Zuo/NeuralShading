#include <pbrt/bxdfs.h>
#include <pbrt/options.h>
#include <pbrt/pbrt.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace pbrt;

static Vector3f direction(Float degrees) {
    const Float radians = degrees * Pi / 180.0f;
    return Vector3f(std::sin(radians), 0.0f, std::cos(radians));
}

int main(int argc, char **argv) {
    const int samples = argc > 1 ? std::atoi(argv[1]) : 262144;
    const Float viewAngle = argc > 2 ? std::atof(argv[2]) : 20.0f;
    const int maxDepth = argc > 3 ? std::atoi(argv[3]) : 32;
    const Float opticalThickness = argc > 4 ? std::atof(argv[4]) : 1e-6f;
    const int seed = argc > 5 ? std::atoi(argv[5]) : 1;
    const Float mediumAlbedo = argc > 6 ? std::atof(argv[6]) : 0.0f;
    const Float g = argc > 7 ? std::atof(argv[7]) : 0.0f;

    PBRTOptions options;
    options.quiet = true;
    options.seed = seed;
    InitPBRT(options);

    const TrowbridgeReitzDistribution distribution(0.12f, 0.12f);
    const DielectricBxDF coat(1.5f, distribution);
    const CoatedDiffuseBxDF layered(
        coat,
        DiffuseBxDF(SampledSpectrum(0.5f)),
        opticalThickness,
        SampledSpectrum(mediumAlbedo),
        g,
        maxDepth,
        samples);

    const Float angles[] = {-55.0f, -20.0f, 0.0f, 35.0f, 60.0f};
    std::cout << std::setprecision(9);
    for (Float lightAngle : angles) {
        const SampledSpectrum value = layered.f(
            direction(viewAngle), direction(lightAngle), TransportMode::Radiance);
        const SampledSpectrum direct = coat.f(
            direction(viewAngle), direction(lightAngle), TransportMode::Radiance);
        std::cout << "angle=" << std::setw(6) << lightAngle
                  << " f=" << value[0]
                  << " direct_response=" << direct[0] * std::abs(direction(lightAngle).z)
                  << " response=" << value[0] * std::abs(direction(lightAngle).z)
                  << "\n";
    }

    CleanupPBRT();
    return 0;
}
