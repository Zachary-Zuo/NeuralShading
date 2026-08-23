#include <glm/glm.hpp>
#include "openpbr.h"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

float readFloat() {
    float value;
    if (!(std::cin >> value))
        throw std::runtime_error("unexpected end of OpenPBR probe input");
    return value;
}

vec2 readVec2() {
    const float x = readFloat();
    const float y = readFloat();
    return vec2(x, y);
}

vec3 readVec3() {
    const float x = readFloat();
    const float y = readFloat();
    const float z = readFloat();
    return vec3(x, y, z);
}

OpenPBR_ResolvedInputs readResolvedInputs() {
    OpenPBR_ResolvedInputs inputs = openpbr_make_default_resolved_inputs();

    inputs.base_weight = readFloat();
    inputs.base_color = readVec3();
    inputs.base_diffuse_roughness = readFloat();
    inputs.base_metalness = readFloat();

    inputs.subsurface_weight = readFloat();
    inputs.subsurface_color = readVec3();
    inputs.subsurface_radius = readFloat();
    inputs.subsurface_radius_scale = readVec3();
    inputs.subsurface_scatter_anisotropy = readFloat();

    inputs.specular_weight = readFloat();
    inputs.specular_color = readVec3();
    inputs.specular_roughness = readFloat();
    inputs.specular_roughness_anisotropy = readFloat();
    inputs.specular_ior = readFloat();
    inputs.specular_anisotropy_rotation_cos_sin = readVec2();

    inputs.coat_weight = readFloat();
    inputs.coat_color = readVec3();
    inputs.coat_roughness = readFloat();
    inputs.coat_roughness_anisotropy = readFloat();
    inputs.coat_ior = readFloat();
    inputs.coat_darkening = readFloat();
    inputs.coat_anisotropy_rotation_cos_sin = readVec2();

    inputs.fuzz_weight = readFloat();
    inputs.fuzz_color = readVec3();
    inputs.fuzz_roughness = readFloat();

    inputs.transmission_weight = readFloat();
    inputs.transmission_color = readVec3();
    inputs.transmission_depth = readFloat();
    inputs.transmission_scatter = readVec3();
    inputs.transmission_scatter_anisotropy = readFloat();
    inputs.transmission_dispersion_scale = readFloat();
    inputs.transmission_dispersion_abbe_number = readFloat();

    inputs.thin_film_weight = readFloat();
    inputs.thin_film_thickness = readFloat();
    inputs.thin_film_ior = readFloat();

    inputs.emission_luminance = readFloat();
    inputs.emission_color = readVec3();
    inputs.geometry_opacity = readFloat();
    inputs.geometry_thin_walled = readFloat() != 0.0f;

    inputs.geometry_basis.t = readVec3();
    inputs.geometry_basis.b = readVec3();
    inputs.geometry_basis.n = readVec3();
    inputs.geometry_coat_basis.t = readVec3();
    inputs.geometry_coat_basis.b = readVec3();
    inputs.geometry_coat_basis.n = readVec3();
    return inputs;
}

OpenPBR_PreparedBsdf prepare(const OpenPBR_ResolvedInputs &inputs, const vec3 viewDirection) {
    return openpbr_prepare(inputs, vec3(1.0f), OpenPBR_BaseRgbWavelengths_nm,
                           OpenPBR_VacuumIor, viewDirection);
}

void evaluateBatch(int count) {
    for (int index = 0; index < count; ++index) {
        const OpenPBR_ResolvedInputs inputs = readResolvedInputs();
        const vec3 viewDirection = readVec3();
        const vec3 lightDirection = readVec3();
        const OpenPBR_PreparedBsdf prepared = prepare(inputs, viewDirection);
        const vec3 response = openpbr_get_sum_of_diffuse_specular(
            openpbr_eval(prepared, lightDirection));
        const float pdf = openpbr_pdf(prepared, lightDirection);
        std::cout << response.x << " " << response.y << " " << response.z << " " << pdf << "\n";
    }
}

void sampleBatch(int count) {
    for (int index = 0; index < count; ++index) {
        const OpenPBR_ResolvedInputs inputs = readResolvedInputs();
        const vec3 viewDirection = readVec3();
        const vec3 randomSample = readVec3();
        const OpenPBR_PreparedBsdf prepared = prepare(inputs, viewDirection);
        vec3 lightDirection;
        OpenPBR_DiffuseSpecular weight;
        float pdf;
        OpenPBR_BsdfLobeType lobeType;
        openpbr_sample(prepared, randomSample, lightDirection, weight, pdf, lobeType);
        const vec3 sum = openpbr_get_sum_of_diffuse_specular(weight);
        std::cout << lightDirection.x << " " << lightDirection.y << " " << lightDirection.z << " "
                  << sum.x << " " << sum.y << " " << sum.z << " " << pdf << " "
                  << static_cast<unsigned int>(lobeType) << "\n";
    }
}

}  // namespace

int main() {
    try {
        std::string mode;
        int count;
        if (!(std::cin >> mode >> count) || count < 0)
            throw std::runtime_error("expected mode and nonnegative query count");
        std::cout << std::setprecision(9);
        if (mode == "eval")
            evaluateBatch(count);
        else if (mode == "sample")
            sampleBatch(count);
        else
            throw std::runtime_error("mode must be eval or sample");
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
    return 0;
}
