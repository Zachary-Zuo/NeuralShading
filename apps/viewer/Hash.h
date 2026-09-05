#pragma once

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace ncls
{
std::string canonicalJson(const nlohmann::json& value);
std::string sha256Json(const nlohmann::json& value);
std::string sha256Hex(const void* data, size_t size);
std::string sha256FileHex(const std::filesystem::path& path);
} // namespace ncls
