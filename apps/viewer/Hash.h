#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace ncls
{
std::string sha256Hex(const void* data, size_t size);
std::string sha256FileHex(const std::filesystem::path& path);
} // namespace ncls
