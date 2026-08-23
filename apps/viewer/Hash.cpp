#include "Hash.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>

#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ncls
{
namespace
{
class Algorithm
{
public:
    Algorithm()
    {
        if (BCryptOpenAlgorithmProvider(&mHandle, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
            throw std::runtime_error("BCryptOpenAlgorithmProvider(SHA-256) failed");
    }
    ~Algorithm() { if (mHandle) BCryptCloseAlgorithmProvider(mHandle, 0); }
    BCRYPT_ALG_HANDLE get() const { return mHandle; }

private:
    BCRYPT_ALG_HANDLE mHandle = nullptr;
};

class Hash
{
public:
    explicit Hash(BCRYPT_ALG_HANDLE algorithm)
    {
        ULONG objectSize = 0;
        ULONG transferred = 0;
        if (BCryptGetProperty(
                algorithm,
                BCRYPT_OBJECT_LENGTH,
                reinterpret_cast<PUCHAR>(&objectSize),
                sizeof(objectSize),
                &transferred,
                0) < 0)
            throw std::runtime_error("BCryptGetProperty failed");
        mObject.resize(objectSize);
        if (BCryptCreateHash(algorithm, &mHandle, mObject.data(), objectSize, nullptr, 0, 0) < 0)
            throw std::runtime_error("BCryptCreateHash failed");
    }
    ~Hash() { if (mHandle) BCryptDestroyHash(mHandle); }

    void update(const void* data, size_t size)
    {
        const auto* cursor = static_cast<const uint8_t*>(data);
        while (size > 0)
        {
            const ULONG chunk = static_cast<ULONG>(std::min<size_t>(size, 1u << 30));
            if (BCryptHashData(mHandle, const_cast<PUCHAR>(cursor), chunk, 0) < 0)
                throw std::runtime_error("BCryptHashData failed");
            cursor += chunk;
            size -= chunk;
        }
    }

    std::array<uint8_t, 32> finish()
    {
        std::array<uint8_t, 32> result{};
        if (BCryptFinishHash(mHandle, result.data(), static_cast<ULONG>(result.size()), 0) < 0)
            throw std::runtime_error("BCryptFinishHash failed");
        return result;
    }

private:
    BCRYPT_HASH_HANDLE mHandle = nullptr;
    std::vector<uint8_t> mObject;
};

std::string toHex(const std::array<uint8_t, 32>& digest)
{
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t index = 0; index < digest.size(); ++index)
    {
        result[2 * index] = kHex[digest[index] >> 4];
        result[2 * index + 1] = kHex[digest[index] & 15];
    }
    return result;
}
} // namespace

std::string sha256Hex(const void* data, size_t size)
{
    Algorithm algorithm;
    Hash hash(algorithm.get());
    hash.update(data, size);
    return toHex(hash.finish());
}

std::string sha256FileHex(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file for SHA-256: " + path.string());
    Algorithm algorithm;
    Hash hash(algorithm.get());
    std::vector<char> buffer(1u << 20);
    while (stream)
    {
        stream.read(buffer.data(), buffer.size());
        const auto count = stream.gcount();
        if (count > 0) hash.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!stream.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
    return toHex(hash.finish());
}
} // namespace ncls
