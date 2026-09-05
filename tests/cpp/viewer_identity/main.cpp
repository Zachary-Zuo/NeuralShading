#include "Hash.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    std::ifstream stream(argv[1], std::ios::binary);
    if (!stream) return 3;
    std::cout << ncls::canonicalJson(nlohmann::json::parse(stream));
}
