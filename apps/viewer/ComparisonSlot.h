#pragma once

#include "ScatteringPackage.h"

#include <cstdint>
#include <string>

namespace ncls
{
enum class SlotMode { PathTracing, Deferred };
enum class SlotStatus { Empty, Loading, Ready, Compiling, Unsupported, Error };

struct ComparisonSlot
{
    const ViewerProgram* program = nullptr;
    SlotMode mode = SlotMode::PathTracing;
    SlotStatus status = SlotStatus::Empty;
    std::string diagnostic;
    uint64_t accumulation = 0;

    void bind(const ViewerProgram* candidate);
};

struct PanelLayout
{
    uint32_t panelWidth;
    uint32_t panelHeight;
    uint32_t dividerWidth;
    uint32_t rightOffset;
};

PanelLayout fixedPanelLayout(uint32_t width, uint32_t height);
} // namespace ncls
