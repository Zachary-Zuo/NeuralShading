#include "ComparisonSlot.h"

#include <stdexcept>

namespace ncls
{
void ComparisonSlot::bind(const ViewerProgram* candidate)
{
    program = candidate;
    accumulation = 0;
    diagnostic.clear();
    if (!candidate) { status = SlotStatus::Empty; return; }
    const uint32_t required = mode == SlotMode::PathTracing ? (1u | 2u | 4u | 8u) : (1u | 2u);
    if (!candidate->program || (candidate->program->capabilities & required) != required)
    {
        status = SlotStatus::Unsupported;
        diagnostic = "selected package lacks capabilities required by slot mode";
        return;
    }
    status = SlotStatus::Ready;
}

PanelLayout fixedPanelLayout(uint32_t width, uint32_t height)
{
    if (width < 2u || height == 0u) throw std::runtime_error("viewer extent must be at least 2x1");
    const uint32_t panel = width / 2u;
    const uint32_t divider = width - 2u * panel;
    return {panel, height, divider, panel + divider};
}
} // namespace ncls
