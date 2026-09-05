from ncls.learning.methods.metal.native_uv import UVGroup, UVMapping
from ncls.learning.methods.metal.spatial_encoder import RawSlot
from ncls.learning.methods.metal.spatial_schedule import freeze_spatial_split, spatial_rf_cells


def test_frozen_raw_rf_split_is_disjoint_and_independent_of_resume_request():
    slots = (RawSlot(0, (2048, 2048), ("roughness",)),)
    groups = (UVGroup(UVMapping("native", (1., 0., 0., 0., 1., 0.),
                               "nonrepeat", 8., 1., 935), (0,)),)
    options = {"spatial_train_tiles": 3, "spatial_validation_tiles": 2,
               "spatial_core_texels": 16, "spatial_split_seed": 2026090501}
    train, validation = freeze_spatial_split(slots, groups, options, 4., True)
    resumed = freeze_spatial_split(slots, groups, {**options, "logical_request_index": 71,
                                                  "validation": True}, 4., True)
    assert tuple(c for c, _ in train) == tuple(c for c, _ in resumed[0])
    assert tuple(c for c, _ in validation) == tuple(c for c, _ in resumed[1])
    heldout = [spatial_rf_cells(bundle) for _, bundle in validation]
    assert heldout[0] and heldout[1] and not (heldout[0] & heldout[1])
    for _, bundle in train:
        cells = spatial_rf_cells(bundle)
        assert cells and all(not (cells & selected) for selected in heldout)
