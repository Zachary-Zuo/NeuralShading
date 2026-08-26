import pytest

from ncls.core.source import SourceEditOperation, SourceEditPatch, SourceSnapshot
from ncls.source_materials.families.merl import MerlFamilyDefinition
from ncls.source_materials.merl import MerlMaterial
from ncls.source_materials.families.merl import snapshot_from_merl


def test_patch_rejects_stale_snapshot_and_merl_is_read_only():
    material = MerlMaterial("merl-test", "table.binary", "CC-BY")
    snapshot = snapshot_from_merl(material, source_asset_sha256="a" * 64)
    family = MerlFamilyDefinition()
    view = family.describe_parameters(snapshot)
    assert not view.root.children[0].editable
    patch = SourceEditPatch("b" * 64, (SourceEditOperation("set", "/measurement", "x"),))
    with pytest.raises(ValueError, match="stale"):
        family.apply_edit(snapshot, patch)
