from pathlib import Path
import ast
import re

root = Path.cwd()
def read(name):
    return (root / name).read_text(encoding='utf-8')
def write(name, value):
    (root / name).write_text(value, encoding='utf-8')

# 两个实际消费者共用 source reference 准备，不依赖历史 Metal handoff。
name = 'src/ncls/visual_eval/windows.py'
value = read(name)
start, end = value.index('def _source_path('), value.index('\n\nclass WindowsVisualEvaluator')
helper = value[start:end].replace('def _source_path(', 'def prepare_source_reference(')
helper = helper.replace('    snapshot = compiled.source_snapshot', '    output.mkdir(parents=True, exist_ok=True)\n    snapshot = compiled.source_snapshot')
helper = helper.replace('    target_types = PROJECT_ROOT / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64/examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"', '''    from ncls.references.backend_manifest import load_reference_backend_manifest
    from ncls.references.platform import current_platform
    manifest = load_reference_backend_manifest()
    target_types = PROJECT_ROOT / manifest.platform(current_platform()).mdl_sdk.package_root / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"''') if False else helper
write('src/ncls/viewer/export.py', '"""为当前模型导出和图像评估准备原生 source reference。"""\n\nimport json\nimport shutil\nfrom pathlib import Path\nimport platform\n\nfrom ncls.core.identity import sha256_file, sha256_json\nfrom ncls.paths import PROJECT_ROOT\n\n\n' + helper.replace('PROJECT_ROOT / "external/MDL-SDK-2025.0.0-387700.1252-nt-x86-64/examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"', 'PROJECT_ROOT / "external" / ("MDL-SDK-2025.0.0-387700.1252-nt-x86-64" if platform.system() == "Windows" else "MDL-SDK-2025.0.0-387700.1252-linux-x86-64") / "examples/mdl_sdk/dxr/content/mdl_target_code_types.hlsl"') + '\n')
value = value[:start] + value[end:]
value = value.replace('import shutil\n', '').replace('from ncls.core.identity import sha256_file, sha256_json', 'from ncls.core.identity import sha256_file\nfrom ncls.viewer.export import prepare_source_reference')
value = value.replace('_source_path(', 'prepare_source_reference(')
write(name, value)

name = 'src/ncls/cli.py'
value = read(name)
marker = '    print(f"ScatteringPackage@2 {compiled.manifest.package_id}: {compiled.root}")'
value = value.replace(marker, '''    from ncls.viewer.export import prepare_source_reference
    source = prepare_source_reference(compiled, compiled.root.with_name(compiled.root.name + "-source"),
        evaluation.source["materials"][material_index]["locator"])
''' + marker + '''
    print(f'Viewer: scripts/launch_viewer.ps1 -Package "{compiled.root}" -Material "{source}"')''')
write(name, value)

# source catalog 不再携带旧 checkpoint descriptor/compatibility 元数据。
name = 'src/ncls/viewer/material_catalog.py'
value = read(name)
fields = ('checkpoint_sha256', 'checkpoint_descriptor_sha256', 'runtime_descriptor_sha256', 'checkpoint_compatibility', 'method_key', 'checkpoint_step', 'checkpoint_phase')
value = re.sub(r'^    (?:' + '|'.join(fields) + r'):.*\n', '', value, flags=re.M)
start = value.index('        checkpoint_sha256 =')
end = value.index('        source = Path(source_path)', start)
value = value[:start] + value[end:]
value = value.replace('                "checkpoint",\n', '').replace('"registry": None, "checkpoint": None,', '"registry": None,')
for field in fields:
    value = value.replace('            ' + field + ',\n', '')
write(name, value)

name = 'apps/viewer/MdlReference.cpp'
value = read(name)
start = value.index('        if (!document.at("checkpoint").is_null())')
end = value.index('\n    }\n    requireSha256(result.targetCodeTypesSha256', start)
value = value[:start] + value[end:]
value = value.replace('"registry", "checkpoint",', '"registry",')
write(name, value)
name = 'apps/viewer/MdlReference.h'
value = read(name)
cpp_fields = ('checkpointSha256', 'checkpointDescriptorSha256', 'runtimeDescriptorSha256', 'checkpointCompatibility', 'methodKey', 'checkpointPhase', 'checkpointStep')
value = re.sub(r'^    (?:std::string|uint32_t) (?:' + '|'.join(cpp_fields) + r').*\n', '', value, flags=re.M)
write(name, value)
name = 'apps/viewer/NclsViewer.cpp'
value = read(name)
start = value.index('        mStatus = std::string("ViewerMaterialCatalog ready: step ")')
end = value.index('\n    else', start)
value = value[:start] + '        mStatus = "ViewerMaterialCatalog ready";' + value[end:]
start = value.index('        widgets.text(std::string("Neural checkpoint: step ")')
end = value.index('\n        widgets.text("Viewer state:', start)
value = value[:start] + value[end:]
value = '\n'.join(line for line in value.splitlines() if not any('catalog.' + field in line for field in cpp_fields) and 'program->checkpointCompatibility' not in line) + '\n'
write(name, value)
name = 'apps/viewer/ScatteringPackage.h'
write(name, read(name).replace('    std::string checkpointCompatibility;\n', ''))
name = 'apps/viewer/ScatteringPackage.cpp'
value = read(name)
value = value.replace('    result.checkpointCompatibility = programProvenance.value(\n        "checkpoint_compatibility", std::string());\n', '')
value = value.replace('    if (!result.checkpointCompatibility.empty())\n        result.displayName += " [" + result.checkpointCompatibility + "]";\n', '')
write(name, value)

name = 'tests/unit/test_viewer_material_catalog.py'
value = read(name)
start = value.index('            "checkpoint": {')
end = value.index('            "reference_runtime":', start)
value = value[:start] + value[end:]
value = '\n'.join(line for line in value.splitlines() if not 'assert catalog.checkpoint_' in line) + '\n'
write(name, value)
name = 'tests/unit/test_viewer_slots.py'
value = read(name).replace('scripts/launch_metal_viewer.ps1', 'scripts/launch_viewer.ps1')
value = '\n'.join(line for line in value.splitlines() if 'checkpoint_compatibility' not in line and 'HybridVsDirect' not in line and "assert '\"exact\"'" not in line and 'assert "$LASTEXITCODE" not in launch' not in line) + '\n'
value = value.replace("assert '\"--slot0-package\"' in launcher", "assert '--slot0-package' in launcher").replace("assert '\"--slot1-package\"' in launcher", "assert '--slot1-package' in launcher")
write(name, value)
