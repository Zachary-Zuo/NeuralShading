from pathlib import Path

path = Path('src/ncls/learning/methods/metal/data.py')
text = path.read_text(encoding='utf-8')
start = text.index('    def _sample_tensors(')
end = text.index('\ndef _balanced_one_native_texel_offsets(', start)
text = text[:start] + '''    def native_assets(self) -> NativeAssetCollection:
        return self._assets

    def spatial_contract_for_source(self, source_index: int = 0):
        if not 0 <= source_index < len(self.snapshots):
            raise ValueError("Metal source index is out of range")
        return self._spatial_contracts[source_index]

    def sample_tensors(
        self, source_index: torch.Tensor, generator: torch.Generator,
        options: Mapping[str, Any], *, execution_source_indices: Sequence[int] | None = None,
    ) -> AdaptedConditioning:
        from ncls.learning.methods.metal.spatial_schedule import spatial_cohort

        candidates = tuple(range(len(self.snapshots))) if execution_source_indices is None else tuple(execution_source_indices)
        if not candidates or any(index < 0 or index >= len(self.snapshots) for index in candidates):
            raise ValueError("Metal execution cohort has invalid source indices")
        if len({self._spatial_cohort_keys[index] for index in candidates}) != 1:
            raise ValueError("raw Metal sampling requires one CPU-declared UV/resource cohort")
        asset_index, slots, groups = self._spatial_contracts[candidates[0]]
        cohort = spatial_cohort(slots, groups, options)
        count = int(source_index.shape[0])
        x0, y0, x1, y1 = cohort.bounds
        uv = torch.rand((count, 2), generator=generator, device=self.device)
        uv = uv * uv.new_tensor((x1 - x0, y1 - y0)) + uv.new_tensor((x0, y0))
        recipe = options.get("footprint_recipe", "balanced-zero-one-four-texel@1")
        if recipe == "balanced-zero-one-four-texel@1":
            footprints = uv.new_tensor((0., 1., 4.)).repeat((count + 2) // 3)[:count]
            footprints = footprints[torch.randperm(count, generator=generator, device=self.device)]
            maximum_footprint = 4.0
        elif recipe == "point@1":
            footprints, maximum_footprint = uv.new_zeros(count), 0.0
        else:
            raise ValueError("Metal spatial footprint recipe is unsupported")
        footprint = footprints * cohort.footprint_step
        dx = torch.stack((footprint, torch.zeros_like(footprint)), dim=1)
        dy = torch.stack((torch.zeros_like(footprint), footprint), dim=1)
        paired = bool(options.get("paired_uv", False))
        pair_x, pair_y = cohort.pair_step if paired else (0., 0.)
        if paired and options.get("paired_uv_recipe") != "one-native-texel-axis-balanced@1":
            raise ValueError("Metal paired UV recipe is unsupported")
        # 主/pair 的所有 bilinear 邻点和 learned RF 由同一 bundle 覆盖。
        # bounds 不做 frac；seam 交由每个原生 UV 组自己的 address mode 处理。
        bounds = (x0, y0, x1 + pair_x, y1 + pair_y)
        maximum_dx = (maximum_footprint * cohort.footprint_step, 0.)
        maximum_dy = (0., maximum_footprint * cohort.footprint_step)
        key = sha256_json({"cohort": self._spatial_cohort_keys[candidates[0]],
                           "bounds": bounds, "dx": maximum_dx, "dy": maximum_dy})
        plan = self._spatial_tile_schedules.get(key)
        if plan is None:
            plan = build_spatial_bundle(slots, groups, bounds, maximum_dx, maximum_dy)
            # 这里只缓存不含 tensor 的 RF plan，容量固定，绝不缓存 learned feature。
            if len(self._spatial_tile_schedules) >= 32:
                self._spatial_tile_schedules.pop(next(iter(self._spatial_tile_schedules)))
            self._spatial_tile_schedules[key] = plan
        compiler_names = {
            "metal_graph_index": "graph", "metal_schema_index": "schema",
            "metal_recipe_index": "recipe", "metal_identity_index": "metal",
            "metal_finish_index": "finish", "metal_asset_index": "asset",
            "metal_typed_semantic_id": "semantic", "metal_typed_type_id": "type",
            "metal_typed_responsibility_id": "responsibility", "metal_typed_discrete": "discrete",
            "metal_typed_continuous": "continuous", "metal_typed_presence": "presence",
            "metal_canonical_optical": "optical", "metal_access_state": "access",
            "metal_frame_state": "frame", "metal_distribution_id": "distribution",
        }
        values = {name: self._tables[table].index_select(0, source_index)
                  for name, table in compiler_names.items()}
        values.update({"uv": uv, "uv_dx": dx, "uv_dy": dy,
                       "filter_random": torch.rand(count, generator=generator, device=self.device)})
        if paired:
            # effective native texel 步长按真实仿射/Jacobian 得出，原始 query 空间保持不变。
            axis = torch.arange(count, device=self.device) % 2
            axis = axis[torch.randperm(count, generator=generator, device=self.device)]
            offset = torch.stack((torch.where(axis == 0, pair_x, 0.),
                                  torch.where(axis == 1, pair_y, 0.)), dim=1)
            values.update({"paired_uv": uv + offset, "paired_uv_dx": dx, "paired_uv_dy": dy})
        resource = self._assets.acquire_spatial_bundle(asset_index, plan)
        resources = ConditioningResources((resource,))
        try:
            return AdaptedConditioning(values, {
                "metal_registry_identity": self.registry.identity,
                "native_asset_collection_identity": self._assets.collection_id,
                "spatial_bundle_identity": key,
                "spatial_query_bounds": list(cohort.bounds),
                "spatial_pair_step": list(cohort.pair_step),
                "native_uv_groups": len(groups),
                "native_asset_reads": sum(2 * group.mapping.lookup_count for group in groups),
                "footprint_recipe": recipe,
            }, resources, {"metal_spatial": torch.zeros(count, dtype=torch.int64, device=self.device)})
        except BaseException:
            resources.release()
            raise

''' + text[end:]
path.write_text(text, encoding='utf-8', newline='\n')

path = Path('src/ncls/learning/methods/metal/method.py')
text = path.read_text(encoding='utf-8').replace('    "mip_level",\n    "metal_mip_fraction",\n', '')
path.write_text(text, encoding='utf-8', newline='\n')
