from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from ncls.learning.models.plane_evaluator import (
    ARCHITECTURE_ID,
    PlaneFactorizedModelConfig,
    PlaneFactorizedNeuralEvaluator,
)

from .base import LearningPipelineDescriptor
from .dense_evaluator import AnalyticResidualEnergyShapeE1Pipeline, DenseEnergyShapeE1Pipeline


PIPELINE_ID = "plane-factorized-small-mlp-energy-shape-e1@1"
ANALYTIC_RESIDUAL_PIPELINE_ID = "plane-factorized-analytic-residual-energy-shape-e1@1"


class PlaneFactorizedEnergyShapeE1Pipeline(DenseEnergyShapeE1Pipeline):
    feature_contract = {
        "format_name": "ncls.feature-contract",
        "format_version": 1,
        "feature_contract_id": "ncls.direction-disk-pairwise-planes@1",
        "inputs": {
            "prepare": ["material_latent", "wo", "wo_xy_plane"],
            "evaluate": ["prepared_view_code", "wi", "wi_xy_plane", "four_cross_planes"],
        },
        "direction_space": "source-reference-local-upper-hemisphere-disk",
    }
    descriptor = LearningPipelineDescriptor(
        pipeline_id=PIPELINE_ID,
        candidate_id="ncls.plane-tensor-factorization@1",
        research_role="e1-single-material-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.identity-source-adapter@1",
        feature_transform_id="ncls.direction-disk-pairwise-planes@1",
        target_transform_id=DenseEnergyShapeE1Pipeline.target_transform_id,
        representation_id="ncls.pairwise-direction-plane-factorization@1",
        architecture_id=ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-plane-features@1",
        compiler_id="ncls.none-capacity-study@1",
        loss_id="ncls.standardized-log1p-energy-shape-reciprocity@1",
        metric_suite_id="ncls.evaluator-quality-suite@1",
        exporter_id="ncls.plane-factorized-method-bundle-planned@1",
        supported_family_ids=(
            "ncls.layer-stack@1",
            "merl.measured-brdf@1",
            "openpbr.surface@1.1.1",
            "materialx.textured-surface@1",
        ),
        scope="single-material-complete-directional-evaluator",
    )

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        return PlaneFactorizedNeuralEvaluator(
            PlaneFactorizedModelConfig.from_mapping(model_parameters)
        )


class PlaneFactorizedAnalyticResidualE1Pipeline(AnalyticResidualEnergyShapeE1Pipeline):
    feature_contract = PlaneFactorizedEnergyShapeE1Pipeline.feature_contract
    descriptor = LearningPipelineDescriptor(
        pipeline_id=ANALYTIC_RESIDUAL_PIPELINE_ID,
        candidate_id="ncls.plane-tensor-factorization@1",
        research_role="e1-single-material-capacity",
        response_reader_id="ncls.reference-query-store@1",
        partition_policy_id="ncls.query-role-within-state@1",
        source_adapter_id="ncls.layer-stack-direct-top-adapter@1",
        feature_transform_id="ncls.direction-disk-pairwise-planes@1",
        target_transform_id=AnalyticResidualEnergyShapeE1Pipeline.target_transform_id,
        representation_id="ncls.analytic-direct-top-plus-pairwise-plane-residual@1",
        architecture_id=ARCHITECTURE_ID,
        latent_inference_id="ncls.optimized-plane-features@1",
        compiler_id="ncls.none-capacity-study@1",
        loss_id="ncls.standardized-asinh-residual-energy-shape-reciprocity@1",
        metric_suite_id="ncls.evaluator-quality-suite-with-core-ablation@1",
        exporter_id="ncls.plane-factorized-method-bundle-planned@1",
        supported_family_ids=("ncls.layer-stack@1",),
        scope="single-material-complete-directional-evaluator",
    )

    def create_model(self, model_parameters: Mapping[str, Any]) -> nn.Module:
        return PlaneFactorizedNeuralEvaluator(
            PlaneFactorizedModelConfig.from_mapping(model_parameters)
        )
