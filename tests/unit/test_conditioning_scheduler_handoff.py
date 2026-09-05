from types import SimpleNamespace

import torch

from ncls.data import OnlineStepRequest, ReferenceScheduler
from ncls.learning.batches import TrainingConditioning, TrainingRouteRequest
from ncls.learning.conditioning_resources import ConditioningResource, ConditioningResources
from ncls.learning.producer import OnlineTrainingProducer
from ncls.references import ReferenceConcurrencyCapability
from tests.unit.test_online_training_producer import _AllValidSession


def test_scheduler_release_preserves_ready_batch_resource_owner():
    producer = object.__new__(OnlineTrainingProducer)
    producer._closed = False
    producer.device = torch.device("cpu")
    producer.config = SimpleNamespace(online_query={})
    producer.session = _AllValidSession()
    producer._request_count = {}
    producer._reference_logical_id = 0
    producer._select_group = lambda request: SimpleNamespace(group_id="group")
    released = []
    class Lease:
        def release(self):
            released.append(True)
    def conditioning(request, group, generator, index):
        resources = ConditioningResources((ConditioningResource("raw", {"raw": torch.ones(1,1,8,8)}, lease=Lease()),))
        wo = torch.tensor([[0.2,0.3,1.]]).expand(request.batch_size,-1)
        return TrainingConditioning("fixture", ("a"*64,),
            {"source_index": torch.zeros(request.batch_size,dtype=torch.int64), "wo": wo},
            {"reference_execution_group_id": group.group_id}, resources,
            {"raw": torch.zeros(request.batch_size,dtype=torch.int64)}), wo
    producer._conditioning = conditioning
    producer._reference_scheduler = ReferenceScheduler(producer._dispatch_evaluator_requests,
        capability=ReferenceConcurrencyCapability("global", False, 2, False), batch_steps=2, ready_capacity=2, maximum_inflight=1)
    route = TrainingRouteRequest("evaluator", "reference-evaluator", 4,1,0,3,{})
    batches = producer.produce_steps((OnlineStepRequest(0,"test",{"evaluator":route}),))
    batch = batches[0]["evaluator"]
    producer._reference_scheduler.assert_idle()
    assert not released
    assert len(batch.conditioning.resources) == 1
    torch.testing.assert_close(batch.conditioning.resources.entries[0].tensors["raw"], torch.ones(1,1,8,8))
    batch.release()
    assert released == [True]
    producer._reference_scheduler.close()
