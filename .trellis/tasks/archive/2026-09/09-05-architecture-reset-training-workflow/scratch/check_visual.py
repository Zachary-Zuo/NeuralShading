from pathlib import Path
import json
import numpy as np
import pyexr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import sys
run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/nvidia-layer-stack-smoke/260905-172009-9d5b6c')
paths = sorted(run.glob('eval/*/capture.json'), key=lambda path: path.stat().st_mtime)
path = paths[-1]
capture = json.loads(path.read_text(encoding='utf-8'))
assert all(slot['status'] == 'ready' for slot in capture['slots'])
for slot in capture['slots']:
    assert slot['spp'] == slot['target_spp']
for name in ('slot_0_linear', 'slot_1_linear', 'difference_linear'):
    filename = capture['files'].get(name)
    if filename:
        values = pyexr.read(str(path.parent / filename))
        assert np.isfinite(values).all()
        print(name, values.shape, float(values.mean()))
events = EventAccumulator(str(run / 'tensorboard')).Reload()
assert events.Tags()['images']
print('capture', path)
print('slots', [(slot['status'], slot['mode'], slot['spp']) for slot in capture['slots']])
print('images', events.Tags()['images'])
print('display', path.parent / capture['files']['display'])
