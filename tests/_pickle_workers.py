from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier

import numpy as np
import variopinta as R

from tests._helpers import image


class Images:
    def __init__(self, path):
        self.path = path
        self.output = R.ReturnTensor(name="tensor")
        self.target = R.Image(carrier=R.Path(), outputs=self.output, name="image")
        self.pipeline = R.Pipeline(
            [R.RandomCrop(7, 9), R.GaussianNoise(), R.Normalize()],
            seed=42,
            targets=self.target,
        ).compile()
        self[(0, 0)]

    def __len__(self):
        return 8

    def __getitem__(self, request):
        epoch, index = request
        result = self.pipeline(image=self.target.bind(self.path), key=epoch * len(self) + index)
        return result[self.target][self.output], epoch, index, os.getpid()


class EpochSampler:
    def __init__(self):
        self.epoch = 0

    def __len__(self):
        return 8

    def __iter__(self):
        return iter((self.epoch, index) for index in reversed(range(8)))


def loaders(context, persistent):
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(1)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "input.png"
        path.write_bytes(R.encode_image(image(13, 19), format="png"))
        dataset = Images(path)
        sampler = EpochSampler()
        loader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            num_workers=2,
            multiprocessing_context=context,
            persistent_workers=persistent,
            timeout=30,
        )
        epoch_pids = []
        for epoch in range(2):
            sampler.epoch = epoch
            pids = set()
            seen = set()
            for tensors, epochs, indices, workers in loader:
                for tensor, actual_epoch, index, pid in zip(
                    tensors, epochs, indices, workers, strict=True
                ):
                    assert actual_epoch.item() == epoch
                    assert tensor.dtype == torch.float32 and tensor.is_contiguous()
                    assert torch.equal(tensor, dataset[(epoch, index.item())][0])
                    seen.add(index.item())
                    pids.add(pid.item())
            assert seen == set(range(8))
            assert len(pids) == 2 and os.getpid() not in pids
            epoch_pids.append(pids)
        if persistent:
            assert epoch_pids[0] == epoch_pids[1]
        del loader


def fork_child(pipeline, connection):
    connection.send(pipeline(image(13, 19)))
    connection.close()


def idle_fork():
    pipeline = R.Pipeline([R.GaussianNoise()], seed=42).compile()
    pipeline(image(13, 19))
    context = mp.get_context("fork")
    parent, child = context.Pipe()
    process = context.Process(target=fork_child, args=(pipeline, child))
    process.start()
    child.close()
    assert parent.poll(15)
    np.testing.assert_array_equal(parent.recv(), pipeline(image(13, 19)))
    process.join(15)
    assert process.exitcode == 0


def threads():
    source = image(512, 769)
    encoded = R.encode_image(source, format="png")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "image.png"
        path.write_bytes(encoded)
        for carrier, data in ((R.Encoded(), encoded), (R.Path(), path)):
            target = R.Image(carrier=carrier, outputs=R.Encode("png", name="png"), name="image")
            reference = R.Pipeline([R.GaussianNoise()], seed=42, targets=target)
            for pipeline in (reference, reference.compile()):
                barrier = Barrier(3)

                def implicit(barrier=barrier, pipeline=pipeline, target=target, data=data):
                    barrier.wait()
                    for _ in range(8):
                        pipeline(image=target.bind(data))

                def snapshots(barrier=barrier, pipeline=pipeline, target=target, data=data):
                    barrier.wait()
                    counts = []
                    for _ in range(16):
                        restored = pickle.loads(pickle.dumps(pipeline))
                        count = restored.__getstate__()["next_key"]
                        counts.append(count)
                        actual = restored(image=restored.targets[0].bind(data)).image.png
                        assert actual == pipeline(image=target.bind(data), key=count).image.png
                    assert counts == sorted(counts)

                def explicit(barrier=barrier, pipeline=pipeline, target=target, data=data):
                    barrier.wait()
                    expected = pipeline(image=target.bind(data), key=91).image.png
                    for _ in range(8):
                        assert pipeline(image=target.bind(data), key=91).image.png == expected

                with ThreadPoolExecutor(3) as pool:
                    futures = [pool.submit(job) for job in (implicit, snapshots, explicit)]
                    for future in futures:
                        future.result()
                assert pipeline.__getstate__()["next_key"] == 8


if __name__ == "__main__":
    if sys.argv[1] == "threads":
        threads()
    elif sys.argv[1] == "fork":
        idle_fork()
    else:
        loaders(sys.argv[1], sys.argv[2] == "persistent")
