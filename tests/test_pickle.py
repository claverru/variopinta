from __future__ import annotations

import copy
import importlib.util
import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import variopinta as R
from variopinta import _variopinta
from variopinta.targets import _route
from variopinta.transforms import _TRANSFORM_CATALOG

from tests._helpers import image


class PickleTests(unittest.TestCase):
    def test_catalog_and_empty_round_trips(self):
        source = image(13, 19)
        configurations = [[], *[[fixture(0.5)] for _, fixture in _TRANSFORM_CATALOG.values()]]
        for transforms in configurations:
            reference = R.Pipeline(transforms)
            for pipeline in (reference, reference.compile()):
                for protocol in (4, 5):
                    with self.subTest(
                        transforms=transforms, mode=type(pipeline), protocol=protocol
                    ):
                        for _ in range(3):
                            pipeline(source)
                        restored = pickle.loads(pickle.dumps(pipeline, protocol=protocol))
                        self.assertIs(type(restored), type(pipeline))
                        self.assertEqual(restored.transforms, pipeline.transforms)
                        self.assertEqual(restored.seed, pipeline.seed)
                        self.assertEqual(restored.explain(), pipeline.explain())
                        self.assertIsNot(restored._pipeline, pipeline._pipeline)
                        for _ in range(3):
                            np.testing.assert_array_equal(restored(source), pipeline(source))

    def test_sequence_failures_wraparound_and_compile(self):
        source = image(13, 19)
        reference = R.Pipeline([R.RandomCrop(7, 9), R.GaussianNoise()], seed=42)
        for pipeline in (reference, reference.compile()):
            state = pipeline.__getstate__()
            state["next_key"] = 2**64 - 2
            pipeline.__setstate__(state)
            for key in (None, 71, None, 3, None):
                restored = pickle.loads(pickle.dumps(pipeline))
                before = restored.__getstate__()["next_key"]
                with self.assertRaises(ValueError):
                    restored(image(1, 1))
                self.assertEqual(restored.__getstate__()["next_key"], before)
                np.testing.assert_array_equal(restored(source, key=key), pipeline(source, key=key))
            self.assertEqual(pipeline.__getstate__()["next_key"], 1)
        self.assertEqual(reference.compile().__getstate__()["next_key"], 0)

    def test_port_graph_carriers_and_delivery_failure(self):
        source = image(13, 19)
        labels = source[:, :, 0].copy()
        for carrier in (R.Array(), R.Encoded(), R.Path()):
            rgb = R.ReturnArray(name="rgb")
            encoded = R.Encode("png", compression=1, name="encoded")
            written = R.Write("png", compression=1, name="written")
            photo = R.Image(carrier=carrier, outputs=(rgb, encoded, written), name="photo")
            mask = R.Mask(outputs=R.ReturnArray(name="classes"), fill=17, name="mask")
            reference = R.Pipeline([R.HorizontalFlip(0.5)], seed=42, targets=(photo, mask))
            for pipeline in (reference, reference.compile()):
                for protocol in (4, 5):
                    with self.subTest(carrier=carrier, mode=type(pipeline), protocol=protocol):
                        graph = pickle.loads(
                            pickle.dumps((pipeline, photo, mask, rgb, written), protocol)
                        )
                        restored, new_photo, new_mask, new_rgb, new_written = graph
                        self.assertIs(restored.targets[0], new_photo)
                        self.assertIs(restored.targets[1], new_mask)
                        self.assertIs(new_photo.outputs[0], new_rgb)
                        self.assertIs(new_photo.outputs[2], new_written)
                        with TemporaryDirectory() as directory:
                            root = Path(directory)
                            data = source
                            if isinstance(carrier, R.Encoded | R.Path):
                                data = R.encode_image(source, format="png")
                            if isinstance(carrier, R.Path):
                                (root / "input.png").write_bytes(data)
                                data = root / "input.png"
                            before = restored.__getstate__()["next_key"]
                            with self.assertRaises(ValueError):
                                restored(
                                    photo=new_photo.bind(
                                        data, new_written.bind(root / "missing" / "out.png")
                                    ),
                                    mask=new_mask.bind(labels),
                                )
                            self.assertEqual(restored.__getstate__()["next_key"], before)
                            with self.assertRaises(OSError):
                                restored(
                                    photo=new_photo.bind(
                                        data, new_written.bind(root / ("x" * 240 + ".png"))
                                    ),
                                    mask=new_mask.bind(labels),
                                )
                            self.assertEqual(restored.__getstate__()["next_key"], before)
                            actual = restored(
                                photo=new_photo.bind(data, new_written.bind(root / "new.png")),
                                mask=new_mask.bind(labels),
                            )
                            expected = pipeline(
                                photo=photo.bind(data, written.bind(root / "old.png")),
                                mask=mask.bind(labels),
                            )
                            np.testing.assert_array_equal(
                                actual[new_photo][new_rgb], expected.photo.rgb
                            )
                            np.testing.assert_array_equal(
                                actual.mask.classes, expected.mask.classes
                            )
                            self.assertEqual(actual.photo.encoded, expected.photo.encoded)
                            self.assertEqual(
                                (root / "new.png").read_bytes(), (root / "old.png").read_bytes()
                            )
                            with self.assertRaisesRegex(ValueError, "different port"):
                                restored(
                                    photo=photo.bind(data, written.bind(root / "bad.png")),
                                    mask=new_mask.bind(labels),
                                )

    def test_copy_semantics_and_independent_sequence(self):
        target = R.Image(name="image", outputs=R.ReturnArray(name="rgb"))
        reference = R.Pipeline([R.GaussianNoise()], targets=target, seed=42)
        source = image(7, 11)
        for pipeline in (reference, reference.compile()):
            pipeline(image=target.bind(source))
            for copier in (copy.copy, copy.deepcopy):
                restored = copier(pipeline)
                if copier is copy.copy:
                    self.assertIs(restored.targets[0], target)
                else:
                    self.assertIsNot(restored.targets[0], target)
                self.assertIsNot(restored._pipeline, pipeline._pipeline)
                count = pipeline.__getstate__()["next_key"]
                actual = restored(image=restored.targets[0].bind(source)).image.rgb
                expected = pipeline(image=target.bind(source), key=count).image.rgb
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(pipeline.__getstate__()["next_key"], count)
                self.assertEqual(restored.__getstate__()["next_key"], count + 1)

    def test_invalid_state_does_not_replace_live_state(self):
        reference = R.Pipeline([R.Invert()], seed=42)
        for pipeline in (reference, reference.compile()):
            valid = pipeline.__getstate__()
            invalid = [None, (), {}, {**valid, "extra": 1}]
            invalid.extend({k: v for k, v in valid.items() if k != field} for field in valid)
            for field in ("seed", "next_key"):
                invalid.extend(
                    {**valid, field: value} for value in (None, True, -1, 2**64, 1.5, "1")
                )
            for field, values in {
                "version": (0, 2, True, 1.0, "1"),
                "explicit_targets": (0, 1, None, "false"),
                "transforms": ([], (object(),)),
                "targets": ([], (), (object(),), (R.Mask(),)),
            }.items():
                invalid.extend({**valid, field: value} for value in values)
            for state in invalid:
                with self.subTest(mode=type(pipeline), state=state):
                    with self.assertRaises((TypeError, ValueError)):
                        pipeline.__setstate__(state)
                    self.assertEqual(pipeline.__getstate__(), valid)
            bad = R.HorizontalFlip()
            object.__setattr__(bad, "p", 2.0)
            with self.assertRaises(ValueError):
                pipeline.__setstate__({**valid, "transforms": (bad,)})
            target = R.Image(name="image", outputs=R.ReturnArray(name="rgb"))
            with self.assertRaisesRegex(ValueError, "same target"):
                pipeline.__setstate__(
                    {**valid, "explicit_targets": True, "targets": (target, target)}
                )
            with self.assertRaisesRegex(ValueError, "name"):
                pipeline.__setstate__({**valid, "explicit_targets": True})

    def test_native_restore_validates_counter_and_semantics(self):
        routes = [_route(R.Image())]
        for counter in (True, -1, 2**64, 1.5, None):
            with self.assertRaises(ValueError):
                _variopinta.Pipeline._restore([], 42, "compiled", routes, counter)
            with self.assertRaises(ValueError):
                _variopinta.Pipeline._restore([], counter, "compiled", routes, 0)
        with self.assertRaises(ValueError):
            _variopinta.Pipeline._restore(
                [{"type": "HorizontalFlip", "p": 2.0}], 42, "compiled", routes, 0
            )
        native = _variopinta.Pipeline._restore([], 42, "compiled", routes, 2**64 - 1)
        self.assertEqual(native._snapshot_next_key(), 2**64 - 1)
        native.apply_targets((image(3, 5),))
        self.assertEqual(native._snapshot_next_key(), 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch is not installed")
    def test_tensor_output_graph(self):
        import torch

        tensor = R.ReturnTensor(name="tensor")
        target = R.Image(outputs=tensor, name="image")
        reference = R.Pipeline([R.Normalize()], seed=42, targets=target)
        source = image(7, 11)
        for pipeline in (reference, reference.compile()):
            expected = pipeline(image=target.bind(source))[target][tensor]
            for protocol in (4, 5):
                restored, port, output = pickle.loads(
                    pickle.dumps((pipeline, target, tensor), protocol)
                )
                actual = restored(image=port.bind(source))[port][output]
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.dtype, torch.float32)

    def test_snapshot_concurrency_in_bounded_process(self):
        subprocess.run(
            [sys.executable, "-m", "tests._pickle_workers", "threads"], check=True, timeout=60
        )


if __name__ == "__main__":
    unittest.main()
