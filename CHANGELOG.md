# Changelog

## 0.1.0

- Compile immutable image-augmentation pipelines with 22 transforms, explicit
  execution plans, deterministic keyed runs, and reusable native workspaces.
- Accept NumPy HWC RGB input and provide optional terminal PyTorch conversion.
- Decode, encode, read, and write JPEG and PNG images through native codecs.
- Support CPython 3.10–3.13 on 64-bit x86 Linux with glibc 2.34 or newer.

The Python API is experimental. Minor `0.y.0` releases may change documented
interfaces; patch releases preserve them except where correctness or security
requires a change.
