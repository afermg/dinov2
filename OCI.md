# DINOv2 Nahual OCI image

The Nix flake builds a reproducible, OCI-compatible image archive containing the
same dependency closure used by `nix run`. The resulting image does not require
Nix at runtime and can be loaded by Podman or Docker.

## Build and load

```console
nix build .#oci-image
podman load < result
# Or: docker load < result
```

The local image name is `nahual/dinov2:local`. It listens on Nahual's NNG TCP
endpoint `tcp://0.0.0.0:5555` by default. Override the endpoint by passing one
as the container argument.

## Run

Persisting `/tmp/nahual` avoids downloading Torch Hub sources and model weights
again after the container is replaced.

```console
podman run --rm --name nahual-dinov2 \
  --device nvidia.com/gpu=all \
  -p 5555:5555 \
  -v nahual-dinov2-cache:/tmp/nahual \
  nahual/dinov2:local
```

For Docker with the NVIDIA Container Toolkit, replace the CDI option with
`--gpus all`. DINOv2 also falls back to CPU when no GPU is exposed, so the
`--device`/`--gpus` option may be omitted.

## Full smoke inference

In another environment with Python (Nix is not needed):

```console
python3 -m venv .venv
. .venv/bin/activate
pip install 'nahual==0.0.8' numpy
NAHUAL_ADDRESS=tcp://127.0.0.1:5555 python oci/smoke_test.py
```

The smoke test loads the pretrained ViT-S/14 model, sends an NCZYX tensor over
TCP, performs inference in the container, and validates the `(1, 384)` result.
