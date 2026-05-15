"""Request/Reply is used for synchronous communications where each question is responded with a single answer,
for example remote procedure calls (RPCs).
Like Pipeline, it also can perform load-balancing.
This is the only reliable messaging pattern in the suite, as it automatically will retry if a request is not matched with a response.

"""

import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import channel_chunks_rigid3, validate_input_shape
from nahual.server import responder

# address = "ipc:///tmp/dinov2.ipc"
address = sys.argv[1]


def setup(
    repo_or_dir: str = "facebookresearch/dinov2",
    model_name: str = "dinov2_vits14",
    execution_params: dict = {},
    device: int | None = None,
    pretrained: bool = True,
) -> dict:
    """Set up the repo/dir and configuration, following `torch.hub.load`.

    Parameters
    ----------
    repo_or_dir : str
        The repository or directory from which to load a model.
    model : str
        The name of the pre-trained model to load. Defaults to "general_2d".

    Returns
    -------
    dict
        A dictionary containing the device information and configuration parameters.
    """
    assert torch.cuda.is_available(), "Cuda is not available"
    if device is None:
        device = 0
    device = torch.device(int(device))

    loaded_model = torch.hub.load(repo_or_dir, model_name, pretrained=pretrained).to(
        device
    )
    loaded_model.eval()

    setup_kwargs = dict()

    setup_kwargs["repo_or_dir"] = repo_or_dir
    setup_kwargs["model_name"] = model_name

    execution_params["expected_channels"] = 3
    execution_params["expected_tile_size"] = 14

    processor = partial(
        process, processor=loaded_model, device=device, **execution_params
    )
    info = {"device": str(device), **setup_kwargs}
    return processor, info


def process(
    pixels: numpy.ndarray,
    processor: Callable,
    expected_tile_size: tuple[int],
    expected_channels: int,
    device: torch.device,
) -> numpy.ndarray:
    """Run DINOv2 on an NCZYX numpy array.

    DINOv2 is a rigid 3-channel ImageNet ViT. Inputs with C ≠ 3 are split
    into ``ceil(C/3)`` overlapping 3-channel chunks via
    :func:`nahual.preprocess.channel_chunks_rigid3` (recycling leading
    channels for the trailing chunk), the backbone is run on each chunk,
    and per-chunk cls tokens are concatenated along the feature axis.
    Final shape is ``(N, D · ceil(C/3))``.
    """
    if pixels.ndim != 5:
        raise ValueError(f"Expected NCZYX (5D) array, got shape {pixels.shape}")
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    chunks = channel_chunks_rigid3(pixels)
    outs = []
    # [N, 3, M*14, M*14] (divisible by 14)
    with torch.no_grad():
        for chunk in chunks:
            torch_tensor = torch.from_numpy(chunk).float().cuda().to(device)
            result = processor(torch_tensor)
            if hasattr(result, "detach"):
                result = result.detach().cpu().numpy()
            outs.append(result)
    return numpy.concatenate(outs, axis=1)


async def main():
    """Main function for the asynchronous server.

    This function sets up a nng connection using pynng and starts a nursery to handle
    incoming requests asynchronously.

    Parameters
    ----------
    address : str
        The network address to listen on.

    Returns
    -------
    None
    """

    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"Server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
