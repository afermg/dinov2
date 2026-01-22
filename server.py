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
from nahual.preprocess import pad_channel_dim, validate_input_shape
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
    """Process a tensor of pixels using a given processor.

    This function validates the input shape, pads the channel dimension if
    necessary, converts the data to a PyTorch tensor, and runs it through
    the provided processor on a CUDA device.

    Parameters
    ----------
    pixels : numpy.ndarray
        The input image data as a NumPy array. The expected shape is
        (batch, channels, z, y, x).
    processor : Callable
        A PyTorch model or other callable that accepts a PyTorch tensor and
        returns the processed result.
    expected_tile_size : tuple[int]
        A tuple representing the expected size of the last two dimensions (yx).
    expected_channels : int
        The number of channels the input tensor should have. If the input
        `pixels` has fewer channels, it will be padded.

    Returns
    -------
    numpy.ndarray
        The result from the processor.
    """
    input_channels = pixels.shape[2]

    # Case when # channels < 3
    # if input_channels > expected_channels:
    #     pixels = pixels[:, :, channels]

    _, input_channels, _, *input_yx = pixels.shape

    validate_input_shape(input_yx, expected_tile_size)

    # This one is necessary to have the correct shape of all dims
    # [N, 3, M*14, M*14] (divisible by 14)
    pixels = pad_channel_dim(pixels, expected_channels)

    torch_tensor = torch.from_numpy(pixels).float().cuda().to(device)

    print(f"Input shape {torch_tensor.shape}")
    result = processor(torch_tensor)

    return result


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
