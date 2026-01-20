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

guardrail_shapes = {
    "dinov2": (3, (1, 420, 420)),
}


def setup(
    repo_or_dir: str = "facebookresearch/dinov2",
    model_name: str = "dinov2_vits14_lc",
    execution_params: dict = {},
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model = processor = torch.hub.load(repo_or_dir, model_name).to(device)

    setup_kwargs = dict()

    setup_kwargs["repo_or_dir"] = repo_or_dir
    setup_kwargs["model_name"] = model_name

    expected_channels, expected_zyx = guardrail_shapes[model_name.split("_")[0]]
    execution_params["expected_channels"] = expected_channels
    execution_params["expected_zyx"] = expected_zyx

    processor = partial(process, processor=loaded_model, **execution_params)
    info = {"device": device, **setup_kwargs}
    return processor, info


def process(
    pixels: numpy.ndarray,
    processor: Callable,
    expected_zyx: tuple[int],
    expected_channels: int,
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
    expected_zyx : tuple[int]
        A tuple representing the expected spatial dimensions (depth, height, width)
        of the input pixels.
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
    if input_channels > expected_channels:
        pixels = pixels[:, :, channels]

    _, input_channels, *input_zyx = pixels.shape

    validate_input_shape(input_zyx, expected_zyx)

    pixels = pad_channel_dim(pixels, expected_channels)

    torch_tensor = torch.from_numpy(pixels).float().cuda()

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
