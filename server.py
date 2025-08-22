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

PARAMETERS = {}

# address = "ipc:///tmp/dinov2.ipc"
address = sys.argv[1]


def setup(
    repo_or_dir: str = "facebookresearch/dinov2", model: str = "dinov2_vits14_lc"
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
    loaded_model = processor = torch.hub.load(repo_or_dir, model).to(device)

    processor = partial(process, processor=loaded_model)

    PARAMETERS["repo_or_dir"] = repo_or_dir
    PARAMETERS["model"] = model

    info = {"device": device, **PARAMETERS}
    return processor, info


def process(img: numpy.ndarray, processor: Callable) -> dict:
    """Process an image and masks to generate a graph-based tracking representation.

    Parameters
    ----------
    img : array-like
        The input image data.
    processor : torch.Model
        Loaded torch model

    Returns
    -------
    dict
        A dictionary containing the edge table representation of the tracking graph.
    """
    torch_tensor = torch.from_numpy(img).float().cuda()
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
    from nahual_server import responder

    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"Server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_wsetup = partial(responder, setup=setup)
            nursery.start_soon(responder_wsetup, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
