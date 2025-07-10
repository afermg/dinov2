"""Request/Reply is used for synchronous communications where each question is responded with a single answer,
for example remote procedure calls (RPCs).
Like Pipeline, it also can perform load-balancing.
This is the only reliable messaging pattern in the suite, as it automatically will retry if a request is not matched with a response.

"""

import json
import sys
import time

import numpy
import pynng
import torch
import trio
from nahual.serial import deserialize_numpy, serialize_numpy

PARAMETERS = {}

# address = "ipc:///tmp/reqrep.ipc"
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
    processor = torch.hub.load(repo_or_dir, model).to(device)

    PARAMETERS["repo_or_dir"] = repo_or_dir
    PARAMETERS["model"] = model

    info = {"device": device, **PARAMETERS}
    return processor, info


async def responder(sock, processor):
    """Asynchronous responder function for handling model setup and data processing.

        This function continuously listens for incoming messages via a socket. It handles two
        modes: initializing a model based on received parameters and processing data using
        an already loaded model.

        Parameters
        ----------
            sock: pynng. (object): The socket object used for receiving and sending messages.

        Returns
        -------
            None: This function does not return a value but sends responses via the socket.

        Raises
        ------
            Exception: If an error occurs during message handling or processing.


    Notes:
        - The function uses JSON for message serialization.
        - The 'setup' function is called to initialize the model.
        - The 'process' function is used to compute results from input data.
    """

    while True:
        if processor is None:
            try:
                msg = await sock.arecv_msg()
                if len(msg.bytes) == 1:
                    print("Exiting")
                    break
                content = msg.bytes.decode()
                parameters = json.loads(content)
                if "model" in parameters:  # Start
                    print("NODE0: RECEIVED REQUEST")
                    processor, info = setup(**parameters)
                    info_str = f"Loaded model with parameters {info}"
                    print(info_str)
                    print("Sending model info back")
                    await sock.asend(json.dumps(info).encode())
                    print("Model loaded. Will wait for data.")

            except Exception as e:
                print(f"Waiting for parameters: {e}")
                time.sleep(1)
        else:
            try:
                # Receive data
                msg = await sock.arecv_msg()
                if len(msg.bytes) == 1:
                    print("Exiting")
                    break
                img = deserialize_numpy(msg.bytes)
                # Add data processing here
                result = process(img, processor=processor)
                result_np = result.cpu().detach().numpy()
                await sock.asend(serialize_numpy(result_np))

            except Exception as e:
                print(f"Waiting for data: {e}")


def process(img: numpy.ndarray, processor) -> dict:
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
    processor = None
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"Server listening on {address}")
        async with trio.open_nursery() as nursery:
            nursery.start_soon(responder, sock, processor)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        # that's the way the program *should* end
        pass
