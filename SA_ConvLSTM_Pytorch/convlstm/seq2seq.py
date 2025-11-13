from typing import TypedDict#, NotRequired
import sys
import torch
from torch import nn

dir = '/data/zr/pen/work/smcl/SA_ConvLSTM_Pytorch'
if dir not in sys.path:
    sys.path.append(dir)

from convlstm.model import ConvLSTM, ConvLSTMParams


class Seq2SeqParams(TypedDict):
    input_seq_length: int
    num_layers: int
    num_kernels: int
    # return_sequences: NotRequired[bool]
    return_sequences: bool
    convlstm_params: ConvLSTMParams


class Seq2Seq(nn.Module):
    """The sequence to sequence model implementation using ConvLSTM."""

    def __init__(
        self,
        input_seq_length: int,
        num_layers: int,
        num_kernels: int,
        convlstm_params: ConvLSTMParams,
        return_sequences: bool = False,
        use_layernorm: bool = True,
        layernorm_mode: int = 0, # 0: normal, 1: feature
    ) -> None:
        """

        Args:
            input_seq_length (int): Number of input frames.
            num_layers (int): Number of ConvLSTM layers.
            num_kernels (int): Number of kernels.
            return_sequences (int): If True, the model predict the next frames that is the same length of inputs. If False, the model predicts only one next frame.
        """
        super().__init__()
        self.input_seq_length = input_seq_length
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.return_sequences = return_sequences
        self.use_layernorm = use_layernorm
        self.in_channels = convlstm_params["in_channels"]
        self.kernel_size = convlstm_params["kernel_size"]
        self.padding = convlstm_params["padding"]
        self.activation = convlstm_params["activation"]
        self.frame_size = convlstm_params["frame_size"]
        self.out_channels = convlstm_params["out_channels"]
        self.weights_initializer = convlstm_params["weights_initializer"]

        self.sequential = nn.Sequential()

        # Add first layer (Different in_channels than the rest)
        self.sequential.add_module(
            "convlstm1",
            ConvLSTM(
                in_channels=self.in_channels,
                out_channels=self.num_kernels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation=self.activation,
                frame_size=self.frame_size,
                weights_initializer=self.weights_initializer,
            ),
        )

        if self.use_layernorm:
            if layernorm_mode == 0:
                self.sequential.add_module(
                    "layernorm1",
                    nn.LayerNorm([self.num_kernels, self.input_seq_length, *self.frame_size]),
                )
            elif layernorm_mode == 1:
                self.sequential.add_module(
                    "layernorm1",
                    nn.LayerNorm([*self.frame_size]),
                )

        # Add the rest of the layers
        for layer_idx in range(2, self.num_layers + 1):
            self.sequential.add_module(
                f"convlstm{layer_idx}",
                ConvLSTM(
                    in_channels=self.num_kernels,
                    out_channels=self.num_kernels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    frame_size=self.frame_size,
                    weights_initializer=self.weights_initializer,
                ),
            )

            if self.use_layernorm:
                if layernorm_mode == 0:
                    self.sequential.add_module(
                        f"layernorm{layer_idx}",
                        nn.LayerNorm(
                            [self.num_kernels, self.input_seq_length, *self.frame_size]
                        ),
                    )
                elif layernorm_mode == 1:
                    self.sequential.add_module(
                        f"layernorm{layer_idx}",
                        nn.LayerNorm(
                            [*self.frame_size]
                        ),
                    )

        self.sequential.add_module(
            "conv3d",
            nn.Conv3d(
                in_channels=self.num_kernels,
                out_channels=self.out_channels,
                kernel_size=(3, 3, 3),
                padding="same",
            ),
        )

        self.sequential.add_module("sigmoid", nn.Sigmoid())

    def forward(self, X: torch.Tensor):
        # Forward propagation through all the layers
        output = self.sequential(X)

        if self.return_sequences is True:
            return output

        return output[:, :, -1:, ...]
