import torch
import numpy as np

def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True, progress=1.0
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    alpha = int(progress * num_encoding_functions)
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def get_weight(alpha, k):
        if alpha < k:
            weight = 0.0
        elif alpha - k >= 0 and alpha - k < 1:
            weight = (1 - torch.cos(torch.tensor((alpha - k) * math.pi))) / 2
        else:
            weight = 1.0
        return weight

    for i, freq in enumerate(frequency_bands):
        weight = get_weight(alpha, i)
        for func in [torch.sin, torch.cos]:
            encoding.append(weight * func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True, alpha=6
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x, progress=1.0: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling, progress
    )

if __name__ == "__main__":
    # TODO: Test positional embedding
    pass
