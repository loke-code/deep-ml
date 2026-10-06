import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    e_v = torch.linalg.eigvals(matrix)
    e_v = e_v.real
    ev, _ = torch.sort(e_v, descending=True)
    return ev