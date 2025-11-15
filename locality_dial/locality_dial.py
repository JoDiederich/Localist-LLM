import torch


class LocalityDial:
    """
    A simple wrapper for a scalar 'locality dial' parameter in [0, 1].

    - 0.0 = fully global (no locality bias)
    - 1.0 = strongly local (maximum locality bias)

    This is a minimalist, educational implementation intended to illustrate
    the *concept* of tunable locality. Proprietary methods are intentionally
    omitted.
    """

    def __init__(self, initial_value: float = 0.0, trainable: bool = False):
        # clamp to valid range
        value = torch.tensor(float(initial_value)).clamp(0.0, 1.0)

        if trainable:
            # trainable parameter (optional)
            self.value = torch.nn.Parameter(value)
        else:
            # non-trainable tensor
            self.value = value

    def set(self, new_value: float):
        """Set the dial to a new value in [0, 1]."""
        with torch.no_grad():
            self.value[...] = float(new_value)
            self.value.clamp_(0.0, 1.0)

    def get(self) -> torch.Tensor:
        """Return the dial value (0-D tensor)."""
        return self.value

    def __repr__(self):
        return f"LocalityDial(value={self.value.item():.3f})"
