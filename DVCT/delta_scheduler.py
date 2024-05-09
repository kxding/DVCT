import torch


def get_delta_weights(
        time_step=50,
        inclination="pose",
        mode="linear",
        coef_ref=1.0,
        coef_src=1.0,
):
    if inclination == "pose":
        if mode == "linear":
            _delta_weights_src = torch.linspace(0, 1, time_step)
            _delta_weights_ref = torch.linspace(1, 0, time_step)
        elif mode == "cosine":
            _delta_weights_src = torch.cos(
                torch.linspace(1, 0, time_step) * 3.14159265
            ) / 2 + 0.5
            _delta_weights_ref = torch.cos(
                torch.linspace(0, 1, time_step) * 3.14159265
            ) / 2 + 0.5
        else:
            raise ValueError("Invalid mode")
    elif inclination == "texture":
        if mode == "linear":
            _delta_weights_src = torch.linspace(1, 0, time_step)
            _delta_weights_ref = torch.linspace(1, 0, time_step)
        elif mode == "cosine":
            _delta_weights_src = torch.cos(
                torch.linspace(0, 1, time_step) * 3.14159265
            ) / 2 + 0.5
            _delta_weights_ref = torch.cos(
                torch.linspace(0, 1, time_step) * 3.14159265
            ) / 2 + 0.5
        else:
            raise ValueError("Invalid mode")
    elif inclination == "none":
        _delta_weights_src = torch.ones(time_step)
        _delta_weights_ref = torch.linspace(1, 0, time_step)
    else:
        raise ValueError("Invalid inclination")
    _delta_weights_ref *= coef_ref
    _delta_weights_src *= coef_src
    _delta_weights = torch.stack([_delta_weights_src, _delta_weights_ref], dim=1)
    return _delta_weights
