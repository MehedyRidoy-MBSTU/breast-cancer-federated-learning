import torch


def clip_gradients(
    model,
    max_norm=1.0
):
    """
    Clip gradients for DP-SGD.
    """

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    )



def add_gaussian_noise(
    model,
    noise_multiplier=1.0,
    max_norm=1.0
):
    """
    Add Gaussian noise to gradients.
    """

    std = (
        noise_multiplier
        *
        max_norm
    )


    for param in model.parameters():

        if param.grad is not None:

            noise = torch.normal(
                mean=0.0,
                std=std,
                size=param.grad.shape,
                device=param.grad.device
            )


            param.grad += noise



def dp_gradient_step(
    model,
    optimizer,
    max_norm=1.0,
    noise_multiplier=1.0
):
    """
    Complete DP-SGD gradient step.
    """

    clip_gradients(
        model,
        max_norm
    )


    add_gaussian_noise(
        model,
        noise_multiplier,
        max_norm
    )


    optimizer.step()