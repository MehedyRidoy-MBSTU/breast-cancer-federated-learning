from opacus.accountants import RDPAccountant



# ============================================================
# DIFFERENTIAL PRIVACY ACCOUNTANT
# ============================================================


def calculate_privacy_epsilon(
    noise_multiplier,
    max_grad_norm,
    sample_rate,
    steps,
    delta=1e-5,
):

    """
    Calculate epsilon using Opacus RDP accountant.

    Parameters
    ----------
    noise_multiplier : float
        Gaussian noise multiplier (sigma)

    max_grad_norm : float
        Gradient clipping norm

    sample_rate : float
        Client sampling probability

    steps : int
        Total DP optimization steps

    delta : float
        Target delta

    Returns
    -------
    epsilon
    """



    accountant = RDPAccountant()



    for _ in range(
        steps
    ):

        accountant.step(

            noise_multiplier=noise_multiplier,

            sample_rate=sample_rate

        )



    epsilon = accountant.get_epsilon(
        delta
    )



    return epsilon




# ============================================================
# DP-FEDAVG CONFIGURATION REPORT
# ============================================================


def print_privacy_report():

    """
    DP-FedAvg DenseNet-121 privacy configuration
    """


    NOISE_MULTIPLIER = 1.0

    MAX_GRAD_NORM = 1.0

    DELTA = 1e-5


    # BreaKHis configuration

    TOTAL_TRAIN_SAMPLES = 6288

    BATCH_SIZE = 16

    NUM_CLIENTS = 5

    LOCAL_EPOCHS = 1

    ROUNDS = 5



    sample_rate = (

        BATCH_SIZE

        /

        TOTAL_TRAIN_SAMPLES

    )



    steps_per_round = (

        TOTAL_TRAIN_SAMPLES

        //

        BATCH_SIZE

    )



    total_steps = (

        steps_per_round

        *

        LOCAL_EPOCHS

        *

        ROUNDS

    )



    epsilon = calculate_privacy_epsilon(

        noise_multiplier=NOISE_MULTIPLIER,

        max_grad_norm=MAX_GRAD_NORM,

        sample_rate=sample_rate,

        steps=total_steps,

        delta=DELTA

    )



    print("=" * 70)

    print(
        "DIFFERENTIAL PRIVACY ACCOUNTING"
    )

    print("=" * 70)


    print(
        "Algorithm: DP-FedAvg DenseNet-121"
    )


    print(
        f"Noise multiplier: {NOISE_MULTIPLIER}"
    )


    print(
        f"Max gradient norm: {MAX_GRAD_NORM}"
    )


    print(
        f"Delta (δ): {DELTA}"
    )


    print(
        f"Sampling rate: {sample_rate:.6f}"
    )


    print(
        f"Training steps: {total_steps}"
    )


    print()

    print(
        "Privacy guarantee:"
    )


    print(
        f"Epsilon (ε): {epsilon:.4f}"
    )


    print(
        f"Delta (δ): {DELTA}"
    )


    print("=" * 70)



    return epsilon



if __name__ == "__main__":

    print_privacy_report()