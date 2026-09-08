import os

import numpy as np
import scipy as sp

from . import parameters


def generate_signals(p: parameters) -> tuple[np.ndarray, ...]:
    """
    Given desired parameters `p`, generate microphone and echo signals as well
    as the corresponding RIRs.

    It is assumed there is only 1 desired source, 1 noise source and 1 loudspeaker
    for the echo signal.

    Catered towards [`VCTK` corpus](
    https://datashare.ed.ac.uk/items/3ea66d32-4541-4cf4-9571-443cc042466b) for audio and
    [`hearpiece`](https://zenodo.org/records/3733191) for RIRs.

    Parameters
    ----------
    p : parameters
        The parameters struct containing all required information.

    Returns
    -------
    A tuple containing `ndarray`s:
        - The desired signal contribution to the microphone signals
        - The noise contribution to the microphone signals
        - The echo signal contribution to the microphone signals
        - The echo signal itself
        - The echo paths from all loudspeakers to all microphones

    Shape of returned microphone signal contributions: `[T * fs x M]`,
    shape of returned echo paths: `[rir_length x M x L]`. `M` is the number of
    microphones, `L` the number of loudspeakers.
    """
    ## Setup
    rng = np.random.default_rng()  # No seed for some randomness
    L_sig = int(p.T * p.fs)

    audio_data = np.zeros((L_sig, 1))
    noise_data = np.zeros_like(audio_data)
    echo_data = np.zeros_like(audio_data)

    micsigs_d = np.zeros((L_sig, len(p.channels)))
    micsigs_n = np.zeros_like(micsigs_d)
    micsigs_e = np.zeros_like(micsigs_d)

    ## Load audio
    for dest, file in zip(
        [audio_data, noise_data, echo_data],
        [p.audio_source, p.noise_source, p.echo_source],
        strict=True,
    ):
        offset = rng.integers(low=0, high=p.fs // 2)
        fs, data = sp.io.wavfile.read(os.path.join(p.audio_dir, file + ".wav"))
        data = data / -np.iinfo(data.dtype).min

        if fs != p.fs:
            data = sp.signal.resample_poly(data, up=p.fs, down=fs)

        dest[:, 0] = data[offset : offset + L_sig]

    ## Load RIRs & convolve
    rir_data = sp.io.loadmat(
        os.path.join(
            p.rir_dir, "_".join(["HRIR", p.subjectID, p.deviceID, str(p.rep)]) + ".mat"
        )
    )

    source_rir = rir_data["M_data"][
        :, np.all(p.source_direction == rir_data["M_dirs_sph"], axis=1), p.channels
    ]
    noise_rir = rir_data["M_data"][
        :, np.all(p.noise_direction == rir_data["M_dirs_sph"], axis=1), p.channels
    ]
    echo_rir = rir_data["M_data"][
        :, np.all(p.echo_direction == rir_data["M_dirs_sph"], axis=1), p.channels
    ]

    if rir_data["srate"] != p.fs:
        source_rir = sp.signal.resample_poly(
            source_rir, up=p.fs, down=rir_data["srate"].item()
        )[: p.rir_length, ...]
        if source_rir.shape[0] < p.rir_length:
            pad_shape = (p.rir_length - source_rir.shape[0], source_rir.shape[1])
            source_rir = np.concatenate((source_rir, np.zeros(pad_shape)))

        noise_rir = sp.signal.resample_poly(
            noise_rir, up=p.fs, down=rir_data["srate"].item()
        )[: p.rir_length, ...]
        if noise_rir.shape[0] < p.rir_length:
            pad_shape = (p.rir_length - noise_rir.shape[0], noise_rir.shape[1])
            noise_rir = np.concatenate((noise_rir, np.zeros(pad_shape)))

        echo_rir = sp.signal.resample_poly(
            echo_rir, up=p.fs, down=rir_data["srate"].item()
        )[: p.rir_length, ...]
        if echo_rir.shape[0] < p.rir_length:
            pad_shape = (p.rir_length - echo_rir.shape[0], echo_rir.shape[1])
            echo_rir = np.concatenate((echo_rir, np.zeros(pad_shape)))

    for i in range(len(p.channels)):
        micsigs_d[:, i] = sp.signal.convolve(source_rir[:, i], audio_data[:, 0])[:L_sig]
        micsigs_n[:, i] = sp.signal.convolve(noise_rir[:, i], noise_data[:, 0])[:L_sig]
        micsigs_e[:, i] = sp.signal.convolve(echo_rir[:, i], echo_data[:, 0])[:L_sig]

    ## Rescale signals to obtain desired SER
    std_d = np.std(micsigs_d[:, 0])
    std_n = np.std(micsigs_n[:, 0])
    std_e = np.std(micsigs_e[:, 0])

    std_n_des = std_d * 10 ** (-p.SNR / 20)
    std_meas_n_des = std_d * 10 ** (-p.meas_SNR / 20)
    std_e_des = std_d * 10 ** (-p.SER / 20)

    micsigs_n *= std_n_des / std_n
    echo_data *= std_e_des / std_e
    micsigs_e *= std_e_des / std_e

    # add measurement noise for stability
    micsigs_n += std_meas_n_des * rng.normal(size=micsigs_n.shape)

    return micsigs_d, micsigs_n, micsigs_e, echo_data, echo_rir
