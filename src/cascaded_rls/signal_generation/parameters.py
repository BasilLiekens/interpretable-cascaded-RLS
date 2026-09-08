import os
from dataclasses import dataclass, field
from typing import Self

import yaml


@dataclass
class parameters:
    ## Signal parameters
    fs: int = int(16e3)  # sampling frequency [Hz]
    T: float = 10  # Signal duration [s]
    SNR: float = 10  # SNR of interfering sources
    meas_SNR: float = 10  # Measurement noise SNR [dB]
    SER: float = 10  # Signal-to-echo ratio [dB]

    ## Data settings, catered towards `VCTK` and `hearpiece`.
    # It is assumed that there is only 1 source of each desired signal, noise and
    # echo.
    audio_dir: str = os.path.join("path", "to", "directory")
    audio_source: str = "source_1"
    noise_source: str = "source_2"
    echo_source: str = "source_3"

    rir_dir: str = os.path.join("path", "to", "directory")

    subjectID: str = "KEMAR"
    deviceID: str = "DV0001"
    rep: int = 5

    source_direction: tuple[float, float] = (90.0, 0.0)
    noise_direction: tuple[float, float] = (0.0, 0.0)
    echo_direction: tuple[float, float] = (180.0, 0.0)
    rir_length: int = 160
    channels: list[int] = field(
        default_factory=lambda: [0, 9]
    )  # which of the recording channels to use

    ## Simulation settings
    RLS_impl: str = "HRLS"
    lmbd: float = 0.995
    L_sc: int = 10

    @classmethod
    def load_from_yaml(cls, path: str) -> Self:
        """Load the parameters from YAML file"""
        with open(path) as file:  # `open` is read-only by default
            data = yaml.safe_load(file)

        p = cls()

        for key, value in data.items():
            setattr(p, key, value)

        return p

    def __repr__(self):
        """Return string representation of the object"""
        return "Parameters:\n >> " + "\n >> ".join(
            [f"{key}: {value}" for key, value in self.__dict__.items()]
        )
