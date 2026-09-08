# Intepretable cascaded RLS

This repository contains a toolbox with the code used for the paper ["On cascaded recursive least squares: an interpretable and cheaper alternative to the standard implementation"](https://ftp.esat.kuleuven.be/stadius/bliekens/26-119.pdf) by Basil Liekens, Arnout Roebben, Toon van Waterschoot and Marc Moonen, submitted to ICASSP 2027.

## Setup

This repository is set up as a package contained in `src/cascaded_rls` which can be installed as `pip install .` once in this repository.
This will then automatically download all required dependencies.
Relevant scripts can be found in `scripts/`.

Alternatively, a `uv.lock` file is provided that enables running scripts as `uv run scripts/xyz.py` without any prior setup if `uv` is installed.

The `geic.py` experiment relies on data from the [Hearpiece dataset](https://zenodo.org/records/3733191) for impulse responses and the [CSTR VCTK Corpus](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928) for audio files. 
Other sources can also be used provided they follow the same conventions and are located correctly using the config file.

## Folder structure

The project is set up with a src-layout: the main package containing utils and algorithms is located under `src/cascaded_rls`. 
The provided algorithms are a standard, possibly weighted, RLS update [1,2], a Householder RLS update [3] and the cascaded implementation derived in the associated paper. 

Example scripts using the package can be found in the `scripts` folder, the `geic.py` script is driven by a config file of which an example can be found in `config/config_example.yml`.

## Citation

If you found this work useful and want to use it in your own research, please consider citing the following paper, also available [here]().

```bibtex
@inproceedings{liekensCascadedRLS2027,
    author={Liekens, Basil and Roebben, Arnout and van Waterschoot, Toon and Moonen, Marc},
    booktitle={Submitted to 2027 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    title={On Cascaded Recursive Least Squares: An Interpretable and Cheaper Alternative to the Standard Implementation},
    year={2027-05},
    location={Toronto, Canada}
}
```

## Contact
Basil Liekens, Arnout Roebben, Toon van Waterschoot, and Marc Moonen
Department of Electrical Engineering (ESAT)
STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics
KU Leuven
Leuven, Belgium
E-mail: basil.liekens@esat.kuleuven.be

## Acknowledgements
This research was carried out at the ESAT Laboratory of KU Leuven, in the frame of Research Council KU Leuven Project C14-21-0075 ”A holistic approach to the design of integrated and distributed digital signal processing algorithms for audio and speech communication devices”, and Aspirant Grant 11PDH24N (for A. Roebben) from the Research Foundation - Flanders (FWO).

## References

[1] S. S. Haykin, Adaptive Filter Theory, 5. ed. Upper Saddle River, NJ: Pearson, 2014.

[2] T. Yoshioka, H. Tachibana, T. Nakatani and M. Miyoshi, "Adaptive dereverberation of speech signals with speaker-position change detection," 2009 IEEE International Conference on Acoustics, Speech and Signal Processing, Taipei, Taiwan, 2009, pp. 3733-3736, doi: 10.1109/ICASSP.2009.4960438.

[3] J. Wung et al., "Robust Multichannel Linear Prediction for Online Speech Dereverberation Using Weighted Householder Least Squares Lattice Adaptive Filter," in IEEE Transactions on Signal Processing, vol. 68, pp. 3559-3574, 2020, doi: 10.1109/TSP.2020.2997201.
