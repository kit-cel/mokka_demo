# MOKka Demo - Constellation Shaping \& Equalization

This repository contains the full source code for the demonstration presented at
the Hack Your Research session at the OFC 2024 in San Diego.

This demo utilizes implementations for end-to-end optimization of communcation systems in the open-source project [MOKka](https://github.com/kit-cel/mokka) in combination with GUI libraries around PyQt/PySide to show the abilities of the MOKka library interactively.

Currently it implements geometric constellation shaping on an AWGN channel and a Wiener phase noise channel including carrier phase synchronization in an end-to-end optiimzation fashion which have been presented in \[1\,\[2\], and \[3\]. Also this interactive GUI implements equalization with a variational autoencoder, as presented in \[4\].

## Usage

Simply clone/download this repository and install necessary dependencies with `pip install -r requirements.txt`. Note: Currently this requires also `git` since some of the required Python packages are not yet released and therefore use a live version on GitHub. This can be seen in the requirements.txt

After installing the required dependencies (preferably in a Python virtualenv) the program can be executed by running `python interactive_training.py`. This opens a GUI in which a simulation of the shaping code and the equalization code can be started.

## References

[\[1\] A. Rode, B. Geiger, and L. Schmalen, ‘Geometric constellation shaping for phase-noise channels using a differentiable blind phase search’, in Optical Fiber Communications Conference (OFC), Mar. 2022, p. Th2A.32. doi: 10.1364/OFC.2022.Th2A.32.](http://opg.optica.org/abstract.cfm?URI=OFC-2022-Th2A.32)

[\[2\] A. Rode, B. Geiger, S. Chimmalgi, and L. Schmalen, ‘End-to-end optimization of constellation shaping for Wiener phase noise channels with a differentiable blind phase search’, Journal of Lightwave Technology, pp. 1–11, 2023, doi: 10.1109/JLT.2023.3265308.](https://ieeexplore.ieee.org/document/10093964/)

[\[3\] A. Rode, W. A. Gebrehiwot, S. Chimmalgi, and L. Schmalen, ‘Optimized geometric constellation shaping for Wiener phase noise channels with Viterbi-Viterbi carrier phase estimation’.](https://arxiv.org/abs/2307.01367)

[\[4\] V. Lauinger, F. Buchali, and L. Schmalen, ‘Blind equalization and channel estimation in coherent optical communications using variational autoencoders’, IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2529–2539, Sep. 2022, doi: 10.1109/JSAC.2022.3191346.](https://ieeexplore.ieee.org/abstract/document/9831780)

## Acknowledgment
This  work  has  received  funding  from  the  European  Re-search Council (ERC) under the European Union's Horizon2020 research and innovation programme (grant agreement No. 101001899).
