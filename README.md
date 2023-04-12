# Demucs - Music Source Separation

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine) ![tests badge](https://github.com/facebookresearch/demucs/workflows/tests/badge.svg) ![linter badge](https://github.com/facebookresearch/demucs/workflows/linter/badge.svg)

This is the 4th release of Demucs (v4), featuring Hybrid Transformer based source separation.
**For the classic Hybrid Demucs (v3):** [Go this commit][Demucs_v3].
If you are experiencing issues and want the old Demucs back, please fill an issue, and then you can get back to the v3 with
`git checkout v3`. You can also go [Demucs v2][Demucs_v2].


Demucs is a state-of-the-art music source separation model, currently capable of separating
drums, bass, and vocals from the rest of the accompaniment.
Demucs is based on a U-Net convolutional architecture inspired by [Wave-U-Net][Wave_U_Net].
The v4 version features [Hybrid Transformer Demucs][htdemucs], a hybrid spectrogram/waveform separation model using Transformers.
It is based on [Hybrid Demucs][Hybrid_Paper] (also provided in this repository) with the innermost layers are
replaced by a cross-domain Transformer Encoder. This Transformer uses self-attention within each domain,
and cross-attention across domains.
The model achieves a SDR of 9.00 dB on the MUSDB HQ test set. Moreover, when using sparse attention
kernels to extend its receptive field and per source fine-tuning, we achieve state-of-the-art 9.20 dB of SDR.

Samples are available [on our sample page](https://ai.honu.io/papers/htdemucs/index.html).
Checkout [our paper][htdemucs] for more information.
It has been trained on the [MUSDB HQ][MusDB] dataset + an extra training dataset of 800 songs.
This model separates drums, bass and vocals and other stems for any song.


As Hybrid Transformer Demucs is brand new, it is not activated by default, you can activate it in the usual
commands described hereafter with `-n htdemucs_ft`.
The single, non fine-tuned model is provided as `-n htdemucs`, and the retrained baseline
as `-n hdemucs_mmi`. The Sparse Hybrid Transformer model decribed in our paper is not provided as its
requires custom CUDA code that is not ready for release yet.
We are also releasing an experimental 6 sources model, that adds a `guitar` and `piano` source.
Quick testing seems to show okay quality for `guitar`, but a lot of bleeding and artifacts for the `piano` source.


<p align="center">
<img src="demucs.png" alt="Schema representing the structure of Hybrid Transformer Demucs, with a dual U-Net structure, one branch for the temporal domain, and one branch for the spectral domain. There is a cross-domain Transformer between the Encoders and Decoders."
width="800px"></p>



## Important news if you are already using Demucs

See the [release notes](./docs/release.md) for more details.

- 22/02/2023
  - Added support for the [SDX 2023 Challenge](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023) (see the [dedicated doc page](./docs/sdx23.md))

- 07/12/2022
  - Added Demucs v4 on PyPI
  - Released `htdemucs_6s`
  - **htdemucs** model now used by default.

- 16/11/2022
  - Added the new **Hybrid Transformer Demucs v4** models.
  - Adding support for the [torchaudio implementation of HDemucs](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html).

- 30/08/2022
  - Added reproducibility and ablation grids along with an updated version of the paper.

- 17/08/2022
  - Releasing v3.0.5
  - Set split segment length to reduce memory
  - Compatible with pyTorch 1.12

- 24/02/2022
  - Releasing v3.0.4
  - `--two-stems` split method (i.e. karaoke mode).
  - `float32` or `int24` export support

- 17/12/2021
    - Releasing v3.0.3
    - Bug fixes  (@keunwoochoi)
    - Memory drastically reduced on GPU (@famzah)
    - New multi-core evaluation on CPU (`-j` flag).

- 12/11/2021
  - Releasing **Demucs v3** with hybrid domain separation.
  - Strong improvements on all sources. This is the model that won Sony MDX challenge.

- 11/05/2021
  - Adding support for MusDB-HQ and arbitrary wav set, for the MDX challenge. For more information on joining the challenge with Demucs see [the Demucs MDX instructions](docs/mdx.md)


## Comparison with other models
We provide hereafter a summary of the different metrics presented in the paper.

You can also compare:
- Hybrid Demucs (v3)
- [KUIELAB-MDX-Net][KuieLab]
- [Spleeter][Spleeter]
- Open-Unmix
- Demucs (v1)
- Conv-Tasnet on one of my favorite songs on my [soundcloud playlist][SoundCloud].

### Accuracy comparison
We refer the reader to [our paper][Hybrid_Paper] for more details.

| Model | Domain | Extra data | Overall<br>SDR<sup>1</sup> | MOS<br>Quality<sup>2</sup> | MOS<br>Contamination<sup>3</sup> |
|-|:-:|:-:|:-:|:-:|:-:|
| [Wave-U-Net][Wave_U_Net] | Waveform | ❌ | 3.2 | - | - |
| [Open-Unmix][OpenUnmix] | Spectrogram | ❌ | 5.3 | - | - |
| [D3_Net][D3_Net] | Spectrogram | ❌ | 6.0 | - | - |
| [Conv-Tasnet][Demucs_v2] | Waveform | ❌ | 5.7 | - | - |
| [Demucs (v2)][Demucs_v2] | Waveform | ❌ | 6.3 | 2.37 | 2.36 |
| [ResUNetDecouple+][DeCouple] | Spectrogram | ❌ | 6.7 | - | - |
| [KUIELAB-MDX-Net][KuieLab] | Hybrid | ❌ | 7.5 | **2.86** | 2.55 |
| [Band-Spit RNN][BandSplit] | Spectrogram | ❌ | **8.2** | - | - |
| [**Hybrid Demucs (v3)**](demucs3) | Hybrid | ❌ | 7.7 | **2.83** | **3.04** |
| [MMDenseLSTM][mmdenselstm] | Spectrogram | 804 songs | 6.0 | - | - |
| [D3_Net][D3_Net] | Spectrogram | 1500 songs | 6.7 | - | - |
| [Spleeter][Spleeter] | Spectrogram | 25000 songs | 5.9 | - | - |
| [Band-Spit RNN][BandSplit] | Spectrogram | 1700 mixes | **9.0** | - | - |
| [**HT Demucs f.t. (v4)**](demucs4) | Hybrid | 800 songs | **9.0** | - | - |

<sup>1</sup> - Mean of the SDR for each of the 4 sources.<br>
<sup>2</sup> - Rating from 1 to 5 of the naturalness and absence of artifacts given by human listeners. (5 = no artifacts)<br>
<sup>3</sup> - Rating from 1 to 5 with 5 being zero contamination by other sources.

## Requirements

You will need at least Python 3.7. See `requirements_minimal.txt` for requirements for separation only,
and `environment-[cpu|cuda].yml` (or `requirements.txt`) if you want to train a new model.

### For Windows users

Everytime you see `python3`, replace it with `sys.executable`/`python.exe`. You should always run commands from the
Anaconda console.

### For musicians

If you just want to use Demucs to separate tracks, you can install it with

```bash
# Basic installation
python3 -m pip install -U demucs

# Bleeding edge versions - directly from this repository
python3 -m pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
```

Advanced OS support are provided on the following page, **you must read the page for your OS before posting an issues**:
- [**Windows**](docs/windows.md).
- [**Mac OS X**](docs/mac.md).
- [**Linux**](docs/linux.md).

### For machine learning scientists

If you have anaconda installed, you can run from the root of this repository. This will create a `demucs` environment with all the dependencies installed:
```bash
conda env update -f environment-cpu.yml    # if you don't have GPUs
conda env update -f environment-cuda.yml   # if you have GPUs
conda activate demucs
pip install -e .
```

You will also need to install [soundtouch](https://www.surina.net/soundtouch/soundstretch.html) for pitch/tempo augmentation:
- Linux
  - `sudo apt-get install soundstretch`
- Mac OS X
  - `brew install sound-touch`

## Running Remotely

- ## Docker
  - Thanks to @xserrat, there is now a [Docker image definition ready for using Demucs](https://github.com/xserrat/docker-facebook-demucs). This can ensure all libraries are correctly installed without interfering with the host OS.

- ## Google Colaboratory
  - Please note that transfer speeds with Colab are slow for large media files, but it will allow you to use Demucs without installing anything.

    - ✨ [**Enhanced Google Colab version**](https://colab.research.google.com/github/kubinka0505/colab-notebooks/blob/master/Notebooks/AI/Music_Separation/Demucs.ipynb) 
    - 🔗 [Simple Google Colab version](https://colab.research.google.com/drive/1dC9nVxk3V_VPjUADsnFu8EiT-xnU1tGH?usp=sharing)

- ## HuggingFace
  - [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/demucs)

- ## Graphical Interface
  - @CarlGao4 has released a [GUI for Demucs](https://github.com/CarlGao4/Demucs-Gui). Downloads for Windows and macOS is available [here](https://github.com/CarlGao4/Demucs-Gui/releases). Use [FossHub mirror](https://fosshub.com/Demucs-GUI.html) to speed up your download.
  - @Anjok07 is providing a self contained GUI in [UVR (Ultimate Vocal Remover)](https://github.com/facebookresearch/demucs/issues/334) that supports Demucs.

- ## Websites
  - [MVSep](https://mvsep.com/)
    - Free online separation with multiple Demucs models.
  - [AudioStrip](https://audiostrip.co.uk)
    - Free online separation with Demucs. 


## Separating tracks
In order to try Demucs, you can just run from any folder (as long as you properly installed it)

```bash
demucs PATH_TO_AUDIO_FILE_1 [PATH_TO_AUDIO_FILE_2 ...]    # for Demucs

# If you used "pip install --user" you might need to replace demucs with python3 -m demucs
# If your filename contain spaces don't forget to quote it!
python3 -m demucs --mp3 --mp3-bitrate BITRATE "PATH_TO_AUDIO_FILE_1"  # Output files saved as MP3

# You can select different models (listed below) with the "-n" flag
demucs -n mdx_q "File.mp3"

# If you only want to separate vocals out of an audio, use `--two-stems=vocal` (You can also set to drums or bass)
demucs --two-stems=vocals "File.mp3"
```

If you have a GPU, but you run out of memory, please use `--segment SEGMENT` to reduce length of each split. `SEGMENT` should be changed to a integer. Personally recommend not less than 10 (the bigger the number is, the more memory is required, but quality may increase). Create an environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is also helpful. If this still cannot help, please add `-d cpu` to the command line. See the section hereafter for more details on the memory requirements for GPU acceleration.

Separated tracks are stored in the `separated/MODEL_NAME/TRACK_NAME` folder. There you will find four stereo wave files sampled at 44 100 Hz: `drums.wav`, `bass.wav`,
`other.wav`, `vocals.wav` (or `.mp3` if you used the `--mp3` option).

All audio formats supported by `torchaudio` can be processed (i.e. WAV, MP3, FLAC, Ogg/Vorbis on Linux/Mac OS X etc.). On Windows, `torchaudio` has limited support, so we rely on `FFmpeg`, which should support pretty much anything.
Audio is resampled on the fly if necessary.
The output will be a wave file encoded as int16.
- You can save as float32 wav files with `--float32`, or 24 bits integer wav with `--int24`.
- You can pass `--mp3` to save as mp3 instead, and set the bitrate with `--mp3-bitrate` (default is 320 kb/s).

It can happen that the output would need clipping, in particular due to some separation artifacts.
Demucs will automatically rescale each output stem so as to avoid clipping. This can however break
the relative volume between stems. If instead you prefer hard clipping, pass `--clip-mode clamp`.
You can also try to reduce the volume of the input mixture before feeding it to Demucs.

The list of pre-trained models:
| Code name | Description |
|:-:|-|
| `htdemucs` | First version of Hybrid Transformer Demucs.<br>Trained on MusDB + 800 songs.<br>Default model. |
| **`htdemucs_ft`** | Fine-tuned version of `htdemucs`<br>Separation will take 4 times more than `htdemucs` at cost of better quality.<br>Same training set as `htdemucs`. | 
| `htdemucs_6s` | 6 sources version of `htdemucs`, with `piano` and `guitar` being added as sources.<br>Note that the `piano` source is not working great at the moment. | 
| **`hdemucs_mmi`** | Hybrid Demucs v3<br>Retrained on MusDB + 800 songs. |
| `mdx` | Trained only on MusDB HQ<br>Winning model on track A at the [MDX][MDX] challenge. |
|  **`mdx_extra`** | Trained with extra training data (**including MusDB test set**)<br>Ranked 2nd on the track B of the [MDX][MDX] challenge.
| `mdx_q`<br>`mdx_extra_q` | Quantized version of the previous models.<br>Smaller download and storage at cost of worse quality.
- `SIG`: where `SIG` is a single model from the [model zoo](docs/training.md#model-zoo).

## Flags

- `--two-stems=STEM`Separate `STEM` from the rest.
  - `STEM` is a value into any source in the selected model. (i.e. `vocals`)
  - This will mix the files after separating the mix fully, so this won't be faster or use less memory.

- `--shifts=n`
  - Performs multiple predictions with random shifts (shift trick) of the input and average them.
    - This makes prediction `n` times slower.
  - Don't use it unless you have a GPU!

- `--overlap`
  - Controls the amount of overlap between prediction windows. Default is 0.25 (25%) which is probably fine.
  - It can probably be reduced to 0.1 to improve a bit speed.

- `-j`
  - Specify a number of parallel jobs. Default is `1`.
  - This will multiply by the same amount the RAM used so be careful!


### Memory requirements for GPU acceleration

If you want to use GPU acceleration, you will need at least 3GB of RAM on your GPU for `demucs`. However, about 7GB of RAM will be required if you use the default arguments. Add `--segment SEGMENT` to change size of each split. If you only have 3GB memory, set SEGMENT to 8 (though quality may be worse if this argument is too small). Creating an environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` can help users with even smaller RAM such as 2GB (I separated a track that is 4 minutes but only 1.5GB is used), but this would make the separation slower.

If you do not have enough memory on your GPU, simply add `-d cpu` to the command line to use the CPU. With Demucs, processing time should be roughly equal to 1.5 times the duration of the track.


## Training Demucs

If you want to train (Hybrid) Demucs, please follow the [training doc](docs/training.md).

## MDX Challenge reproduction

In order to reproduce the results from the Track A and Track B submissions, please check out the [MDX Hybrid Demucs submission][MDX_Submission] repository.



## How to cite

```
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

## License

Demucs is released under the MIT license as found in the [LICENSE](LICENSE) file.

[Hybrid_Paper]: https://arxiv.org/abs/2111.03600
[Wave_U_Net]: https://github.com/f90/Wave-U-Net
[MusDB]: https://sigsep.github.io/datasets/musdb.html
[OpenUnmix]: https://github.com/sigsep/open-unmix-pytorch
[mmdenselstm]: https://arxiv.org/abs/1805.02410
[Demucs_v2]: https://github.com/facebookresearch/demucs/tree/v2
[Demucs_v3]: https://github.com/facebookresearch/demucs/tree/v3
[Spleeter]: https://github.com/deezer/spleeter
[SoundCloud]: https://soundcloud.com/honualx/sets/source-separation-in-the-waveform-domain
[D3_Net]: https://arxiv.org/abs/2010.01733
[MDX]: https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
[KuieLab]: https://github.com/kuielab/mdx-net-submission
[DeCouple]: https://arxiv.org/abs/2109.05418
[MDX_Submission]: https://github.com/adefossez/mdx21_demucs
[BandSplit]: https://arxiv.org/abs/2209.15174
[htdemucs]: https://arxiv.org/abs/2211.08553
