# WeSpeaker finetuning

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.8%7C3.9-brightgreen)](https://github.com/wenet-e2e/wespeaker)

[**Roadmap**](ROADMAP.md)
| [**Docs**](http://wenet.org.cn/wespeaker)
| [**Paper**](https://arxiv.org/abs/2210.17016)
| [**Runtime**](https://github.com/wenet-e2e/wespeaker/tree/master/runtime)
| [**Pretrained Models**](docs/pretrained.md)
| [**Huggingface Demo**](https://huggingface.co/spaces/wenet/wespeaker_demo)
| [**Modelscope Demo**](https://www.modelscope.cn/studios/wenet/Speaker_Verification_in_WeSpeaker/summary)


**TL;DR:** Practical, config-driven fine-tuning for WeSpeaker models (ResNet, ECAPA-TDNN, CAMPPlus, …) with:
* Flexible layer-freezing strategies (exclude/only/none/all)
* Pattern-based layer selection (by layer name substrings)
* Optional embedding-dimension change during fine-tuning (e.g., 256→128)
* Clean VoxCeleb-style recipes & runnable commands
* Full [fine-tuning guide](examples/voxceleb/v2/README_finetune.md).

## Why this repo? 
[WeSpeaker](https://github.com/wenet-e2e/wespeaker/) already provides strong training pipelines and docs. This fork adds a focused set of fine-tuning utilities & recipes to adapt pretrained models to your target domains (new speakers, languages, channels) without re-training from scratch. See the upstream docs & paper for the broader toolkit.

## Features 
* 🔧 Layer freezing, your way — choose from multiple strategies and specify trainable/frozen parts via name patterns (e.g., layer4, pool, xvector.dense).
* 🧩 Architecture-aware — examples for ResNet (18–293), ECAPA-TDNN, CAMPPlus, etc.
* 🪄 Change embedding size during fine-tune (e.g., 256→128/64) to speed up deployment and reduce memory, while keeping upstream feature extractors intact.
* 🧪 Batteries included — VoxCeleb-style data layout, quickstart configs, and simple run scripts.
Details and YAML snippets are in the [fine-tuning guide](examples/voxceleb/v2/README_finetune.md).

## Installation 

### Install for development & deployment
``` sh
# Option A: use upstream pip install of wespeaker
pip install git+https://github.com/wenet-e2e/wespeaker.git

# Option B: dev env for this repo
# Create conda env: pytorch version >= 1.12.1 is recommended !!!
git clone https://github.com/wa3dbk/wespeaker-finetuning.git
cd wespeaker-finetuning
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
pre-commit install  # for clean and tidy code
```

For general WeSpeaker usage and environment notes, see [upstream docs](https://wenet.org.cn/wespeaker/).

## Usage

**Command-line usage** (use `-h` for parameters):

``` sh
$ wespeaker --task embedding --audio_file audio.wav --output_file embedding.txt
$ wespeaker --task embedding_kaldi --wav_scp wav.scp --output_file /path/to/embedding
$ wespeaker --task similarity --audio_file audio.wav --audio_file2 audio2.wav
$ wespeaker --task diarization --audio_file audio.wav
```

**Python programming usage**:

``` python
import wespeaker

model = wespeaker.load_model('chinese')
embedding = model.extract_embedding('audio.wav')
utt_names, embeddings = model.extract_embedding_list('wav.scp')
similarity = model.compute_similarity('audio1.wav', 'audio2.wav')
diar_result = model.diarize('audio.wav')
```

Please refer to [python usage](docs/python_package.md) for more command line and python programming usage.


## 🔥 News
* 2025.10.14: Add the possibility to change embed_dim during model fine-tuning, see [fine-tuning guidelines](examples/voxceleb/v2/README_finetune.md) for more details.
* 2025.10.04: Add support for model fine-tuning, see [fine-tuning guidelines](examples/voxceleb/v2/README_finetune.md) for more details.
* 2025.02.23: Add support for the Xi-vector, see [#404](https://github.com/wenet-e2e/wespeaker/pull/404).
* 2024.09.03: Support the SimAM_ResNet and the model pretrained on VoxBlink2, check [Pretrained Models](docs/pretrained.md) for the pretrained model, [VoxCeleb Recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v2) for the super performance, and [python usage](docs/python_package.md) for the command line usage!
* 2024.08.30: We support whisper_encoder based frontend and propose the [Whisper-PMFA](https://arxiv.org/pdf/2408.15585) framework, check [#356](https://github.com/wenet-e2e/wespeaker/pull/356).
* 2024.08.20: Update diarization recipe for VoxConverse dataset by leveraging umap dimensionality reduction and hdbscan clustering, see [#347](https://github.com/wenet-e2e/wespeaker/pull/347) and [#352](https://github.com/wenet-e2e/wespeaker/pull/352).
* 2024.08.18: Support using ssl pre-trained models as the frontend. The [WavLM recipe](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/run_wavlm.sh) is also provided, see [#344](https://github.com/wenet-e2e/wespeaker/pull/344).
* 2024.05.15: Add support for [quality-aware score calibration](https://arxiv.org/pdf/2211.00815), see [#320](https://github.com/wenet-e2e/wespeaker/pull/320).
* 2024.04.25: Add support for the gemini-dfresnet model, see [#291](https://github.com/wenet-e2e/wespeaker/pull/291).
* 2024.04.23: Support MNN inference engine in runtime, see [#310](https://github.com/wenet-e2e/wespeaker/pull/310).
* 2024.04.02: Release [Wespeaker document](http://wenet.org.cn/wespeaker) with detailed model-training tutorials, introduction of various runtime platforms, etc.
* 2024.03.04: Support the [eres2net-cn-common-200k](https://www.modelscope.cn/models/iic/speech_eres2net_sv_zh-cn_16k-common/summary) and [campplus-cn-common-200k](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) of damo [#281](https://github.com/wenet-e2e/wespeaker/pull/281), check [python usage](https://github.com/wenet-e2e/wespeaker/blob/master/docs/python_package.md) for details.
* 2024.02.05: Support the ERes2Net [#272](https://github.com/wenet-e2e/wespeaker/pull/272) and Res2Net [#273](https://github.com/wenet-e2e/wespeaker/pull/273) models.
* 2023.11.13: Support CLI usage of wespeaker, check [python usage](https://github.com/wenet-e2e/wespeaker/blob/master/docs/python_package.md) for details.
* 2023.07.18: Support the kaldi-compatible PLDA and unsupervised adaptation, see [#186](https://github.com/wenet-e2e/wespeaker/pull/186).
* 2023.07.14: Support the [NIST SRE16 recipe](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016), see [#177](https://github.com/wenet-e2e/wespeaker/pull/177).

## Recipes

* [VoxCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb): Speaker Verification recipe on the [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
    * 🔥 UPDATE 2024.05.15: We support score calibration for Voxceleb and achieve better performance!
    * 🔥 UPDATE 2023.07.10: We support self-supervised learning recipe on Voxceleb! Achieving **2.627%** (ECAPA_TDNN_GLOB_c1024) EER on vox1-O-clean test set without any labels.
    * 🔥 UPDATE 2022.10.31: We support deep r-vector up to the 293-layer version! Achieving **0.447%/0.043** EER/mindcf on vox1-O-clean test set
    * 🔥 UPDATE 2022.07.19: We apply the same setups as the CNCeleb recipe, and obtain SOTA performance considering the open-source systems
      - EER/minDCF on vox1-O-clean test set are **0.723%/0.069** (ResNet34) and **0.728%/0.099** (ECAPA_TDNN_GLOB_c1024), after LM fine-tuning and AS-Norm
* [CNCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2): Speaker Verification recipe on the [CnCeleb dataset](http://cnceleb.org/)
    * 🔥 UPDATE 2024.05.16: We support score calibration for Cnceleb and achieve better EER.
    * 🔥 UPDATE 2022.10.31: 221-layer ResNet achieves **5.655%/0.330**  EER/minDCF
    * 🔥 UPDATE 2022.07.12: We migrate the winner system of CNSRC 2022 [report](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082.pdf) [slides](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082-ZhengyangChen.pdf)
      - EER/minDCF reduction from 8.426%/0.487 to **6.492%/0.354** after large margin fine-tuning and AS-Norm
* [NIST SRE16](https://github.com/wenet-e2e/wespeaker/tree/master/examples/sre/v2): Speaker Verification recipe for the [2016 NIST Speaker Recognition Evaluation Plan](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016). Similar recipe can be found in [Kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16).
   * 🔥 UPDATE 2023.07.14: We support NIST SRE16 recipe. After PLDA adaptation, we achieved 6.608%, 10.01%, and 2.974% EER on trial Pooled, Tagalog, and Cantonese, respectively.
* [VoxConverse](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxconverse): Diarization recipe on the [VoxConverse dataset](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)


## Citations
If you find wespeaker useful, please cite it as
```bibtex
@article{wang2024advancing,
  title={Advancing speaker embedding learning: Wespeaker toolkit for research and production},
  author={Wang, Shuai and Chen, Zhengyang and Han, Bing and Wang, Hongji and Liang, Chengdong and Zhang, Binbin and Xiang, Xu and Ding, Wen and Rohdin, Johan and Silnova, Anna and others},
  journal={Speech Communication},
  volume={162},
  pages={103104},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{wang2023wespeaker,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
## Looking for contributors

If you are interested to contribute, feel free to contact @wsstriving or @robin1001
