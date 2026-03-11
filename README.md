# Deep Autocorrelation Modeling for Time-series Forecasting: Papers and Resources


<h3 align="center">Welcome to AutoTSF</h3>

<p align="center"><i>A list of awesome time-series forecasting papers featured by autocorrelation modeling.</i></p>

<p align="center">
    <a href="https://github.com/Master-PLC/AutoTSF">
        <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-E9BB41?logo=opensourceinitiative&logoColor=white">
    </a>
    <a href="https://star-history.com/#Master-PLC/AutoTSF">
        <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Master-PLC/AutoTSF">
    </a>
    <a href="https://github.com/Master-PLC/PyITS/network/members">
        <img alt="GitHub Repo forks" src="https://img.shields.io/github/forks/Master-PLC/AutoTSF">
    </a>
   <a href="https://github.com/Master-PLC/PyITS/blob/main/README.md">
        <img alt="README in English" src="https://pypots.com/figs/pypots_logos/readme/US.svg">
    </a>
</p>

## 1. Introduction

AutoTSF is a carefully curated repository of research papers on time-series forecasting, with a particular focus on innovative strategies for modeling autocorrelation. To our knowledge, it is the first repository that systematically summarizes recent advances in both neural architectures and loss functions for time-series forecasting.


✨ AutoTSF highlights methods that explicitly improve the modeling of autocorrelation, a central challenge in time-series forecasting. While related topics such as denoised learning and channel dependency modeling are also valuable, they address more general machine learning problems and therefore fall outside the primary scope of it.

🚩 We will continue to maintain and update this repository. If you notice any missing papers or code resources, please feel free to open an issue or submit a PR.






## 2. Model Architectures

### 2.1. Non-Transformers

#### Recurrent Neural Networks

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| P-sLSTM | [Unlocking the power of lstm for long term time series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/33303) | 2025 | AAAI | [Github](https://github.com/Eleanorkong/P-sLSTM) |
| DSTMamba | [Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/33281) | 2025 | AAAI | [Github](https://github.com/Anle-He/DST-Mamba) |
| MixMamba | [MixMamba: Time series modeling with adaptive expertise](https://www.sciencedirect.com/science/article/abs/pii/S1566253524003671) | 2024 | Inf. Fusion | Not available |
| TVFSSM | [Heterogeneous Multivariate Functional Time Series Modeling: A State Space Approach](https://ieeexplore.ieee.org/document/10713887) | 2024 | IEEE TKDE | Not available |
| SpaceTime | [Effectively Modeling Time Series with Simple Discrete State Spaces](https://openreview.net/forum?id=2EpjkjzdCAa) | 2023 | ICLR | [Github](https://github.com/HazyResearch/spacetime) |
| Mamba | [Mamba: Linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752) | 2023 | CoLM | [GitHub](https://github.com/state-spaces/mamba) |
| LRU | [Resurrecting recurrent neural networks for long sequences](https://proceedings.mlr.press/v202/orvieto23a.html) | 2023 | ICML | Not available|
| S4 | [Efficiently Modeling Long Sequences with Structured State Spaces](https://openreview.net/forum?id=uYLFoz1vlAC) | 2021 | ICLR | [GitHub](https://github.com/state-spaces/s4) |
| LMU | [Legendre memory units: Continuous-time representation in recurrent neural networks](https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html) | 2019 | NeurIPS | [GitHub](https://github.com/abr/lmu) |
| DeepAR | [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888) | 2020 | Int. J. Forecast | [GitHub](https://github.com/awslabs/gluon-ts) |
| DeepSSM | [Deep state space models for time series forecasting](https://proceedings.neurips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html) | 2018 | NeurIPS | Not available |
| SRU | [Simple Recurrent Units for Highly Parallelizable Recurrence](https://aclanthology.org/D18-1477/) | 2018 | EMNLP | [GitHub](https://github.com/taolei87/sru) |
| GLU | [Language modeling with gated convolutional networks](https://proceedings.mlr.press/v70/dauphin17a.html) | 2017 | ICML | [Github](https://github.com/jojonki/Gated-Convolutional-Networks) |
| GRU | [On the properties of neural machine translation: Encoder-decoder approaches](https://arxiv.org/abs/1409.1259) | 2014 | Arxiv | [Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) |
| LSTM | [Long short-term memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735) | 1997 | Neural Comput. | [Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) |

#### Convolution Neural Networks

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| WaveTS | [Multi-Order Wavelet Derivative Transform for Deep Time Series Forecasting](https://arxiv.org/abs/2505.11781) | 2025 | Arxiv | [Github](https://github.com/zhouziyu02/WaveTS) |
| AdaWaveNet | [AdaWaveNet: Adaptive wavelet network for time series analysis](https://openreview.net/forum?id=m4bE9Y9FlX) | 2024 | TMLR | [Github](https://github.com/comp-well-org/AdaWaveNet) |
| WFTNet | [WFTNet: Exploiting Global and Local Periodicity in Long-Term Time Series Forecasting](https://arxiv.org/abs/2309.11319v1) | 2024 | ICASSP | [Github](https://github.com/Hank0626/WFTNet) |
| ModernTCN | [Moderntcn: A modern pure convolution structure for general time series analysis](https://openreview.net/forum?id=vpJMJerXHU) | 2024 | ICLR | [GitHub](https://github.com/luodhhh/ModernTCN) |
| MICN | [Micn: Multi-scale local and global context modeling for long-term series forecasting](https://openreview.net/forum?id=zt53IDUR1U) | 2023 | ICLR | [GitHub](https://github.com/wanghq21/MICN) |
| TimesNet | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/forum?id=ju_Uqw384Oq) | 2023 | ICLR | [GitHub](https://github.com/thuml/TimesNet) |
| SCINet | [SCINet: time series modeling and forecasting with sample convolution and interaction](https://openreview.net/forum?id=AyajSjTAzmg) | 2022 | NeurIPS | [GitHub](https://github.com/cure-lab/SCINet) |
| DESCINet | [DESCINet: A hierarchical deep convolutional neural network with skip connection for long time series forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0957417423007480?via%3Dihub) | 2023 | ESWA | Not available |
| FiLM | [Film: Frequency improved legendre memory model for long-term time series forecasting](https://openreview.net/forum?id=zTQdHSQUQWc) | 2022 | NeurIPS | [Github](https://github.com/tianzhou2011/FiLM/) |
| FilterNet | [Filternet: Harnessing frequency filters for time series forecasting](https://openreview.net/forum?id=ugL2D9idAD) | 2024 | NeurIPS | [Github](https://github.com/aikunyi/FilterNet) |
| TCN | [An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org/abs/1803.01271) | 2018 | Arxiv | [GitHub](https://github.com/locuslab/TCN) |
| WaveNet | [Conditional time series forecasting with convolutional neural networks](https://arxiv.org/abs/1703.04691) | 2017 | ICANN | [Github](https://github.com/basveeling/wavenet) |

#### Dense Neural Networks

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| TimeMixer++ | [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://openreview.net/forum?id=1CLzLXSFNn) | 2025 | ICLR | Not available |
| SparseTSF | [SparseTSF: Lightweight and Robust Time Series Forecasting via Sparse Modeling](https://ieeexplore.ieee.org/abstract/document/11141354) | 2025 | IEEE TPAMI | [GitHub](https://github.com/lss-1138/SparseTSF) |
| WPMixer | [Wpmixer: Efficient multi-resolution mixing for long-term time series forecasting](https://arxiv.org/abs/2412.17176) | 2025 | AAAI | [Github](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer) |
| CycleNet | [Cyclenet: Enhancing time series forecasting through modeling periodic patterns](https://neurips.cc/virtual/2024/poster/94391) | 2024 | NeurIPS | [Github](https://github.com/ACAT-SCUT/CycleNet) |
| RLinear | [An Analysis of Linear Time Series Forecasting Models](https://proceedings.mlr.press/v235/toner24a.html) | 2024 | ICML | [Github](https://github.com/sir-lab/linear-forecasting) |
| TimeMixer | [Timemixer: Decomposable multiscale mixing for time series forecasting](https://openreview.net/forum?id=7oLshfEIC2) | 2024 | ICLR | [GitHub](https://github.com/kwuking/TimeMixer) |
| FITS | [FITS: Modeling Time Series with $10 k $ Parameters](https://openreview.net/forum?id=bWcnvZ3qMb) | 2024 | ICLR | [GitHub](https://github.com/VEWOXIC/FITS) |
| MoLE | [Mixture-of-Linear-Experts for Long-term Time Series Forecasting](https://proceedings.mlr.press/v238/ni24a/ni24a.pdf) | 2024 | AISTATS | [Github](https://github.com/RogerNi/MoLE) |
| RPMixer | [RPMixer: Shaking up time series forecasting with random projections for large spatial-temporal data](https://dl.acm.org/doi/10.1145/3637528.3671881) | 2024 | SIGKDD | Not available |
| FreTS | [Frequency-domain MLPs are More Effective Learners in Time Series Forecasting](https://neurips.cc/virtual/2023/poster/70726) | 2023 | NeurIPS | [Github](https://github.com/aikunyi/FreTS) |
| TSMixer | [TSMixer: An all-MLP Architecture for Time Series Forecasting](https://openreview.net/forum?id=wbpxTuXgm0) | 2023 | TMLR | [Github](https://github.com/google-research/google-research/tree/master/tsmixer) |
| TiDE | [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://openreview.net/forum?id=pCbC3aQB5W) | 2023 | TMLR | [GitHub](https://github.com/google-research/google-research/tree/master/tide) |
| DLinear | [Are Transformers Effective for Time Series Forecasting?](https://www.aaai.org/ojs/index.php/AAAI/article/view/26317) | 2023 | AAAI | [GitHub](https://github.com/cure-lab/LTSF-Linear) |

### 2.2 Transformers

#### Standard Self-attention Models

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Duet | (Reference not found in bib files) | - | - | Not available |
| Timer-XL | [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://openreview.net/forum?id=KMCJXjlDDr&noteId=7jSaM7SC1f) | 2025 | ICLR | [Github](https://github.com/thuml/Timer-XL) |
| Sundial | [Sundial: A Family of Highly Capable Time Series Foundation Models](https://openreview.net/forum?id=LO7ciRpjI5) | 2025 | ICML | [Github](https://github.com/thuml/Sundial) |
| LLM4TS | [Llm4ts: Aligning pre-trained llms as data-efficient time-series forecasters](https://dl.acm.org/doi/10.1145/3719207) | 2025 | ACM TIST | [Github](https://github.com/blacksnail789521/LLM4TS) |
| TEMPO | [TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://proceedings.iclr.cc/paper_files/paper/2024/file/5132940b1bced8a7b28e9695d49d435a-Paper-Conference.pdf) | 2024 | ICLR | [Github](https://github.com/DC-research/TEMPO) |
| GPT4mts | [Gpt4mts: Prompt-based large language model for multimodal time-series forecasting](https://dl.acm.org/doi/10.1609/aaai.v38i21.30383) | 2024 | AAAI | [Github](https://github.com/Flora-jia-jfr/GPT4MTS-Prompt-based-Large-Language-Model-for-Multimodal-Time-series-Forecasting) |
| Moirai | [Unified training of universal time series forecasting transformers](https://openreview.net/pdf?id=Yd8eHMY1wz) | 2024 | ICML | [GitHub](https://github.com/SalesforceAIResearch/uni2ts) |
| UniTS | [Units: A unified multi-task time series model](https://openreview.net/forum?id=nBOdYBptWW&referrer=%5Bthe%20profile%20of%20Owen%20Queen%5D(%2Fprofile%3Fid%3D~Owen_Queen1)) | 2024 | NeurIPS | [GitHub](https://github.com/mims-harvard/UniTS) |
| Timer | [Timer: generative pre-trained transformers are large time series models](https://dl.acm.org/doi/10.5555/3692070.3693383) | 2024 | ICML | [Github](https://github.com/thuml/timer) |
| TimesFM | [A decoder-only foundation model for time-series forecasting](https://openreview.net/forum?id=jn2iTJas6h) | 2024 | ICML | [Github](https://github.com/google-research/timesfm) |
| Chronos | [Chronos: Learning the Language of Time Series](https://openreview.net/forum?id=gerNCVqqtR) | 2024 | TMLR | [GitHub](https://github.com/amazon-science/chronos-forecasting) |
| S²IP-LLM | [S2IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting](https://dl.acm.org/doi/10.5555/3692070.3693658) | 2024 | ICML | [Github](https://github.com/panzijie825/S2IP-LLM) |
| Time-FFM | [Time-ffm: Towards lm-empowered federated foundation model for time series forecasting](https://openreview.net/forum?id=HS0faHRhWD&referrer=%5Bthe%20profile%20of%20Yuxuan%20Liang%5D(%2Fprofile%3Fid%3D~Yuxuan_Liang1)) | 2024 | NeurIPS | [Github](https://github.com/yuppielqx/Time-FFM) |
| Time-LLM | [Time-llm: Time series forecasting by reprogramming large language models](https://openreview.net/forum?id=Unb5CVPtae) | 2024 | ICLR | [GitHub](https://github.com/KimMeen/Time-LLM) |
| PatchTST | [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://openreview.net/forum?id=Jbdc0vTOcol) | 2023 | ICLR | [GitHub](https://github.com/yuqinie98/PatchTST) |
| LLMTime | [Large language models are zero-shot time series forecasters](https://openreview.net/forum?id=md68e8iZK1) | 2023 | NeurIPS | [Github](https://github.com/ngruver/llmtime) |
| PromptCast | [PromptCast: A new prompt-based learning paradigm for time series forecasting](https://dl.acm.org/doi/abs/10.1109/TKDE.2023.3342137) | 2023 | IEEE TKDE | [Github](https://github.com/cruiseresearchgroup/PISA-PromptCast) |
| GPT4TS | [One fits all: Power general time series analysis by pretrained lm](https://openreview.net/forum?id=gMS6FVZvmF) | 2023 | NeurIPS | [Github](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) |

#### Modified Attention Models

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| TQNet | [Temporal Query Network for Efficient Multivariate Time Series Forecasting](https://openreview.net/forum?id=e24CueVty2&noteId=R07EYunNvR) | 2025 | ICML | [Github](https://github.com/ACAT-SCUT/TQNet) |
| TimeBridge | [TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting](https://openreview.net/forum?id=pyKO0ZZ5lz) | 2025 | ICML | [GitHub](https://github.com/hank0626/timebridge) |
| Freeformer | [FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting](https://www.ijcai.org/proceedings/2025/0401.pdf) | 2025 | IJCAI | [Github](https://github.com/jackyue1994/FreEformer) |
| FredFormer | [Fredformer: Frequency debiased transformer for time series forecasting](https://dl.acm.org/doi/10.1145/3637528.3671928) | 2024 | SIGKDD | [Github](https://github.com/chenzRG/Fredformer) |
| PDF | [Periodicity decoupling framework for long-term series forecasting](https://openreview.net/forum?id=dp27P5HBBt) | 2024 | ICLR | [Github](https://github.com/Hank0626/PDF) |
| DeformableTST | [DeformableTST: Transformer for time series forecasting without over-reliance on patching](https://neurips.cc/virtual/2024/poster/96221) | 2024 | NeurIPS | [Github](https://github.com/luodhhh/DeformableTST) |
| SAMformer | [SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://dl.acm.org/doi/10.5555/3692070.3692911) | 2024 | ICML | [Github](https://github.com/romilbert/samformer) |
| iTransformer | [itransformer: Inverted transformers are effective for time series forecasting](https://openreview.net/forum?id=JePfAI8fah) | 2024 | ICLR | [GitHub](https://github.com/thuml/iTransformer) |
| AttentionMixer | [An Accurate and Interpretable Framework for Trustworthy Process Monitoring](https://ieeexplore.ieee.org/document/10265128) | 2023 | IEEE TAI | Not available |
| CrossFormer | [Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting](https://openreview.net/forum?id=vSVLM2j9eie) | 2023 | ICLR | [GitHub](https://github.com/Thinklab-SJTU/Crossformer) |
| FedFormer | [FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting](https://proceedings.mlr.press/v162/zhou22g.html) | 2022 | ICML | [GitHub](https://github.com/MAZiqing/FEDformer) |
| Pyraformer | [Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://openreview.net/forum?id=0EXmFzUn5I) | 2022 | ICLR | [GitHub](https://github.com/ant-research/Pyraformer) |
| Informer | [Informer: Beyond efficient transformer for long sequence time-series forecasting](https://cdn.aaai.org/ojs/17325/17325-13-20819-1-2-20210518.pdf) | 2021 | AAAI | [GitHub](https://github.com/zhouhaoyi/Informer2020) |
| Autoformer | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://openreview.net/forum?id=J4gRj6d5Qm) | 2021 | NeurIPS | [GitHub](https://github.com/thuml/Autoformer) |
| LogTrans | [Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting](https://openreview.net/pdf/9755681fa6f4f183ca07e57f5b4bdacfbb058895.pdf) | 2019 | NeurIPS | Not available |
| DSANet | [Dsanet: Dual self-attention network for multivariate time series forecasting](https://dl.acm.org/doi/10.1145/3357384.3358132) | 2019 | CIKM | [Github](https://github.com/bighuang624/DSANet) |

### 2.3 Plug-ins

#### Normalization plug-ins

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| SIN | [SIN: Selective and interpretable normalization for long-term time series forecasting](https://proceedings.mlr.press/v235/han24e.html) | 2024 | ICML | Not available |
| FAN | [Frequency adaptive normalization for non-stationary time series forecasting](https://openreview.net/forum?id=T0axIflVDD&noteId=h8wN5bnm3k) | 2024 | NeurIPS | [Github](http://github.com/icannotnamemyself/FAN) |
| DDN | [DDN: Dual-domain dynamic normalization for non-stationary time series forecasting](https://openreview.net/forum?id=RVZfra6sZo&referrer=%5Bthe%20profile%20of%20Zexuan%20Zhu%5D(%2Fprofile%3Fid%3D~Zexuan_Zhu1)) | 2024 | NeurIPS | [Github](https://github.com/Hank0626/DDN) |
| Dish-TS | [Dish-ts: a general paradigm for alleviating distribution shift in time series forecasting](https://dl.acm.org/doi/10.1609/aaai.v37i6.25914) | 2023 | AAAI | [Github](https://github.com/weifantt/Dish-TS) |
| SAN | [Adaptive normalization for non-stationary time series forecasting: A temporal slice perspective](https://openreview.net/forum?id=5BqDSw8r5j) | 2023 | NeurIPS | [Github](https://github.com/icantnamemyself/SAN) |
| RevIN | [Reversible instance normalization for accurate time-series forecasting against distribution shift](https://openreview.net/forum?id=cGDAkQo1C0p) | 2021 | ICLR | [GitHub](https://github.com/ts-kim/RevIN) |

#### Decomposition plug-ins

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Autoformer | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://openreview.net/forum?id=J4gRj6d5Qm) | 2021 | NeurIPS | [GitHub](https://github.com/thuml/Autoformer) |
| xPatch | [xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition](https://dl.acm.org/doi/10.1609/aaai.v39i19.34270) | 2025 | AAAI | [Github](https://github.com/stitsyuk/xPatch) |
| Patchmlp | [Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting](https://dl.acm.org/doi/10.1609/aaai.v39i12.33378) | 2025 | AAAI | [Github](https://github.com/TangPeiwang/PatchMLP) |
| TEMPO | [TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://proceedings.iclr.cc/paper_files/paper/2024/file/5132940b1bced8a7b28e9695d49d435a-Paper-Conference.pdf) | 2024 | ICLR | [Github](https://github.com/DC-research/TEMPO) |
| MICN | [Micn: Multi-scale local and global context modeling for long-term series forecasting](https://openreview.net/forum?id=zt53IDUR1U) | 2023 | ICLR | [GitHub](https://github.com/wanghq21/MICN) |
| Times2d | [Times2d: Multi-period decomposition and derivative mapping for general time series forecasting](https://dl.acm.org/doi/10.1609/aaai.v39i18.34164) | 2025 | AAAI | [Github](https://github.com/Tims2D/Times2D) |
| TimesNet | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/forum?id=ju_Uqw384Oq) | 2023 | ICLR | [GitHub](https://github.com/thuml/TimesNet) |
| MSGNet | [Msgnet: Learning multi-scale inter-series correlations for multivariate time series forecasting](https://dl.acm.org/doi/10.1609/aaai.v38i10.28991) | 2024 | AAAI | [Github](https://github.com/YoZhibo/MSGNet) |
| SCINet | [SCINet: time series modeling and forecasting with sample convolution and interaction](https://openreview.net/forum?id=AyajSjTAzmg) | 2022 | NeurIPS | [GitHub](https://github.com/cure-lab/SCINet) |
| DESCINet | [DESCINet: A hierarchical deep convolutional neural network with skip connection for long time series forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0957417423007480?via%3Dihub) | 2023 | ESWA | Not available |
| TimeMixer | [Timemixer: Decomposable multiscale mixing for time series forecasting](https://openreview.net/forum?id=7oLshfEIC2) | 2024 | ICLR | [GitHub](https://github.com/kwuking/TimeMixer) |
| TimeMixer++ | [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://openreview.net/forum?id=1CLzLXSFNn) | 2025 | ICLR | Not available |

#### Tokenization plug-ins

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| WaveToken | [Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization](https://openreview.net/forum?id=B6WalMoQJW) | 2025 | ICML | [Github](https://github.com/amazon-science/chronos-forecasting/tree/wavetoken) |
| Patchmlp | [Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting](https://dl.acm.org/doi/10.1609/aaai.v39i12.33378) | 2025 | AAAI | [Github](https://github.com/TangPeiwang/PatchMLP) |
| SimpleTM | [SimpleTM: A Simple Baseline for Multivariate Time Series Forecasting](https://openreview.net/forum?id=oANkBaVci5) | 2025 | ICLR | [Github](https://github.com/vsingh-group/SimpleTM) |
| ElasTST | [ElasTST: Towards robust varied-horizon forecasting with elastic time-series transformer](https://neurips.cc/virtual/2024/poster/93264) | 2024 | NeurIPS | [Github](https://github.com/microsoft/ProbTS/tree/elastst) |
| MTST | [Multi-resolution time-series transformer for long-term forecasting](https://proceedings.mlr.press/v238/zhang24l/zhang24l.pdf) | 2024 | AISTATS | [Github](https://github.com/VenkatachalamSubramanianPeriyaSubbu/multiresolution-time-series-transformer) |
| TimeSQL | [TimeSQL: Improving multivariate time series forecasting with multi-scale patching and smooth quadratic loss](https://www.sciencedirect.com/science/article/abs/pii/S0020025524005656) | 2024 | Inf. Sci. | Not available |
| PatchTST | [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://openreview.net/forum?id=Jbdc0vTOcol) | 2023 | ICLR | [GitHub](https://github.com/yuqinie98/PatchTST) |

## 3. Learning Objectives

### 3.1 Likelihood Estimation Objectives

#### Label Transformation

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Time-o1 | [Time-o1: Time-Series Forecasting Needs Transformed Label Alignment](https://openreview.net/forum?id=RxWILaXuhb) | 2025 | NeurIPS | [Github](https://github.com/Master-PLC/Time-o1) |
| OLMA | [One Loss for More Accurate Time Series Forecasting](https://arxiv.org/html/2505.11567v2) | 2025 | Arxiv | [Github](https://github.com/Yuyun1011/OLMA-One-Loss-for-More-Accurate-Time-Series-Forecasting) |
| FreDF | [FreDF: Learning to Forecast in the Frequency Domain](https://openreview.net/forum?id=4A9IdSa1ul) | 2025 | ICLR | [Github](https://github.com/Master-PLC/FreDF) |
| DBLoss | [DBLoss: Decomposition-based Loss Function for Time Series Forecasting](https://neurips.cc/virtual/2025/loc/san-diego/poster/117918) | 2025 | NeurIPS | [Github](https://github.com/decisionintelligence/DBLoss) |
| TDAlign | [Modeling temporal dependencies within the target for long-term time series forecasting](https://ieeexplore.ieee.org/document/11160655) | 2025 | IEEE TKDE | [Github](https://github.com/XQ-edu/TDAlign) |
| TimeSQL | [TimeSQL: Improving multivariate time series forecasting with multi-scale patching and smooth quadratic loss](https://www.sciencedirect.com/science/article/abs/pii/S0020025524005656) | 2024 | Inf. Sci. | Not available |
| AutoMSE | [Adjusting for autocorrelated errors in neural networks for time series](https://proceedings.neurips.cc/paper_files/paper/2021/file/f8e6ba1db0f3c4054afec1684ba8fb26-Paper.pdf) | 2021 | NeurIPS | [Github](https://github.com/Daikon-Sun/AdjustAutocorrelation) |

#### Covariance Modeling 

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| MMKE | [Multivariate probabilistic time series forecasting with correlated errors](https://openreview.net/forum?id=cAFvxVFaii&noteId=ZL0AiqWDNn) | 2024 | NeurIPS | [Github](https://github.com/rottenivy/mv_pts_correlatederr) |
| MKE | [Better batch for deep probabilistic time series forecasting](https://proceedings.mlr.press/v238/zheng24a/zheng24a.pdf) | 2024 | AISTATS | [Github](https://github.com/rottenivy/betterbatch) |
| QDF | [Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models](https://openreview.net/forum?id=vpO8n9AqEG) | 2026 | ICLR | [Github](https://github.com/Master-PLC/QDF) |

### 3.2 Shape Alignment Objectives

#### Dynamic Time Wrapping

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| STRIPE | [Deep Time Series Forecasting With Shape and Temporal Criteria](https://ieeexplore.ieee.org/document/9721108) | 2023 | IEEE TPAMI | [Github](https://github.com/vincent-leguen/DILATE) |
| GDTW | [Gdtw: A novel differentiable dtw loss for time series tasks](https://ieeexplore.ieee.org/document/9413895) | 2021 | ICASSP | Not available |
| DSDTW | [Differentiable divergences between time series](https://proceedings.mlr.press/v130/blondel21a/blondel21a.pdf) | 2021 | AISTATS | [Github](https://github.com/google-research/soft-dtw-divergences) |
| GromovDTW | [Aligning time series on incomparable spaces](https://proceedings.mlr.press/v130/cohen21a.html) | 2021 | AISTATS | [Github](https://github.com/samcohen16/Aligning-Time-Series) |
| Dilate | [Shape and time distortion loss for training deep time series forecasting models](https://openreview.net/forum?id=r1ld_NBxIB) | 2019 | NeurIPS | [Github](https://github.com/vincent-leguen/DILATE) |
| ShapeDTW | [shapeDTW: Shape dynamic time warping](https://www.sciencedirect.com/science/article/abs/pii/S0031320317303710) | 2018 | PR | [Github](https://github.com/jiapingz/shapeDTW) |
| LDTWs | [Dynamic time warping under limited warping path length](https://www.sciencedirect.com/science/article/abs/pii/S0020025517304176) | 2017 | Inf. Sci. | [Github](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw_limited_warping_length.html) |
| SoftDTW | [Soft-dtw: a differentiable loss function for time-series](https://dl.acm.org/doi/10.5555/3305381.3305474) | 2017 | ICML | [GitHub](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.soft_dtw.html) |
| DTW | [Dynamic programming algorithm optimization for spoken word recognition](https://ieeexplore.ieee.org/document/1163055) | 2003 | IEEE TSAP | [Github](https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html) |

### 3.3 Distribution Balancing Objectives

#### Discrepancy Minimization

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| DistDF | [DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment](hhttps://openreview.net/forum?id=VrdLwUmzBy) | 2026 | ICLR | [Github](https://github.com/Master-PLC/DistDF) |
| PSLoss | [Patch-wise Structural Loss for Time Series Forecasting](https://openreview.net/forum?id=p1KkW2kgDp) | 2025 | ICML | [Github](https://github.com/Dilfiraa/PS_Loss%7D) |
| PSW | [Optimal Transport for Time Series Imputation](https://openreview.net/forum?id=xPTzjpIQNp) | 2025 | ICLR | [Github](https://github.com/FMLYD/PSW-I) |

#### Adversarial Training 

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| AttnWGAIN | [AttnWGAIN: Attention-Based Wasserstein Generative Adversarial Imputation Network for IoT Multivariate Time Series](https://ieeexplore.ieee.org/document/11131293) | 2025 | IEEE TCE | [Github](https://github.com/wzq-come-on/AttnWGAIN.git) |
| SDGCN | [Transformer-Based Generative Adversarial Network for Traffic Forecasting](https://ieeexplore.ieee.org/document/11133428) | 2025 | IEEE TCE | Not available |
| WRCGAN | [Generative representation learning in Recurrent Neural Networks for causal timeseries forecasting](https://ieeexplore.ieee.org/document/10643032) | 2024 | IEEE TAI | Not available |
| TrendGCN | [Enhancing the robustness via adversarial learning and joint spatial-temporal embeddings in traffic forecasting](https://dl.acm.org/doi/10.1145/3583780.3614868) | 2023 | CIKM | [Github](https://github.com/juyongjiang/TrendGCN) |
| AST | [Adversarial sparse transformer for time series forecasting](https://proceedings.neurips.cc/paper/2020/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html) | 2020 | NeurIPS | [Github](https://github.com/hihihihiwsf/AST) |

### 3.4 Conditional Generation Objectives

#### Diffusion-based Generation

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Sundial | [Sundial: A Family of Highly Capable Time Series Foundation Models](https://openreview.net/forum?id=LO7ciRpjI5) | 2025 | ICML | [Github](https://github.com/thuml/Sundial) |
| TimeDiff | [Non-autoregressive conditional diffusion models for time series prediction](https://dl.acm.org/doi/10.5555/3618408.3619692) | 2023 | ICML | Not available |
| StochDif | [Stochastic Diffusion: A Diffusion Based Model for Stochastic Time Series Forecasting](https://dl.acm.org/doi/10.1145/3711896.3737137) | 2025 | SIGKDD | Not available |
| D3VAE | [Generative time series forecasting with diffusion, denoise, and disentanglement](https://dl.acm.org/doi/10.5555/3600270.3601942) | 2022 | NeurIPS | [Github]( https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE) |
| CSDI | [Csdi: Conditional score-based diffusion models for probabilistic time series imputation](https://openreview.net/forum?id=VzuIzbRDrum) | 2021 | NeurIPS | [GitHub](https://github.com/ermongroup/CSDI) |
| TimeWeaver | [Time Weaver: A Conditional Time Series Generation Model](https://openreview.net/forum?id=WpKDeixmFr) | 2024 | ICML | Not available |
| D3M | [Probabilistic time series modeling with decomposable denoising diffusion model](https://openreview.net/forum?id=BNH8spaR3l) | 2024 | ICML | Not available |
| TimeGrad | [Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting](https://proceedings.mlr.press/v139/rasul21a/rasul21a.pdf) | 2021 | ICML | [Github](https://github.com/zalandoresearch/pytorch-ts) |
| SSD-TS | [SSD-TS: Exploring the potential of linear state space models for diffusion models in time series imputation](https://dl.acm.org/doi/10.1145/3711896.3737135) | 2025 | SIGKDD | [Github](https://github.com/decisionintelligence/SSD-TS) |
| SSSD | [Diffusion-based time series imputation and forecasting with structured atate apace models](https://openreview.net/forum?id=hHiIbk7ApW) | 2023 | TMLR | [Github](https://github.com/AI4HealthUOL/SSSD) |
| Diffusion-TS | [Diffusion-TS: Interpretable Diffusion for General Time Series Generation](https://openreview.net/forum?id=4h1apFjO99) | 2025 | ICLR | [Github](https://github.com/Y-debug-sys/Diffusion-TS) |
| TMDM | [Transformer-modulated diffusion models for probabilistic multivariate time series forecasting](https://openreview.net/forum?id=qae04YACHs) | 2024 | ICLR | [Github](https://github.com/LiYuxin321/TMDM) |
| D3U | [Diffusion-based decoupled deterministic and uncertain framework for probabilistic multivariate time series forecasting](https://openreview.net/forum?id=HdUkF1Qk7g) | 2025 | ICLR | [Github](https://github.com/Torea-L/D3U) |
| TimeDart | [Timedart: A diffusion autoregressive transformer for self-supervised time series representation](https://openreview.net/forum?id=v2G9HML7ep&noteId=kMFf8iFNdu) | 2025 | ICML | [Github](https://github.com/Melmaphother/TimeDART%7D) |
| NSDiff | [Non-stationary Diffusion For Probabilistic Time Series Forecasting](https://openreview.net/forum?id=afpc1MFMYU&noteId=dHebxGz0sZ) | 2025 | ICML | [Github](https://github.com/wwy155/NsDiff) |
| CNDiff | [Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting](https://openreview.net/forum?id=kcUNMKqrCg) | 2025 | ICML | [Github](https://github.com/quest-lab-iisc/CNDiff) |
| MG-TSD | [MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process](https://openreview.net/forum?id=CZiY6OLktd) | 2024 | ICLR | [Github](https://github.com/Hundredl/MG-TSD) |
| TCDM | [TCDM: A Temporal Correlation-Empowered Diffusion Model for Time Series Forecasting](https://www.ijcai.org/proceedings/2025/0749.pdf) | 2025 | IJCAI | Not available |

#### Autoregression-based Generation

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Timer-XL | [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://openreview.net/forum?id=KMCJXjlDDr&noteId=7jSaM7SC1f) | 2025 | ICLR | [Github](https://github.com/thuml/Timer-XL) |
| TimeBase | [TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting](https://openreview.net/forum?id=GhTdNOMfOD) | 2025 | ICML | [Github](https://github.com/hqh0728/TimeBase%7D) |
| MoLA | [Mixture of Low Rank Adaptation with Partial Parameter Sharing for Time Series Forecasting](https://arxiv.org/abs/2505.17872) | 2025 | Arxiv | [Github](https://anonymous.4open.science/r/MoLA-BC92) |
| LangTime | [LangTime: A Language-Guided Unified Model for Time Series Forecasting with Proximal Policy Optimization](https://openreview.net/forum?id=VfoKOD65Zq&noteId=3cn4eRwRCU) | 2025 | ICML | [Github](https://github.com/niuwz/LangTime) |
| AutoTimes | [Autotimes: Autoregressive time series forecasters via large language models](https://openreview.net/forum?id=FOvZztnp1H&referrer=%5Bthe%20profile%20of%20Jianmin%20Wang%5D(%2Fprofile%3Fid%3D~Jianmin_Wang1)) | 2024 | NeurIPS | [Github](https://github.com/thuml/AutoTimes) |
| Timer | [Timer: generative pre-trained transformers are large time series models](https://dl.acm.org/doi/10.5555/3692070.3693383) | 2024 | ICML | [Github](https://github.com/thuml/Large-Time-Series-Model) |
| DeepAR | [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888) | 2020 | Int. J. Forecast | [GitHub](https://github.com/awslabs/gluon-ts) |
| LSTNet | [Modeling long-and short-term temporal patterns with deep neural networks](https://dl.acm.org/doi/10.1145/3209978.3210006) | 2018 | SIGIR | [GitHub](https://github.com/laiguokun/LSTNet) |

## 4. Related Surveys

| Survey | Year | Venue | Code |
|------------|-------|------|-------------------|
| [Deep learning for time series forecasting: Tutorial and literature survey](https://dl.acm.org/doi/10.1145/3533382) | 2024 | ACM Comput. Surv. | Not Available |
| [Deep time series models: A comprehensive survey and benchmark]() | 2024 | Arxiv | [Github](https://github.com/thuml/Time-Series-Library) |
| [Transformers in time series: a survey]() | 2023 | IJCAI | Not Available |
| [A survey on deep learning based time series analysis with frequency transformation]() | 2025 | SIGKDD | [Github](https://github.com/qingsongedu/time-series-transformers-review?tab=readme-ov-file) |
| [Foundation models for time series analysis: A tutorial and survey]() | 2024 | SIGKDD | Not Available |
| [Self-supervised learning for time series analysis: Taxonomy, progress, and prospect]() | 2024 | TPAMI | [Github](https://github.com/qingsongedu/Awesome-SSL4TS) |
| [Survey on research of rnn-based spatio-temporal sequence prediction algorithms](https://www.techscience.com/jbd/v3n3/45671) | 2021 | J. Big Data | Not Available | 
| [Time series data augmentation for deep learning: A survey](https://www.ijcai.org/proceedings/2021/0631.pdf) | 2021 | IJCAI | Not Available |
| [A survey on diffusion models for time series and spatio-temporal data](https://arxiv.org/abs/2404.18886) | ACM Comput. Surv. | 2024 | [Github](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model) |
| [Diffusion models for time-series applications: a survey](https://arxiv.org/abs/2305.00624) | Front. Inf. Technol. Electron. Eng. | 2024 | Not Available |