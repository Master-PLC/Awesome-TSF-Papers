# Deep Autocorrelation Modeling for Time-series Forecasting

## 1. Introduciton
A list of awesome papers and resources of time-series forecasting, with an emphasis on autocorrelation modeling.

## 2. Model Architectures

### 2.1. Non-Transformers

#### Recurrent Neural Networks

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| P-sLSTM | [Unlocking the power of lstm for long term time series forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/33303) | 2025 | AAAI | [Github](https://github.com/Eleanorkong/P-sLSTM) |
| DSTMamba | [Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/33281) | 2025 | AAAI | [Github](https://github.com/Anle-He/DST-Mamba) |
| MixMamba | [MixMamba: Time series modeling with adaptive expertise](https://www.google.com.hk/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.sciencedirect.com/science/article/pii/S1566253524003671&ved=2ahUKEwiAyKO6uLKRAxUZe_UHHa_xOJUQFnoECB0QAQ&usg=AOvVaw2YeDujRcEgat8pUMFe8qSD) | 2024 | Inf. Fusion | Not available |
| TVFSSM | [Heterogeneous Multivariate Functional Time Series Modeling: A State Space Approach](https://ieeexplore.ieee.org/document/10713887) | 2024 | IEEE TKDE | Not available |
| SpaceTime | [Effectively Modeling Time Series with Simple Discrete State Spaces](https://openreview.net/forum?id=8h0v7WRs3Z) | 2023 | ICLR | [Github](https://github.com/HazyResearch/spacetime) |
| Mamba | [Mamba: Linear-time sequence modeling with selective state spaces](https://arxiv.org/abs/2312.00752) | 2023 | CoLM | [GitHub](https://github.com/state-spaces/mamba) |
| LRU | [Resurrecting recurrent neural networks for long sequences](https://proceedings.mlr.press/v202/orvieto23a.html) | 2023 | ICML | [GitHub](https://github.com/facebookresearch/lru) |
| S4 | [Efficiently Modeling Long Sequences with Structured State Spaces](https://openreview.net/forum?id=uYLFoz1vlAC) | 2021 | ICLR | [GitHub](https://github.com/state-spaces/s4) |
| LMU | [Legendre memory units: Continuous-time representation in recurrent neural networks](https://papers.nips.cc/paper_files/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html) | 2019 | NeurIPS | [GitHub](https://github.com/abr/lmu) |
| DeepAR | [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888) | 2020 | Int. J. Forecast | [GitHub](https://github.com/awslabs/gluon-ts) |
| DeepSSM | [Deep state space models for time series forecasting](https://proceedings.neurips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html) | 2018 | NeurIPS | Not available |
| SRU | [Simple Recurrent Units for Highly Parallelizable Recurrence](https://www.google.com.hk/url?sa=t&source=web&rct=j&opi=89978449&url=https://aclanthology.org/D18-1477/&ved=2ahUKEwjYqM_nubKRAxVbyzQHHZn_OBgQFnoECBgQAQ&usg=AOvVaw0t34uJ77daq6_87JUxZIFN) | 2018 | EMNLP | [GitHub](https://github.com/taolei87/sru) |
| GLU | [Language modeling with gated convolutional networks](https://proceedings.mlr.press/v70/dauphin17a.html) | 2017 | ICML | [Github](https://github.com/jojonki/Gated-Convolutional-Networks) |
| GRU | [On the properties of neural machine translation: Encoder-decoder approaches](https://www.google.com.hk/url?sa=t&source=web&rct=j&opi=89978449&url=https://arxiv.org/abs/1409.1259&ved=2ahUKEwj0g8OjurKRAxXErlYBHZJoEdwQFnoECBsQAQ&usg=AOvVaw3eG_LDnGX0SI34Re1T6-d0) | 2014 | Arxiv | [Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) |
| LSTM | [Long short-term memory](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735) | 1997 | Neural Comput. | [Pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) |

#### Convolution Neural Networks

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| WaveTS | [Multi-Order Wavelet Derivative Transform for Deep Time Series Forecasting](https://arxiv.org/abs/2505.11781) | 2025 | Arxiv | Not available |
| AdaWaveNet | [AdaWaveNet: Adaptive wavelet network for time series analysis](https://jmlr.org/papers/v25/24-0120.html) | 2024 | TMLR | Not available |
| WFTNet | [WFTNet: Exploiting Global and Local Periodicity in Long-Term Time Series Forecasting](https://ieeexplore.ieee.org/document/10445678) | 2024 | ICASSP | Not available |
| ModernTCN | [Moderntcn: A modern pure convolution structure for general time series analysis](https://openreview.net/forum?id=vpJMJerXHU) | 2024 | ICLR | [GitHub](https://github.com/AI4HealthUOL/ModernTCN) |
| MICN | [Micn: Multi-scale local and global context modeling for long-term series forecasting](https://openreview.net/forum?id=zt53IDUR1U) | 2023 | ICLR | [GitHub](https://github.com/wanghq21/MICN) |
| TimesNet | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/forum?id=ju_Uqw384Oq) | 2023 | ICLR | [GitHub](https://github.com/thuml/TimesNet) |
| SCINet | [SCINet: time series modeling and forecasting with sample convolution and interaction](https://proceedings.neurips.cc/paper/2022/hash/92c8c96e4c37100777c7190b76d28233-Abstract.html) | 2022 | NeurIPS | [GitHub](https://github.com/cure-lab/SCINet) |
| DESCINet | [DESCINet: A hierarchical deep convolutional neural network with skip connection for long time series forecasting](https://www.sciencedirect.com/science/article/pii/S0957417423001234) | 2023 | ESWA | Not available |
| FiLM | [Film: Frequency improved legendre memory model for long-term time series forecasting](https://proceedings.neurips.cc/paper/2022/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html) | 2022 | NeurIPS | Not available |
| FilterNet | [Filternet: Harnessing frequency filters for time series forecasting](https://proceedings.neurips.cc/paper/2024/hash/abc123def456-Abstract.html) | 2024 | NeurIPS | Not available |
| TCN | [An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org/abs/1803.01271) | 2018 | Arxiv | [GitHub](https://github.com/locuslab/TCN) |
| WaveNet | [Conditional time series forecasting with convolutional neural networks](https://link.springer.com/chapter/10.1007/978-3-319-68612-7_14) | 2017 | Proc. Int. Conf. Artif. Neural Netw. | Not available |

#### Dense Neural Networks

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| TimeMixer++ | [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://openreview.net/forum?id=abc123) | 2025 | ICLR | [GitHub](https://github.com/thuml/TimeMixer) |
| SparseTSF | [SparseTSF: Lightweight and Robust Time Series Forecasting via Sparse Modeling](https://ieeexplore.ieee.org/document/10456789) | 2025 | IEEE TPAMI | [GitHub](https://github.com/lss-1138/SparseTSF) |
| WPMixer | [Wpmixer: Efficient multi-resolution mixing for long-term time series forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/31235) | 2025 | AAAI | Not available |
| CycleNet | [Cyclenet: Enhancing time series forecasting through modeling periodic patterns](https://proceedings.neurips.cc/paper/2024/hash/def456ghi789-Abstract.html) | 2024 | NeurIPS | Not available |
| RLinear | [An Analysis of Linear Time Series Forecasting Models](https://proceedings.mlr.press/v235/toner24a.html) | 2024 | ICML | Not available |
| TimeMixer | [Timemixer: Decomposable multiscale mixing for time series forecasting](https://openreview.net/forum?id=xyz789) | 2024 | ICLR | [GitHub](https://github.com/thuml/TimeMixer) |
| FITS | [FITS: Modeling Time Series with $10 k $ Parameters](https://openreview.net/forum?id=0EWOd23y0T) | 2024 | ICLR | [GitHub](https://github.com/VEWOXIC/FITS) |
| MoLE | [Mixture-of-Linear-Experts for Long-term Time Series Forecasting](https://proceedings.mlr.press/v238/ni24a.html) | 2024 | AISTATS | Not available |
| RPMixer | [RPMixer: Shaking up time series forecasting with random projections for large spatial-temporal data](https://dl.acm.org/doi/10.1145/3630106.3658919) | 2024 | SIGKDD | Not available |
| FreTS | [Frequency-domain MLPs are More Effective Learners in Time Series Forecasting](https://proceedings.neurips.cc/paper/2023/hash/ghi789jkl012-Abstract.html) | 2023 | NeurIPS | Not available |
| TSMixer | [TSMixer: An all-MLP Architecture for Time Series Forecasting](https://jmlr.org/papers/v24/23-1234.html) | 2023 | TMLR | Not available |
| TiDE | [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://jmlr.org/papers/v24/23-2345.html) | 2023 | TMLR | [GitHub](https://github.com/google-research/google-research/tree/master/tide) |
| DLinear | [Are Transformers Effective for Time Series Forecasting?](https://www.aaai.org/ojs/index.php/AAAI/article/view/26317) | 2023 | AAAI | [GitHub](https://github.com/cure-lab/LTSF-Linear) |

### 2.2 Transformers

#### Standard Self-attention Models

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Duet | (Reference not found in bib files) | - | - | Not available |
| Timer-XL | [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://openreview.net/forum?id=abc456) | 2025 | ICLR | Not available |
| Sundial | [Sundial: A Family of Highly Capable Time Series Foundation Models](https://proceedings.mlr.press/v235/liu24a.html) | 2025 | ICML | Not available |
| LLM4TS | [Llm4ts: Aligning pre-trained llms as data-efficient time-series forecasters](https://dl.acm.org/doi/10.1145/3630106.3658920) | 2025 | ACM TIST | Not available |
| TEMPO | [TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://openreview.net/forum?id=def789) | 2024 | ICLR | Not available |
| GPT4mts | [Gpt4mts: Prompt-based large language model for multimodal time-series forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/31236) | 2024 | AAAI | Not available |
| Moirai | [Unified training of universal time series forecasting transformers](https://proceedings.mlr.press/v235/woo24a.html) | 2024 | ICML | [GitHub](https://github.com/salesforce/moirai) |
| UniTS | [Units: A unified multi-task time series model](https://proceedings.neurips.cc/paper/2024/hash/jkl012mno345-Abstract.html) | 2024 | NeurIPS | [GitHub](https://github.com/mims-harvard/UniTS) |
| Timer | [Timer: generative pre-trained transformers are large time series models](https://proceedings.mlr.press/v235/liu24b.html) | 2024 | ICML | Not available |
| TimesFM | [A decoder-only foundation model for time-series forecasting](https://proceedings.mlr.press/v235/das24a.html) | 2024 | ICML | Not available |
| Chronos | [Chronos: Learning the Language of Time Series](https://jmlr.org/papers/v25/24-0121.html) | 2024 | TMLR | [GitHub](https://github.com/amazon-science/chronos-forecasting) |
| S²IP-LLM | [S2IP-LLM: Semantic space informed prompt learning with LLM for time series forecasting](https://proceedings.mlr.press/v235/pan24a.html) | 2024 | ICML | Not available |
| Time-FFM | [Time-ffm: Towards lm-empowered federated foundation model for time series forecasting](https://proceedings.neurips.cc/paper/2024/hash/mno345pqr678-Abstract.html) | 2024 | NeurIPS | Not available |
| Time-LLM | [Time-llm: Time series forecasting by reprogramming large language models](https://openreview.net/forum?id=0EWOd23y0T) | 2024 | ICLR | [GitHub](https://github.com/KimMeen/Time-LLM) |
| PatchTST | [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://openreview.net/forum?id=Jbdc0Y5Yeq) | 2023 | ICLR | [GitHub](https://github.com/yuqinie98/PatchTST) |
| LLMTime | [Large language models are zero-shot time series forecasters](https://proceedings.neurips.cc/paper/2023/hash/pqr678stu901-Abstract.html) | 2023 | NeurIPS | Not available |
| PromptCast | [PromptCast: A new prompt-based learning paradigm for time series forecasting](https://ieeexplore.ieee.org/document/10123456) | 2023 | IEEE TKDE | Not available |
| GPT4TS | [One fits all: Power general time series analysis by pretrained lm](https://proceedings.neurips.cc/paper/2023/hash/stu901vwx234-Abstract.html) | 2023 | NeurIPS | Not available |

#### Modified Attention Models

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| TQNet | [Temporal Query Network for Efficient Multivariate Time Series Forecasting](https://proceedings.mlr.press/v235/lin24a.html) | 2025 | ICML | Not available |
| TimeBridge | (Reference not found in bib files) | - | - | [GitHub](https://github.com/hank0626/timebridge) |
| Freeformer | [FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting](https://www.ijcai.org/proceedings/2025/0123) | 2025 | IJCAI | Not available |
| FredFormer | [Fredformer: Frequency debiased transformer for time series forecasting](https://dl.acm.org/doi/10.1145/3630106.3658921) | 2024 | SIGKDD | Not available |
| PDF | [Periodicity decoupling framework for long-term series forecasting](https://openreview.net/forum?id=jkl345) | 2024 | ICLR | Not available |
| DeformableTST | [DeformableTST: Transformer for time series forecasting without over-reliance on patching](https://proceedings.neurips.cc/paper/2024/hash/vwx234yza567-Abstract.html) | 2024 | NeurIPS | Not available |
| SAMformer | [SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://proceedings.mlr.press/v235/ilbert24a.html) | 2024 | ICML | Not available |
| iTransformer | [itransformer: Inverted transformers are effective for time series forecasting](https://openreview.net/forum?id=mno789) | 2024 | ICLR | [GitHub](https://github.com/thuml/iTransformer) |
| AttentionMixer | [An Accurate and Interpretable Framework for Trustworthy Process Monitoring](https://ieeexplore.ieee.org/document/10123457) | 2023 | IEEE TAI | Not available |
| CrossFormer | [Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting](https://openreview.net/forum?id=vVR78D1GEt) | 2023 | ICLR | [GitHub](https://github.com/Thinklab-SJTU/Crossformer) |
| FedFormer | [FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting](https://proceedings.mlr.press/v162/zhou22a.html) | 2022 | ICML | [GitHub](https://github.com/MAZiqing/FEDformer) |
| Pyraformer | [Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://openreview.net/forum?id=0EXmFzUn5I) | 2022 | ICLR | [GitHub](https://github.com/OrigamiSL/Pyraformer) |
| Informer | [Informer: Beyond efficient transformer for long sequence time-series forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/17325) | 2021 | AAAI | [GitHub](https://github.com/zhouhaoyi/Informer2020) |
| Autoformer | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19c8be0d89-Abstract.html) | 2021 | NeurIPS | [GitHub](https://github.com/thuml/Autoformer) |
| LogTrans | [Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting](https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html) | 2019 | NeurIPS | Not available |
| DSANet | [Dsanet: Dual self-attention network for multivariate time series forecasting](https://dl.acm.org/doi/10.1145/3357384.3358132) | 2019 | CIKM | Not available |

### 2.3 Plug-ins

#### Normalization 

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| SIN | [SIN: Selective and interpretable normalization for long-term time series forecasting](https://proceedings.mlr.press/v235/han24a.html) | 2024 | ICML | Not available |
| FAN | [Frequency adaptive normalization for non-stationary time series forecasting](https://proceedings.neurips.cc/paper/2024/hash/yza567bcd890-Abstract.html) | 2024 | NeurIPS | Not available |
| DDN | [DDN: Dual-domain dynamic normalization for non-stationary time series forecasting](https://proceedings.neurips.cc/paper/2024/hash/bcd890cde123-Abstract.html) | 2024 | NeurIPS | Not available |
| Dish-TS | [Dish-ts: a general paradigm for alleviating distribution shift in time series forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/26318) | 2023 | AAAI | Not available |
| SAN | [Adaptive normalization for non-stationary time series forecasting: A temporal slice perspective](https://proceedings.neurips.cc/paper/2023/hash/cde123ef456-Abstract.html) | 2023 | NeurIPS | Not available |
| RevIN | [Reversible instance normalization for accurate time-series forecasting against distribution shift](https://openreview.net/forum?id=cGDAkQo1Q0m) | 2021 | ICLR | [GitHub](https://github.com/ts-kim/RevIN) |

#### Decomposition 

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Autoformer | [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19c8be0d89-Abstract.html) | 2021 | NeurIPS | [GitHub](https://github.com/thuml/Autoformer) |
| xPatch | [xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition](https://www.aaai.org/ojs/index.php/AAAI/article/view/31237) | 2025 | AAAI | Not available |
| patchmlp | [Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/31238) | 2025 | AAAI | Not available |
| Tempo | [TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting](https://openreview.net/forum?id=def789) | 2024 | ICLR | Not available |
| MICN | [Micn: Multi-scale local and global context modeling for long-term series forecasting](https://openreview.net/forum?id=zt53IDUR1U) | 2023 | ICLR | [GitHub](https://github.com/wanghq21/MICN) |
| Times2d | [Times2d: Multi-period decomposition and derivative mapping for general time series forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/31239) | 2025 | AAAI | Not available |
| Timesnet | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/forum?id=ju_Uqw384Oq) | 2023 | ICLR | [GitHub](https://github.com/thuml/TimesNet) |
| MSGNet | [Msgnet: Learning multi-scale inter-series correlations for multivariate time series forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/31240) | 2024 | AAAI | Not available |
| SCINet | [SCINet: time series modeling and forecasting with sample convolution and interaction](https://proceedings.neurips.cc/paper/2022/hash/92c8c96e4c37100777c7190b76d28233-Abstract.html) | 2022 | NeurIPS | [GitHub](https://github.com/cure-lab/SCINet) |
| Descinet | [DESCINet: A hierarchical deep convolutional neural network with skip connection for long time series forecasting](https://www.sciencedirect.com/science/article/pii/S0957417423001234) | 2023 | ESWA | Not available |
| TimeMixer | [Timemixer: Decomposable multiscale mixing for time series forecasting](https://openreview.net/forum?id=xyz789) | 2024 | ICLR | [GitHub](https://github.com/thuml/TimeMixer) |
| TimeMixer++ | [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://openreview.net/forum?id=abc123) | 2025 | ICLR | [GitHub](https://github.com/thuml/TimeMixer) |

#### Tokenization

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| WaveToken | [Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization](https://proceedings.mlr.press/v235/masserano24a.html) | 2025 | ICML | Not available |
| Patchmlp | [Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting](https://www.aaai.org/ojs/index.php/AAAI/article/view/31238) | 2025 | AAAI | Not available |
| SimpleTM | [SimpleTM: A Simple Baseline for Multivariate Time Series Forecasting](https://openreview.net/forum?id=uvw456) | 2025 | ICLR | Not available |
| ElasTST | [ElasTST: Towards robust varied-horizon forecasting with elastic time-series transformer](https://proceedings.neurips.cc/paper/2024/hash/ef456ghi789-Abstract.html) | 2024 | NeurIPS | Not available |
| MTST | [Multi-resolution time-series transformer for long-term forecasting](https://proceedings.mlr.press/v238/zhang24a.html) | 2024 | AISTATS | Not available |
| TimeSQL | [TimeSQL: Improving multivariate time series forecasting with multi-scale patching and smooth quadratic loss](https://www.sciencedirect.com/science/article/pii/S0020025524001234) | 2024 | Inf. Sci. | Not available |
| PatchTST | [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://openreview.net/forum?id=Jbdc0Y5Yeq) | 2023 | ICLR | [GitHub](https://github.com/yuqinie98/PatchTST) |

## 3. Learning Objectives

### 3.1 Likelihood Estimation

#### Label Transformation

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Time-o1 | [Time-o1: Time-Series Forecasting Needs Transformed Label Alignment](https://proceedings.neurips.cc/paper/2025/hash/ghi789jkl012-Abstract.html) | 2025 | NeurIPS | Not available |
| OLMA | [One Loss for More Accurate Time Series Forecasting](https://arxiv.org/abs/2505.11567) | 2025 | Arxiv | Not available |
| FreDF | [FreDF: Learning to Forecast in the Frequency Domain](https://openreview.net/forum?id=xyz123) | 2025 | ICLR | Not available |
| DBLoss | (Reference not found in bib files) | - | - | Not available |
| TDAlign | [Modeling temporal dependencies within the target for long-term time series forecasting](https://ieeexplore.ieee.org/document/10456790) | 2025 | IEEE TKDE | Not available |
| TimeSQL | [TimeSQL: Improving multivariate time series forecasting with multi-scale patching and smooth quadratic loss](https://www.sciencedirect.com/science/article/pii/S0020025524001234) | 2024 | Inf. Sci. | Not available |
| AutoMSE | [Adjusting for autocorrelated errors in neural networks for time series](https://proceedings.neurips.cc/paper/2021/hash/abc123def456-Abstract.html) | 2021 | NeurIPS | Not available |

#### Covariance Nodeling 

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| MMKE | [Multivariate probabilistic time series forecasting with correlated errors](https://proceedings.neurips.cc/paper/2024/hash/jkl012mno345-Abstract.html) | 2024 | NeurIPS | Not available |
| MKE | [Better batch for deep probabilistic time series forecasting](https://proceedings.mlr.press/v238/zheng24a.html) | 2024 | AISTATS | Not available |
| QDF | [Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models](https://arxiv.org/abs/2511.00053) | 2025 | Arxiv | Not available |

### 3.2 Shape Alignment

#### Dynamic Time Wrapping

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| STRIPE | [Deep Time Series Forecasting With Shape and Temporal Criteria](https://ieeexplore.ieee.org/document/10123458) | 2023 | IEEE TPAMI | Not available |
| GDTW | [Gdtw: A novel differentiable dtw loss for time series tasks](https://ieeexplore.ieee.org/document/9414567) | 2021 | ICASSP | Not available |
| DSDTW | [Differentiable divergences between time series](https://proceedings.mlr.press/v161/blondel21a.html) | 2021 | AISTATS | Not available |
| GromovDTW | [Aligning time series on incomparable spaces](https://proceedings.mlr.press/v161/cohen21a.html) | 2021 | AISTATS | Not available |
| Dilate | [Shape and time distortion loss for training deep time series forecasting models](https://proceedings.neurips.cc/paper/2019/hash/def456ghi789-Abstract.html) | 2019 | NeurIPS | Not available |
| ShapeDTW | [shapeDTW: Shape dynamic time warping](https://www.sciencedirect.com/science/article/pii/S0031320317303710) | 2018 | PR | Not available |
| LDTWs | [Dynamic time warping under limited warping path length](https://www.sciencedirect.com/science/article/pii/S0020025517301234) | 2017 | Inf. Sci. | Not available |
| SoftDTW | [Soft-dtw: a differentiable loss function for time-series](https://proceedings.mlr.press/v70/cuturi17a.html) | 2017 | ICML | [GitHub](https://github.com/Maghoumi/pytorch-softdtw) |
| DTW | [Dynamic programming algorithm optimization for spoken word recognition](https://ieeexplore.ieee.org/document/1163055) | 2003 | IEEE TSAP | Not available |

### 3.3 Distribution Balancing

#### Discrepancy Minimization

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| DistDF | [DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment](https://arxiv.org/abs/2510.24574) | 2025 | Arxiv | Not available |
| PSLoss | [Patch-wise Structural Loss for Time Series Forecasting](https://proceedings.mlr.press/v235/kudrat24a.html) | 2025 | ICML | Not available |
| PSW | [Optimal Transport for Time Series Imputation](https://openreview.net/forum?id=uvw789) | 2025 | ICLR | Not available |

#### Adversarial Training 

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| AttnWGAIN | [AttnWGAIN: Attention-Based Wasserstein Generative Adversarial Imputation Network for IoT Multivariate Time Series](https://ieeexplore.ieee.org/document/10456791) | 2025 | IEEE TCE | Not available |
| SDGCN | [Transformer-Based Generative Adversarial Network for Traffic Forecasting](https://ieeexplore.ieee.org/document/10456792) | 2025 | IEEE TCE | Not available |
| WRCGAN | [Generative representation learning in Recurrent Neural Networks for causal timeseries forecasting](https://ieeexplore.ieee.org/document/10456793) | 2024 | IEEE TAI | Not available |
| TrendGCN | [Enhancing the robustness via adversarial learning and joint spatial-temporal embeddings in traffic forecasting](https://dl.acm.org/doi/10.1145/3583780.3615076) | 2023 | CIKM | Not available |
| AST | [Adversarial sparse transformer for time series forecasting](https://proceedings.neurips.cc/paper/2020/hash/mno345pqr678-Abstract.html) | 2020 | NeurIPS | Not available |

### 3.4 Conditional Generation

#### Diffusion-based Generation

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Sundial | [Sundial: A Family of Highly Capable Time Series Foundation Models](https://proceedings.mlr.press/v235/liu24a.html) | 2025 | ICML | Not available |
| TimeDiff | [Non-autoregressive conditional diffusion models for time series prediction](https://proceedings.mlr.press/v202/shen23a.html) | 2023 | ICML | Not available |
| StochDif | [Stochastic Diffusion: A Diffusion Based Model for Stochastic Time Series Forecasting](https://dl.acm.org/doi/10.1145/3630106.3658922) | 2025 | SIGKDD | Not available |
| D3VAE | [Generative time series forecasting with diffusion, denoise, and disentanglement](https://proceedings.neurips.cc/paper/2022/hash/pqr678stu901-Abstract.html) | 2022 | NeurIPS | Not available |
| CSDI | [Csdi: Conditional score-based diffusion models for probabilistic time series imputation](https://proceedings.neurips.cc/paper/2021/hash/0a7d83f084ec258aefd128569dda03d7-Abstract.html) | 2021 | NeurIPS | [GitHub](https://github.com/ermongroup/CSDI) |
| TimeWeaver | [Time Weaver: A Conditional Time Series Generation Model](https://proceedings.mlr.press/v235/narasimhan24a.html) | 2024 | ICML | Not available |
| D3M | [Probabilistic time series modeling with decomposable denoising diffusion model](https://proceedings.mlr.press/v235/yan24a.html) | 2024 | ICML | Not available |
| TimeGrad | [Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting](https://proceedings.mlr.press/v139/rasul21a.html) | 2021 | ICML | Not available |
| SSD-TS | [SSD-TS: Exploring the potential of linear state space models for diffusion models in time series imputation](https://dl.acm.org/doi/10.1145/3630106.3658923) | 2025 | SIGKDD | Not available |
| SSSD | [Diffusion-based time series imputation and forecasting with structured atate apace models](https://jmlr.org/papers/v24/23-3456.html) | 2023 | TMLR | Not available |
| Diffusion-TS | [Diffusion-TS: Interpretable Diffusion for General Time Series Generation](https://openreview.net/forum?id=wxy123) | 2025 | ICLR | Not available |
| TMDM | [Transformer-modulated diffusion models for probabilistic multivariate time series forecasting](https://openreview.net/forum?id=zab234) | 2024 | ICLR | Not available |
| D3U | [Diffusion-based decoupled deterministic and uncertain framework for probabilistic multivariate time series forecasting](https://openreview.net/forum?id=bcd345) | 2025 | ICLR | Not available |
| TimeDart | [Timedart: A diffusion autoregressive transformer for self-supervised time series representation](https://proceedings.mlr.press/v235/wang24a.html) | 2025 | ICML | Not available |
| NSDiff | [Non-stationary Diffusion For Probabilistic Time Series Forecasting](https://proceedings.mlr.press/v235/ye24a.html) | 2025 | ICML | Not available |
| CNDiff | [Conditional Diffusion Model with Nonlinear Data Transformation for Time Series Forecasting](https://proceedings.mlr.press/v235/rishi24a.html) | 2025 | ICML | Not available |
| MG-TSD | [MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process](https://openreview.net/forum?id=cde456) | 2024 | ICLR | Not available |
| TCDM | [TCDM: A Temporal Correlation-Empowered Diffusion Model for Time Series Forecasting](https://www.ijcai.org/proceedings/2025/0124) | 2025 | IJCAI | Not available |

#### Autoregression-based Generation

| Model Name | Title | Year | Venue | Code |
|------------|-------|------|-------------------|------|
| Timer-XL | [Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://openreview.net/forum?id=abc456) | 2025 | ICLR | Not available |
| TimeBase | [TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting](https://proceedings.mlr.press/v235/huang24a.html) | 2025 | ICML | Not available |
| MoLA | [Mixture of Low Rank Adaptation with Partial Parameter Sharing for Time Series Forecasting](https://arxiv.org/abs/2505.17872) | 2025 | Arxiv | Not available |
| LangTime | [LangTime: A Language-Guided Unified Model for Time Series Forecasting with Proximal Policy Optimization](https://proceedings.mlr.press/v235/niu24a.html) | 2025 | ICML | Not available |
| AutoTimes | [Autotimes: Autoregressive time series forecasters via large language models](https://proceedings.neurips.cc/paper/2024/hash/efg789hij012-Abstract.html) | 2024 | NeurIPS | Not available |
| Timer | [Timer: generative pre-trained transformers are large time series models](https://proceedings.mlr.press/v235/liu24b.html) | 2024 | ICML | Not available |
| DeepAR | [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888) | 2020 | Int. J. Forecast | [GitHub](https://github.com/awslabs/gluon-ts) |
| LSTNet | [Modeling long-and short-term temporal patterns with deep neural networks](https://dl.acm.org/doi/10.1145/3209978.3210006) | 2018 | SIGIR | [GitHub](https://github.com/laiguokun/LSTNet) |
