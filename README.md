
# Bayes-Adaptive RL for LLM Reasoning

Code for [Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning](https://arxiv.org/abs/2505.20561).

Authors: [Shenao Zhang](https://shenao-zhang.github.io)¹, [Yaqing Wang](https://yaqingwang.github.io/)², [Yinxiao Liu](https://scholar.google.com/citations?user=c7HKsEsAAAAJ&hl=en)², [Tianqi Liu](https://scholar.google.com/citations?user=pUKhiMIAAAAJ&hl=en)², [Peter Grabowski](https://scholar.google.com/citations?user=c9APALsAAAAJ&hl=en)³, [Eugene Ie](https://scholar.google.com/citations?user=jNCbl2IAAAAJ&hl=en)³, [Zhaoran Wang](https://zhaoranwang.github.io)¹, [Yunxian Li](https://scholar.google.com/citations?user=Nun8Dy0AAAAJ&hl=en)³.

¹Northwestern University,  ²Google Deepmind,  ³Google.

 We introduce a principled RL framework for stitching together plausible strategies, analogous to linearized best-of-N reasoning, but with explicit step-level guidance on when and how LLMs should reflectively explore.

### Installation

```python
pip install -e .
```
### Run the Code

```bash
bash train_barl.sh
```

## Citation

```bibtex
@article{zhang2025beyond,
  title={Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning},
  author={Zhang, Shenao and Wang, Yaqing and Liu, Yinxiao and Liu, Tianqi and Grabowski, Peter and Ie, Eugene and Wang, Zhaoran and Li, Yunxuan},
  journal={arXiv preprint arXiv:2505.20561},
  year={2025}
}
```

## Acknowledgement
This repository is built upon the OpenRLHF framework. We thank the authors for their great work.
