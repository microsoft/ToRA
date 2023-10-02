
<h1 align="center">
<img src="./docs/static/images/tora_logo.png" width="100" alt="ToRA" />
<br>
ToRA: A Tool-Integrated Reasoning Agent
</h1>

<div align="center">

![](https://img.shields.io/badge/Task-Mathematical%20Reasoning-orange)
![](https://img.shields.io/badge/Model-Release%20Soon-blue)
![](https://img.shields.io/badge/Code%20License-MIT-green)
<br>

</div>

<p align="center">
  <!-- <a href="#-quick-start">Quick Start</a> • -->
  <a href="https://microsoft.github.io/ToRA/">Project Page</a> •
  <a href="https://arxiv.org/pdf/2309.17452.pdf">Paper</a>
  <!-- <a href="#%EF%B8%8F-citation">Citation</a> -->
</p>

<p align="center">
Repo for "ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving"
</p>

<p align="center">
    <img src="./docs/static/images/math_gsm_hist.png" width="1000">
        <br>
    <em>Figure 1: Comparing ToRA with baselines on LLaMA-2 base models from 7B to 70B.</em>
</p>


<h2 align="center">
Tool-Integrated Reasoning Format
</h2>

<p align="center">
<!-- > add img caption for the following figure: a basic example of single-round tool interaction -->
    <img src="./docs/static/images/example.png" width="800">
    <br>
    <em>Figure 2: A basic example of single-round tool interaction, which interleaves rationales with program-based tool use.</em>
</p>

<h2 align="center">
ToRA Training Pipeline
</h2>

<p align="center">
    <img src="./docs/static/images/pipeline.png" width="1000">
    <br>
    <em>Figure 3: Training ToRA contains ① Imitation Learning, and ② output space shaping.</em>
</p>



## Code & Models

🏝️ The code will be cleaned and uploaded within a few days, all ToRA models will be released.


<!-- ## Models


## 🚀 Quick Start

### ⚙️ Setup

```sh
conda create -n tora python=3.10
conda activate tora
pip install -r requirements.txt
```

### ⚡️ Training


### ⚖️ Evaluation


-->

## ☕️ Citation

If you find this repository helpful, please consider citing our paper:

```
@misc{gou2023tora,
      title={ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving}, 
      author={Zhibin Gou and Zhihong Shao and Yeyun Gong and yelong shen and Yujiu Yang and Minlie Huang and Nan Duan and Weizhu Chen},
      year={2023},
      eprint={2309.17452},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

<!-- ## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments. -->
