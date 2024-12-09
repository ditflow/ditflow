<div align="center">

## Video Motion Transfer with Diffusion Transformers

<p>
⚡ <b> Training-free </b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
⚙️ <b> Optimization-based </b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
🔄 <b> Zero-shot injection</b>
</p>

<a href="https://arxiv.org/abs/"><img src='https://img.shields.io/badge/arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://ditflow.github.io"><img src='https://img.shields.io/badge/Project-Page-Green'></a>

<!-- Official PyTorch implementation of DiTFlow for CogVideoX models -->

**Code release coming soon...**

</div>

## Transfer motion from video to new content

<img src="assets/git_teaser.png">

🔍 **AMF Extraction**: Process reference videos through pre-trained DiT to extract Attention Motion Flow (AMF)

⚙️ **Motion Optimization**: Guide latent denoising with AMF loss in a training-free manner to reproduce reference motion

🔄 **Zero-shot Motion Injection**: Optimized transformer positional embeddings can be *injected in new generation* for zero-shot motion transfer

📊 **Evaluation**: Outperforms existing methods (<a href="https://arxiv.org/abs/2405.14864v2" target="_blank">MOFT</a>, <a href="https://github.com/diffusion-motion-transfer/diffusion-motion-transfer" target="_blank">SMM</a>) across multiple metrics and human evaluation when implemented for DiTs

## News 📰
**[09/12/2024]** 🔥🔥🔥 Our paper, *Video Motion Transfer with Diffusion Transformers*, has been archived.