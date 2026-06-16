# 图像描述评估方法综述

这是一个关于图像描述评估方法的综述博客，总结了近期的代表性工作和我们正在探索的研究方向。

## 在线阅读

- [博客页面](blog/index.html)
- [GitHub项目](https://github.com/yourusername/granularity-eval)

## 内容概览

本博客涵盖以下内容：

1. **评估范式概览**
   - VQA-based 评估（CapsBench）
   - Matching-based 评估（CAPTURE, CompreScore）
   - VLM-as-a-Judge 评估（CAPArena）

2. **范式对比与互补性**
   - 不同范式的优势和局限
   - 准确性 vs 完整性的权衡
   - 评估粒度的差异

3. **未来方向：粒度偏差问题**
   - 问题定义和动机
   - 3×3 实验设计矩阵
   - 初步实验结果

## 相关论文

- [CapsBench: A Benchmark for Evaluating Image Captioning Models](https://arxiv.org/abs/2405.19092)
- [CAPTURE: Comprehensive Attributes and Precision-Based Target Evaluation](https://arxiv.org/abs/2412.08614)
- [CompreScore: Comprehensive Scene Graph-Based Image Captioning Evaluation](https://arxiv.org/abs/2503.12329v1)
- [CAPArena: Benchmarking Image Captioning with Large Language Models](https://arxiv.org/abs/2409.10695)

## 实验代码

粒度偏差验证实验的代码位于本仓库：

- `pilot.py` - 小规模验证实验（20张图像）
- `generate.py` - 描述生成脚本
- `compute_metrics.py` - 评估指标计算
- `analysis.py` - 结果分析

## 本地运行博客

```bash
# 克隆仓库
git clone https://github.com/yourusername/granularity-eval.git
cd granularity-eval

# 在浏览器中打开
open blog/index.html  # macOS
# 或直接用浏览器打开 blog/index.html 文件
```

## GitHub Pages 部署

将此仓库推送到GitHub后，在仓库设置中启用GitHub Pages：

1. 进入仓库 Settings > Pages
2. Source 选择 `main` 分支
3. Folder 选择 `/blog`
4. 保存后即可通过 `https://yourusername.github.io/granularity-eval/` 访问

## 联系方式

如有问题或建议，欢迎提issue或联系作者。

## 许可

MIT License
