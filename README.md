这是一个用于实验基础媒体编码与压缩算法的小项目。

当前项目已实现 `LZW` 和 `LZSS`，并提供了 `report/lzw_lzss_demo.ipynb` 用于演示文本与图像数据的编码效果。

现在也支持从指定目录读取文本、图片和音频样本，或直接使用 Hugging Face `datasets` 流水线取样，统计 `LZW/LZSS` 的压缩比。

## 运行实验

命令行入口：

```bash
.venv/bin/python main.py --image-dir path/to/images --audio-dir path/to/audio
```

或使用项目内置的数据集管理目录：

```bash
.venv/bin/python main.py --text-datasets synthetic_structured public_samples --image-datasets synthetic hf_mnist --audio-datasets synthetic hf_librispeech_dummy --limit-per-dataset 10
```

也可以直接使用 Hugging Face `datasets` 流水线而不落盘到项目目录：

```bash
.venv/bin/python main.py --hf-text-dataset wikitext --hf-config-name wikitext-2-raw-v1 --hf-split "train[:10]" --hf-text-column text --limit-per-dataset 10
.venv/bin/python main.py --hf-image-dataset mnist --hf-split "train[:10]" --limit-per-dataset 10
.venv/bin/python main.py --hf-audio-dataset hf-internal-testing/librispeech_asr_dummy --hf-split "validation[:10]" --hf-audio-column audio --limit-per-dataset 10
```

列出当前已发现的数据集：

```bash
.venv/bin/python main.py datasets-list
```

从 Hugging Face 导入前几个样本：

```bash
.venv/bin/python main.py hf-import --category image --dataset-id mnist --dataset-name hf_mnist --split train --limit 10
.venv/bin/python main.py hf-import --category audio --dataset-id speech_commands --dataset-name hf_speech --split train --limit 10
```

说明：

1. `--text-dir`、`--image-dir` 和 `--audio-dir` 至少提供一个，或使用受管数据集选项。
2. 文本数据目前支持 `txt/md`。
3. 音频目前支持 `wav/flac`。
4. 图片默认支持 `bmp/pgm/ppm` 直接读入；若环境已安装 `Pillow`，也支持 `png/jpg/jpeg`。
5. 对文本，程序统计 UTF-8 字节流的压缩效果；对图片，优先统计像素字节流；对音频，统计采样字节流。
6. 实验输出还会给出基于熵的 `structure_score`，可用于绘制“结构度-压缩比”关系图，并计算相关系数与线性拟合结果。
7. 若使用数据集管理模式，目录约定为 `assets/text/<dataset_name>/...`、`assets/images/<dataset_name>/...` 和 `assets/audio/<dataset_name>/...`。
8. 若使用 `--hf-*-dataset` 选项，样本会直接通过 `datasets` 流水线加载，不需要先下载到项目目录。
9. 若文本样本中存在大量空样本或短文本，可通过 `--min-text-bytes` 过滤过短文本；对于 Hugging Face 文本流水线，程序会持续取样直到满足指定 `limit` 或数据耗尽。

示例：

```bash
.venv/bin/python main.py --text-dir ./assets/text --image-dir ./assets/images --audio-dir ./assets/audio
```

导出表格和绘图：

```bash
.venv/bin/python main.py --text-datasets synthetic_structured synthetic_unstructured --image-datasets synthetic --audio-datasets synthetic --limit-per-dataset 10 --csv-out ./report/results/metrics.csv --plot-out ./report/results/structure_vs_ratio.pdf --corr-out ./report/results/correlations.csv
```

推荐的数据集目录结构：

```text
assets/
├── text/
│   ├── synthetic_structured/
│   ├── synthetic_unstructured/
│   └── public_samples/
├── images/
│   ├── synthetic/
│   ├── hf_mnist/
│   └── hf_cifar10/
└── audio/
    ├── synthetic/
    ├── hf_speech/
    └── hf_librispeech/
```

结构度定义：

1. 统计字节熵 `byte_entropy`。
2. 统计 2 字节窗口熵 `window_entropy`。
3. 将两者归一化后取平均，并定义 `structure_score = 1 - average_entropy`。
4. `structure_score` 越高，表示数据越有规律、重复性越强，理论上越容易被 `LZW/LZSS` 压缩。

数据集管理说明：

1. `datasets-list` 会扫描 `assets/text/*`、`assets/images/*` 和 `assets/audio/*`。
2. `hf-import` 会把导入样本落到 `assets/images/<dataset_name>/` 或 `assets/audio/<dataset_name>/`。
3. 导入后即可直接用 `--text-datasets`、`--image-datasets` 或 `--audio-datasets` 跑实验。
4. `--csv-out` 会导出逐样本实验结果表。
5. `--plot-out` 会导出包含散点图和最佳拟合直线的 `PDF` 图，可直接插入实验报告。
6. `--corr-out` 会导出相关系数与拟合参数表。

## 要求原文

### 本次实验作业

1.  **程序实现 LZW、LZSS 算法**的编解码算法。验证程序的正确性。
2.  通过**压缩比**等指标论证 LZW、LZSS 适合的数据对象。如：图片、字符串、音频采样等。
3.  **4 月 26 日之前**，代码和实验报告提交到 `xieyun@bupt.edu.cn`；并在邮件中注明是否愿意做课堂展示。
4.  除主动报名展示的同学外，也会随机抽取一部分同学.
