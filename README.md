# photocircuit-py

给手绘电路图拍一张照，程序会自动认出里面的元件和连线，然后帮你生成 LaTeX 的 CircuiTikZ 代码。

这是 Android 应用 PhotoCircuit 的 Python 版。

---

## 你需要准备什么

- Python 3.10 或更高版本
- 一张清晰的手绘电路图照片（笔迹清楚、纸上没有太多杂物）

## 安装

打开命令行，进入本项目文件夹，运行：

```
pip install -r requirements.txt
```

## 怎么用

**最简单的方式**——直接告诉它图片在哪里，代码会打印到屏幕上：

```
python -m photocircuit 你的电路图.jpg
```

看到类似这样的输出就成功了（这是 CircuiTikZ 代码）：

```
\begin{circuitikz}[american,x=0.01cm,y=0.01cm]
\draw (189, 709) to [short] (48, 709);
...
\end{circuitikz}
```

**想保存成文件？** 加一个 `-o` 参数：

```
python -m photocircuit 你的电路图.jpg -o 结果.tex
```

**想看看程序是怎么一步步识别出来的？** 加 `--debug-dir`，它会生成中间图片（阈值图、检测框、每个元件的裁剪图），方便你排查为什么没认对：

```
python -m photocircuit 你的电路图.jpg --debug-dir 检查/
```

## 常见问题

**识别错了几个元件怎么办？**

这个版本用的还是原来的方法（传统图像处理 + 分类模型），跟手机 App 一样，对照片质量敏感。请尽量拍正、拍清晰、让图纸占满画面。想更准的话，可以换用 YOLO 模型重新训练——代码结构已经留好了接缝。

**没有模型文件能跑吗？**

能，加 `--no-model` 可以只做图像处理部分的调试，但不会输出元件识别结果。

**照片是竖着拍的？**

加 `--portrait`：

```
python -m photocircuit --portrait 你的电路图.jpg
```

## 测试

跑一下自带的测试，确认程序正常：

```
python -m pytest
```

测试里有一张示例电路图，会检查程序识别出的元件种类和数量是否正确。

---

给开发者的架构说明和与原 Java 版的对照，见 `CLAUDE.md`。
