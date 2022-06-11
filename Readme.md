# Pybmpc

## Pytorch based BNN Multi-Parites Computation

我们采用二值神经网络（Binary Neural Net）作为模型基础，针对BNN权重二值化的特点设计了安全协议Pybmpc。算法创新主要有，在非二值化层采用基于加噪的Beaver-Triple的算术秘密交换协议（Arithmetic Secret Share），在二值化层采用二值不经意传输协议（Binary Obviously Transfer）。

我们将其封装为Pytorch的API，通过Pytorch->Onnx->CrypTen的模式生成安全加密模型。使用者只需要往相应接口feed模型和参数即可复现。

效果上达到模型体积、通信开销和计算时间的优化。

## 使用方法

要求先：

    pip install crypten

crypten支持文件在crypten地址下，包含了SS部分的实现方法。
BNN模型格式在models下。
训练ResNet18和VGG16相应的BNN模型可以参照main_binary.py文件。训练时可以使用命令如下：

    python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10

更多命令请查看main_binary.py的parser。

训练MLP相应的BNN模型可以参照main_mnist.py文件。训练是可以使用命令如下：

    python main_mnist.py --batch-size 64 --epochs 100 --lr 0.001

更多命令请查看main_mnist.py的parser 。
数据集可以参考data.py自行下载。模型请按照上一点的方式自行训练。
Pybmpc包含了具体的调用流程，crypten内包含模型转换、加密和SS实现，BOT包含OT实现，其余程序实现相应的辅助功能。



Example是样例，可以按照样例的模式复现Pybmpc框架。




