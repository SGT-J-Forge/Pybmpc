要求先pip install crypten
crypten支持文件在crypten地址下，包含了SS部分的实现方法
请自行下载相应的数据集
同时data内存放了MNIST数据集（较小的）,可供示例程序运行
BNN模型格式在models下
训练ResNet18和VGG16相应的BNN模型可以参照main_binary.py文件。训练时可以使用命令如下：
       python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10
       更多命令请查看main_binary.py的parser
训练MLP相应的BNN模型可以参照main_mnist.py文件。训练是可以使用命令如下：
       python main_mnist.py --batch-size 64 --epochs 100 --lr 0.001
       更多命令请查看main_mnist.py的parser 
保存模型在results地址下，这里因为模型较大所以只提交了MLP模型。其余的请按照上一点的方式自行训练模型
Pybmpc包含了具体的调用流程，crypten内包含模型转换、加密和SS实现，BOT包含OT实现，其余程序实现相应的辅助功能
Example程序可直接运行，是MLP模型安全计算的实例
ResNet18和VGG16模型必须先下载好数据集才能进行训练。训练好模型后，可以按照示例Example.py的方法进行安全计算