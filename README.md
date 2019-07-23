# ssd.pytorch

#### 介绍
1. 用pytorch复现ssd并在自己的数据集上进行行人检测

2. 在docker环境下运行免除安装pytorch以及各种依赖环境的痛苦

    源码来源于https://github.com/amdegroot/ssd.pytorch.git 和 https://github.com/acm5656/ssd_pytorch.git

    根据自己需求进行了差异化改动,按照下边的使用说明可直接运行出结果

#### 安装教程

1. 安装docker以及nvidia-docker

        安装docker参照官网 https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1

        安装nvidia-docker参照 https://github.com/NVIDIA/nvidia-docker

        操作docker参照 http://www.runoob.com/docker/docker-command-manual.html

2. 下载docker镜像

        docker push qinzhenyi1314/pytorch:1.1.0-cuda10.0-cudnn7.5-py3-vnc-jpd

3. 下载本项目代码

        git clone https://gitee.com/qinzhenyi1314/ssd.pytorch.git

#### 使用说明

1. 运行镜像

        docker run --runtime=nvidia -it --rm -w /data -v /home/test/qzy/deeplearning/:/data qinzhenyi1314/pytorch:1.1.0-cuda10.0-cudnn7.5-py3-vnc-jpd

            -v 是为了将服务器路径挂载到容器

            /home/test/qzy/deeplearning/是服务器路径

            /data是运行起来的容器里的路径

2. 下载数据集

        原始的voc2007以及voc2012都是20+1(背景)类，由于自己做的是行人检测 1+1(背景)类

        公司的数据集所以不能分享，暂提供一个很小的数据进行验证

        链接：https://pan.baidu.com/s/1-luJwOIhJhLWRHICoJItcw 提取码：42mh 

        放入data/VOCdevkit下

        格式参照data/VOCdevkit/readme.txt

3. 下载预训练模型

        链接：https://pan.baidu.com/s/1t4uG3YjCy2uIKFG3IZXQKA 提取码：5qhg 

        主要使用

        1. vgg16_reducedfc.pth 用来训练用
        2. ssd300_VOC_17000.pth 用来测试 自己模型在50000张训练17000次得到 map0.68左右，还得继续优化！

4. 运行测试

        python test.py

        运行后将框画在图片上并保存在test文件夹下

5. 运行训练

        python train.py
        
        运行后会在weights生成相应的训练模型 xxx.pth

6. 运行评价

        python eval.py
        
        运行后会测试及画pr曲线以及map值
