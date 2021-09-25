#ifndef NETWORK_H
#define NETWORK_H
#include "layer.h"

#include <iostream>
#include <vector>
class network
{
private:
    std::vector<layer*> layers;//记录神经网络所有层
    int size;
public:
    network(std::vector<int> layerInfo)//layerInfo中存储着每一层的节点数
    {
        size=layerInfo.size();
        auto *layer0=new layer(layerInfo[0]);
        layers.push_back(layer0);
        for (int i=1;i<size-1;i++)
        {
            layer0=layer0->genNextLayer(layerInfo[i]);//依次生成下一层
            layers.push_back(layer0);
        }
        layer0=layer0->genNextLayer(layerInfo[size-1],layer::activateFunc);//依次生成下一层
        layers.push_back(layer0);
    }
    layer::MatrixType feed(layer::MatrixType &data,layer::MatrixType &y0)//训练数据,data表示数据,y0表示标签值
    {
        layer::MatrixType y=layers[0]->forward(data);//从输入层开始前向传播
        layer::MatrixType delta=y-y0;//计算误差
        layers[size-1]->backward(delta);//反向传播更新模型参数        
        return y;
    }
    layer::MatrixType predict(layer::MatrixType data)//预测
    {
        return layers[0]->predict(data);
    }
    ~network()
    {
        for (auto &&_layer:layers)
        {
            delete _layer;
        }
    }
};

#endif
