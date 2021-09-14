#ifndef LAYER_H
#define LAYER_H
#include <functional>
#include <Eigen/Dense>
//每一层全连接层
class layer
{
public:
    using numType=double;
    using activateFuncType=std::function<numType(numType)>;//激活函数的类型
    using MatrixType=Eigen::Matrix<numType,Eigen::Dynamic,Eigen::Dynamic>;//矩阵的类型
    //ReLU    
    static numType sigmoid(numType a)
    {
        return 1/(1+exp(-a));
    }
    static numType ReLU(numType a)
    {
        return a>0?a:0;
    }
/*
 * 全连接层的构造函数
 * int num:当前层的节点数
 * layer* lastLayer:上一层
 * activateFuncType activate:当前层的激活函数
 * numType alpha:学习率
 * numType lambda:正则化系数
*/
    layer(int num,layer* lastLayer=nullptr,activateFuncType&& activate=ReLU,numType alpha=0.001,numType lambda=0.01)
    {
        this->activate=activate;
        this->num=num;
        this->lastLayer=lastLayer;
        if (lastLayer!=nullptr)//当前层不是输入层，初始化w和b，将其赋为0-1间的随机矩阵
        {
            w.resize(num,lastLayer->getNum());
            b.resize(num,1);
            auto *pb=b.data();
            auto *pw=w.data();
            for (int i=0;i<num;i++)
            {
                pb[i]=1.0*rand()/RAND_MAX-0.5;
                for (int j=0;j<lastLayer->getNum();j++)
                    pw[i*lastLayer->getNum()+j]=1.0*rand()/RAND_MAX-0.5;
            }
        }
        z.resize(num,1);
        delta.resize(num,1);
        this->alpha=alpha;
        this->lambda=lambda;
    }
    int getNum()//获得当前层的节点数
    {
        return num;
    }
    numType fdot(numType x)//对当前激活函数近似求导
    {
        numType dt=0.001;
        return (activate(x+dt)-activate(x))/dt;
    }
/*
 * 前向传播
 * MatrixType t:上一层的活性值 
*/
    MatrixType forward(MatrixType &t)
    {
        if (lastLayer==nullptr)//假如是输入层，不需要计算直接输出t
        {
            a=t;
            z=t;
        }
        else
        {
            a=w*t+b;//假如不是输入层，计算净活性值
            auto *ap=a.data();
            auto *zp=z.data();
            for (int i=0;i<num;i++)
                zp[i]=activate(ap[i]);//使用激活函数激活
        }
        if (nextLayer==nullptr)//假如到了最后一次，输出预测值
            return z;
        else
            return nextLayer->forward(z);//否则传播到下一层
    }
    MatrixType predict(MatrixType &t)//预测，和前向传播基本一致
    {
        MatrixType result;
        if (lastLayer==nullptr)
            result=t;
        else
        {
            result=w*t+b;
            auto *ptr=result.data();
            for (int i=0;i<num;i++)
                ptr[i]=activate(ptr[i]);
        }
        if (nextLayer==nullptr)
            return result;
        else
            return nextLayer->predict(result);
    }
    MatrixType& getInactivate()//返回当前层的净活性值a
    {
        return a;
    }
/*
 * 反向传播
 * MatrixType t:上一层的 W^T \times \delta 
*/
    void backward(MatrixType &wMulDelta)
    {
        auto *nextDelta=wMulDelta.data();
        auto *zp=z.data();
        auto *dp=delta.data();
        for (int i=0;i<num;i++)
            dp[i]=fdot(zp[i])*nextDelta[i];//计算当前delta=f'(z)*wmulDelta
        w=w-alpha*(delta*lastLayer->getInactivate().transpose()+lambda*w);//更新W
        b-=alpha*delta;//更新b
        MatrixType result=w.transpose()*delta;//计算当前层的W^T \times \delta
        if (lastLayer->lastLayer!=nullptr)//假如下一层不是输入层，继续反向传播
            lastLayer->backward(result);
    }
    layer* genNextLayer(int num,activateFuncType&& activate=ReLU)//生成当前网络的下一层
    {
        nextLayer=new layer(num,this,std::move(activate));
        return nextLayer;
    }
    ~layer()=default;
private:
    int num;
    layer* lastLayer;
    layer *nextLayer=nullptr;
    MatrixType w;
    MatrixType b,z,a;
    numType alpha,lambda;//alpha:学习率 lambda:正则化系数
    activateFuncType activate;

    MatrixType delta;
};

#endif