/*
*   paillierMatrix类是基于paillier的矩阵运算类
*   提供了两个实现，都可使用，注释第9行使用NTL库，反注释此行使用gmp
*   本类继承自paillier类，后者是在Z_{n*n}上的paillier加密类
*   本来想写支持多次乘法，后来没调通，但是目前只需要计算一次矩阵乘法应该就可以，所以后来干脆也没写
*/
#ifndef PAILLIERMATRIX_H
#define PAILLIERMATRIX_H
#include <Eigen/Dense>
//#define GMPIMP    
#ifdef GMPIMP
#include "paillierGmpImp.h"
using paillier=paillierGmpImp;
#else
#include "paillier.h"
#endif
class paillierMatrix:public paillier
{    
private:
    int trunc;//小数需要乘这个数字取整之后变成整数
    bigInteger invTrunc;//trunc^{-1} mod n^2 ，但是暂时没有用到
public:
    using matrixType=Eigen::Matrix<bigInteger,Eigen::Dynamic,Eigen::Dynamic>;//加密后的矩阵类
    using matrixIType=Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>;//加密前的矩阵类，和main.cpp中的MATRIX是完全一样的
    paillierMatrix(int trunc=1<<12,int bitLength=512);//构造函数
    matrixType encrypt(matrixIType &m,publicKey& key);//加密函数
    matrixIType decrypt(matrixType& m,privateKey& key);//解密函数
    void add(matrixType&,matrixIType&,publicKey&);//同态意义下的矩阵加法
    void add(matrixType&,matrixType&,publicKey&);//同态意义下的矩阵加法
    paillierMatrix::matrixType mul(matrixType,matrixIType,publicKey&);//同态意义下的矩阵乘法
    ~paillierMatrix()=default;
};
/*
*   重写Eigen::NumTraits<T>，让其支持矩阵上的大数运算
*/
namespace Eigen{
	template<> struct NumTraits<paillier::bigInteger>:GenericNumTraits<paillier::bigInteger>
    {
        enum{
            IsComplex=0,
            IsInteger=1,
            IsSigned=1,
            RequireInitialization=1,
            ReadCost=6,
            AddCost=20,
            MulCost=100
        };
    };
}
#endif