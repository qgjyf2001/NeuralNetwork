/*
*   paillier类是基于paillier的加密解密类
*   利用NTL库实现
*/
#ifndef PAILLIER_H
#define PAILLIER_H
#include <NTL/ZZ.h>
#include <NTL/ZZ_pXFactoring.h>
class paillier
{
public:    
    using bigInteger=NTL::ZZ;//大数的类型别名
    using privateKey=std::pair<NTL::ZZ,NTL::ZZ>;//私钥的类型
    using publicKey=std::pair<NTL::ZZ,NTL::ZZ>;//公钥的类型

    paillier(int bitLength=512);
    std::pair<privateKey,publicKey> genKey();//产生密钥
    bigInteger encrypt(bigInteger,publicKey&);//加密
    bigInteger decrypt(bigInteger,privateKey&);//解密
    bigInteger add(bigInteger,bigInteger,publicKey&);//同态加法
    bigInteger mul(bigInteger,bigInteger,publicKey&);//同态数乘
    ~paillier()=default;
    bigInteger n;//模数
    bigInteger n2;//模数的平方
private:
    int bitLength;//加密的n的长度
};

#endif