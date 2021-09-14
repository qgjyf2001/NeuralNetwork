#include "paillierMatrix.h"
paillierMatrix::paillierMatrix(int trunc,int bitLength):paillier(bitLength),trunc(trunc)
{
}
paillierMatrix::matrixType paillierMatrix::encrypt(matrixIType &m,publicKey& key)
{
    int row=m.rows();
    int column=m.cols();
    matrixType result;
    result.resize(row,column);
    auto *md=m.data();
    auto *rd=result.data();//此处使用data可以直接获得Matrix对象在内存中的指针，相比于间接访问节省了80%的时间
    int k=0;
    for (int i=0;i<row;i++)
    {
        for (int j=0;j<column;j++,k++)
            rd[k]=paillier::encrypt(bigInteger(int(md[k]*trunc)),key);//矩阵中的每个元素都进行加密
    }
    return result;
}
paillierMatrix::matrixIType paillierMatrix::decrypt(paillierMatrix::matrixType& m,privateKey& key)
{
    int row=m.rows();
    int column=m.cols();
    matrixIType result;
    result.resize(row,column);
    auto *md=m.data();
    auto *rd=result.data();
    int k=0;
    for (int i=0;i<row;i++)
        for (int j=0;j<column;j++,k++)
        {
            auto decrypted=paillier::decrypt(md[k],key);//对矩阵中的每个元素进行解密
            int res=0;            
            if (decrypted>bigInteger(1000000000))//假如解密的值比较大，认为他应该是负数
                decrypted-=paillier::n;
#ifdef GMPIMP
            res=decrypted.get_ui();
#else  
            NTL::conv(res,decrypted);//转换回int
#endif      
            rd[k]=1.0*res/trunc/trunc;//由于矩阵乘法时多乘了一个trunc，所以这里应该除两次
        }
    return result;
}
void paillierMatrix::add(paillierMatrix::matrixType& cipher,paillierMatrix::matrixIType& num,publicKey& key)
{
    int row=cipher.rows();
    int column=cipher.cols();
    assert(row==num.rows());
    assert(column==num.cols());
    for (int i=0;i<row;i++)
        for (int j=0;j<column;j++)
            cipher(i,j)=paillier::add(cipher(i,j),bigInteger(int(num(i,j)*trunc)),key);
}
void paillierMatrix::add(paillierMatrix::matrixType& cipher,paillierMatrix::matrixType& num,publicKey& key)
{
    int row=cipher.rows();
    int column=cipher.cols();
    assert(row==num.rows());
    assert(column==num.cols());
    for (int i=0;i<row;i++)
        for (int j=0;j<column;j++)
            cipher(i,j)=cipher(i,j)*num(i,j)%(paillier::n*paillier::n);
}
paillierMatrix::matrixType paillierMatrix::mul(paillierMatrix::matrixType cipher,paillierMatrix::matrixIType num,publicKey& key)
{    
    int row=cipher.rows();
    int column=cipher.cols();
    int col2=num.cols();
    assert(column==num.rows());
    matrixType result;
    result.resize(row,col2);
    auto *cd=cipher.data();
    auto *rd=result.data();
    auto *nd=num.data();
    int t=0;
    //此处使用朴素的矩阵乘法进行计算，时间复杂度是 O(n^3)
    for (int i=0;i<row;i++)
        for (int j=0;j<col2;j++,t++)
        {  
            auto temp=paillier::mul(cd[i],bigInteger(int(nd[j*column]*trunc)),key);      
            for (int k=1;k<column;k++)
            {
                temp*=paillier::mul(cd[i+k*row],bigInteger(int(nd[k+j*column]*trunc)),key);//把相应的元素进行同态意义下的数乘，然后再相乘（相当于同态意义下的加法）
                temp%=(paillier::n*paillier::n);   
            }
            rd[j*row+i]=temp;
        }
    return result;
}