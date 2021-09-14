#include <paillier.h>
paillier::paillier(int bitLength)
{
    this->bitLength=bitLength;
}
std::pair<paillier::privateKey,paillier::publicKey> paillier::genKey()
{
    bigInteger p,q,phi;
    const int error=80;
    while (true)
    {
        p=NTL::GenPrime_ZZ(bitLength,error);
        q=NTL::GenPrime_ZZ(bitLength,error);
        n=p*q;
        phi=(p-1)*(q-1);
        if (NTL::GCD(n,phi)==1)
            break;
    }    
    auto g=n+1;
    auto lambda = phi/NTL::GCD(p-1,q-1);
    auto mu=NTL::InvMod(lambda,n);
    auto publicKey=std::make_pair(n,g);
    auto privateKey=std::make_pair(lambda,mu);

    n2=n*n;
    return std::make_pair(privateKey,publicKey);
}
paillier::bigInteger paillier::encrypt(paillier::bigInteger m,paillier::publicKey& key)
{
    bigInteger r;
    auto &&[n,g]=key;
    while (true)
    {
        r=RandomBnd(n);
        if (NTL::GCD(r,n)==1)
            break;
    }
    return (NTL::PowerMod(g,m,n2)*NTL::PowerMod(r,n,n2))%n2;
}
paillier::bigInteger paillier::decrypt(paillier::bigInteger c,paillier::privateKey& key)
{
    auto &&[lambda,mu]=key;
    auto mask=NTL::PowerMod(c,lambda,n2);
    auto power=(mask-1)/n;
    return (power*mu)%n;
}
paillier::bigInteger paillier::add(paillier::bigInteger cipher,paillier::bigInteger num,paillier::publicKey& key)
{
    return (cipher*encrypt(num,key))%n2;
}
paillier::bigInteger paillier::mul(paillier::bigInteger cipher,paillier::bigInteger num,paillier::publicKey& key)
{
    return NTL::PowerMod(cipher,num,n2);
}