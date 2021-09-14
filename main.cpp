/*
*编译方法：
*	sudo apt-get install -y autoconf
*	sudo apt-get install -y m4
*	sudo apt-get install -y libgmp-dev
*	sudo apt-get install -y libgf2x-dev
* 	sudo apt-get install -y libeigen3-dev
*   sudo apt-get install -y libgf2x-dev
*   sudo apt-get install -y libntl-dev
*  	make
*运行：
*	./main
*/

#include <Eigen/Dense>

#include <iostream>
#include <set>
#include <time.h>
#include <fstream>
#include <vector>
#include <math.h>

#include "network.h"

using namespace std;
using namespace Eigen;

// N * E = B * T
#define N 120*150	// number of training data samples
#define D 784	// number of features
#define T 150	// total number of iterations per epoch
#define B_size 120	// size of mini-batch
#define ALPHA 128	// 2^7
#define epoch 10

typedef Matrix<double,Dynamic, Dynamic> MATRIX;
typedef Matrix<double, Dynamic, Dynamic> MATRIXd;
typedef Matrix<double, Dynamic, 1, ColMajor> ColVectorXi64;
typedef Matrix<double, Dynamic, 1, ColMajor> ColVectorXd;
typedef Matrix<double, 1, Dynamic, RowMajor> RowVectorXd;
typedef Matrix<double, 1, Dynamic, RowMajor> RowVectorXi64;




int reverse_int(int i);

void read_MNIST_data(bool train, vector<vector<double>>& vec, int& number_of_images, int& number_of_features);

void read_MNIST_labels(bool train, vector<double>& vec);

void read_MNIST_data_test(bool train, vector<vector<double>>& vec, int& number_of_images, int& number_of_features);

void read_MNIST_labels_test(bool train, vector<double>& vec);

 
struct timeCounter//计算程序运行时间的辅助类
{
	int time;
	timeCounter()
	{
		time=clock();
	}
	void timePrint()
	{
		std::cout<<1000*(clock()-time)/CLOCKS_PER_SEC<<"ms"<<std::endl;
		time=clock();
	}
};
int main() {

// 读取mnist数据集的训练数据
	vector<vector<double>> training_data;
	int param_n= N; int param_d= D;
	read_MNIST_data(true, training_data, param_n, param_d);
	MATRIX X(param_n, param_d);
	for(int i = 0; i < training_data.size(); i++){
        X.row(i) << Map<RowVectorXi64>(training_data[i].data(), training_data[i].size());
    }
	X /= 255;//归一化

// 读取mnist数据集的标签
	vector<double> training_labels;
	read_MNIST_labels(true, training_labels);
	MATRIX Y(N, 1);
	Y << Map<ColVectorXi64>(training_labels.data(), training_labels.size());
// 读取mnist数据集的测试数据
	vector<vector<double> > testing_data;
	int n_;
	read_MNIST_data_test(false, testing_data, n_, param_d);
	MATRIXd Xt(n_, param_d);
	for(int i = 0; i < testing_data.size(); i++){
        Xt.row(i) << Map<RowVectorXd>(testing_data[i].data(), testing_data[i].size());
    }
	Xt /= 255;

// 读取mnist数据集的测试标签
	vector<double> testing_labels;
	read_MNIST_labels_test(false, testing_labels);
	MATRIXd Yt(n_, 1);
	Yt << Map<ColVectorXd>(testing_labels.data(), testing_labels.size());

    std::vector<int> layer={D,120,100,10};
    network nt(layer);//创建一个神经网络，加上输入层一共有四层，D为输入的特征数，最后的输出有10个节点

    for (int T0=0;T0<epoch;T0++)//训练epoch轮
    {
        std::cout<<"epoch:"<<T0<<std::endl;
        for (int i=0;i<N;i++)
        {
            MATRIX t=X.row(i).transpose();//获得数据
            MATRIX feedY(10,1);
            int actualY=Y(i,0);//获得标签值
            auto *ptrY=feedY.data();
            for (int i=0;i<10;i++)
                ptrY[i]=i==actualY;

            nt.feed(t,feedY);//训练
        }
    }

    //检验模型准确度
    int total=10000;
    int corrected=0;
    for (int i=0;i<total;i++)
    {
        MATRIX t=Xt.row(i).transpose();
        int actualY=Yt(i,0);
        MATRIX resultY=nt.predict(t);
        auto *ptrY=resultY.data();
        double minn=-100000000;
        int pred=0;
        for (int i=0;i<10;i++)
        {
            if (minn<ptrY[i])
            {
                minn=ptrY[i];
                pred=i;
            }
        }
        if (pred==actualY)
            corrected+=1;
    }
    std::cout<<1.0*corrected/total;

} // end of main()


int reverse_int(int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void read_MNIST_data(bool train, vector<vector<double>> &vec, int& number_of_images, int& number_of_features){
    std::ifstream file;
	if (train == true)
		file.open("./train-images-idx3-ubyte", std::ios::binary);
	else
		file.open("./t10k-images-idx3-ubyte", std::ios::binary);
    
    if(!file){
        std::cout<<"Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if(train == true)
            number_of_images = N;
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        number_of_features = n_rows * n_cols;
        std::cout << "Number of Images: " << number_of_images << std::endl;
        std::cout << "Number of Features: " << number_of_features << std::endl;
        for(int i = 0; i < number_of_images; ++i){
            std::vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((unsigned long long int) temp));
                }
            vec.push_back(tp);
        }
    }
	file.close();
}


void read_MNIST_labels(bool train, vector<double> &vec){
    std::ifstream file;
	if (train == true) {
		file.open("./train-labels-idx1-ubyte", std::ios::binary);
	}
	else {
		file.open("./t10k-labels-idx1-ubyte", std::ios::binary);
	}
    if(!file){
        std::cout << "Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
		cout << "number of images inside MNIST LAbels: " << number_of_images << endl;

        if(train == true)
            number_of_images = N;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
			//vec.push_back((unsigned long int) temp);
			
                vec.push_back((unsigned long long int) temp);
        }
    }
	file.close();
}


void read_MNIST_data_test(bool train, vector<vector<double>> &vec, int& number_of_images, int& number_of_features){
    std::ifstream file;
	if (train == true)
		file.open("./train-images-idx3-ubyte", std::ios::binary);
	else
		file.open("./t10k-images-idx3-ubyte", std::ios::binary);
    
    if(!file){
        std::cout<<"Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        if(train == true)
            number_of_images = N;
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);
        number_of_features = n_rows * n_cols;
        std::cout << "Number of Images: " << number_of_images << std::endl;
        std::cout << "Number of Features: " << number_of_features << std::endl;
        for(int i = 0; i < number_of_images; ++i){
            std::vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back(((double) temp));
                }
            vec.push_back(tp);
        }
    }
}

void read_MNIST_labels_test(bool train, vector<double> &vec){
    std::ifstream file;
	if (train == true) {
		file.open("./train-labels-idx1-ubyte", std::ios::binary);
	}
	else {
		file.open("./t10k-labels-idx1-ubyte", std::ios::binary);
	}
    if(!file){
        std::cout << "Unable to open file";
        return;
    }
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverse_int(magic_number);
        file.read((char*) &number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
		cout << "number of images inside MNIST LAbels: " << number_of_images << endl;

        if(train == true)
            number_of_images = N;
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
			//vec.push_back((double) temp);
			
                vec.push_back((double) temp);
        }
    }
}

/*
	Modulus
	Epochs, testing accuracy
*/
