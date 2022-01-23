#include <vector>
#include <iostream>
#include <cassert>

#include "kernels/functions.hpp"


int main(){

	std::vector<float> A, B;

	for(int i = 0; i < 4096; ++i){
		A.push_back(float(i));
		B.push_back(float(i));
	}

	auto out = CUaddVectors(A,B);

	for(int i = 0; i < 4096; ++i){
		assert(A.at(i) + B.at(i) == out.at(i));
	}

	std::cout << "Function success!\n";

	return 1;

}
