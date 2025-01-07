#include <iostream>

using namespace std;
extern "C" {
    int test() {
        cout << "Hello hFlux";
        return 42;
    }
}