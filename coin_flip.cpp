#include <iostream>
#include <cstdlib>  // Used <cstdlib> instead of <stdlib.h>
#include <ctime>

int main() {
    srand(time(NULL));  // Seed the random number generator
    int coin = rand() % 2;  // Generate 0 or 1

    if (coin == 0) {
        std::cout << "Heads\n";
    } else {
        std::cout << "Tails\n";
    }

    return 0;  // Best practice to return 0 in main()
}