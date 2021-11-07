#include <pke/gazelle.h>
using namespace lbcrypto;

int main() {
    GRUCell cell(10, 128);
    cell.call();
    return 0;
}
