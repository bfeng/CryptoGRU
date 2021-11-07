#ifndef GRUCELL_HEADER
#define GRUCELL_HEADER

#include <cryptoTools/Common/Defines.h>
#include <iostream>
#include "utils/backend.h"

namespace lbcrypto {
class GRUCell {
   private:
    ui32 m_hidden_size;
    ui32 m_input_size;
    void matmul(ui32 num_rows, ui32 num_cols, ui32 window_size);
    void elem_add(ui32 vec_size);
    void elem_mult(ui32 vec_size);
    void sigmoid(osuCrypto::u64 n_circ);
    void tanh(osuCrypto::u64 n_circ);

   public:
    GRUCell(int input_size, int hidden_size)
        : m_hidden_size(hidden_size), m_input_size(input_size) {
        std::cout << "************************************" << std::endl;
        std::cout << "Input:" << m_input_size << ", Hidden:" << m_hidden_size
                  << std::endl;
        std::cout << "************************************" << std::endl;
    }
    ~GRUCell() = default;
    void call();
};
}  // namespace lbcrypto

#endif