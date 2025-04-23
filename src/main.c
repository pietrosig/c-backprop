#include "tensor.h"
#include "numeric.h"

int main() {
  tensor_t *a;
  tensor_t *b;
  tensor_t *c;

  STORE_GRAD {
    a = TENSOR_ARANGE(10);
    b = TENSOR_ARANGE(0, 20, 2);
  }

  tensor_print(a);
  tensor_print(b);

  c = tensor_sum(tensor_hadamard(a, b), -1);

  tensor_print(c);
  tensor_backward(c);

  tensor_print_grad(a);
  tensor_print_grad(b);

  return 0;
}
