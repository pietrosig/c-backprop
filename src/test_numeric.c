#include "numeric.h"

int main() {
  // true enables gradient storing
  numeric_t *a = create_numeric_(-3.6, true);
  numeric_t *b = create_numeric_(2.312, true);
  numeric_t *c = create_numeric_(7.95, true);
  numeric_t *d = create_numeric_(-271, true);
  numeric_t *e = create_numeric_(-932.229, true);

  // (ReLu(cos(a - b) / c) * d)
  numeric_t *right =
      numeric_mul(numeric_relu(numeric_div(numeric_cos(numeric_sub(a, b)), c)), d);

  // sin(a / c) / e
  numeric_t *left =
      numeric_div(numeric_sin(numeric_div(a, c)), e);

  // (ReLu(cos(a - b) / c) * d) - (sin(a / c) / d)
  numeric_t *loss = numeric_sub(right, left);

  // Backward pass
  backward(loss);

  // Print variables with stored gradients
  // Special format to do tests with PyTorch
  printf("loss: %f\n", loss->n);
  printf("a_grad: %f\n", a->grad);
  printf("b_grad: %f\n", b->grad);
  printf("c_grad: %f\n", c->grad);
  printf("d_grad: %f\n", d->grad);
  printf("e_grad: %f\n", e->grad);

  return 0;
}
