#include "numeric.h"

int main() {
  // true enables gradient storing
  numeric_t *a = create_numeric_(-3.6, true);
  numeric_t *b = create_numeric_(2.312, true);
  numeric_t *c = create_numeric_(2, true);

  // ReLu(cos(a - b) / c)
  numeric_t *loss =
      numeric_relu(numeric_div(numeric_cos(numeric_sub(a, b)), c));

  // Compute gradients via Backpropagation
  backprop(loss);

  // Print variables with stored gradients
  print_numeric(loss);
  print_numeric(a);
  print_numeric(b);
  print_numeric(c);

  return 0;
}
