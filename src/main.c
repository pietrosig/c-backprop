#include "numeric.h"

int main() {
  numeric_t *a = create_numeric_(3.6, true);
  numeric_t *b = create_numeric_(-2.312, true);
  numeric_t *c = create_numeric_(2, true);

  // (a - b) ^ c
  numeric_t *loss =
      numeric_sub(numeric_abs(numeric_div(numeric_sub(a, b), c)), a);
  print_numeric(loss);

  // Compute gradients via Backpropagation
  backprop(loss);

  // Let's print stored gradients
  print_numeric(a);
  print_numeric(b);
  print_numeric(c);

  return 0;
}
