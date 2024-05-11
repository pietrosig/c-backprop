#include "numeric.h"

numeric_t *NUMERIC_POS_ONE;
numeric_t *NUMERIC_NEG_ONE;

void init_numeric_const() {
  NUMERIC_POS_ONE = create_numeric(1);
  NUMERIC_NEG_ONE = create_numeric(-1);
}

// ! GCC specific !
// This will ensure that init_numeric_const() will be called before main()
void __attribute__((constructor)) init_numeric_const_constructor() {
  init_numeric_const();
}
