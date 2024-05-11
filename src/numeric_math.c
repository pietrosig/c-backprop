#include "numeric.h"

// Computes ---> op1 + op2
numeric_t *numeric_add(numeric_t *op1, numeric_t *op2) {

  numeric_t *res = create_numeric(op1->n + op2->n);

  res->op1 = op1;
  res->op2 = op2;

  res->grad_fn = &ADD_GRAD_CALC_;

  return res;
}

// Computes ---> op1 - op2
numeric_t *numeric_sub(numeric_t *op1, numeric_t *op2) {

  // partial = -op2
  numeric_t *partial = numeric_mul(op2, NUMERIC_NEG_ONE);

  return numeric_add(op1, partial);
}

// Computes ---> op1 * op2
numeric_t *numeric_mul(numeric_t *op1, numeric_t *op2) {

  numeric_t *res = create_numeric(op1->n * op2->n);

  res->op1 = op1;
  res->op2 = op2;

  res->grad_fn = &MUL_GRAD_CALC_;

  return res;
}

// Computes ---> op1 / op2
numeric_t *numeric_div(numeric_t *op1, numeric_t *op2) {

  // partial = 1 / op2
  numeric_t *partial = numeric_inv(op2);

  return numeric_mul(op1, partial);
}

// Computes ---> op1 ^ op2
numeric_t *numeric_pow(numeric_t *op1, numeric_t *op2) {

  numeric_t *res = create_numeric(pow(op1->n, op2->n));

  res->op1 = op1;
  res->op2 = op2;

  res->grad_fn = &POW_GRAD_CALC_;

  return res;
}

// Computes ---> 1 / op1
// This can be computed as numeric_pow(op1, -1)
// But directly computing 1 / x is more efficient than computing x ^ -1
numeric_t *numeric_inv(numeric_t *op1) {

  numeric_t *res = create_numeric(1 / op1->n);

  res->op1 = op1;

  res->grad_fn = &INV_GRAD_CALC_;

  return res;
}

// Computes ---> e ^ op1
// This can be computed as numeric_pow(e, op1)
numeric_t *numeric_exp(numeric_t *op1) {

  numeric_t *res = create_numeric(exp(op1->n));

  res->op1 = op1;

  res->grad_fn = &EXP_GRAD_CALC_;

  return res;
}

// Computes ---> abs(op1)
numeric_t *numeric_abs(numeric_t *op1) {
  numeric_t *res = create_numeric(fabs(op1->n));

  res->op1 = op1;

  res->grad_fn = &ABS_GRAD_CALC_;

  return res;
}
