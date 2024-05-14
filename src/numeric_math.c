#include "numeric.h"

numeric_t *numeric_create_result(double result, numeric_t *op1, numeric_t *op2,
                                 grad_calc_t grad_fn) {

  numeric_t *res = create_numeric(result);

  res->op1 = op1;
  res->op2 = op2;

  res->grad_fn = grad_fn;

  return res;
}

// Computes ---> op1 + op2
numeric_t *numeric_add(numeric_t *op1, numeric_t *op2) {
  double res = op1->n + op2->n;
  return numeric_create_result(res, op1, op2, &ADD_GRAD_CALC_);
}

// Computes ---> op1 - op2
numeric_t *numeric_sub(numeric_t *op1, numeric_t *op2) {

  // partial = -op2
  numeric_t *partial = numeric_mul(op2, NUMERIC_NEG_ONE);

  return numeric_add(op1, partial);
}

// Computes ---> op1 * op2
numeric_t *numeric_mul(numeric_t *op1, numeric_t *op2) {
  double res = op1->n * op2->n;
  return numeric_create_result(res, op1, op2, &MUL_GRAD_CALC_);
}

// Computes ---> op1 / op2
numeric_t *numeric_div(numeric_t *op1, numeric_t *op2) {

  // partial = 1 / op2
  numeric_t *partial = numeric_inv(op2);

  return numeric_mul(op1, partial);
}

// Computes ---> op1 ^ op2
numeric_t *numeric_pow(numeric_t *op1, numeric_t *op2) {
  double res = pow(op1->n, op2->n);
  return numeric_create_result(res, op1, op2, &POW_GRAD_CALC_);
}

// Computes ---> 1 / op1
// This can be computed as numeric_pow(op1, -1)
// But directly computing 1 / x is more efficient than computing x ^ -1
numeric_t *numeric_inv(numeric_t *op1) {
  double res = 1 / op1->n;
  return numeric_create_result(res, op1, NULL, &INV_GRAD_CALC_);
}

// Computes ---> e ^ op1
// This can be computed as numeric_pow(e, op1)
numeric_t *numeric_exp(numeric_t *op1) {
  double res = exp(op1->n);
  return numeric_create_result(res, op1, NULL, &EXP_GRAD_CALC_);
}

// Computes ---> log(op1)
numeric_t *numeric_log(numeric_t *op1) {
  double res = log(op1->n);
  return numeric_create_result(res, op1, NULL, &LOG_GRAD_CALC_);
}

// Computes ---> abs(op1)
numeric_t *numeric_abs(numeric_t *op1) {
  double res = fabs(op1->n);
  return numeric_create_result(res, op1, NULL, &ABS_GRAD_CALC_);
}

// Computes ---> sin(op1)
numeric_t *numeric_sin(numeric_t *op1) {
  double res = sin(op1->n);
  return numeric_create_result(res, op1, NULL, &SIN_GRAD_CALC_);
}

// Computes ---> cos(op1)
numeric_t *numeric_cos(numeric_t *op1) {
  double res = cos(op1->n);
  return numeric_create_result(res, op1, NULL, &COS_GRAD_CALC_);
}

// Computes ---> ReLu(op1)
numeric_t *numeric_relu(numeric_t *op1) {
  double res = fmax(0, op1->n);
  return numeric_create_result(res, op1, NULL, &RELU_GRAD_CALC_);
}
