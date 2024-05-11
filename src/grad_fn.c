#include "numeric.h"

double ADD_GRAD_CALC_(numeric_t *output, numeric_t *input) {
  if (output == NULL || input == NULL)
    return 0;

  return 1;
}

double MUL_GRAD_CALC_(numeric_t *output, numeric_t *input) {
  if (output == NULL || input == NULL)
    return 0;

  // Derivative w.r.t. op1 ---> (op1 * op2) / op1 = op2
  // Derivative w.r.t. op2 ---> (op1 * op2) / op2 = op1
  return (output->n / input->n);
}

double POW_GRAD_CALC_(numeric_t *output, numeric_t *input) {
  if (output == NULL || input == NULL)
    return 0;

  // Derivative w.r.t. op1 --> (op1 ^ (op2 - 1)) * op2
  if (output->op1 == input)
    return (output->n / output->op1->n) * output->op2->n;

  // Derivative w.r.t. op2 --> (op1 ^ op2) * log(op1)
  if (output->op2 == input)
    return output->n * log(output->op1->n);

  // TODO: ERROR
  return 0;
}

double INV_GRAD_CALC_(numeric_t *output, numeric_t *input) {
  if (output == NULL || input == NULL)
    return 0;

  // Derivative w.r.t. op1 --> -1 / (op1 ^ 2)
  return -(output->n * output->n);
}

double EXP_GRAD_CALC_(numeric_t *output, numeric_t *input) {
  if (output == NULL || input == NULL)
    return 0;

  // Derivative w.r.t. op1 --> e ^ op1
  return output->n;
}

double ABS_GRAD_CALC_(numeric_t *output, numeric_t *input) {
  if (output == NULL || input == NULL)
    return 0;

  // Derivative w.r.t. op1 --> abs(op1) / op1
  return output->n / input->n;
}
