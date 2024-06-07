#ifndef __NUMERIC_H_
#define __NUMERIC_H_

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct numeric_t numeric_t;

typedef double (*grad_calc_t)(numeric_t *, numeric_t *);

struct numeric_t {
  double n;
  bool store_grad;
  double grad;
  grad_calc_t grad_fn;
  struct numeric_t *op1, *op2;
};

numeric_t *create_numeric(double);
numeric_t *create_numeric_(double, bool);
void destroy_numeric(numeric_t *);
void backward(numeric_t *);
void print_numeric(numeric_t *);
void init_numeric_const();

numeric_t *numeric_add(numeric_t *, numeric_t *);
numeric_t *numeric_sub(numeric_t *, numeric_t *);
numeric_t *numeric_mul(numeric_t *, numeric_t *);
numeric_t *numeric_div(numeric_t *, numeric_t *);
numeric_t *numeric_pow(numeric_t *, numeric_t *);
numeric_t *numeric_inv(numeric_t *);
numeric_t *numeric_exp(numeric_t *);
numeric_t *numeric_log(numeric_t *);
numeric_t *numeric_abs(numeric_t *);
numeric_t *numeric_sin(numeric_t *);
numeric_t *numeric_cos(numeric_t *);
numeric_t *numeric_relu(numeric_t *);

double ADD_GRAD_CALC_(numeric_t *output, numeric_t *input);
double MUL_GRAD_CALC_(numeric_t *output, numeric_t *input);
double POW_GRAD_CALC_(numeric_t *output, numeric_t *input);
double INV_GRAD_CALC_(numeric_t *output, numeric_t *input);
double EXP_GRAD_CALC_(numeric_t *output, numeric_t *input);
double LOG_GRAD_CALC_(numeric_t *output, numeric_t *input);
double ABS_GRAD_CALC_(numeric_t *output, numeric_t *input);
double SIN_GRAD_CALC_(numeric_t *output, numeric_t *input);
double COS_GRAD_CALC_(numeric_t *output, numeric_t *input);
double RELU_GRAD_CALC_(numeric_t *output, numeric_t *input);

extern numeric_t *NUMERIC_POS_ONE;
extern numeric_t *NUMERIC_NEG_ONE;

#endif
