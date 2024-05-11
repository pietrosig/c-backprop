#include "numeric.h"

numeric_t *create_numeric_(double n, bool store_grad) {
  numeric_t *p = malloc(sizeof(numeric_t));

  p->n = n;
  p->store_grad = store_grad;
  p->grad = 0;

  p->op1 = NULL;
  p->op2 = NULL;

  p->grad_fn = NULL;

  return p;
}

numeric_t *create_numeric(double n) { return create_numeric_(n, false); }

void destroy_numeric(numeric_t *numeric) { free(numeric); }

void store_grad(numeric_t *numeric, double grad) {
  if (!numeric->store_grad)
    return;

  numeric->grad += grad;
}

void backprop_(numeric_t *f, double upstream_grad) {
  if (f == NULL)
    return;

  store_grad(f, upstream_grad);

  if (f->op1 != NULL)
    backprop_(f->op1, upstream_grad * f->grad_fn(f, f->op1));

  if (f->op2 != NULL)
    backprop_(f->op2, upstream_grad * f->grad_fn(f, f->op2));
}

void backprop(numeric_t *f) { backprop_(f, 1); }

void print_numeric(numeric_t *n) {
  printf("<numeric: (%f)", n->n);

  if (n->store_grad)
    printf("\n    grad: (%f)", n->grad);

  printf(">\n\n");
}
