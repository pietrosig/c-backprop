#ifndef __TENSOR_H_
#define __TENSOR_H_

#include "numeric.h"
#include "tensor_macros.h"
#include "grad_context.h"


#include <assert.h>

typedef struct tensor_t tensor_t;

// Tensor struct
struct tensor_t {
  numeric_t **data;
  size_t ndim;
  size_t *shape;
  size_t *strides;
  size_t storage_offset;
  bool is_view;
};

size_t tensor_len(const tensor_t *);
tensor_t *tensor_like(const tensor_t *);
tensor_t *tensor_get(const tensor_t *, const ssize_t[]);
void tensor_set(tensor_t *, double, const ssize_t[]);
void tensor_print(tensor_t *);
void tensor_print_shape(tensor_t *);
void tensor_print_grad(tensor_t *);

tensor_t *tensor_zeros(size_t, const size_t[]);
tensor_t *tensor_arange(const size_t, const int[]);
void tensor_backward(tensor_t *);


// Math
tensor_t *tensor_hadamard(const tensor_t *, const tensor_t *);
tensor_t *tensor_add(const tensor_t *, const tensor_t *);
tensor_t *tensor_sum(const tensor_t *, const size_t);

#endif
