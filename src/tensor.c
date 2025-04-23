#define GRAD_CONTEXT_IMPLEMENTATION // This is for the global grad variable
#include "tensor.h"
#include "errors.h"
#include <stdarg.h>
#include <stddef.h>

// For memmove
#include <string.h>


size_t tensor_len(const tensor_t *t) {
    size_t total = 1;
    // Compute total
    for (size_t i = t->ndim; i > 0; --i)
        total *= t->shape[i-1];
    
    return total;
}

tensor_t *tensor_allocate(size_t ndim) {
    tensor_t *t = malloc(sizeof(tensor_t));
    t->ndim = ndim;
    t->shape = malloc(ndim * sizeof(size_t));
    t->strides = malloc(ndim * sizeof(size_t));
    return t;
}

tensor_t *tensor_create_no_data(size_t ndim, const size_t shape[]) {
    tensor_t *t = tensor_allocate(ndim);
    t->is_view = false;
    t->storage_offset = 0;

    // Compute strides
    for (size_t i = ndim; i > 0; --i) {
        size_t dim = shape[i-1];
        t->shape[i-1] = dim;
        t->strides[i-1] = (i == ndim ? 1 : t->strides[i] * t->shape[i]);
    }

    return t;
}


tensor_t *tensor_zeros(size_t ndim, const size_t shape[]) {
    tensor_t *t = tensor_create_no_data(ndim, shape);
    size_t total = ndim == 0 ? 1 : tensor_len(t);

    t->data = malloc(total * sizeof(numeric_t*));
   
    // Store grad
    for (size_t i = 0; i < total; ++i)
        t->data[i] = create_numeric_(0, __grad_enabled);
    

    return t;
}

tensor_t *tensor_ones(size_t ndim, const size_t shape[]) {
    tensor_t *t = tensor_zeros(ndim, shape);
    size_t total = tensor_len(t);

    for (size_t i = 0; i < total; ++i)
        t->data[i]->n = 1;

    return t;
}

tensor_t *tensor_arange_(const int start, const int end, const int step) {
    // TODO: actually handle this
    assert(step != 0);
    assert(start < end);

    size_t total =  (int) round((float)(end - start) / (float) step) ;
    tensor_t *t = tensor_zeros(1, (size_t[]){ total });
   
    // Set the values
    for (size_t i = 0; i < total; ++i) {
        tensor_set(t, start + i * step, (ssize_t[]){ i });
    }
    
    return t;
}

// Behaviour - 3 cases:
//       1. args == { start=0, end }      ->   { start=0, end, step=1 }
//       2. args == { start, end }        ->   { start, end, step=1 }
//       3. args == { start, end, step }  ->   { start, end, step }
tensor_t *tensor_arange(const size_t size, const int args[]) {                
    // TODO: actually handle this
    assert(args != NULL);
    assert(size > 0 && size <= 3);

    // 1.
    if (size == 1)
        return tensor_arange_(0, args[0], 1);

    // 2.
    if (size == 2)
        return tensor_arange_(args[0], args[1], 1);

    // 3.
    return tensor_arange_(args[0], args[1], args[2]);
}


// Returns a tensor with the same shape as t, but with all values set to 0
tensor_t *tensor_like(const tensor_t *t) {
    NULL_POINTER_CHECK_(t);

    // Allocate memory for the result tensor
    tensor_t *result = tensor_zeros(t->ndim, t->shape);

    result->storage_offset = t->storage_offset;

    // Copy shape and strides for the result tensor
    memmove(result->shape, t->shape, t->ndim * sizeof(size_t));
    memmove(result->strides, t->strides, t->ndim * sizeof(size_t));

    return result;
}

tensor_t *tensor_copy(tensor_t *t, bool view) {
    // TODO
    printf("This function is not implemented yet\n");
    assert(false);
    view = !view;
    return t;
}

// For now there is no support for slices set
void tensor_set(tensor_t* t, const double n, const ssize_t indices[]) {
    NULL_POINTER_CHECK_(t);

    // no slices (aka negative indices) and smae dimension as t
    size_t offset = t->storage_offset;
    for (size_t i = 0; i < t->ndim; ++i) {
        if (indices[i] < 0) {
            // TODO actually handle this
            printf("Negative indices not supported yet\n");
            assert(false);
        } 

        if (indices[i] >= t->shape[i]) {
            // TODO actually handle this
            printf("Index out of bounds\n");
            assert(false);
        }

        // Compute the offset
        offset += indices[i] * t->strides[i];
    }

    // Set the value at the specified indices
    t->data[offset]->n = n;
}

// -1 means "slice all", anything >= 0 means "fix to this index"
tensor_t *tensor_get(const tensor_t* t, const ssize_t indices[]) {
    NULL_POINTER_CHECK_(t);

    size_t new_ndim = 0;
    for (size_t i = 0; i < t->ndim; ++i) {
        if (indices[i] < 0) {
            new_ndim++;
        }
    }

    tensor_t *view = tensor_create_no_data(new_ndim, t->shape);
    view->ndim = new_ndim;
    view->data = t->data;
    view->is_view = true;

    size_t offset = t->storage_offset;
    size_t j = 0;
    for (size_t i = 0; i < t->ndim; ++i) {
        if (indices[i] >= 0) {
            offset += indices[i] * t->strides[i];
        } else {
            // Slice over this axis
            view->shape[j] = t->shape[i];
            view->strides[j] = t->strides[i];
            ++j;
        }
    }

    view->storage_offset = offset;
    return view;
}

void tensor_backward(tensor_t *t) {
    if (t == NULL) {
        return;
    }

    // Scalar case
    if (t->ndim == 0) {
        // print addresses if they are not null
        backward(t->data[t->storage_offset]);
        return;
    }

    // Tensor of one element case
    // Maybe it's not a good idea to have this?
    if (t->ndim == 1 && t->shape[0] == 1) {
        backward(t->data[t->storage_offset + t->shape[0] * t->strides[0]]);
        return;
    }

    // Tensor of more than one element case
    // TODO: actually handle this
    printf("Backward pass can be done only for scalars.\n");
    assert(false);
}

void tensor_destroy(tensor_t *t) {
    if (t == NULL)
        return;

    // View tensor use data of another tensor
    if (!t->is_view) {
        size_t total = 1;
        for (size_t i = 0; i < t->ndim; i++)
            total *= t->shape[i];

        // This can potentially break the computational graph
        for (size_t i = 0; i < total; i++)
            destroy_numeric(t->data[i]);  
    }
    free(t->shape);
    free(t->strides);
    free(t);
}

void tensor_print_shape(tensor_t *t) {
    if (t == NULL) {
        printf("<tensor: (NULL)>\n");
        return;
    }

    printf("<tensor: (ndim: %zu, shape: [", t->ndim);
    for (size_t i = 0; i < t->ndim; ++i) {
        printf("%zu", t->shape[i]);
        if (i < t->ndim - 1)
            printf(", ");
    }
    printf("]>\n");
}

// Recursive helper function
void tensor_print_data_(const tensor_t *t, const size_t dim, const size_t offset, const bool is_grad) {
    if (dim == t->ndim - 1) {
        // Print innermost array
        printf("[");
        for (size_t i = 0; i < t->shape[dim]; ++i) {
            size_t index = offset + i * t->strides[dim];
            if (is_grad) {
                printf("%.2f", t->data[index]->grad);
            } else {
                printf("%.2f", t->data[index]->n);
            }
            if (i + 1 < t->shape[dim]) printf(", ");
        }
        printf("]");
    } else {
        // Recurse into the next dimension
        printf("[");
        for (size_t i = 0; i < t->shape[dim]; ++i) {
            size_t new_offset = offset + i * t->strides[dim];
            tensor_print_data_(t, dim + 1, new_offset, is_grad);
            if (i + 1 < t->shape[dim]) printf(", ");
        }
        printf("]");
    }
}

void tensor_print_grad(tensor_t *t) {
    tensor_print_shape(t);
    if (t == NULL) {
        return;
    }

    if (!t->data[t->storage_offset]->store_grad) {
        printf("grad: NULL\n");
        return;
    }

    printf("grad: ");
    // Handle scalar case
    if (t->ndim == 0) {
        printf("%.2f\n", t->data[t->storage_offset]->grad);
        return;
    }

    tensor_print_data_(t, 0, t->storage_offset, true);
    printf("\n");
}

void tensor_print(tensor_t *t) {
    tensor_print_shape(t);
    if (t == NULL) {
        return;
    }

    // Handle scalar case
    if (t->ndim == 0) {
        printf("%.2f\n", t->data[t->storage_offset]->n);
        return;
    }
    tensor_print_data_(t, 0, t->storage_offset, false);
    printf("\n");
}