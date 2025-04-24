#include "tensor.h"
#include "errors.h"

tensor_t *tensor_hadamard(const tensor_t *t1, const tensor_t *t2) {
    NULL_POINTER_CHECK_(t1);
    NULL_POINTER_CHECK_(t2);

    // Ensure both tensors have the same number of dimensions
    // TODO: actually handle this
    assert(t1->ndim == t2->ndim);
    

    // Ensure both tensors have the same shape
    for (size_t i = 0; i < t1->ndim; ++i) {
        // TODO: actually handle this
        assert(t1->shape[i] == t2->shape[i]);
    }

    // Allocate memory for the result tensor
    tensor_t *result = tensor_zeros(t1->ndim, t1->shape);

    // Perform element-wise multiplication
    for (size_t i = 0; i < tensor_len(result); ++i) {
        result->data[i] = numeric_mul(
            t1->data[i], t2->data[i]
        );
    }

    return result;
}

tensor_t *tensor_add(const tensor_t *t1, const tensor_t *t2) {
    NULL_POINTER_CHECK_(t1);
    NULL_POINTER_CHECK_(t2);

    // Ensure both tensors have the same number of dimensions
    // TODO: actually handle this
    assert(t1->ndim == t2->ndim);
    

    // Ensure both tensors have the same shape
    for (size_t i = 0; i < t1->ndim; ++i) {
        // TODO: actually handle this
        assert(t1->shape[i] == t2->shape[i]);
    }

    // Allocate memory for the result tensor
    tensor_t *result = tensor_zeros(t1->ndim, t1->shape);

    // Perform element-wise multiplication
    for (size_t i = 0; i < tensor_len(result); ++i) {
        result->data[i] = numeric_add(
            t1->data[i], t2->data[i]
        );
    }

    return result;
}

// Will sum each element of the tensor on a given dimension
tensor_t *tensor_sum(const tensor_t *t, const ssize_t dim) {
    NULL_POINTER_CHECK_(t);

    // Ensure the dimension is valid
    if (dim < -1 || dim >= (ssize_t) t->ndim) {
        printf("Invalid dimension: %zd for a tensor with %zu dimensions\n", dim, t->ndim);
        assert(false);
    }

    // Result tensor
    tensor_t *result;

    // If dim == -1, sum all elements
    if (dim == -1) {
        // Allocate memory for the result tensor (scalar)
        result = tensor_zeros(0, (size_t[]){});

        // Perform sum reduction on all elements
        for (size_t i = 0; i < tensor_len(t); ++i) {
            result->data[result->storage_offset] = numeric_add(
                t->data[i], result->data[result->storage_offset]
            );
        }
    } else {
        size_t new_shape[t->ndim - 1];
        for (size_t i = 0; i < t->ndim; ++i) {
            if (i == dim) continue;
            new_shape[i] = t->shape[i];
        }
        result = tensor_zeros(t->ndim-1, new_shape);

        // Perform sum reduction on all elements
        tensor_t *view;
        for (size_t i = 0; i < result->ndim; ++i) {
            //view = tensor_get(t, );
            for (size_t j = 0; j < result->shape[i]; ++j) {
                result->data[result->storage_offset + j * result->strides[j]] = numeric_add(
                    t->data[j], result->data[j]
                );
            }

            result->data[result->storage_offset] = numeric_add(
                t->data[i], result->data[result->storage_offset]
            );
        }
    }


    return result;
}