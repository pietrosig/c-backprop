#ifndef __TENSOR_MACROS_H_
#define __TENSOR_MACROS_H_

#define TENSOR_ZEROS(...)                                   \
  tensor_zeros(                                             \
    sizeof((size_t[]){ __VA_ARGS__ }) / sizeof(size_t),     \
    (size_t[]){ __VA_ARGS__ }                               \
  )

#define TENSOR_GET(t, ...)                                  \
  tensor_get(                                               \
    t,                                                      \
    (ssize_t[]){ __VA_ARGS__ }                              \
  )

#define TENSOR_SET(t, n, ...)                               \
  tensor_set(                                               \
    t,                                                      \
    n,                                                      \
    (ssize_t[]){ __VA_ARGS__ }                              \
  )

#define TENSOR_ARANGE(...)                                  \
  tensor_arange(                                            \
    sizeof((int[]){ __VA_ARGS__ }) / sizeof(int),           \
    (int[]){ __VA_ARGS__ }                                  \
  )

#endif