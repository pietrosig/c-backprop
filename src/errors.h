#ifndef __MACROS_H_
#define __MACROS_H_

#include <errno.h>

// Error messages
#define NULL_POINTER_ERR_(p) "pointer " #p " cannot be NULL"

#define BOTH_NULL_POINTERS_ERR_(p1, p2)                                        \
  "both pointers" #p1 " " #p2 " cannot be NULL at the same time"

#define GRAD_FN_INPUT_MISMATCH_ERR_(input, output)                             \
  #input " of grad_calc_t function needs to be an operand of " #output


// Error macros
// Runtime errors if the pointer p is NULL
#define NULL_POINTER_CHECK_(p)                                                 \
  do {                                                                         \
    if ((p) == NULL) {                                                         \
      RUNTIME_ERROR_(NULL_POINTER_ERR_(p));                                    \
    }                                                                          \
  } while (0)

// Runtime errors if both pointers p1 and p2 are NULL
#define BOTH_NULL_POINTERS_CHECK_(p1, p2)                                      \
  do {                                                                         \
    if ((p1) == NULL && (p2) == NULL) {                                        \
      RUNTIME_ERROR_(BOTH_NULL_POINTERS_ERR_(p1, p2));                         \
    }                                                                          \
  } while (0)

// Runtime errors if pointer input or output is NULL
//                OR
//                if input is not an operand of output
#define GRAD_FN_INPUT_CHECK_(output, input)                                    \
  do {                                                                         \
    NULL_POINTER_CHECK_(input);                                                \
    NULL_POINTER_CHECK_(output);                                               \
                                                                               \
    if ((output->op1) != (input) && (output->op2) != (input)) {                \
      RUNTIME_ERROR_(GRAD_FN_INPUT_MISMATCH_ERR_(output, input));              \
    }                                                                          \
  } while (0)

// General Runtime error
#define RUNTIME_ERROR_(msg)                                                    \
  do {                                                                         \
    fprintf(stderr, "Runtime Error: at '%s' in line %d\n", __FILE__,           \
            __LINE__);                                                         \
    fprintf(stderr, "      Message: %s\n", msg);                               \
    fprintf(stderr, "         Type: %s", #msg);                                \
    exit(-1);                                                                  \
  } while (0)

#endif
