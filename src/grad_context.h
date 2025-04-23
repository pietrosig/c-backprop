#ifndef __GRAD_CONTEXT_H
#define __GRAD_CONTEXT_H

#include <stdbool.h>

#ifdef GRAD_CONTEXT_IMPLEMENTATION
  // This is a global variable
  _Thread_local bool __grad_enabled = false;
#else
  extern _Thread_local bool __grad_enabled;
#endif


static inline bool _grad_set(bool on) {
    __grad_enabled = on;
    return true;
}

#define STORE_GRAD                                                      \
  for (bool _once = _grad_set(true);                                    \
       _once;                                                           \
       _once = (_grad_set(false), false))

#endif 
