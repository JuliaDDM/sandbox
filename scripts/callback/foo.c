#include "foo.h"

double bar(const double a, const double b, void *thunk, callback func)
{
    return func(thunk, a, b);
}

