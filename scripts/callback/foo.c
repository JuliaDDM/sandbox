#include "foo.h"

double bar(const double x, const double y, void *thunk, callback func)
{
    return func(thunk, x, y);
}

