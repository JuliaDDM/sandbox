A minimal working example showing pure Julia functions, even anonymous function, can be called back from a C library.

The code is composed of...

1. A C library (`libfoo.so`) with a single function (`bar`) that takes a callback function (`func`) and applies it to a couple of doubles (`x` and `y`) while passing a closure (`thunk`),
1. A Julia module with a single function (also called `bar`) that wraps the aforementioned C function.

Just run 
```bash
make
```
to see how this works (assuming both GCC and Julia are installed).

