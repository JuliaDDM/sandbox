default: libfoo.so
	@julia Foo.jl

libfoo.so: foo.o
	@gcc -shared -o libfoo.so foo.o

foo.o: foo.c
	@gcc -c -Wall -Werror -fpic foo.c

clean:
	@rm -f libfoo.so
	@rm -f foo.o

