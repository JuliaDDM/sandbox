
# D'après «Multi-Threading Using Julia for Enterprises», Jeff Bezanson,
# https://www.youtube.com/watch?v=FzhipiZO4Jk
#
# A lancer via par ex. : $ julia -t4 mandel.jl 

using Images
using Base.Threads
using ThreadsX


#@async println("Hello from thread ", Threads.threadid())
#Base.Threads.@spawn println("Hello from thread ", Threads.threadid())

# The Mandelbrot set escape time function
function escapetime(z; maxiter=1000)
    c = z
    for n = 1:maxiter
        if abs(z) > 2
            return n-1
        end
        z = z^2 + c
    end
    return maxiter
end
    
# sequential code :
function mandel_seq(;width=800, height=800, maxiter=1000)
    out = zeros(Int, height, width)
    real = range(-2.0, 2.0, length=width)
    imag = range(-2.0, 2.0, length=height)
    for x in 1:width
        for y in 1:height
            z = real[x] + imag[y]*im
            out[y,x] = escapetime(z, maxiter=maxiter)
        end
    end
    return out
end
println("Sequential: ")
@time m = mandel_seq(width=800, height=800, maxiter=1000);
@time m = mandel_seq(width=800, height=800, maxiter=1000);
img = Gray.((m.%400)./100)
save("img_seq.png", clamp.(img, 0, 1))
println("")

# threaded code:
function mandel_thread(;width=800, height=800, maxiter=1000)
    out = zeros(Int, height, width)
    real = range(-2.0, 2.0, length=width)
    imag = range(-2.0, 2.0, length=height)
    @threads for x in 1:width
        for y in 1:height
            z = real[x] + imag[y]*im
            out[y,x] = escapetime(z, maxiter=maxiter)
        end
    end
    return out
end
println("Multi-threaded: ")
@time m = mandel_thread(width=800, height=800, maxiter=1000);
@time m = mandel_thread(width=800, height=800, maxiter=1000);
img = Gray.((m.%400)./100)
save("img_thread.png", clamp.(img, 0, 1))
println("")



# task-based code:
function mandel_task(;width=800, height=800, maxiter=1000)
    out = zeros(Int, height, width)
    real = range(-2.0, 2.0, length=width)
    imag = range(-2.0, 2.0, length=height)
    @sync for x in 1:width
        Base.Threads.@spawn for y in 1:height
            z = real[x] + imag[y]*im
            out[y,x] = escapetime(z, maxiter=maxiter)
        end
    end
    return out
end
println("Task-based: ")
@time m = mandel_task(width=800, height=800, maxiter=1000);
@time m = mandel_task(width=800, height=800, maxiter=1000);
img = Gray.((m.%400)./100)
save("img_task.png", clamp.(img, 0, 1))
println("")


# ThreadsX code:
function mandel_ThreadsX(;width=800, height=800, maxiter=1000)
    out = zeros(Int, height, width)
    real = range(-2.0, 2.0, length=width)
    imag = range(-2.0, 2.0, length=height)
    ThreadsX.foreach(1:width) do x
        for y in 1:height
            z = real[x] + imag[y]*im
            out[y,x] = escapetime(z, maxiter=maxiter)
        end    
    end
    return out
end
println("ThreadsX: ")
@time m = mandel_ThreadsX(width=800, height=800, maxiter=1000);
@time m = mandel_ThreadsX(width=800, height=800, maxiter=1000);
img = Gray.((m.%400)./100)
save("img_ThreadsX.png", clamp.(img, 0, 1))
println("")

