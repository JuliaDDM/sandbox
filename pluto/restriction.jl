### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ a54a4980-1c7a-4cc0-aba0-440360d47dc9
using LinearAlgebra

# ╔═╡ fb25196c-0c33-47a7-8492-12deacf02d5e
using BenchmarkTools

# ╔═╡ 949f8319-9676-46f4-afe7-6fc2b7748492
import LinearAlgebra: mul!

# ╔═╡ 6d44144c-2066-44ca-8db2-c30232ff819b
indices(x::CartesianIndices) = x.indices

# ╔═╡ a8400937-394b-47e4-9b26-0c19f3833a0b
struct GhostedArray{T,N,AA<:AbstractArray{T,N}} <: AbstractArray{T,N}
	parent::AA
	width::NTuple{N,NTuple{2,Int}}
end

# ╔═╡ 3717745e-f80a-426e-9df0-353a13070c5c
Base.parent(x::GhostedArray) = x.parent

# ╔═╡ ed229e82-e690-44a1-9d35-abab5b1bec8c
width(x::GhostedArray) = x.width

# ╔═╡ 7380f7d6-8eb0-48bb-9f0f-1c0aef2413f0
Base.size(x::GhostedArray) = size(parent(x)) .- sum.(width(x))

# ╔═╡ 90138179-720f-411d-aace-ee00f88a51dc
function Base.getindex(x::GhostedArray, i...)
	getindex(parent(x), i .+ first.(width(x))...)
end

# ╔═╡ cf4949d9-52b3-4949-9bb2-b13eb68ff3de
# Restriction just removes ghost

# ╔═╡ 57acf85e-ed59-45b1-b156-bbc28a3f3908
begin
	struct Restriction{T,N,R} <: AbstractMatrix{T} where {R<:NTuple{N,<:AbstractUnitRange}}
		# Row space
		ranges::R
		# Column space
		dims::NTuple{N,Int}
	end

	Restriction{T}(x::NTuple{N}...) where {T,N} =
		Restriction{T,N,typeof(first(x))}(x...)
end

# ╔═╡ ba130356-61c2-43f6-a060-21576216ffa8
flatten(A::Restriction) = A.ranges, A.dims

# ╔═╡ 55b84b1e-20e1-4494-8f0b-6cdcad4c4cfc
function Base.size(A::Restriction)
	ranges, dims = flatten(A)
	prod(length.(ranges)), prod(dims)
end

# ╔═╡ 64e45c5b-b7b2-495d-8753-68dfb388192b
import Base.OneTo, Base.ReshapedArray

# ╔═╡ 3c685005-e25d-4249-aa2b-c9146509982d
function Base.getindex(A::Restriction, i, j)
	ranges, dims = flatten(A)

	row = reshape(CartesianIndices(ranges), prod(length.(ranges)))[i]
	col = reshape(CartesianIndices(OneTo.(dims)), prod(dims))[j]

	row == col ? one(eltype(A)) : zero(eltype(A))
end

# ╔═╡ f8f2e4a7-98df-493c-9ad8-bca55deb5550
function LinearAlgebra.:*(A::Restriction, x::AbstractVector)
	ranges, dims = flatten(A)
	y = similar(x, prod(length.(ranges)))
	mul!(y, A, x)
end

# ╔═╡ 8af17c38-8b86-4de2-aa73-7b1f3c9ac007
function LinearAlgebra.mul!(y::AbstractVector, A::Restriction, x::AbstractVector)
	ranges, dims = flatten(A)

	x̄ = ReshapedArray(x, dims, ())
	ȳ = ReshapedArray(y, length.(ranges), ())

	ȳ .= view(x̄, ranges...)

	y
end

# ╔═╡ fed28d89-39ef-43ae-88e4-bd3818b3df0a
ranges = (2:3, 5:7)

# ╔═╡ 8a925d91-5fd7-4a5b-beb7-e4553693f14c
dims = (4, 8)

# ╔═╡ 2cb731e1-29f2-41cd-bd70-99f846740639
A = Restriction{Float64}(ranges, dims)

# ╔═╡ 38629dee-09fe-475e-adc4-e422f941de0d
x = rand(prod(dims))

# ╔═╡ 8d0bdf18-abeb-4f67-b1ad-5703d31102b6
y = similar(x, prod(length.(ranges)))

# ╔═╡ aa3b3e49-9d47-4002-a703-124cab55d133
collect(A) * x == mul!(y, A, x) == A * x

# ╔═╡ c4967239-69a5-43b1-b992-ae118c55b46c
@btime (*)($A, $x)

# ╔═╡ 0b0ff2b0-d99e-47b1-b065-e5e0df2f13ca
@btime mul!($y, $A, $x)

# ╔═╡ 8a97eac0-b795-4181-84e3-434cb380c609
"""
Returns a `GhostedArray` with zeros in the ghost region.

"""
#=
function LinearAlgebra.:*(A::Restriction, x)

	col = CartesianIndices(x)
	row = map(indices(col), width(A)) do range, (a, b)
		UnitRange(
			first(range) + a,
			last(range) - b
		)
	end |> CartesianIndices

	y = zero(x)
	y[row] = x[row]

	GhostedArray(y, width(A))
end
=#

# ╔═╡ dae546eb-978a-4a07-8b71-21cfd6f05e35
md"""
Should we throw error on :
```
LinearAlgebra.:*(A::Restriction, x::GhostedArray)
```

"""

# ╔═╡ 888afb10-3648-4960-bf1b-bfb848125f49
#const R = Restriction(((1, 1), (1, 1)))

# ╔═╡ e4edf4e2-92b0-4759-acf5-b16c80597269
struct Extension end

# ╔═╡ f978d2df-54e6-4f53-91b8-26d5cd465ea9
function LinearAlgebra.:*(A::Extension, x::GhostedArray)
	parent(x)
end

# ╔═╡ 5dc91842-0fda-4a34-b4a1-a9e5eca7c065
const Rᵀ = Extension()

# ╔═╡ 669b08df-fa43-46da-97a0-68d735831548
md"""
# In action

"""

# ╔═╡ 9995f8fe-6c67-429f-b779-fdab35c9a115
foo = rand(4, 8)

# ╔═╡ 8b466c51-227b-4f76-9df5-9bd67b866f81
bar = GhostedArray(foo, ((1, 1), (1, 1)))

# ╔═╡ 646c0aae-3fde-452b-a72d-45654716417c
parent(bar)

# ╔═╡ d3680351-1f3d-4971-92d6-72af4c5f2da8
#Rᵀ * (R * foo)

# ╔═╡ f36d9f36-20bc-48f7-8168-ebf33fc0b721
setdiff

# ╔═╡ 8ce5963f-0880-41a1-a99f-8155317512b4
Base.IteratorsMD.CartesianPartition

# ╔═╡ Cell order:
# ╠═a54a4980-1c7a-4cc0-aba0-440360d47dc9
# ╠═949f8319-9676-46f4-afe7-6fc2b7748492
# ╠═6d44144c-2066-44ca-8db2-c30232ff819b
# ╠═a8400937-394b-47e4-9b26-0c19f3833a0b
# ╠═3717745e-f80a-426e-9df0-353a13070c5c
# ╠═ed229e82-e690-44a1-9d35-abab5b1bec8c
# ╠═7380f7d6-8eb0-48bb-9f0f-1c0aef2413f0
# ╠═90138179-720f-411d-aace-ee00f88a51dc
# ╠═cf4949d9-52b3-4949-9bb2-b13eb68ff3de
# ╠═57acf85e-ed59-45b1-b156-bbc28a3f3908
# ╠═ba130356-61c2-43f6-a060-21576216ffa8
# ╠═55b84b1e-20e1-4494-8f0b-6cdcad4c4cfc
# ╠═64e45c5b-b7b2-495d-8753-68dfb388192b
# ╠═3c685005-e25d-4249-aa2b-c9146509982d
# ╠═f8f2e4a7-98df-493c-9ad8-bca55deb5550
# ╠═8af17c38-8b86-4de2-aa73-7b1f3c9ac007
# ╠═fed28d89-39ef-43ae-88e4-bd3818b3df0a
# ╠═8a925d91-5fd7-4a5b-beb7-e4553693f14c
# ╠═2cb731e1-29f2-41cd-bd70-99f846740639
# ╠═38629dee-09fe-475e-adc4-e422f941de0d
# ╠═8d0bdf18-abeb-4f67-b1ad-5703d31102b6
# ╠═aa3b3e49-9d47-4002-a703-124cab55d133
# ╠═fb25196c-0c33-47a7-8492-12deacf02d5e
# ╠═c4967239-69a5-43b1-b992-ae118c55b46c
# ╠═0b0ff2b0-d99e-47b1-b065-e5e0df2f13ca
# ╠═8a97eac0-b795-4181-84e3-434cb380c609
# ╟─dae546eb-978a-4a07-8b71-21cfd6f05e35
# ╠═888afb10-3648-4960-bf1b-bfb848125f49
# ╠═e4edf4e2-92b0-4759-acf5-b16c80597269
# ╠═f978d2df-54e6-4f53-91b8-26d5cd465ea9
# ╠═5dc91842-0fda-4a34-b4a1-a9e5eca7c065
# ╟─669b08df-fa43-46da-97a0-68d735831548
# ╠═9995f8fe-6c67-429f-b779-fdab35c9a115
# ╠═8b466c51-227b-4f76-9df5-9bd67b866f81
# ╠═646c0aae-3fde-452b-a72d-45654716417c
# ╠═d3680351-1f3d-4971-92d6-72af4c5f2da8
# ╠═f36d9f36-20bc-48f7-8168-ebf33fc0b721
# ╠═8ce5963f-0880-41a1-a99f-8155317512b4
