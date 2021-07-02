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

# ╔═╡ 64e45c5b-b7b2-495d-8753-68dfb388192b
import Base.OneTo, Base.ReshapedArray

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

# ╔═╡ a3863afc-ea79-401d-9bff-58cf95ea81cb
md"""
# Restriction

"""

# ╔═╡ 57acf85e-ed59-45b1-b156-bbc28a3f3908
struct Restriction{T,N,R<:NTuple{N}} <: AbstractMatrix{T}
	# rows
	ranges::R
	# columns
	dims::NTuple{N,Int}
end

# ╔═╡ 5d863fda-b7fd-4da4-8be5-c967902b0282
restriction(::Type{T}, args::NTuple{N}...) where {T,N} =
		Restriction{T,N,typeof(first(args))}(args...)

# ╔═╡ ba130356-61c2-43f6-a060-21576216ffa8
flatten(A::Restriction) = A.ranges, A.dims

# ╔═╡ a0b0c3f4-095e-4d78-966e-23d1187b752a
md"""
## Test

"""

# ╔═╡ 5ce1f0f7-ef14-4de9-a4ed-08814ce5b5a5
function test(::Type{Restriction};
		ranges = (2:3, 5:7),
		dims = (4, 8))

	A = restriction(Float64, ranges, dims)
	x = rand(prod(dims))
	y = similar(x, prod(length.(ranges)))

	@btime (*)($A, $x)
	@btime mul!($y, $A, $x)

	@assert collect(A) * x == mul!(y, A, x) == A * x
end

# ╔═╡ 464ce150-ee8f-4508-b22a-b3232d8cdcc9
md"""
# Extension

"""

# ╔═╡ 22c6d3b7-cac0-4bf3-a76b-c92e345e9ac0
struct Extension{T,N,R<:NTuple{N}} <: AbstractMatrix{T}
	# columns
	ranges::R
	# rows
	dims::NTuple{N,Int}
end

# ╔═╡ 8f9b46b5-efce-45d2-9b4e-dce5617ecd01
extension(::Type{T}, args::NTuple{N}...) where {T,N} =
		Extension{T,N,typeof(first(args))}(args...)

# ╔═╡ f4efaf92-ad4d-4d2c-895b-14d6fb6f6787
flatten(A::Extension) = A.ranges, A.dims

# ╔═╡ 55b84b1e-20e1-4494-8f0b-6cdcad4c4cfc
function Base.size(A::Restriction)
	ranges, dims = flatten(A)
	prod(length.(ranges)), prod(dims)
end

# ╔═╡ 3c685005-e25d-4249-aa2b-c9146509982d
function Base.getindex(A::Restriction, i, j)
	ranges, dims = flatten(A)

	row = reshape(CartesianIndices(ranges), prod(length.(ranges)))[i]
	col = reshape(CartesianIndices(OneTo.(dims)), prod(dims))[j]

	row == col ? one(eltype(A)) : zero(eltype(A))
end

# ╔═╡ f8f2e4a7-98df-493c-9ad8-bca55deb5550
function LinearAlgebra.:*(A::Restriction, x::AbstractVector)
	ranges, _ = flatten(A)
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

# ╔═╡ 78eac17d-05c4-41d5-8919-6bf6a703eefa
function Base.size(A::Extension)
	ranges, dims = flatten(A)
	prod(dims), prod(length.(ranges))
end

# ╔═╡ d511f224-5061-43a6-a363-b562a08e400f
function Base.getindex(A::Extension, i, j)
	ranges, dims = flatten(A)

	row = reshape(CartesianIndices(OneTo.(dims)), prod(dims))[i]
	col = reshape(CartesianIndices(ranges), prod(length.(ranges)))[j]

	row == col ? one(eltype(A)) : zero(eltype(A))
end

# ╔═╡ c64cfdfa-b65f-42ed-9ec6-0b26ca5bc2a0
function LinearAlgebra.:*(A::Extension, x::AbstractVector)
	_, dims = flatten(A)
	y = similar(x, prod(dims))
	mul!(y, A, x)
end

# ╔═╡ d9ce0a2b-f9ec-4335-8578-2f9132a35378
function LinearAlgebra.mul!(y::AbstractVector, A::Extension, x::AbstractVector)
	ranges, dims = flatten(A)

	x̄ = ReshapedArray(x, length.(ranges), ())
	ȳ = ReshapedArray(y, dims, ())

	ȳ .= 0
	view(ȳ, ranges...) .= x̄

	y
end

# ╔═╡ 5b652bcd-bad6-44c2-852d-86d79baf2ae2
function test(::Type{Extension};
		ranges = (2:3, 5:7),
		dims = (4, 8))

	A = extension(Float64, ranges, dims)
	x = rand(prod(length.(ranges)))
	y = similar(x, prod(dims))

	@btime (*)($A, $x)
	@btime mul!($y, $A, $x)

	@assert collect(A) * x == mul!(y, A, x) == A * x
end

# ╔═╡ ebf2fe32-8885-4829-8f44-c4ba76ae96e0
md"""
# Mask

"""

# ╔═╡ 11e843ca-b03f-461c-b196-4378fe2f45e9
struct Mask{T,N,R<:NTuple{N}} <: AbstractMatrix{T}
	# columns
	ranges::R
	# rows
	dims::NTuple{N,Int}
end

# ╔═╡ 2c5576a4-f369-4a9f-bbef-3028173949b3
function test(::Type{Mask};
		ranges = (2:3, 5:7),
		dims = (4, 8))

	R = restriction(Float64, ranges, dims)
	Rᵀ = extension(Float64, ranges, dims)
	x = rand(prod(dims))

	x̄ = ReshapedArray(x, dims, ())
	x̲ = x̄[ranges...]

	y = Rᵀ * (R * x)
	x .= 0; x̄[ranges...] .= x̲

	@assert x == y
end

# ╔═╡ 446a7c1c-c597-424d-9deb-c30d9ae7dc72
md"""
# Multi-diagonal matrices

"""

# ╔═╡ 5f7456b6-e92a-4004-a9a5-e3b2be463bba
md"""
!!! note "`Tridiagonal`"

	Are `Tridiagonal` matrices always square ?

"""

# ╔═╡ edb9a561-1095-4fc5-a5ce-b30aec943fdb
function test(ranges = (2:3, 5:7), dims = (4, 8))

	R = restriction(Float64, ranges, dims)
	Rᵀ = extension(Float64, ranges, dims)
	x = rand(prod(dims))

	Rᵀ * R
end

# ╔═╡ 0595da82-4241-4c4f-ba59-06d3214c47df
test(Restriction)

# ╔═╡ 515a5644-7a7d-4f09-8303-cfe0d9875193
test(Extension)

# ╔═╡ 86d3fcd3-fb66-431c-b883-52b73210a95e
test(Mask)

# ╔═╡ a47400cd-dc3b-4d1b-9791-03b21fa6edd1
function LinearAlgebra.:*(::Extension, ::Restriction)
	"coucou"
end

# ╔═╡ cc62cfe8-9f53-4013-a1ca-4a637ec124e3
test()

# ╔═╡ 669b08df-fa43-46da-97a0-68d735831548
md"""
# Misc.

"""

# ╔═╡ f36d9f36-20bc-48f7-8168-ebf33fc0b721
setdiff

# ╔═╡ 8ce5963f-0880-41a1-a99f-8155317512b4
Base.IteratorsMD.CartesianPartition

# ╔═╡ Cell order:
# ╠═a54a4980-1c7a-4cc0-aba0-440360d47dc9
# ╠═fb25196c-0c33-47a7-8492-12deacf02d5e
# ╠═949f8319-9676-46f4-afe7-6fc2b7748492
# ╠═64e45c5b-b7b2-495d-8753-68dfb388192b
# ╠═6d44144c-2066-44ca-8db2-c30232ff819b
# ╠═a8400937-394b-47e4-9b26-0c19f3833a0b
# ╠═3717745e-f80a-426e-9df0-353a13070c5c
# ╠═ed229e82-e690-44a1-9d35-abab5b1bec8c
# ╠═7380f7d6-8eb0-48bb-9f0f-1c0aef2413f0
# ╠═90138179-720f-411d-aace-ee00f88a51dc
# ╠═cf4949d9-52b3-4949-9bb2-b13eb68ff3de
# ╟─a3863afc-ea79-401d-9bff-58cf95ea81cb
# ╠═57acf85e-ed59-45b1-b156-bbc28a3f3908
# ╠═5d863fda-b7fd-4da4-8be5-c967902b0282
# ╠═ba130356-61c2-43f6-a060-21576216ffa8
# ╠═55b84b1e-20e1-4494-8f0b-6cdcad4c4cfc
# ╠═3c685005-e25d-4249-aa2b-c9146509982d
# ╠═f8f2e4a7-98df-493c-9ad8-bca55deb5550
# ╠═8af17c38-8b86-4de2-aa73-7b1f3c9ac007
# ╟─a0b0c3f4-095e-4d78-966e-23d1187b752a
# ╠═5ce1f0f7-ef14-4de9-a4ed-08814ce5b5a5
# ╠═0595da82-4241-4c4f-ba59-06d3214c47df
# ╟─464ce150-ee8f-4508-b22a-b3232d8cdcc9
# ╠═22c6d3b7-cac0-4bf3-a76b-c92e345e9ac0
# ╠═8f9b46b5-efce-45d2-9b4e-dce5617ecd01
# ╠═f4efaf92-ad4d-4d2c-895b-14d6fb6f6787
# ╠═78eac17d-05c4-41d5-8919-6bf6a703eefa
# ╠═d511f224-5061-43a6-a363-b562a08e400f
# ╠═c64cfdfa-b65f-42ed-9ec6-0b26ca5bc2a0
# ╠═d9ce0a2b-f9ec-4335-8578-2f9132a35378
# ╠═5b652bcd-bad6-44c2-852d-86d79baf2ae2
# ╠═515a5644-7a7d-4f09-8303-cfe0d9875193
# ╟─ebf2fe32-8885-4829-8f44-c4ba76ae96e0
# ╠═11e843ca-b03f-461c-b196-4378fe2f45e9
# ╠═a47400cd-dc3b-4d1b-9791-03b21fa6edd1
# ╠═2c5576a4-f369-4a9f-bbef-3028173949b3
# ╠═86d3fcd3-fb66-431c-b883-52b73210a95e
# ╟─446a7c1c-c597-424d-9deb-c30d9ae7dc72
# ╠═5f7456b6-e92a-4004-a9a5-e3b2be463bba
# ╠═edb9a561-1095-4fc5-a5ce-b30aec943fdb
# ╠═cc62cfe8-9f53-4013-a1ca-4a637ec124e3
# ╟─669b08df-fa43-46da-97a0-68d735831548
# ╠═f36d9f36-20bc-48f7-8168-ebf33fc0b721
# ╠═8ce5963f-0880-41a1-a99f-8155317512b4
