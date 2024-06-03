### A Pluto.jl notebook ###
# v0.19.21

using Markdown
using InteractiveUtils

using SparseArrays, Test, Tullio, BenchmarkTools

# ╔═╡ ebfea7cc-298a-457c-909e-ae629c883395
# ╠═╡ skip_as_script = true
#=╠═╡
using Plots
  ╠═╡ =#

# ╔═╡ 87e5ed49-30b1-443d-b1fc-851c26451dfa
# m1 = (P * m) * Q
@inline function apply_proj_matrix!(m1::AbstractArray{T, 2}, m::AbstractArray{T, 2}, P, Q) where {T}
	@tullio m1[i, j] = P[k, i] * m[k, l] * Q[l, j]
end

# same as above, but for 3D
@inline function apply_proj_matrix!(m1::AbstractArray{T, 3}, m::AbstractArray{T, 3}, P, Q, R) where {T}
	@tullio m1[i, j, o] = P[k,i] * m[k, l, n] * Q[l, j] * R[n, o]
end

md"---"

l = LinearIndices((5,5))

l[1]

r1=range(1,5,length=500)

N1=100; N2=100

Y=zeros(N1, N2)

begin
	x1 = rand(2,3,4)
	x2 = rand(2,3,4)
	x3 = rand(3,4)
	@tullio x3[j, k] = x1[i, j, k] * x2[i, j, k];
end

begin
	f(x,y) = sin(3x) * cos(x+y)
	
	x = range(0,5, length=20)
	y = range(0,5, length=25)
	X = broadcast(Iterators.product(x,y)) do (x, y)
		f(x, y)
	end
end

# ╔═╡ 78c2e09e-843c-4b32-a09a-df4027d10895
function get_neighbour_indices(arr, val)
    len = length(arr)
    idx = searchsortedfirst(arr, val)
    if idx == 1
        return [1, 2]
    elseif idx >= len
        return [len - 1, len]
    else
        return [idx - 1, idx]
    end
end

# ╔═╡ 49d5dc87-0256-404c-9b59-f1db4450a3e3
"""
This function takes in three vectors x, y, z, and three values xi, yi, zi, corresponding to the interpolation point. It first checks if the interpolation point is inside the grid, then finds the indices of the 8 neighbouring grid points using the get_neighbour_indices function. It then computes the interpolation weights and stores them in a sparse vector with values of 1 and the specified weights at the corresponding indices. The resulting sparse vector has length n * m * p and a single column.
"""
function bilinear_interp(x, y, z, zi, yi, xi; number=Data.Number)

    n, m, p = length(x), length(y), length(z)
    x_idx1, x_idx2 = get_neighbour_indices(x, xi)
    y_idx1, y_idx2 = get_neighbour_indices(y, yi)
    z_idx1, z_idx2 = get_neighbour_indices(z, zi)
    l = LinearIndices((n, m, p))
    i000 = l[x_idx1, y_idx1, z_idx1]
    i001 = l[x_idx1, y_idx1, z_idx2]
    i010 = l[x_idx1, y_idx2, z_idx1]
    i011 = l[x_idx1, y_idx2, z_idx2]
    i100 = l[x_idx2, y_idx1, z_idx1]
    i101 = l[x_idx2, y_idx1, z_idx2]
    i110 = l[x_idx2, y_idx2, z_idx1]
    i111 = l[x_idx2, y_idx2, z_idx2]
	if(xi < minimum(x))
		dx = 0
	elseif(xi > maximum(x))
		dx = 1
	else
		dx = (xi - x[x_idx1]) / (x[x_idx2] - x[x_idx1])
	end
    if(yi < minimum(y))
		dy = 0
	elseif(yi > maximum(y))
		dy = 1
	else
		dy = (yi - y[y_idx1]) / (y[y_idx2] - y[y_idx1])
	end
	if(zi < minimum(z))
		dz = 0
	elseif(zi > maximum(z))
		dz = 1
	else
		dz = (zi - z[z_idx1]) / (z[z_idx2] - z[z_idx1])
	end    
    w000 = (1 - dx) * (1 - dy) * (1 - dz)
    w001 = (1 - dx) * (1 - dy) * dz
    w010 = (1 - dx) * dy * (1 - dz)
    w011 = (1 - dx) * dy * dz
    w100 = dx * (1 - dy) * (1 - dz)
    w101 = dx * (1 - dy) * dz
    w110 = dx * dy * (1 - dz)
    w111 = dx * dy * dz
    weights = [w000, w001, w010, w011, w100, w101, w110, w111]
    indices = [i000, i001, i010, i011, i100, i101, i110, i111]
    return sparse(indices, fill(1, 8), number.(weights), n * m * p, 1)
end

# ╔═╡ f8a70920-ac87-4126-980f-c3cbd223269a
"""
This function takes as input the x and y coordinates of the grid points, the corresponding function values z, and the x and y coordinates of the point to interpolate (xi, yi). It returns a sparse matrix with the weights for each of the four grid points used in the bilinear interpolation.

Note that if the interpolation point is outside the grid, the function will throw an error.
"""
function bilinear_interp(x, z, xi, zi; number=Data.Number)

    n, m = length(x), length(z)
    x_idx1, x_idx2 = get_neighbour_indices(x, xi)
    z_idx1, z_idx2 = get_neighbour_indices(z, zi)

    l = LinearIndices((n, m))

    i00 = l[x_idx1, z_idx1]
    i01 = l[x_idx1, z_idx2]
    i10 = l[x_idx2, z_idx1]
    i11 = l[x_idx2, z_idx2]

	if(xi < minimum(x))
		dx = 0
	elseif(xi > maximum(x))
		dx = 1
	else
		dx = (xi - x[x_idx1]) / (x[x_idx2] - x[x_idx1])
	end
    if(zi < minimum(z))
		dz = 0
	elseif(zi > maximum(z))
		dz = 1
	else
		dz = (zi - z[z_idx1]) / (z[z_idx2] - z[z_idx1])
	end
    w00 = (1 - dx) * (1 - dz)
    w01 = (1 - dx) * dz
    w10 = dx * (1 - dz)
    w11 = dx * dz

    weights = [w00, w01, w10, w11]
    indices = [i00, i01, i10, i11]

    return sparse(indices, fill(1, 4), number.(weights), n * m, 1)
end

# ╔═╡ 4fa99e86-d6fb-4da1-91d9-8065245c1191
"""
This function returns a sparse matrix with four nonzero elements, which are the weights for the four surrounding points used in bilinear interpolation. If the given xi value is outside the range of x, the function returns a sparse matrix with a single nonzero element, which is the index of the closest boundary point.
"""
function bilinear_interp(x, xi; number=Data.Number)

    n = length(x)
    x_idx1, x_idx2 = get_neighbour_indices(x, xi)

    l = LinearIndices((n,))

    i00 = l[x_idx1]
    i01 = l[x_idx2]
	if(xi < minimum(x))
		dx = 0
	elseif(xi > maximum(x))
		dx = 1
	else
		dx = (xi - x[x_idx1]) / (x[x_idx2] - x[x_idx1])
	end

    w00 = 1 - dx
    w01 = dx

    weights = [w00, w01]
    indices = [i00, i01]

    return sparse(indices, fill(1, 2), number.(weights), n, 1)
end

# ╔═╡ 1c90e052-ed95-11ed-0cad-ff4e7234f520
function get_proj_matrix(migrid::T, mmgrid::T; use_gpu=_fd_use_gpu, number=Data.Number) where {T}
    mat = mapreduce(sparse_hcat, Iterators.product(migrid...)) do P
        bilinear_interp(mmgrid..., P..., number=number)
    end
    if (use_gpu)
        return CuSparseMatrixCSC(mat)
    else
        return mat
    end
end

# ╔═╡ 137e4dfd-60f0-460b-8239-082680b09b77
function get_proj_matrix(Ps, mmgrid; use_gpu=_fd_use_gpu, number=Data.Number)
    mat = mapreduce(sparse_hcat, Ps) do P
        bilinear_interp(mmgrid..., P..., number=number)
    end
    if (use_gpu)
        return CuSparseMatrixCSC(mat)
    else
        return mat
    end
end

# ╔═╡ 205e6872-a5fa-4b2d-9d1d-8f0c55793f30
get_proj_matrix([[1.2], [1.5]], [1:10], use_gpu=false, number=Float32)

# ╔═╡ bdd0def5-2d64-429f-bd7f-85529b9dab40
get_proj_matrix([range(1,5,length=8)], [range(2,3,length=4)], use_gpu=false, number=Float32)

# ╔═╡ d0de00fa-9fa7-4159-9756-97e4c4bc33e2
@time get_proj_matrix([range(-0.75, 0.5, length=3)], [range(-1, 1, length=10)], use_gpu=false, number=Float32)

# ╔═╡ eecdb64c-c123-4678-b449-66cdef070244
get_proj_matrix([range(-0.75, 0.5, length=3), range(-0.75, 0.5, length=3)], [range(-1, 1, length=10), range(-1, 1, length=10)], use_gpu=false, number=Float32)'

# ╔═╡ 3acf2968-8591-49bf-af62-ff5926752ee2
get_proj_matrix([range(-0.75, 0.5, length=500)], [range(-1, 1, length=100)], use_gpu=false, number=Float32)'

# ╔═╡ de0c3b35-3efc-4eb6-8fac-c39ff7a0c735
# with extrap
get_proj_matrix([range(-0.5, 0.5, length=6), range(-0.25, 0.25, length=2)], [range(-0.25, 0.25, length=3), range(-0.25, 0.25, length=3)], use_gpu=false, number=Float32)'

# ╔═╡ 4416f66b-4774-4043-97f5-3ff7b2d46666
P1=get_proj_matrix( [range(-0.5, 0.5, length=N1), range(-0.5, 0.5, length=N2)], [range(-0.5, 0.25, length=20), range(-0.25, 0.5, length=25)],use_gpu=false, number=Float32)

# ╔═╡ e44e25c9-11e2-4cca-93a6-cb2373795af8
P2=get_proj_matrix([range(-0.5, 0.25, length=20), range(-0.25, 0.5, length=25)] ,[range(-0.5, 0.5, length=N1), range(-0.5, 0.5, length=N2)],use_gpu=false, number=Float32);

# ╔═╡ 17858bea-9661-46f3-b08f-e2084b0f4aef
#=╠═╡
plot(heatmap(X), heatmap(reshape(P1'*vec(X), N1, N2)), heatmap(reshape(P2' * P1'*vec(X), 20, 25)), heatmap(Y))
  ╠═╡ =#

# ╔═╡ 6f844582-16e7-4a8b-848c-f001896b7b50
P11=get_proj_matrix( [range(-0.5, 0.5, length=N1)], [range(-0.5, 0.25, length=20)],use_gpu=false, number=Float32)

# ╔═╡ f27f724a-2fe4-4dc7-9d72-3fd5c301c3ef
P11

# ╔═╡ f06fc36a-91ed-4285-9880-a559cc1592c1
P12=get_proj_matrix( [range(-0.5, 0.5, length=N2)], [range(-0.25, 0.5, length=25)],use_gpu=false, number=Float32)

# ╔═╡ f2c65505-34b1-4de7-9fc0-33bd087b2b74
@btime apply_proj_matrix!(Y, X, P11, P12)

# ╔═╡ be183231-697e-408f-aa08-ad0d6db1f663
@time bilinear_interp(1:10, 1.8, number=Float32)

# ╔═╡ 08a62f0b-5eee-4cd8-82cb-7d5c2317c257
@btime bilinear_interp(r1, 0.3, number=Float32)

# ╔═╡ 95ff1284-db71-4927-b509-e4f76da8c892
@test sum(bilinear_interp(1:10, range(10, 20, length=7), 2.2, 14.4, number=Float64)) ≈ 1

# ╔═╡ e1c0c307-2fb4-4fe8-a6d0-cb26624c9075
@test sum(bilinear_interp(1:10, range(10, 20, length=7), range(30, 41, length=5), 5.2, 14.2, 35.0, number=Float64)) ≈ 1

