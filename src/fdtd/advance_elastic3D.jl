@inbounds @fastmath function advance!(pac::T, pap) where {T<:P_common{<:FdtdElastic}}
    for ipw in pac.activepw
        # store p for the last two steps
        # pppppp!(pap[ipw],pac.attrib_mod)
        advance_kernel!(pap[ipw], pac)
    end
    return nothing
end


# these relative indices of the arrays point to same location
# [ix,iy,iz]     --> tauxx, tauyy, and tauzz grid [1:nx,1:ny,1:nz]
# [ix+1/2,iy+1/2,iz+1]      --> tauxy
# [ix+1/2,iy+1,iz+1/2]      -->  tauxz
# [ix-1/2,iy,iz]       --> vx
# [ix,iy-1/2,iz]       --> vy
# [ix,iy,iz-1/2]       --> vz

@parallel function compute_dtau!(
    tauxx::Data.Array,
    tauyy::Data.Array,
    tauzz::Data.Array,
    tauxy::Data.Array,
    tauxz::Data.Array,
    tauyz::Data.Array,
    dtauxxdx,
    dtauxydx,
    dtauxzdx,
    dtauyydy,
    dtauxydy,
    dtauyzdy,
    dtauzzdz,
    dtauyzdz,
    dtauxzdz,
    dxI,
    dyI,
    dzI,
)

    @all(dtauxxdx) = @d_zi(tauxx) * dxI # at [ix+1/2,iy+1,iz+1] with indices []
    @all(dtauxydx) = @d_za(tauxy) * dxI # at [ix+1,iy+1/2,iz+1]
    @all(dtauxzdx) = @d_za(tauxz) * dxI # at [ix+1,iy+1,iz+1/2]


    @all(dtauyydy) = @d_yi(tauyy) * dyI # at [ix+1,iy+1/2,iz+1]
    @all(dtauxydy) = @d_ya(tauxy) * dyI # at [ix+1/2,iy+1,iz+1]
    @all(dtauyzdy) = @d_ya(tauyz) * dyI # at [ix+1,iy+1,iz+1/2]


    @all(dtauzzdz) = @d_xi(tauzz) * dzI # at [ix+1,iy+1,iz+1/2]
    @all(dtauxzdz) = @d_xa(tauxz) * dzI # at [ix+1/2,iy+1,iz+1]
    @all(dtauyzdz) = @d_xa(tauyz) * dzI # at [ix+1,iy+1/2,iz+1]

    return
end
@parallel function compute_v!(
    vx,
    vy,
    vz,
    dtauxxdx,
    dtauxydx,
    dtauxzdx,
    dtauyydy,
    dtauxydy,
    dtauyzdy,
    dtauzzdz,
    dtauyzdz,
    dtauxzdz,
    dt,
    rho,
)

    @inn(vx) =
        @inn(vx) - dt / @av_xi(rho) * (@all(dtauxxdx) + @all(dtauxydy) + @all(dtauxzdz))
    @inn(vy) =
        @inn(vy) - dt / @av_yi(rho) * (@all(dtauxydx) + @all(dtauyydy) + @all(dtauyzdz))
    @inn(vz) =
        @inn(vz) - dt / @av_zi(rho) * (@all(dtauxzdx) + @all(dtauyzdy) + @all(dtauzzdz))

    return
end


@parallel function compute_dv!(
    vx,
    vy,
    vz,
    dvxdx,
    dvydy,
    dvzdz,
    dvxdy,
    dvxdz,
    dvydx,
    dvydz,
    dvzdx,
    dvzdy,
    dxI,
    dyI,
    dzI,
)
    @all(dvxdx) = @d_za(vx) * dxI # at [ix,iy,iz]
    @all(dvydy) = @d_ya(vy) * dyI # at      "
    @all(dvzdz) = @d_xa(vz) * dzI # at      "

    @all(dvxdy) = @d_yi(vx) * dyI # at [ix+1/2,iy+1/2,iz+1]
    @all(dvxdz) = @d_xi(vx) * dzI # at [ix+1/2,iy+1,iz+1/2]

    @all(dvydz) = @d_xi(vy) * dzI # at [ix+1,iy+1/2,iz+1/2]
    @all(dvydx) = @d_zi(vy) * dxI # at [ix+1/2,iy+1/2,iz+1]

    @all(dvzdx) = @d_zi(vz) * dxI # at [ix+1/2,iy+1,iz+1/2]
    @all(dvzdy) = @d_yi(vz) * dyI # at [ix+1,iy+1/2,iz+1/2]

    return
end




@parallel function compute_tauii!(tauxx, tauyy, tauzz, dvxdx, dvydy, dvzdz, dt, M, lambda)

    @all(tauxx) =
        @all(tauxx) -
        dt * ((@all(M) * @all(dvxdx)) + (@all(lambda) * (@all(dvydy) + @all(dvzdz))))
    @all(tauyy) =
        @all(tauyy) -
        dt * ((@all(M) * @all(dvydy)) + (@all(lambda) * (@all(dvxdx) + @all(dvydy))))
    @all(tauzz) =
        @all(tauzz) -
        dt * ((@all(M) * @all(dvzdz)) + (@all(lambda) * (@all(dvydy) + @all(dvxdx))))
    return
end
@parallel function compute_tauij!(
    tauxy,
    tauxz,
    tauyz,
    dvxdy,
    dvxdz,
    dvydx,
    dvydz,
    dvzdx,
    dvzdy,
    dt,
    mu,
)
    @all(tauxz) = @all(tauxz) - dt * (@av_xzi(mu) * (@all(dvxdz) + @all(dvzdx)))
    @all(tauxy) = @all(tauxy) - dt * (@av_xyi(mu) * (@all(dvxdy) + @all(dvydx)))
    @all(tauyz) = @all(tauyz) - dt * (@av_yzi(mu) * (@all(dvydz) + @all(dvzdy)))

    return
end


for dim in [:z, :y, :x]
    fname = Symbol("memory", string(dim), "!")
    i = Symbol("i", string(dim))
    @eval @parallel_indices (iz, iy, ix) function $fname(
        memory::Data.Array,
        d::Data.Array,
        a,
        b,
        kI,
    )
        memory[iz, iy, ix] = b[$i] * memory[iz, iy, ix] + a[$i] * d[iz, iy, ix]
        d[iz, iy, ix] = d[iz, iy, ix] * kI[$i] + memory[iz, iy, ix]
        return
    end
    fname = Symbol("memory1", string(dim), "!")
    @eval @parallel_indices (iz, iy, ix) function $fname(
        memory::Data.Array,
        d::Data.Array,
        a,
        b,
        kI,
    )
        memory[iz, iy, ix] = b[$i+1] * memory[iz, iy, ix] + a[$i+1] * d[iz, iy, ix]
        d[iz, iy, ix] = d[iz, iy, ix] * kI[$i+1] + memory[iz, iy, ix]
        return
    end
    fname = Symbol("dirichlet", string(dim), "!")
    is = [:iz, :iy, :ix]

    is1, is2, is3 = [replace(is, i => ii) for ii in [:1, :2, :3]]
    isn = replace(is, i => :n)
    isnm1 = replace(is, i => :(n - 1))
    isnp1 = replace(is, i => :(n + 1))
    i1, i2 = filter(x -> x != i, is)
    v = Symbol("v", string(dim))
    vs = (:vx, :vy, :vz)
    v1, v2 = filter(x -> x != v, vs)

    @eval @parallel_indices ($i1, $i2) function $fname($v, $v1, $v2, n)
        # along other dimensions velocity grid matches tauii grid
        $v1[$(is1...)] = 0
        $v1[$(isn...)] = 0
        $v2[$(is1...)] = 0
        $v2[$(isn...)] = 0

        # relative to the tauii grid, velocity at i=2 should be zero, so making use of ghost cells 
        $v[$(is1...)] = -$v[$(is2...)]
        $v[$(isnp1...)] = -$v[$(isn...)]
        return
    end
end





function advance_kernel!(pap, pac::T) where {T<:P_common{FdtdElastic}}
    w1t = pap.w1[:t]
    mw = pap.memory_pml
    pml = pac.pml

    @parallel compute_dtau!(
        w1t[:tauxx],
        w1t[:tauyy],
        w1t[:tauzz],
        w1t[:tauxy],
        w1t[:tauxz],
        w1t[:tauyz],
        w1t[:dtauxxdx],
        w1t[:dtauxydx],
        w1t[:dtauxzdx],
        w1t[:dtauyydy],
        w1t[:dtauxydy],
        w1t[:dtauyzdy],
        w1t[:dtauzzdz],
        w1t[:dtauyzdz],
        w1t[:dtauxzdz],
        pac.fc[:dxI],
        pac.fc[:dyI],
        pac.fc[:dzI],
    )

    @parallel (
        1:size(w1t[:dtauxxdx], 1),
        1:size(w1t[:dtauxxdx], 2),
        1:size(w1t[:dtauxxdx], 3),
    ) memoryx!(
        mw[:dtauxxdx],
        w1t[:dtauxxdx],
        pml[:x][:a_half],
        pml[:x][:b_half],
        pml[:x][:k_halfI],
    )
    @parallel (
        1:size(w1t[:dtauxydx], 1),
        1:size(w1t[:dtauxydx], 2),
        1:size(w1t[:dtauxydx], 3),
    ) memory1x!(mw[:dtauxydx], w1t[:dtauxydx], pml[:x][:a], pml[:x][:b], pml[:x][:kI])
    @parallel (
        1:size(w1t[:dtauxzdx], 1),
        1:size(w1t[:dtauxzdx], 2),
        1:size(w1t[:dtauxzdx], 3),
    ) memory1x!(mw[:dtauxzdx], w1t[:dtauxzdx], pml[:x][:a], pml[:x][:b], pml[:x][:kI])

    @parallel (
        1:size(w1t[:dtauyydy], 1),
        1:size(w1t[:dtauyydy], 2),
        1:size(w1t[:dtauyydy], 3),
    ) memoryy!(
        mw[:dtauyydy],
        w1t[:dtauyydy],
        pml[:y][:a_half],
        pml[:y][:b_half],
        pml[:y][:k_halfI],
    )
    @parallel (
        1:size(w1t[:dtauxydy], 1),
        1:size(w1t[:dtauxydy], 2),
        1:size(w1t[:dtauxydy], 3),
    ) memory1y!(mw[:dtauxydy], w1t[:dtauxydy], pml[:y][:a], pml[:y][:b], pml[:y][:kI])
    @parallel (
        1:size(w1t[:dtauyzdy], 1),
        1:size(w1t[:dtauyzdy], 2),
        1:size(w1t[:dtauyzdy], 3),
    ) memory1y!(mw[:dtauyzdy], w1t[:dtauyzdy], pml[:y][:a], pml[:y][:b], pml[:y][:kI])

    @parallel (
        1:size(w1t[:dtauzzdz], 1),
        1:size(w1t[:dtauzzdz], 2),
        1:size(w1t[:dtauzzdz], 3),
    ) memoryz!(
        mw[:dtauzzdz],
        w1t[:dtauzzdz],
        pml[:z][:a_half],
        pml[:z][:b_half],
        pml[:z][:k_halfI],
    )
    @parallel (
        1:size(w1t[:dtauyzdz], 1),
        1:size(w1t[:dtauyzdz], 2),
        1:size(w1t[:dtauyzdz], 3),
    ) memory1z!(mw[:dtauyzdz], w1t[:dtauyzdz], pml[:z][:a], pml[:z][:b], pml[:z][:kI])
    @parallel (
        1:size(w1t[:dtauxzdz], 1),
        1:size(w1t[:dtauxzdz], 2),
        1:size(w1t[:dtauxzdz], 3),
    ) memory1z!(mw[:dtauxzdz], w1t[:dtauxzdz], pml[:z][:a], pml[:z][:b], pml[:z][:kI])

    @parallel compute_v!(
        w1t[:vx],
        w1t[:vy],
        w1t[:vz],
        w1t[:dtauxxdx],
        w1t[:dtauxydx],
        w1t[:dtauxzdx],
        w1t[:dtauyydy],
        w1t[:dtauxydy],
        w1t[:dtauyzdy],
        w1t[:dtauzzdz],
        w1t[:dtauyzdz],
        w1t[:dtauxzdz],
        pac.fc[:dt],
        pac.mod[:rho],
    )

    @parallel (1:pac.ic[:nz], 1:pac.ic[:ny]) dirichletx!(
        w1t[:vx],
        w1t[:vz],
        w1t[:vy],
        pac.ic[:nx],
    )
    @parallel (1:pac.ic[:nz], 1:pac.ic[:nx]) dirichlety!(
        w1t[:vy],
        w1t[:vz],
        w1t[:vx],
        pac.ic[:ny],
    )
    @parallel (1:pac.ic[:ny], 1:pac.ic[:nx]) dirichletz!(
        w1t[:vz],
        w1t[:vy],
        w1t[:vx],
        pac.ic[:nz],
    )
    @parallel compute_dv!(
        w1t[:vx],
        w1t[:vy],
        w1t[:vz],
        w1t[:dvxdx],
        w1t[:dvydy],
        w1t[:dvzdz],
        w1t[:dvxdy],
        w1t[:dvxdz],
        w1t[:dvydx],
        w1t[:dvydz],
        w1t[:dvzdx],
        w1t[:dvzdy],
        pac.fc[:dxI],
        pac.fc[:dyI],
        pac.fc[:dzI],
    )
    @parallel (1:size(w1t[:dvxdx], 1), 1:size(w1t[:dvxdx], 2), 1:size(w1t[:dvxdx], 3)) memoryx!(
        mw[:dvxdx],
        w1t[:dvxdx],
        pml[:x][:a],
        pml[:x][:b],
        pml[:x][:kI],
    )
    @parallel (1:size(w1t[:dvydy], 1), 1:size(w1t[:dvydy], 2), 1:size(w1t[:dvydy], 3)) memoryy!(
        mw[:dvydy],
        w1t[:dvydy],
        pml[:y][:a],
        pml[:y][:b],
        pml[:y][:kI],
    )
    @parallel (1:size(w1t[:dvzdz], 1), 1:size(w1t[:dvzdz], 2), 1:size(w1t[:dvzdz], 3)) memoryz!(
        mw[:dvzdz],
        w1t[:dvzdz],
        pml[:z][:a],
        pml[:z][:b],
        pml[:z][:kI],
    )

    @parallel (1:size(w1t[:dvxdy], 1), 1:size(w1t[:dvxdy], 2), 1:size(w1t[:dvxdy], 3)) memoryy!(
        mw[:dvxdy],
        w1t[:dvxdy],
        pml[:y][:a_half],
        pml[:y][:b_half],
        pml[:y][:k_halfI],
    )
    @parallel (1:size(w1t[:dvxdz], 1), 1:size(w1t[:dvxdz], 2), 1:size(w1t[:dvxdz], 3)) memoryz!(
        mw[:dvxdz],
        w1t[:dvxdz],
        pml[:z][:a_half],
        pml[:z][:b_half],
        pml[:z][:k_halfI],
    )
    @parallel (1:size(w1t[:dvydx], 1), 1:size(w1t[:dvydx], 2), 1:size(w1t[:dvydx], 3)) memoryx!(
        mw[:dvydx],
        w1t[:dvydx],
        pml[:x][:a_half],
        pml[:x][:b_half],
        pml[:x][:k_halfI],
    )

    @parallel (1:size(w1t[:dvydz], 1), 1:size(w1t[:dvydz], 2), 1:size(w1t[:dvydz], 3)) memoryz!(
        mw[:dvydz],
        w1t[:dvydz],
        pml[:z][:a_half],
        pml[:z][:b_half],
        pml[:z][:k_halfI],
    )
    @parallel (1:size(w1t[:dvzdx], 1), 1:size(w1t[:dvzdx], 2), 1:size(w1t[:dvzdx], 3)) memoryx!(
        mw[:dvzdx],
        w1t[:dvzdx],
        pml[:x][:a_half],
        pml[:x][:b_half],
        pml[:x][:k_halfI],
    )
    @parallel (1:size(w1t[:dvzdy], 1), 1:size(w1t[:dvzdy], 2), 1:size(w1t[:dvzdy], 3)) memoryy!(
        mw[:dvzdy],
        w1t[:dvzdy],
        pml[:y][:a_half],
        pml[:y][:b_half],
        pml[:y][:k_halfI],
    )

    @parallel compute_tauii!(
        w1t[:tauxx],
        w1t[:tauyy],
        w1t[:tauzz],
        w1t[:dvxdx],
        w1t[:dvydy],
        w1t[:dvzdz],
        pac.fc[:dt],
        pac.mod[:M],
        pac.mod[:lambda],
    )
    @parallel compute_tauij!(
        w1t[:tauxy],
        w1t[:tauxz],
        w1t[:tauyz],
        w1t[:dvxdy],
        w1t[:dvxdz],
        w1t[:dvydx],
        w1t[:dvydz],
        w1t[:dvzdx],
        w1t[:dvzdy],
        pac.fc[:dt],
        pac.mod[:mu],
    )

    
end
