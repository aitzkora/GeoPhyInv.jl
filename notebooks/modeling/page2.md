```@meta
EditURL = "https://github.com/TRAVIS_REPO_SLUG/blob/master/"
```

the aim is to save snapshots, at given `t` in Fdtd
we start with loading some packages

```@example page2
using JuMIT
using Statistics
using Plots
pyplot()
```

create simple (almost) homogeneous acoustic model

```@example page2
model=J.Gallery.Seismic(:acou_homo1)
J.Models.Seismic_addon!(model, randn_perc=0.01)
```

a simple acquisition geometry

```@example page2
acqgeom = JuMIT.Gallery.Geom(model.mgrid,:xwell);
```

plot the model and source, receivers

```@example page2
p1=JP.seismic(model)
JP.geom!(acqgeom)
plot(p1)
```

generate time grid

```@example page2
tgrid = range(0.0,stop=2.0,length=1000)
```

Ricker wavelet

```@example page2
wav = JuMIT.Utils.Wavelets.ricker(10.0, tgrid, tpeak=0.25,);
```

distribute the same source wavelet to all the supsersources

```@example page2
acqsrc=JuMIT.Acquisition.Src_fixed(acqgeom.nss,1,[:P],wav,tgrid);
```

create `Fdtd.Param` object to prepare forward modelling
* npw corresponds to the number of independently propagating wavefields (1 in most cases)

```@example page2
pa=JuMIT.Fdtd.Param(npw=1,model=model,
	acqgeom=[acqgeom], acqsrc=[acqsrc],
	sflags=[2], rflags=[1],
	snaps_flag=true,
	tsnaps=[0.3, 0.4, 0.5],
	tgridmod=tgrid, verbose=true);
```

Once the `Param` object is created, do the modelling "without any memory allocations" using `mod!`

```@example page2
@time JuMIT.Fdtd.mod!(pa);
```

plotting snapshots of the first supersource

```@example page2
p1=[heatmap(pa.p.localpart.ss[1].snaps[:,:,ii]) for ii in 1:3]
plot(p1..., layout=(1,3), aspect_ratio=:equal)
```

plotting snapshots of the second supersource

```@example page2
p1=[heatmap(pa.p.localpart.ss[2].snaps[:,:,ii]) for ii in 1:3]
plot(p1..., layout=(1,3), aspect_ratio=:equal)
```

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
