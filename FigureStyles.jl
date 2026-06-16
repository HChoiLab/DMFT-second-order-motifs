module FigureStyles
using CairoMakie

export DoubleColumnTheme, SingleColumnTheme, MyTheme, phase_colors

α = 1.0
phase_colors = [
    CairoMakie.RGBA(1.0,      1.0,      1.0,      0.0),
    CairoMakie.RGBA(137/255, 161/255, 180/255, α),
    CairoMakie.RGBA(220/255, 226/255, 229/255, α),
    CairoMakie.RGBA(162/255, 191/255, 211/255, α),
    CairoMakie.RGBA(191/255, 202/255, 206/255, α),
    CairoMakie.RGBA(145/255, 171/255, 150/255, α),
    CairoMakie.RGBA(202/255, 216/255, 203/255, α),
]

MyTheme = Theme(fontsize=22, font = "Helvetica", 
Axis=(rightspinevisible=false, 
topspinevisible=false, 
xgridvisible=false,
ygridvisible=false, 
xlabelsize=24,
ylabelsize=24, 
xticklabelsize=20,
yticklabelsize=20), 
Legend=(framevisible=false, labelsize=16))

DoubleColumnTheme = Theme(fontsize=12, font = "Helvetica", 
Figure=(figure_padding = (1, 2, 1, 1)),
Axis=(rightspinevisible=false, 
topspinevisible=false, 
xgridvisible=false,
ygridvisible=false, 
xlabelsize=12,
ylabelsize=12, 
xticklabelsize=11,
yticklabelsize=11), 
Legend=(framevisible=false, labelsize=8))

SingleColumnTheme = Theme(fontsize=10, font = "Helvetica", 
Figure=(figure_padding = (0.5, 0.5, 0.2, 0.2)),
Axis=(rightspinevisible=false, 
topspinevisible=false, 
xgridvisible=false,
ygridvisible=false, 
xlabelsize=11,
ylabelsize=11, 
xticklabelsize=9,
yticklabelsize=9,
xticksize=3,
yticksize=3,
spinewidth=0.8,
xtickwidth=0.8,
ytickwidth=0.8
), 
Legend=(framevisible=false, labelsize=8))

end