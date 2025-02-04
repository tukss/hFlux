module hFlux

const lib = joinpath(@__DIR__, "..", "build", "lib", "libhFlux.so")

function initialize()
    @ccall lib.initialize()::Cvoid 
end

function setRawFieldData()
    @ccall lib.setRawFieldData()::Cvoid 
end

function interpolate()
    @ccall lib.interpolate()::Cvoid 
end

function evaluate()
    @ccall lib.evaluate()::Cvoid 
end

function plotPoincareSection()
    @ccall lib.plotPoincareSection()::Cvoid 
end

end
