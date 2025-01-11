module hFlux


const lib = joinpath(@__DIR__, "..", "build", "lib", "libhFlux.so")
# Define Julia functions that wrap library calls
function test()
    @ccall lib.test()::Cint 
end

function initialize()
    @ccall lib.initialize()::Cvoid 
end

function setRaFieldData()
    @ccall lib.initialize()::Cvoid 
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
