module hFlux


const lib = joinpath(@__DIR__, "..", "build", "lib", "libhFlux.so")
# Define Julia functions that wrap library calls
function test()
    @ccall lib.test()::Cint 
end
function printLib()
    print(lib, "\n")
end

end
