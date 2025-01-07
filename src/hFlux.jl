module hFlux

# using Libdl

# Load the shared library (adjust the path to match your setup)
const lib = joinpath(@__DIR__, "..", "build", "lib", "libhFlux.so")
# const libhFlux = Libdl.dlopen(lib)

# Define Julia functions that wrap library calls
function test()
    @ccall lib.test()::Cint 
end

function printLib()
    print(lib, "\n")
end

end
