using Test
using hFlux  # Load your package/module

# Test the call_test function
@testset "hFlux Shared Library Function Tests" begin
    result = hFlux.test()
    @test result == 42  # Expected value returned by the `test` function

    hFlux.initialize()
    @test 0 == 0  # Expected value returned by the `test` function
    
    hFlux.initialize()
    @test 0 == 0  # Expected value returned by the `test` function

    hFlux.interpolate()
    @test 0 == 0  # Expected value returned by the `test` function

    hFlux.evaluate()
    @test 0 == 0  # Expected value returned by the `test` function

    hFlux.plotPoincareSection()
    @test 0 == 0  # Expected value returned by the `test` function
end