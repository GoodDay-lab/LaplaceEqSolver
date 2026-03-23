using LaplaceEqSolver
using Test

function SolveOddCase(n)
    u = x -> sin(4x)*cos(3x)
    L = 1.0
    f = x -> - 25*sin(4x)*cos(3x) - 24*cos(4x)*sin(3x)

    x, u_num = SolveLaplace1d(n, f, u(0.0), u(L); L=L)
    result = ComputeErrors1d(u_num, u, x)

    return result
end

@testset "LaplaceEqSolver" begin
    @testset "Нечетный случай;  n = 10" begin
        result = SolveOddCase(10)

        println(result)
    end

    @testset "Нечетный случай;  n = 20" begin
        result = SolveOddCase(50)

        println(result)
    end

    @testset "Нечетный случай;  n = 100" begin
        result = SolveOddCase(100)

        println(result)
    end

    @testset "Нечетный случай;  n = 250" begin
        result = SolveOddCase(250)

        println(result)
    end
end
