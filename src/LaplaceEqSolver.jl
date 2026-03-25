module LaplaceEqSolver

using LinearAlgebra
using Statistics
using Plots

export main
export SolveLaplace1d, ComputeErrors1d

"""
    SolveLaplace1d(n, f, u0, uL)

Решение одномерного уравнения Лапласа:
    u''(x) = f(x), x ∈ [0, L]
    u(0) = u0, u(L) = uL

# Arguments
- `n`: количество внутренних узлов (шагов n+1)
- `f`: функция правой части f(x)
- `u0`: граничное условие слева
- `uL`: граничное условие справа
- `L`: длина интервала (по умолчанию 1.0)

# Returns
- `x`: массив узлов сетки
- `u`: массив значений решения
"""
function SolveLaplace1d(n, f, u0, uL; L=1.0)
    h = L / (n + 1)
    x = LinRange(0.0, L, n + 2)

    a = zeros(n)
    b = zeros(n)
    c = zeros(n)
    d = zeros(n)

    for i in 1:n
        xi = x[i+1]
        a[i] = 1.0
        b[i] = -2.0
        c[i] = 1.0
        d[i] = h^2 * f(xi)
    end

    d[1] -= u0
    d[end] -= uL

    for i in 2:n
        w = a[i] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]
    end

    u = zeros(n + 2)
    u[1] = u0
    u[end] = uL
    u[n+1] = d[n] / b[n]

    for i in n-1:-1:1
        u[i+1] = (d[i] - c[i] * u[i+2]) / b[i]
    end

    return x, u
end

"""
    ComputeErrors1d(u_numerical, u_exact, x)

Вычисляет ошибки между численным и точным решением в 1D.

# Arguments
- `u_numerical`: массив численного решения
- `u_exact`: массив точного решения или функция exact(x)
- `x`: массив узлов сетки

# Returns
- `errors`: Dict с ключами:
    - `inf_norm`: максимальная абсолютная ошибка (Чебышёвская норма)
    - `l2_norm`: среднеквадратичная ошибка (евклидова норма)
    - `relative_l2`: относительная ошибка в L2
    - `relative_inf`: относительная ошибка в L∞
"""
function ComputeErrors1d(u_numerical, u_exact, x)
    if u_exact isa Function
        u_exact_vals = u_exact.(x)
    else
        u_exact_vals = u_exact
    end

    abs_errors = abs.(u_numerical .- u_exact_vals)

    inf_norm = maximum(abs_errors)
    l2_norm = sqrt(sum(abs_errors.^2))

    u_norm_inf = maximum(abs.(u_exact_vals))
    u_norm_l2 = sqrt(sum(u_exact_vals.^2))

    relative_inf = inf_norm / (u_norm_inf + eps())
    relative_l2 = l2_norm / (u_norm_l2 + eps())

    return Dict(
        "inf_norm" => inf_norm,
        "l2_norm" => l2_norm,
        "relative_inf" => relative_inf,
        "relative_l2" => relative_l2,
        "max_error" => inf_norm,
        "rms_error" => sqrt(mean(abs_errors.^2))
    )
end


function main()
    u_exact = x -> sin(4x)*cos(3x)
    f_exact = x -> -25*sin(4x)*cos(3x) - 24*cos(4x)*sin(3x)
    n_values = 5:5:1000
    L = 1.0

    inf_norm_history = Float64[]
    l2_norm_history  = Float64[]

    for n in n_values
        x, u_num = SolveLaplace1d(n, f_exact, u_exact(0), u_exact(L); L=L)
        
        result = ComputeErrors1d(u_num, u_exact, x)
        push!(inf_norm_history, result["inf_norm"])
        push!(l2_norm_history, result["l2_norm"])
    end

    p = plot(xlabel="Количество узлов сетки",
             ylabel="Ошибка",
             title="Сходимость метода",
             yscale=:log10,
             xscale=:log10,
             grid=true)

    plot!(p, n_values, inf_norm_history,
          label="Чебышёвская норма",
          linecolor=:red,
          linewidth=2)

    plot!(p, n_values, l2_norm_history,
          label="L2 норма",
          linecolor=:blue,
          linewidth=2)

    savefig(p, "result.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module LaplaceEqSolver
