import numpy as np
from scipy.sparse import diags


def Ji(A0, Ha):
    return abs(A0 / Ha)


def Je(BMp1, Ha, RM, RMp1):
    return abs(2 * BMp1 / Ha * np.sqrt(np.log(RMp1 / RM) / (RMp1 ** 4 - RM ** 4)))


def J(A0, BMp1, Ha, RM, RMp1):
    return (Ji(A0, Ha) + Je(BMp1, Ha, RM, RMp1)) / 2


def get_solver_and_functional(Ha, μi, μe, a, b, shell_size, RMp1, problem):  # shell_size M
    if problem not in {'shielding', 'external cloaking', 'full cloaking'}:
        raise ValueError(f"unknown problem '{problem}'")

    R = np.hstack((np.linspace(a, b, shell_size + 1), RMp1))  # M+2

    def solve_direct_problem(μ):  # M
        μ = np.hstack((μi, μ, μe))  # M+2
        # totry to rename c,d to a,b
        c = np.hstack((
            np.vstack((  # A
                diags((R[:-1] ** 2, -R[:-2] ** 2), (0, 1)).toarray(),  # A R
                diags((μ[:-1] * R[:-1] ** 2, -μ[1:-1] * R[:-2] ** 2), (0, 1)).toarray()  # A μR
            )),
            np.roll(  # B cyclically shifted on 1 column to right
                np.vstack((  # B
                    diags((-1, 1), (0, -1), (shell_size + 1, shell_size + 1)).toarray(),  # B -1 1
                    diags((μ[1:], -μ[1:-1]), (0, -1)).toarray()  # B μ
                )),
                1,
                1
            )
        ))
        d = np.hstack((
            np.zeros(shell_size), -Ha * R[-2] ** 2,
            np.zeros(shell_size), -μ[-1] * Ha * R[-2] ** 2
        ))
        return np.hsplit(np.linalg.solve(c, d), 2)  # BM+1==B0!!!

    def calculate_functionals(A0, BMp1):
        return {
            'shielding': Ji(A0, Ha),
            'external cloaking': Je(BMp1, Ha, *R[-2:]),
            'full cloaking': J(A0, BMp1, Ha, *R[-2:])
        }

    def calculate_problem_functional(μ):  # M
        A, B = solve_direct_problem(μ)
        if problem == 'shielding':
            return Ji(A[0], Ha)
        elif problem == 'external cloaking':
            return Je(B[0], Ha, *R[-2:])
        elif problem == 'full cloaking':
            return J(A[0], B[0], Ha, *R[-2:])

    return solve_direct_problem, calculate_functionals, calculate_problem_functional


# todo type annotations
# totry to use python 3.8 with := anywhere
# totry to use coconut python
def PSO(f, ε, iter_num, swarm_size, particle_size, b_lo, b_up, ω, φp, φg):
    def F(v):
        if v.ndim == 1:
            return f(v)
        elif v.ndim == 2:
            return np.apply_along_axis(f, 1, v)

    # todo hold p,F(p),g,F(g) for optimizing and recount F(p)[c] only
    rng = np.random.default_rng()

    b_diff = b_up - b_lo
    x = b_diff * rng.random((swarm_size, particle_size)) + b_lo
    p = x.copy()
    g = p[F(p).argmin()].copy()
    v = rng.random((swarm_size, particle_size)) * 2 * abs(b_diff) - abs(b_diff)  # ???choose max velocity more carefully

    for i in range(iter_num):
        # print(i)
        rp, rg = rng.random(2)
        v = ω * v + φp * rp * (p - x) + φg * rg * (g - x)  # g-vector minus x-matrix works correctly
        x += v
        # возвращаем вышедшие за пределы компоненты
        x[x < b_lo] = b_lo
        x[x > b_up] = b_up

        c = F(x) < F(p)
        p[c] = x[c].copy()  # when c.any()==False works correctly

        fp = F(p)
        if fp.min() < F(g):
            g = p[fp.argmin()].copy()

        if F(g) < ε:
            break

    return g


def solve_inverse_problem(Ha, μi, μe, a, b, shell_size, RMp1, problem, ε, iter_num, swarm_size, b_lo, b_up, ω, φp, φg):
    solve_direct_problem, calculate_functionals, calculate_problem_functional = \
        get_solver_and_functional(Ha, μi, μe, a, b, shell_size, RMp1, problem)

    g = PSO(calculate_problem_functional, ε, iter_num, swarm_size, shell_size, b_lo, b_up, ω, φp, φg)

    print('optimum shell:', g)
    A, B = solve_direct_problem(g)
    print('; '.join(f"{k} : {'{:1.6e}'.format(v)}" for k, v in calculate_functionals(A[0], B[0]).items()))
    # for k, v in calculate_functionals(A[0], B[0]).items():
    #     print(k, ':', '{:1.2e}'.format(v), end='; ')
    # print()
    # print(calculate_functionals(*solve_direct_problem(np.full(shell_size, b_lo))))
    # print(calculate_functionals(*solve_direct_problem(np.full(shell_size, b_up))))
    # print(calculate_functionals(*solve_direct_problem(np.array(shell_size // 2 * [b_lo, b_up]))))
    # print(calculate_functionals(*solve_direct_problem(np.array((shell_size // 2 - 1) * [b_lo, b_up] + [b_lo, 57.93]))))
    # print(calculate_functionals(*solve_direct_problem(np.array((shell_size // 2 - 1) * [b_up, b_lo] + [b_up, 20.04]))))


# ω, φp, φg = 0.7298, 1.49618, 1.49618
# ω, φp, φg = 0.5, 1, 1.5
'''
for ω, φp, φg in ((0.7298, 1.49618, 1.49618), (0.5, 1, 1.5)):
    print(f'ω={ω}, φp={φp}, φg={φg}')
    for i in range(1, 11):
        print(i)
        μmin = ε = 10 ** -i
        solve_inverse_problem(
            # 1488, 1, 1, 0.04, 0.05, 16, 0.1, 'full cloaking', 1e-7, 100, 160, 0.0045, 70, 0.7928, 1.49618, 1.49618
            # 1488, 1, 1, 0.04, 0.05, 10, 0.1, 'shielding', 1e-7, 100, 160, 0.0045, 40, 0.7928, 1.49618, 1.49618
            # 1000, 1, 1, 0.01, 0.05, 2, 0.1, 'full cloaking', ε, 100, 20, μmin, 10, 0.7928, 1.49618, 1.49618
            1000, 1, 1, 0.01, 0.05, 2, 0.1, 'full cloaking', ε, 100, 20, μmin, 10, ω, φp, φg
        )
    print('=' * 80)
# '''

# '''
for ω, φp, φg in ((0.7298, 1.49618, 1.49618), (0.5, 1, 1.5)):
    print(f'ω={ω}, φp={φp}, φg={φg}')
    for i in range(1, 9):
        M = i * 2
        print(f'M={M}')
        solve_inverse_problem(
            1000, 1, 1, 0.04, 0.05, M, 0.1, 'full cloaking', 1e-10, 100, 20 * i, 0.0045, 40, ω, φp, φg
        )
    print('=' * 80)
# '''
