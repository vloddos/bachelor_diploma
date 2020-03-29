import numpy as np
from scipy.sparse import diags

'''
todo random vector generator by web with n^m vectors
m layers, n points on segment [σmin,σmax], n^m vectors in total
k=swarm_size
take k random unique vectors if n^m>k
else take all the n^m vectors
'''


# tocheck can functional value be <0????????????????
def Ji(A0, Ha):
    return A0 / Ha


def Je(BMp1, Ha, RM, RMp1):
    return 2 * BMp1 / Ha * np.sqrt(np.log(RMp1 / RM) / (RMp1 ** 4 - RM ** 4))


def J(A0, BMp1, Ha, RM, RMp1):
    return (Ji(A0, Ha) + Je(BMp1, Ha, RM, RMp1)) / 2


def get_solver_and_functional(Ha, μi, μe, a, b, shell_size, RMp1, problem):  # shell_size=M
    if problem not in {'shielding', 'external cloaking', 'full cloaking'}:
        raise ValueError(f"unknown problem '{problem}'")

    R = np.hstack((np.linspace(a, b, shell_size + 1), RMp1))  # len = M+2

    def solve_direct_problem(μ):  # BM+1==B0!!!!!!!!!!!!!!!!!!!
        μ = np.hstack((μi, μ, μe))  # len = M+2
        # totry to rename c,d to a,b
        c = np.hstack((  # totry use for loop to initialize c
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
        return np.hsplit(np.linalg.solve(c, d), 2)  # ???is solve mutator

    def calculate_functionals(A, B):
        return {
            'shielding': Ji(A[0], Ha),
            'external cloaking': Je(B[0], Ha, *R[-2:]),
            'full cloaking': J(A[0], B[0], Ha, *R[-2:])
        }

    def calculate_problem_functional(μ):  # (μi, μ, μe) len = M+2
        return calculate_functionals(*solve_direct_problem(μ))[problem]

    # A,B=solve_direct_problem(μ)
    # if problem=='shielding':
    #     return Ji(A[0],Ha)
    # elif problem=='external cloaking':
    #     return Je(B[0],Ha,*R[-2:])
    # elif problem=='full cloaking':
    #     return J(A[0],B[0],Ha,*R[-2:])

    return solve_direct_problem, calculate_functionals, calculate_problem_functional


# tocheck all copy/assignment operations
# todo random function with boundaries
# todo type annotations
# totry to use python 3.8 with := anywhere
# totry to use coconut python
# todo rename args to independent from task names
def PSO(f, ε, iter_num, swarm_size, shell_size, b_lo, b_up, web_points_num, ω, φp, φg):
    def F(v):
        if v.ndim == 1:
            return f(v)
        elif v.ndim == 2:
            # return np.array([f(i) for i in v])
            return np.apply_along_axis(f, 1, v)

    # todo hold p,F(p),g,F(g) for optimizing and recount F(p)[c] only
    b_diff = b_up - b_lo
    x = np.random.rand(swarm_size, shell_size) * b_diff + b_lo
    # x = np.random.choice(  # ???
    #     np.linspace(b_lo, b_up, web_points_num),
    #     (swarm_size, shell_size)
    # )

    p = x.copy()

    # print(p)
    # print(F(p).argmin())
    g = p[F(p).argmin()].copy()
    # ???choose max velocity more carefully
    v = np.random.rand(swarm_size, shell_size) * 2 * abs(b_diff) - abs(b_diff)

    for i in range(iter_num):
        # ???make rp pg scalars
        print(i)
        rp, rg = np.random.rand(2)  # np.split(  # like vsplit by default
        # np.random.rand(2 * swarm_size, shell_size),
        # 2
        # )
        v = ω * v + φp * rp * (p - x) + φg * rg * (g - x)  # g-vector minus x-matrix works correctly
        x += v
        print(f'particles beyond the boundaries: {((x < b_lo) | (x > b_up)).sum() / x.size * 100}%')  # fordebug
        print(f'vector of percent exceeding borders: {x[(x < b_lo) | (x > b_up)] / b_diff * 100}')  # fordebug
        bbi = np.where((x < b_lo) | (x > b_up))
        x[bbi] = abs(x[bbi]) % b_diff + b_lo

        c = F(x) < F(p)
        p[c] = x[c].copy()  # when c.any()==False works correctly

        # totry reduce code
        fp = F(p)
        if (fp.min() < F(g)).any():
            g = p[fp.argmin()].copy()

        if abs(F(g)) < ε:
            break

    return g


solve_direct_problem, calculate_functionals, calculate_problem_functional = \
    get_solver_and_functional(
        5000, 1, 1, 0.04, 0.05, 2, 0.1, 'full cloaking'  # tocheck Ha
    )

ω, φp, φg = 0.7928, 1.49618, 1.49618
# ω, φp, φg = 0.2, 1.4, 2.8
g = PSO(calculate_problem_functional, 1e-5, 100, 20, 2, 1e-1, 10, 100, ω, φp, φg)

print('optimum shell:', g)
print(calculate_functionals(*solve_direct_problem(g)))
