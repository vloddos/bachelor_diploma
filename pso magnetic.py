from collections import namedtuple
import pickle
import time

import numpy as np
from scipy.sparse import diags

InverseProblemParameters = namedtuple(
    'InverseProblemParameters',
    ('Ea', 'σi', 'σe', 'a', 'b', 'shell_size', 'RMp1', 'problem')
)
InverseProblemSolution = namedtuple(
    'InverseProblemSolution',
    ('optimum_shell', 'functionals')
)


def Ji(A0, Ea):
    return abs(A0 / Ea)


def Je(BMp1, Ea, RM, RMp1):
    return abs(2 * BMp1 / Ea * np.sqrt(np.log(RMp1 / RM) / (RMp1 ** 4 - RM ** 4)))


def J(A0, BMp1, Ea, RM, RMp1):
    return (Ji(A0, Ea) + Je(BMp1, Ea, RM, RMp1)) / 2


def get_solver_and_functional(Ea, σi, σe, a, b, shell_size, RMp1, problem):  # shell_size M
    if problem not in {'shielding', 'external cloaking', 'full cloaking'}:
        raise ValueError(f"unknown problem '{problem}'")

    R = np.hstack((np.linspace(a, b, shell_size + 1), RMp1))  # M+2

    def solve_direct_problem(σ):  # M
        σ = np.hstack((σi, σ, σe))  # M+2
        # totry to rename c,d to a,b
        c = np.hstack((
            np.vstack((  # A
                diags((R[:-1] ** 2, -R[:-2] ** 2), (0, 1)).toarray(),  # A R
                diags((σ[:-1] * R[:-1] ** 2, -σ[1:-1] * R[:-2] ** 2), (0, 1)).toarray()  # A μR
            )),
            np.roll(  # B cyclically shifted on 1 column to right
                np.vstack((  # B
                    diags((-1, 1), (0, -1), (shell_size + 1, shell_size + 1)).toarray(),  # B -1 1
                    diags((σ[1:], -σ[1:-1]), (0, -1)).toarray()  # B μ
                )),
                1,
                1
            )
        ))
        d = np.hstack((
            np.zeros(shell_size), -Ea * R[-2] ** 2,
            np.zeros(shell_size), -σ[-1] * Ea * R[-2] ** 2
        ))
        # print(c)
        # print(d)
        # print(np.linalg.cond(c))
        # print(np.hstack((c, d[:, np.newaxis])))
        # np.savetxt(r'C:\Users\Windows\Desktop\4sloy_Abblyad.txt', np.hstack((c, d[:, np.newaxis])), '%.18f')
        # np.savetxt(r'C:\Users\Windows\Desktop\4sloy_Ab.txt', np.hstack((c, d[:, np.newaxis])), '%.18f', newline=' ')
        # np.savetxt(r'C:\Users\Windows\Desktop\4sloy_A.txt', c, '%.18f')
        # np.savetxt(r'C:\Users\Windows\Desktop\4sloy_b.txt', d, '%.18f')
        return np.hsplit(np.linalg.solve(c, d), 2)  # BM+1==B0!!!

    def calculate_functionals(A0, BMp1):
        return {
            'shielding': Ji(A0, Ea),
            'external cloaking': Je(BMp1, Ea, *R[-2:]),
            'full cloaking': J(A0, BMp1, Ea, *R[-2:])
        }

    def calculate_problem_functional(σ):  # M
        A, B = solve_direct_problem(σ)
        if problem == 'shielding':
            return Ji(A[0], Ea)
        elif problem == 'external cloaking':
            return Je(B[0], Ea, *R[-2:])
        elif problem == 'full cloaking':
            return J(A[0], B[0], Ea, *R[-2:])

    return solve_direct_problem, calculate_functionals, calculate_problem_functional


# todo type annotations
# totry to use python 3.8 with := anywhere
# totry to use coconut python
def PSO(f, eps, iter_num, swarm_size, particle_size, b_lo, b_up, ω, φp, φg):
    def F(v):
        if v.ndim == 1:
            return f(v)
        elif v.ndim == 2:
            return np.apply_along_axis(f, 1, v)

    rng = np.random.default_rng()

    b_diff = b_up - b_lo
    # x = b_diff * rng.random((swarm_size, particle_size)) + b_lo
    half_swarm_size = swarm_size // 2
    x = np.empty((swarm_size, particle_size))
    x[:half_swarm_size, 0::2] = b_lo
    x[:half_swarm_size, 1:-1:2] = b_up
    x[half_swarm_size:, 0::2] = b_up
    x[half_swarm_size:, 1:-1:2] = b_lo
    x[:, -1] = b_diff * rng.random(swarm_size) + b_lo
    # print(x)

    p = x.copy()
    Fp = F(p)

    imin1 = Fp[:half_swarm_size].argmin()
    g1 = p[imin1].copy()
    Fg1 = Fp[imin1]

    imin2 = Fp[half_swarm_size:].argmin() + half_swarm_size
    g2 = p[imin2].copy()
    Fg2 = Fp[imin2]

    v = rng.random((swarm_size, particle_size)) * 2 * abs(b_diff) - abs(b_diff)  # ???choose max velocity more carefully

    for i in range(iter_num):
        # print(i)
        rp, rg = rng.random(2)
        # v = ω * v + φp * rp * (p - x) + φg * rg * (g - x)  # g-vector minus x-matrix works correctly
        v[:half_swarm_size] = \
            ω * v[:half_swarm_size] + \
            φp * rp * (p[:half_swarm_size] - x[:half_swarm_size]) + \
            φg * rg * (g1 - x[:half_swarm_size])  # g-vector minus x-matrix works correctly
        v[half_swarm_size:] = \
            ω * v[half_swarm_size:] + \
            φp * rp * (p[half_swarm_size:] - x[half_swarm_size:]) + \
            φg * rg * (g2 - x[half_swarm_size:])  # g-vector minus x-matrix works correctly

        bbi = np.where((v < -abs(b_diff)) | (v > abs(b_diff)))
        v[bbi] = abs(v[bbi]) % (2 * abs(b_diff)) - abs(b_diff)
        # x += v
        x[:, -1] += v[:, -1]
        # возвращаем вышедшие за пределы компоненты
        x[x < b_lo] = b_lo
        x[x > b_up] = b_up

        Fx = F(x)
        c = Fx < Fp
        p[c] = x[c].copy()  # when c.any()==False works correctly
        Fp[c] = Fx[c].copy()

        imin1 = Fp[:half_swarm_size].argmin()
        if Fp[imin1] < Fg1:
            g1 = p[imin1].copy()
            Fg1 = Fp[imin1]

        imin2 = Fp[half_swarm_size:].argmin() + half_swarm_size
        if Fp[imin2] < Fg2:
            g2 = p[imin2].copy()
            Fg2 = Fp[imin2]

        if Fg1 <= eps or Fg2 <= eps:
            break

    if Fg1 < Fg2:
        return g1
    else:
        return g2


def solve_inverse_problem(
        Ea, σi, σe, a, b, shell_size, RMp1, problem, eps, iter_num, swarm_size, b_lo, b_up, ω, φp, φg
):
    solve_direct_problem, calculate_functionals, calculate_problem_functional = \
        get_solver_and_functional(Ea, σi, σe, a, b, shell_size, RMp1, problem)

    g = PSO(calculate_problem_functional, eps, iter_num, swarm_size, shell_size, b_lo, b_up, ω, φp, φg)
    A, B = solve_direct_problem(g)
    return InverseProblemSolution(g, calculate_functionals(A[0], B[0]))
    # A, B = solve_direct_problem([0.004500, 40.000000, 0.004500, 40.000000, 0.004500, 40.000000, 0.004500, 34.804877])
    # A, B = solve_direct_problem(
    #     [0.004500000000000000, 40.000000000000000000, 0.004500000000000000, 40.000000000000000000, 0.004500000000000000,
    #      40.000000000000000000, 0.004500000000000000, 33.521181256597074594])
    # print(A[0])
    # print(B[0])
    # print(calculate_functionals(A[0], B[0]))

    # print('optimum shell:', g)
    # A, B = solve_direct_problem(g)
    # print('; '.join(f"{k} : {'{:1.6e}'.format(v)}" for k, v in calculate_functionals(A[0], B[0]).items()))

    # print(calculate_functionals(*solve_direct_problem(np.full(shell_size, b_lo))))
    # print(calculate_functionals(*solve_direct_problem(np.full(shell_size, b_up))))

    # A, B = solve_direct_problem(np.array(shell_size // 2 * [b_lo, b_up]))
    # A, B = solve_direct_problem(np.array((shell_size // 2 - 1) * [b_lo, b_up] + [b_lo, 17.947036677753093]))
    # A, B = solve_direct_problem(np.array((shell_size // 2 - 1) * [b_up, b_lo] + [b_up, 20.04]))
    # print('; '.join(f"{k} : {'{:1.6e}'.format(v)}" for k, v in calculate_functionals(A[0], B[0]).items()))


ω, φp, φg = 0.5, 1, 1.5
print(solve_inverse_problem(1, 1, 1, 0.035, 0.05, 16, 0.1, 'full cloaking', 0, 100, 40, 0.1, 40, 0.5, 1, 1.5))
'''
print(f'TIME = {time.asctime()}')
inverse_problem_solution_dict = {}
ipp = InverseProblemParameters(1, 1, 1, 0.01, 0.05, 2, 0.1, 'full cloaking')
for i in range(-1, -11, -1):
    εmin = 10 ** i
    ips = solve_inverse_problem(
        *ipp, 0, 100, 2000, εmin, 10, ω, φp, φg
    )
    inverse_problem_solution_dict[εmin] = ips

    print(εmin)
    print(ips)
    print('-' * 80)

print(f'TIME = {time.asctime()}')

with open(f'full cloaking m=2.pickle', 'wb') as f:
    pickle.dump(inverse_problem_solution_dict, f)
'''

'''
log_file = open('pso log.txt', 'a')
print('TIME = ', time.asctime(), file=log_file)
for problem in 'shielding', 'external cloaking':
    for a, b in (0.01, 0.05), (0.03, 0.05), (0.04, 0.05):
        for b_up in 40, 70:
            print('a b b_up', a, b, b_up)
            inverse_problem_solution_dict = {}
            for i in range(1, 9):
                print(i)
                ipp = InverseProblemParameters(1, 1, 1, a, b, i * 2, 0.1, problem)
                ips = solve_inverse_problem(
                    *ipp, 0, 100, 2000, 0.0045, b_up, ω, φp, φg
                )
                inverse_problem_solution_dict[ipp] = ips

                # print(ipp, file=log_file)
                # print(ips, file=log_file)
                # print('-' * 80, file=log_file)

                print(ipp)
                print(ips)
                print('-' * 80)

            with open(f'{problem} a={a} b={b} b_up={b_up}.pickle', 'wb') as f:
                pickle.dump(inverse_problem_solution_dict, f)

            # print('=' * 100, file=log_file)
            # print('TIME = ', time.asctime(), file=log_file)

            print('=' * 100)
            print('TIME = ', time.asctime())
log_file.close()
'''
