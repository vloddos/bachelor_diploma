from collections import namedtuple
import pickle
import time

import numpy as np
from scipy.sparse import diags

InverseProblemParameters = namedtuple(
    'InverseProblemParameters',
    ('Ea', 'ei', 'ee', 'a', 'b', 'shell_size', 'RMp1', 'problem')
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


def get_solver_and_functional(Ea, ei, ee, a, b, shell_size, RMp1, problem):  # shell_size M
    if problem not in {'shielding', 'external cloaking', 'full cloaking'}:
        raise ValueError(f"unknown problem '{problem}'")

    R = np.hstack((np.linspace(a, b, shell_size + 1), RMp1))  # M+2

    def solve_direct_problem(e):  # M
        e = np.hstack((ei, e, ee))  # M+2
        # totry to rename c,d to a,b
        c = np.hstack((
            np.vstack((  # A
                diags((R[:-1] ** 2, -R[:-2] ** 2), (0, 1)).toarray(),  # A R
                diags((e[:-1] * R[:-1] ** 2, -e[1:-1] * R[:-2] ** 2), (0, 1)).toarray()  # A μR
            )),
            np.roll(  # B cyclically shifted on 1 column to right
                np.vstack((  # B
                    diags((-1, 1), (0, -1), (shell_size + 1, shell_size + 1)).toarray(),  # B -1 1
                    diags((e[1:], -e[1:-1]), (0, -1)).toarray()  # B μ
                )),
                1,
                1
            )
        ))
        d = np.hstack((
            np.zeros(shell_size), -Ea * R[-2] ** 2,
            np.zeros(shell_size), -e[-1] * Ea * R[-2] ** 2
        ))
        return np.hsplit(np.linalg.solve(c, d), 2)  # BM+1==B0!!!

    def calculate_functionals(A0, BMp1):
        return {
            'shielding': Ji(A0, Ea),
            'external cloaking': Je(BMp1, Ea, *R[-2:]),
            'full cloaking': J(A0, BMp1, Ea, *R[-2:])
        }

    def calculate_problem_functional(e):  # M
        A, B = solve_direct_problem(e)
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
def PSO(f, eps, iter_num, swarm_size, particle_size, b_lo, b_up, w, c1, c2, rng):
    def F(v):
        if v.ndim == 1:
            return f(v)
        elif v.ndim == 2:
            return np.apply_along_axis(f, 1, v)

    b_diff = b_up - b_lo
    # delta = (b_up - b_lo) / 100
    delta = 1
    t = b_lo, b_up - delta
    x = np.pad(
        np.tile([t, t[::-1]], (swarm_size // 2, particle_size // 2)),
        ((0, swarm_size % 2), (0, particle_size % 2)),
        'reflect'
    ) + delta * rng.random((swarm_size, particle_size))
    # print(x)

    p = x.copy()
    Fp = F(p)

    imin = Fp.argmin()
    g = p[imin].copy()
    Fg = Fp[imin]

    v_min, v_max = -b_diff, b_diff
    v = (v_max - v_min) * rng.random((swarm_size, particle_size)) + v_min
    # v = 2 * b_diff * rng.random((swarm_size, particle_size)) - b_diff  # init???
    # x v beyond boundaries???

    for i in range(iter_num):
        rp, rg = rng.random(2)

        v = w * v + c1 * rp * (p - x) + c2 * rg * (g - x)  # g-vector minus x-matrix works correctly
        v[v < v_min] = v_min
        v[v > v_max] = v_max

        x += v
        x[x < b_lo] = b_lo
        x[x > b_up] = b_up

        Fx = F(x)
        c = Fx < Fp
        p[c] = x[c].copy()  # when c.any()==False works correctly
        Fp[c] = Fx[c].copy()

        imin = Fp.argmin()
        if Fp[imin] < Fg:
            g = p[imin].copy()
            Fg = Fp[imin]

        if Fg <= eps:
            break

    return g


def solve_inverse_problem(
        Ea, ei, ee, a, b, shell_size, RMp1, problem, eps, iter_num, swarm_size, b_lo, b_up, w, c1, c2, rng
):
    solve_direct_problem, calculate_functionals, calculate_problem_functional = \
        get_solver_and_functional(Ea, ei, ee, a, b, shell_size, RMp1, problem)

    g = PSO(calculate_problem_functional, eps, iter_num, swarm_size, shell_size, b_lo, b_up, w, c1, c2, rng)
    A, B = solve_direct_problem(g)
    return InverseProblemSolution(g, calculate_functionals(A[0], B[0]))


w, c1, c2 = 0.5, 1, 1.5
rng = np.random.default_rng()
solution_list = []
for i in range(20):
    print(i)
    solution_list.append(
        solve_inverse_problem(
            1, 1, 1, 0.04, 0.05, 16, 0.1, 'full cloaking', 0, 100, 40, 0.0045, 70, w, c1, c2, rng
        )
    )
    print(solution_list[-1])

with open('fc 0.04 0.05 70 16 inner.pickle', 'wb') as f:
    pickle.dump(solution_list, f)

'''
print('BEGIN TIME = ', time.asctime())

for problem in 'shielding', 'external cloaking', 'full cloaking':
    for a, b in (0.01, 0.05), (0.03, 0.05), (0.04, 0.05):
        for b_up in 40, 70:
            inverse_problem_solution_dict = {}
            for M in range(2, 18, 2):
                ipp = InverseProblemParameters(1, 1, 1, a, b, M, 0.1, problem)
                for i in range(20):
                    ips_new = solve_inverse_problem(
                        *ipp, 0, 100, 40, 0.0045, b_up, w, c1, c2, rng
                    )
                    ips_old = inverse_problem_solution_dict.get(ipp)

                    if ips_old is None or \
                            ips_new.functionals[ipp.problem] < ips_old.functionals[ipp.problem] or \
                            (np.array(tuple(ips_new.functionals.values())) <=
                             np.array(tuple(ips_old.functionals.values()))).all():
                        inverse_problem_solution_dict[ipp] = ips_new

            print(inverse_problem_solution_dict)
            print('=' * 100)
            print('TIME = ', time.asctime())

            with open(f'{problem} a={a} b={b} b_up={b_up}.pickle', 'wb') as f:
                pickle.dump(inverse_problem_solution_dict, f)

print('END TIME = ', time.asctime())
'''

'''
fc 0.01 0.05 70 16 inner
-2
InverseProblemSolution(optimum_shell=array([7.00000000e+01, 4.50000000e-03, 7.00000000e+01, 4.50000000e-03,
       7.00000000e+01, 4.50000000e-03, 6.99984848e+01, 4.50000000e-03,
       7.00000000e+01, 4.50000000e-03, 7.00000000e+01, 4.50000000e-03,
       7.00000000e+01, 4.50000000e-03, 7.00000000e+01, 6.90930886e-02]), functionals={'shielding': 6.041091589762917e-16, 'external cloaking': 9.249297156456155e-17, 'full cloaking': 3.483010652704266e-16})

fc 0.03 0.05 40 16 inner
9
InverseProblemSolution(optimum_shell=array([4.00000000e+01, 4.50000000e-03, 3.19648247e+01, 4.50000000e-03,
       4.00000000e+01, 4.50000000e-03, 3.79972431e+01, 4.50000000e-03,
       4.00000000e+01, 4.50000000e-03, 3.14406668e+01, 4.50000000e-03,
       4.00000000e+01, 4.50000000e-03, 3.16905971e+01, 1.70197337e+00]), functionals={'shielding': 7.044492338761227e-08, 'external cloaking': 2.3489244728792817e-17, 'full cloaking': 3.522246170555076e-08})

fc 0.03 0.05 70 16 inner
5
InverseProblemSolution(optimum_shell=array([7.00000000e+01, 4.50000000e-03, 6.03563553e+01, 4.50000000e-03,
       7.00000000e+01, 4.50000000e-03, 5.19757138e+01, 4.50000000e-03,
       7.00000000e+01, 4.50000000e-03, 6.83606785e+01, 4.50000000e-03,
       7.00000000e+01, 4.50000000e-03, 1.53285702e+01, 1.76522804e+01]), functionals={'shielding': 2.716789575488618e-09, 'external cloaking': 2.117471087400077e-16, 'full cloaking': 1.3583948936178634e-09})

fc 0.04 0.05 70 14 inner
4
InverseProblemSolution(optimum_shell=array([4.50000000e-03, 7.00000000e+01, 4.50000000e-03, 7.00000000e+01,
       4.50000000e-03, 6.99803242e+01, 4.50000000e-03, 7.00000000e+01,
       4.50000000e-03, 7.00000000e+01, 4.50000000e-03, 7.00000000e+01,
       4.50000000e-03, 5.22391937e+01]), functionals={'shielding': 7.104668239347539e-06, 'external cloaking': 2.927199615134163e-16, 'full cloaking': 3.5523341198201295e-06})

fc 0.04 0.05 70 16 inner       
14
InverseProblemSolution(optimum_shell=array([4.50000000e-03, 7.00000000e+01, 4.50000000e-03, 6.55556329e+01,
       4.50000000e-03, 5.22605322e+01, 4.50000000e-03, 7.00000000e+01,
       4.50000000e-03, 7.00000000e+01, 4.50000000e-03, 7.00000000e+01,
       4.50000000e-03, 7.00000000e+01, 4.50000000e-03, 5.79339623e+01]), functionals={'shielding': 6.140665608438516e-06, 'external cloaking': 2.2587837805243685e-10, 'full cloaking': 3.070445743408284e-06})
'''
