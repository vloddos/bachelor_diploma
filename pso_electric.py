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
    ) + delta * rng.random((swarm_size, particle_size))  # tocheck even/odd reflect type
    # print(x)

    p = x.copy()
    Fp = F(p)

    imin = Fp.argmin()
    g = p[imin].copy()
    Fg = Fp[imin]

    v_min, v_max = -b_diff, b_diff
    # v_min, v_max = -b_diff/1000, b_diff/1000
    # v = (v_max - v_min) * rng.random((swarm_size, particle_size)) + v_min  # у алекса zeros
    v = np.zeros((swarm_size, particle_size))  # tocheck
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
        Ea, ei, ee, a, b, shell_size, RMp1, problem,
        eps, iter_num, swarm_size, b_lo, b_up, w, c1, c2, rng
):
    solve_direct_problem, calculate_functionals, calculate_problem_functional = \
        get_solver_and_functional(Ea, ei, ee, a, b, shell_size, RMp1, problem)

    g = PSO(
        calculate_problem_functional,
        eps, iter_num, swarm_size, shell_size, b_lo, b_up, w, c1, c2, rng
    )

    A, B = solve_direct_problem(g)
    return InverseProblemSolution(g, calculate_functionals(A[0], B[0]))


def SAME_get_2_layer_full_cloaking_problem_solutions(w, c1, c2, rng):
    with open(r'1 parameter\1 parameter full cloaking m=2.pickle', 'rb') as f:
        inverse_problem_solution_dict = pickle.load(f)

    for RMp1 in 0.07, 0.15, 0.3, 0.6:
        print(f'RMp1={RMp1}')

        for e_min, ips in inverse_problem_solution_dict.items():
            inverse_problem_solution_dict[e_min] = solve_inverse_problem(
                1, 1, 1, 0.01, 0.05, 2, RMp1, 'full cloaking', 0, 100, 20, e_min, 10, w, c1, c2, rng,
                ips.optimum_shell
            )

            print(f'e_min={e_min}')
            print(inverse_problem_solution_dict[e_min])
            print('-' * 80)

        print('=' * 100)

        with open(f'SAME full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle', 'wb') as f:
            pickle.dump(inverse_problem_solution_dict, f)


def get_2_layer_full_cloaking_problem_solutions(w, c1, c2, rng):
    print('BEGIN TIME = ', time.asctime())

    # for RMp1 in 0.07, 0.15, 0.3, 0.6:
    for RMp1 in 0.3, 0.6:
        print(f'RMp1={RMp1}')

        inverse_problem_solution_dict = {}
        for i in range(-1, -11, -1):
            e_min = 10 ** i
            for j in range(20):
                ips_new = solve_inverse_problem(
                    1, 1, 1, 0.01, 0.05, 2, RMp1, 'full cloaking', 0, 100, 20, e_min, 10, w, c1, c2, rng
                )
                ips_old = inverse_problem_solution_dict.get(e_min)

                if ips_old is None or \
                        ips_new.functionals['full cloaking'] < ips_old.functionals['full cloaking'] or \
                        (np.array(tuple(ips_new.functionals.values())) <=
                         np.array(tuple(ips_old.functionals.values()))).all():
                    inverse_problem_solution_dict[e_min] = ips_new

            print(f'e_min={e_min}')
            print(inverse_problem_solution_dict[e_min])
            print('-' * 80)

        print('=' * 100)
        print('TIME = ', time.asctime())

        with open(f'full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle', 'wb') as f:
            pickle.dump(inverse_problem_solution_dict, f)

    print('END TIME = ', time.asctime())


def get_multi_layer_all_problems_solutions(w, c1, c2, rng):
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


def get_individual_problem_solution(w, c1, c2, rng):
    J_list = []
    solution_list = []
    for i in range(20):
        print(i)
        solution_list.append(
            solve_inverse_problem(
                1, 1, 1, 0.01, 0.05, 2, 0.6, 'full cloaking', 0, 100, 20, 1e-1, 3, w, c1, c2, rng
            )
        )
        print(solution_list[-1])
        print('{:.15f}'.format(solution_list[-1].optimum_shell[1]))
        J_list.append(solution_list[-1].functionals['full cloaking'])

    imin = np.argmin(np.array(J_list))
    print('imin=', imin)

    with open(fr'updated 2 layer full cloaking\fc M=2 a=0.01 b=0.05 RMp1=0.6 emin=0.01 imin={imin}.pickle', 'wb') as f:
        pickle.dump(solution_list, f)


def get_2_layer_full_cloaking_RMp1_dependence_plot():
    import matplotlib.pyplot as pp

    def f(x):
        return np.sqrt(np.log(x / 0.05) / (x ** 4 - 0.05 ** 4))

    inverse_problem_solution_dict_list = []
    x_list, y_list = [], []

    for RMp1, marker in zip((0.07, 0.1, 0.15, 0.3, 0.6), ('o', '^', 's', '+', 'x')):
        # with open(
        #         'same solutions different Je 2 layer full cloaking\\' +
        #         f'same solutions different Je full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle',
        #         'rb'
        # ) as file:
        with open(
                'updated 2 layer full cloaking\\' +
                f'updated full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle',
                'rb'
        ) as file:
            inverse_problem_solution_dict_list.append(pickle.load(file))

        x, y = [], []
        for e_min, ips in inverse_problem_solution_dict_list[-1].items():
            x.append(np.log10(e_min))
            y.append(ips.functionals['external cloaking'])

        x_list.append(np.array(x))
        y_list.append(np.array(y))

        print(y_list[0] / y_list[-1])  # fordebug
        fxy = np.nan_to_num(y_list[0] / y_list[-1], nan=1)
        pp.scatter(x_list[-1], fxy, marker=marker, label=fr'$R_{{M+1}}={RMp1}$')

        asymptote = f(0.07) / f(RMp1)
        pp.hlines(asymptote, x_list[-1].min(), x_list[-1].max())

    # мб лучше убрать титл и написать только в техе вместе с одз
    # pp.title(
    #     r'$ f(x, y) = \frac'
    #     r'{J_{e_{R_{M+1} = 0.07, \varepsilon_{min} = x}} (\mathbf{e}^{opt})}'
    #     r'{J_{e_{R_{M+1} = y, \varepsilon_{min} = x}} (\mathbf{e}^{opt})}$',
    #     fontsize=30
    # )
    pp.xlabel(r'$\lg \varepsilon_{min}$', fontsize=30)
    pp.ylabel('$f(x,y)$', fontsize=30)
    pp.legend()
    pp.show()


if __name__ == '__main__':
    # w, c1, c2 = 0.5, 1, 1.5
    # rng = np.random.default_rng()
    # SAME_get_2_layer_full_cloaking_problem_solutions(w, c1, c2, rng)
    get_2_layer_full_cloaking_RMp1_dependence_plot()
    # get_individual_problem_solution(w, c1, c2, rng)
