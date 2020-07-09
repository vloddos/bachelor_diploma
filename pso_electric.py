from collections import namedtuple
import pickle
import time
import pathlib

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as pp

InverseProblemParameters = namedtuple(
    'InverseProblemParameters',
    ('Ea', 'ei', 'ee', 'a', 'b', 'shell_size', 'RMp1', 'problem')
)
InverseProblemSolution = namedtuple(
    'InverseProblemSolution',
    ('optimum_shell', 'functionals')
)
InverseProblem = namedtuple(
    'InverseProblem',
    ('parameters', 'solution')
)
ProblemDesign = namedtuple(
    'ProblemDesign',
    ('one_parameter', 'scheme')  # layer???
)


def Ji(A0, Ea):
    return abs(A0 / Ea)


def Je(BMp1, Ea, RM, RMp1):
    return abs(2 * BMp1 / Ea * np.sqrt(np.log(RMp1 / RM) / (RMp1 ** 4 - RM ** 4)))


def J(A0, BMp1, Ea, RM, RMp1):
    return (Ji(A0, Ea) + Je(BMp1, Ea, RM, RMp1)) / 2


# def get_solver_and_functional(Ea, ei, ee, a, b, shell_size, RMp1, problem):  # shell_size M
def get_solver_and_functional(inverse_problem_parameters):  # shell_size M
    Ea, ei, ee, a, b, shell_size, RMp1, problem = inverse_problem_parameters

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
def PSO(f, eps, iter_num, swarm_size, particle_size, b_lo, b_up, w, c1, c2, rng, problem_design):
    def F(v):
        if v.ndim == 1:
            return f(v)
        elif v.ndim == 2:
            return np.apply_along_axis(f, 1, v)

    # b_diff = b_up - b_lo
    if 1 / (b_up - b_lo) < 0.2:
        delta = 1
    else:  # elif 1 / (b_up - b_lo) <= 1:
        delta = 10 ** np.log10([
            -b_lo + 0.1 * (b_lo + b_up) / 2,
            b_up - 1.9 * (b_lo + b_up) / 2
        ]).min()
    # print('delta', delta)  # fordebug

    if problem_design.one_parameter:
        if problem_design.scheme == 1:
            t = b_lo, b_up
        elif problem_design.scheme == 2:
            t = b_up, b_lo

        x = np.pad(
            np.tile(t, (swarm_size, particle_size // 2)),
            ((0, 0), (0, particle_size % 2)),
            'reflect'
        )
        x[:, -1] = (b_up - b_lo) * rng.random(swarm_size) + b_lo
    else:
        t = b_lo, b_up - delta
        x = np.pad(
            np.tile([t, t[::-1]], (swarm_size // 2, particle_size // 2)),
            ((0, swarm_size % 2), (0, particle_size % 2)),
            'reflect'
        ) + delta * rng.random((swarm_size, particle_size))  # tocheck even/odd reflect type
    # print(x)  # fordebug

    p = x.copy()
    Fp = F(p)

    imin = Fp.argmin()
    g = p[imin].copy()
    Fg = Fp[imin]

    # v_min, v_max = -b_diff, b_diff  # tocheck убрать??? fordebug
    # v_min, v_max = -b_diff/1000, b_diff/1000
    # v = (v_max - v_min) * rng.random((swarm_size, particle_size)) + v_min  # у алекса zeros
    v = np.zeros((swarm_size, particle_size))
    # v = 2 * b_diff * rng.random((swarm_size, particle_size)) - b_diff  # init???
    # v beyond boundaries???

    for i in range(iter_num):
        rp, rg = rng.random(2)

        v = w * v + c1 * rp * (p - x) + c2 * rg * (g - x)  # g-vector minus x-matrix works correctly
        # v[v < v_min] = v_min  # fordebug
        # v[v > v_max] = v_max  # fordebug

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
        inverse_problem_parameters,
        eps, iter_num, swarm_size, b_lo, b_up, w, c1, c2, rng, problem_design
):
    solve_direct_problem, calculate_functionals, calculate_problem_functional = \
        get_solver_and_functional(inverse_problem_parameters)

    g = PSO(
        calculate_problem_functional,
        eps, iter_num, swarm_size, inverse_problem_parameters.shell_size, b_lo, b_up, w, c1, c2, rng, problem_design
    )

    A, B = solve_direct_problem(g)
    return InverseProblemSolution(g, calculate_functionals(A[0], B[0]))


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


def get_multi_layer_problems_solutions(problem_design, a_b_pairs, b_lo_b_up_pairs, w, c1, c2, rng):
    # not ternary because without slashes
    if problem_design.one_parameter:
        root = pathlib.Path('1-parameter', f'{problem_design.scheme} scheme')
    else:
        root = pathlib.Path('multi-parameter')

    print('BEGIN TIME = ', time.asctime())

    # for problem in 'shielding', 'external cloaking', 'full cloaking':
    for problem in 'full cloaking',:
        print(problem)

        for a, b in a_b_pairs:
            print(a, b)

            for b_lo, b_up in b_lo_b_up_pairs:
                print(b_lo, b_up)

                d = (root /
                     f'{problem}' /
                     # f'Ea={ipp.Ea} ei={ipp.ei} ee={ipp.ee} RMp1={ipp.RMp1}' /
                     f'a={a} b={b}' /
                     f'b_lo={b_lo} b_up={b_up}')
                d.mkdir(parents=True, exist_ok=True)

                for M in range(2, 18, 2):
                    ipp, ips_min = InverseProblemParameters(1, 1, 1, a, b, M, 0.1, problem), None

                    for i in range(20):
                        ips = solve_inverse_problem(
                            ipp, 0, 200, 40, b_lo, b_up, w, c1, c2, rng, problem_design
                        )

                        if ips_min is None or \
                                ips.functionals[ipp.problem] < ips_min.functionals[ipp.problem] or \
                                (np.array(tuple(ips.functionals.values())) <=
                                 np.array(tuple(ips_min.functionals.values()))).all():
                            ips_min = ips

                    print(ipp)
                    print(ips_min)
                    print('-' * 80)

                    (d / f'M={M}.pickle').write_bytes(pickle.dumps(InverseProblem(ipp, ips_min)))

                print('=' * 100)
                print('TIME = ', time.asctime())

    print('END TIME = ', time.asctime())


def get_individual_problem_solution(w, c1, c2, rng):
    J_list = []
    solution_list = []
    for i in range(20):
        print(i)
        solution_list.append(
            solve_inverse_problem(
                # 1, 1, 1, 0.03, 0.05, 8, 0.1, 'full cloaking', 0, 100, 40, 0.01, 86, w, c1, c2, rng
                1, 1, 1, 5, 5.5, 54, 6.5, 'full cloaking', 0, 100, 40, 0.001, 200, w, c1, c2, rng, True
            )
        )
        print(solution_list[-1])
        print('{:.15f}'.format(solution_list[-1].optimum_shell[1]))
        J_list.append(solution_list[-1].functionals['full cloaking'])

    imin = np.argmin(np.array(J_list))
    print('imin=', imin)

    # with open(fr'updated 2 layer full cloaking\fc M=2 a=0.01 b=0.05 RMp1=0.07 emin=1e-10 imin={imin}.pickle', 'wb') as f:
    #     pickle.dump(solution_list, f)


def J_e_sqrt_part(x):
    return np.sqrt(np.log(x / 0.05) / (x ** 4 - 0.05 ** 4))


def show_2_layer_full_cloaking_RMp1_dependence_plot():
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

        pp.scatter(x_list[-1], y_list[0] / y_list[-1], s=100, marker=marker, label=f'$R_{{M+1}}={RMp1}$')

        asymptote = J_e_sqrt_part(0.07) / J_e_sqrt_part(RMp1)
        pp.hlines(asymptote, x_list[-1].min(), x_list[-1].max())

    pp.xlabel(r'$\lg \varepsilon_{min}$', fontsize=30)
    pp.xticks(fontsize=30)

    pp.ylabel('$f(x,y)$', fontsize=30)
    pp.yticks(fontsize=30)

    pp.legend(fontsize=30)

    pp.show()


def show_J_e_sqrt_part_plot():
    x = np.array([0.07, 0.1, 0.15, 0.3, 0.6])

    pp.plot(x, J_e_sqrt_part(x), label='$y(x)$')

    pp.xlabel('x', fontsize=30)
    pp.xticks(fontsize=30)

    pp.ylabel('y', fontsize=30)
    pp.yticks(fontsize=30)

    pp.legend(fontsize=30)

    pp.show()


# def show_multi_layer_triple_plots():
#     ips_dict_list = []
#     RMp1_tuple = 0.07, 0.1, 0.15, 0.3, 0.6
#     functional_names = 'J_i', 'J_e', 'J'
#     # colors = 'red', 'orange', 'yellow', 'green', 'blue'  # tocheck
#     linestyles = '-', '--', ':'  # Ji Je J
#
#     for RMp1 in RMp1_tuple:
#         with open(
#                 'updated 2 layer full cloaking\\'
#                 f'updated full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle',
#                 'rb'
#         ) as f:
#             ips_dict_list.append(pickle.load(f))
#
#     for ips_dict, RMp1, color in zip(ips_dict_list, RMp1_tuple, colors):
#         x = np.array([ipp.shell_size for ipp in ips_dict.keys()])
#         y_array = np.array([[*ips.functionals.values()] for ips in ips_dict.values()]).T  # Ji Je J
#         for y, linestyle, functional_name in zip(y_array, linestyles, functional_names):
#             pp.plot(
#                 x, y,
#                 linestyle=linestyle, color=color,
#                 label=f'${functional_name}; R_{{M+1}}={RMp1}$'
#             )
#         # print(x)  # fordebug
#
#     pp.xlabel('$M$', fontsize=20)
#     pp.xticks(fontsize=20)
#
#     pp.ylabel(f"${';'.join(functional_names)}$", fontsize=20)
#     pp.yticks(fontsize=20)
#
#     pp.legend(fontsize=20)
#
#     pp.show()


def show_multi_layer_plots():
    functional_names = 'J_i', 'J_e', 'J'
    colors = 'red', 'orange', 'yellow', 'green', 'blue', 'black'
    linestyles = '-', '--', ':'  # Ji Je J

    # for problem in 'shielding', 'external cloaking', 'full cloaking':
    for problem in 'full cloaking',:
        colors_it = iter(colors)

        pp.figure()

        for a in 0.01, 0.03, 0.04:
            for b_up in 40, 70:
                with open(
                        'multi parameter a=0.01,0.03,0.04 b=0.05 b_lo=0.0045 b_up=40,70\\'
                        f'{problem} a={a} b=0.05 b_up={b_up}.pickle',
                        'rb'
                ) as f:
                    # with open(
                    #         '1 parameter\\'
                    #         f'1 parameter {problem} a={a} b=0.05 b_up={b_up}.pickle',
                    #         'rb'
                    # ) as f:
                    ips_dict = pickle.load(f)

                x = np.array([ipp.shell_size for ipp in ips_dict.keys()])
                y_array = np.array([[*ips.functionals.values()] for ips in ips_dict.values()]).T  # Ji Je J

                # HARDCODE
                # y_array = np.array([[*ips.functionals.values()] for ips in ips_dict.values()])  # Ji Je J
                #
                # if a == 0.01 and b_up == 70:
                #     y_array[-1] = 6.041e-16, 9.249e-17, 3.483e-16
                # elif a == 0.03:
                #     if b_up == 40:
                #         y_array[-1] = 7.044e-8, 2.349e-17, 3.522e-8
                #     elif b_up == 70:
                #         y_array[-1] = 2.717e-9, 2.117e-16, 1.358e-9
                # elif a == 0.04 and b_up == 70:
                #     y_array[-2] = 7.105e-6, 2.927e-16, 3.522e-6
                #     y_array[-1] = 6.141e-6, 2.259e-10, 3.070e-6
                #
                # y_array = y_array.T
                # HARDCODE

                color = next(colors_it)

                for y, linestyle, functional_name in zip(y_array, linestyles, functional_names):
                    pp.plot(
                        x, y,
                        linestyle=linestyle, color=color,
                        label=fr'${functional_name}; a={a}; \varepsilon_{{max}}={b_up}$'
                    )

        pp.title(problem)  # fordebug

        pp.xlabel('$M$', fontsize=20)
        pp.xticks(np.arange(2, 18, 2), fontsize=20)

        pp.ylabel('$' + r'\quad;\quad'.join(functional_names) + '$', fontsize=20)
        pp.yticks(fontsize=20)
        pp.yscale('log', basey=10)

        pp.legend(fontsize=20)

    pp.show()


def show_2_layer_full_cloaking_plot():
    ips_dict_list = []
    RMp1_tuple = 0.07, 0.1, 0.15, 0.3, 0.6
    functional_names = 'J_i', 'J_e', 'J'

    colors = 'red', 'orange', 'yellow', 'green', 'blue'
    linestyles = '-', '--', ':'  # Ji Je J

    for RMp1 in RMp1_tuple:
        with open(
                'updated 2 layer full cloaking\\'
                f'updated full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle',
                'rb'
        ) as f:
            ips_dict_list.append(pickle.load(f))

    for ips_dict, RMp1, color in zip(ips_dict_list, RMp1_tuple, colors):
        x = np.array([*ips_dict.keys()])  # sorting???
        # print(x)  # fordebug

        y_array = np.array([[*ips.functionals.values()] for ips in ips_dict.values()]).T  # Ji Je J
        for y, linestyle, functional_name in zip(y_array, linestyles, functional_names):
            # if functional_name=='J_e':continue
            pp.plot(
                x, y,
                linestyle=linestyle, color=color,
                label=f'${functional_name}; R_{{M+1}}={RMp1}$'
            )

    pp.xlabel(r'$\varepsilon_{min}$', fontsize=20)
    pp.xticks(fontsize=20)
    pp.xscale('log', basex=10)

    pp.ylabel('$' + r'\quad;\quad'.join(functional_names) + '$', fontsize=20)
    pp.yticks(fontsize=20)
    pp.yscale('log', basey=10)

    pp.legend(fontsize=20)

    pp.show()


if __name__ == '__main__':
    w, c1, c2 = 0.5, 1, 1.5
    rng = np.random.default_rng()  # tocheck x delta 0.01 8 0.01 86

    a_b_pairs = (0.03, 0.05),
    b_lo_b_up_pairs = (0.005, 8), (0.005, 30), (0.005, 150), (0.005, 200)

    get_multi_layer_problems_solutions(
        ProblemDesign(False, None),
        a_b_pairs,
        b_lo_b_up_pairs,
        w, c1, c2, rng
    )

    # get_multi_layer_problems_solutions(
    #     ProblemDesign(True, 1),
    #     a_b_pairs,
    #     b_lo_b_up_pairs,
    #     w, c1, c2, rng
    # )
    #
    # get_multi_layer_problems_solutions(
    #     ProblemDesign(True, 2),
    #     a_b_pairs,
    #     b_lo_b_up_pairs,
    #     w, c1, c2, rng
    # )

    # get_individual_problem_solution(w, c1, c2, rng)
