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

    # print('optimum shell:', g)
    # A, B = solve_direct_problem(g)
    # print('; '.join(f"{k} : {'{:1.6e}'.format(v)}" for k, v in calculate_functionals(A[0], B[0]).items()))

    # print(calculate_functionals(*solve_direct_problem(np.full(shell_size, b_lo))))
    # print(calculate_functionals(*solve_direct_problem(np.full(shell_size, b_up))))

    # A, B = solve_direct_problem(np.array(shell_size // 2 * [b_lo, b_up]))
    # A, B = solve_direct_problem(np.array((shell_size // 2 - 1) * [b_lo, b_up] + [b_lo, 17.947036677753093]))
    # A, B = solve_direct_problem(np.array((shell_size // 2 - 1) * [b_up, b_lo] + [b_up, 20.04]))
    # print('; '.join(f"{k} : {'{:1.6e}'.format(v)}" for k, v in calculate_functionals(A[0], B[0]).items()))

    # print(calculate_functionals(A[0], B[0]))
    # print(calculate_functionals(*solve_direct_problem(np.array((shell_size // 2 - 1) * [b_lo, b_up] + [b_lo, 57.93]))))
    # print(calculate_functionals(*solve_direct_problem(np.array((shell_size // 2 - 1) * [b_up, b_lo] + [b_up, 20.04]))))


# ω, φp, φg = 0.7298, 1.49618, 1.49618
'''
# for ω, φp, φg in ((0.7298, 1.49618, 1.49618), (0.5, 1, 1.5)):
#     print(f'ω={ω}, φp={φp}, φg={φg}')
for i in [6]:
    print(i)
    μmin = ε = 10 ** -i
    solve_inverse_problem(
            # 1488, 1, 1, 0.04, 0.05, 16, 0.1, 'full cloaking', 1e-7, 100, 160, 0.0045, 70, 0.7928, 1.49618, 1.49618
            # 1488, 1, 1, 0.04, 0.05, 10, 0.1, 'shielding', 1e-7, 100, 160, 0.0045, 40, 0.7928, 1.49618, 1.49618
            # 1000, 1, 1, 0.01, 0.05, 2, 0.1, 'full cloaking', ε, 100, 20, μmin, 10, 0.7928, 1.49618, 1.49618
        1000, 1, 1, 0.01, 0.05, 2, 0.1, 'full cloaking', ε, 100, 20, μmin, 10, ω, φp, φg
    )
    # print('=' * 80)
# '''

# '''
# for ω, φp, φg in ((0.7298, 1.49618, 1.49618), (0.5, 1, 1.5)):
#     print(f'ω={ω}, φp={φp}, φg={φg}')
ω, φp, φg = 0.5, 1, 1.5
print('TIME = ', time.asctime())
for a, b in (0.01, 0.05), (0.03, 0.05), (0.04, 0.05):
    for b_up in 40, 70:
        print('a b b_up', a, b, b_up)
        inverse_problem_solution_dict = {}
        for i in range(1, 9):
            print(i)
            ipp = InverseProblemParameters(1, 1, 1, a, b, i * 2, 0.1, 'full cloaking')
            ips = solve_inverse_problem(
                # 1, 1, 1, 0.01, 0.05, M, 0.1, 'full cloaking', 1e-10, 100, 20 * i, 0.0045, 70, ω, φp, φg
                *ipp, 0, 100, 2000 * i, 0.0045, b_up, ω, φp, φg
            )
            if ipp not in inverse_problem_solution_dict or \
                    ips.functionals[ipp.problem] < inverse_problem_solution_dict[ipp].functionals[ipp.problem]:
                inverse_problem_solution_dict[ipp] = ips

        with open(f'full cloaking a={a} b={b} b_up={b_up}.pickle', 'wb') as f:
            pickle.dump(inverse_problem_solution_dict, f)

        for k, v in inverse_problem_solution_dict.items():
            print(k)
            print(v)
            print('-' * 80)

        print('=' * 100)
        print('TIME = ', time.asctime())
# print(inverse_problem_solution_dict)
'''
{InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=2, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045, 1.    ]), functionals={'shielding': 0.020024320897405565, 'external cloaking': 0.15304655972869177, 'full cloaking': 0.08653544031304866}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=4, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045, 1.    , 1.    , 1.    ]), functionals={'shielding': 0.0236448484648455, 'external cloaking': 0.06776939024658171, 'full cloaking': 0.0457071193557136}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=6, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([1.    , 0.0045, 1.    , 1.    , 1.    , 1.    ]), functionals={'shielding': 0.03575701865275852, 'external cloaking': 0.09109736636425671, 'full cloaking': 0.06342719250850762}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=8, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045, 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ]), functionals={'shielding': 0.03165854887080865, 'external cloaking': 0.037807399433046694, 'full cloaking': 0.03473297415192767}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=10, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045    , 1.        , 1.        , 1.        , 0.88164457,
       1.        , 1.        , 1.        , 1.        , 1.        ]), functionals={'shielding': 0.03588063625932177, 'external cloaking': 0.03530616251291336, 'full cloaking': 0.035593399386117566}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=12, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045, 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    ]), functionals={'shielding': 0.039860809651242966, 'external cloaking': 0.029619480338332643, 'full cloaking': 0.03474014499478781}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=14, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045, 0.0045, 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    ]), functionals={'shielding': 0.02961997571161948, 'external cloaking': 0.04158118891586229, 'full cloaking': 0.035600582313740886}), InverseProblemParameters(Ea=1, σi=1, σe=1, a=0.01, b=0.05, shell_size=16, RMp1=0.1, problem='full cloaking'): InverseProblemSolution(optimum_shell=array([0.0045    , 0.45231201, 0.79812088, 0.0045    , 0.0045    ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        ]), functionals={'shielding': 0.006133974884359569, 'external cloaking': 0.08415744088590815, 'full cloaking': 0.04514570788513386})}
'''
'''
{'shielding': 0.014787955284087412, 'external cloaking': 1.2660982569758628e-16, 'full cloaking': 0.007393977642043769}
{'shielding': 2.6830224542942666e-05, 'external cloaking': 0.0, 'full cloaking': 1.3415112271471333e-05}
{'shielding': 1.7583433492344072e-07, 'external cloaking': 0.0, 'full cloaking': 8.791716746172036e-08}
{'shielding': 2.5091804195164257e-09, 'external cloaking': 0.0, 'full cloaking': 1.2545902097582128e-09}
{'shielding': 6.22988392123863e-11, 'external cloaking': 1.0049775799438554e-11, 'full cloaking': 3.6174307505912427e-11}
{'shielding': 2.3695144682396352e-12, 'external cloaking': 2.8874881321642302e-11, 'full cloaking': 1.562219789494097e-11}
{'shielding': 1.2712640761473922e-13, 'external cloaking': 8.222643847487434e-12, 'full cloaking': 4.174885127551087e-12}
{'shielding': 9.077846995021184e-15, 'external cloaking': 1.3688862571471389e-10, 'full cloaking': 6.844885178085445e-11}

2 2.1157242550480544
4 4.485990791379854
6 6.849022163547021
8 9.167404941959008
10 11.436047904310172
12 13.654641613210709
14 15.824358186060602
16 17.947036677753093
'''
