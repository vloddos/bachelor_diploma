import sympy as sp

Ea = sp.symbols('Ea')


def init_vars(M):
    global A, B, R, ε
    A = sp.symbols(f'A0:{M + 1}')
    B = sp.symbols(f'B0:{M + 1}')
    R = sp.symbols(f'R0:{M + 1}')
    ε = sp.symbols(f'εi ε1:{M + 1} εe')


def init_SLAE(M):
    global SLAE
    SLAE = \
        [
            sp.Eq(R[0] ** 2 * (A[0] - A[1]) - B[1], 0),
            sp.Eq(R[0] ** 2 * (ε[0] * A[0] - ε[1] * A[1]) + ε[1] * B[1], 0)
        ] + \
        [sp.Eq(R[k] ** 2 * (A[k] - A[k + 1]) + B[k] - B[k + 1], 0) for k in range(1, M)] + \
        [
            sp.Eq(R[k] ** 2 * (ε[k] * A[k] - ε[k + 1] * A[k + 1]) - ε[k] * B[k] + ε[k + 1] * B[k + 1], 0)
            for k in range(1, M)
        ] + \
        [
            sp.Eq(R[M] ** 2 * A[M] + B[M] - B[0], -Ea * R[M] ** 2),
            sp.Eq(ε[M] * R[M] ** 2 * A[M] - ε[M] * B[M] + ε[M + 1] * B[0], -ε[M + 1] * Ea * R[M] ** 2)
        ]


M = 3
init_vars(M)
init_SLAE(M)
print(A, B, R, ε)
solution = sp.solve(SLAE, *A, *B)
# print(solution)
for k, v in solution.items():
    print(k)
    print(sp.expand(v))
    # print(v)
    print('-' * 80)
