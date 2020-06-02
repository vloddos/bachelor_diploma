import sympy as sp

Ea = sp.symbols('Ea')


def init_vars(M):
    global A, B, R, ε
    A = sp.symbols(f'A0:{M + 1}')
    B = sp.symbols(f'B3 B1:{M + 1}')
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


M = 2
init_vars(M)
init_SLAE(M)
print(A, B, R, ε)

SLAE_l, SLAE_r = sp.linear_eq_to_matrix(SLAE, *A, *B)
det = SLAE_l.det()

tmp = SLAE_l.copy()
tmp[:, 0] = SLAE_r
A0det = tmp.det()

tmp = SLAE_l.copy()
tmp[:, 3] = SLAE_r
B3det = tmp.det()

B30ε2 = (B3det / det).xreplace({ε[1]: 0}).simplify()

for i in det, A0det, B3det, B30ε2:
    print(
        sp.latex(
            i,
            mode='equation',
            symbol_names={
                Ea: 'E_a',
                **dict(
                    zip(
                        ε,
                        (r'\varepsilon_i', r'\varepsilon_1', r'\varepsilon_2', r'\varepsilon_e')
                    )
                )
            }
        )
    )

print(sp.solve(sp.Eq(B30ε2, 0), ε[2]))
