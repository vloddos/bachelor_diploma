from collections import namedtuple
import pickle
import numpy as np


def frexp10(x):
    e = int(np.log10(x))
    m = x / 10 ** e
    if abs(m) < 1:
        m *= 10
        e -= 1

    return m, e


InverseProblemParameters = namedtuple(
    'InverseProblemParameters',
    ('Ea', 'σi', 'σe', 'a', 'b', 'shell_size', 'RMp1', 'problem')
)
InverseProblemSolution = namedtuple(
    'InverseProblemSolution',
    ('optimum_shell', 'functionals')
)

tr_d = {
    'shielding': 'Задача экранирования: ',
    'external cloaking': 'Задача внешней маскировки: ',
    'full cloaking': 'Задача полной маскировки: '
}
ips_dict_list = []
# M=2
with open(f'full cloaking m=2.pickle', 'rb') as f:
    ips_dict_list.append(pickle.load(f))

print(f'% full cloaking m=2')
print(r'\begin{table}[H]')
print('\t' r'\begin{tabular}{ | c | c | c | c | c | }')
print('\t\t' r'\hline')
print(
    '\t\t'
    r'\(\varepsilon_1^{opt}\) & \(\varepsilon_2^{opt}\) & \(J_i(\mathbf{e}^{opt})\) & \(J_e(\mathbf{e}^{opt})\) & \(J(\mathbf{e}^{opt})\)'
)
print('\t\t' r'\\ \hline')

for emin, ips in ips_dict_list[-1].items():
    print(
        '\t\t'
        r'\({}\) & '.format(ips.optimum_shell[0]) +
        r'\({} \times 10^{{{}}}\) & '.format(*frexp10(ips.optimum_shell[-1])) +
        ' & '.join(
            r'\(0.0\)' if i == 0 else r'\({:1.4} \times 10^{{{}}}\)'.format(*frexp10(i))
            for i in ips.functionals.values()
        )
    )
    print('\t\t' r'\\ \hline')

print('\t' r'\end{tabular}')
print(r'\end{table}')

exit()
# M=2,4,6,...,16
for problem in 'shielding', 'external cloaking', 'full cloaking':
    for a, b in (0.01, 0.05), (0.03, 0.05), (0.04, 0.05):
        for b_up in 40, 70:
            with open(f'{problem} a={a} b={b} b_up={b_up}.pickle', 'rb') as f:
                ips_dict_list.append(pickle.load(f))

            continue

            print(f'% {problem} a={a} b={b} b_up={b_up}')
            # print(r'\begin{center}')
            print(r'\begin{table}[H]')
            caption = \
                tr_d[problem] + \
                ', '.join((
                    fr'\( a = {a} \)',
                    fr'\( b = {b} \)',
                    r'\( \varepsilon_{min} = 0.0045 \)',
                    fr'\( \varepsilon_{{max}} = {b_up} \)',
                    fr'\( \varepsilon_{{max}} / \varepsilon_{{min}} \approx {int(np.ceil(b_up / 0.0045))} \)'
                ))
            print('\t' fr'\caption{{{caption}}}')  # todo label
            print('\t' r'\begin{tabular}{ | c | c | c | c | c | c | }')
            print('\t\t' r'\hline')
            print(
                '\t\t'
                r'\(M\) & \(\varepsilon_1^{opt}\) & \(\varepsilon_M^{opt}\) & \(J_i(\mathbf{e}^{opt})\) & \(J_e(\mathbf{e}^{opt})\) & \(J(\mathbf{e}^{opt})\)'
            )
            print('\t\t' r'\\ \hline')

            for ipp, ips in ips_dict_list[-1].items():
                print(
                    '\t\t'
                    r'{} & \({}\) & '.format(
                        ipp.shell_size,
                        ips.optimum_shell[0]
                    ) +
                    r'\({:1.7} \times 10^{{{}}}\) & '.format(*frexp10(ips.optimum_shell[-1])) +
                    ' & '.join(
                        r'\(0.0\)' if i == 0 else r'\({:1.4} \times 10^{{{}}}\)'.format(*frexp10(i))
                        for i in ips.functionals.values()
                    )
                )
                print('\t\t' r'\\ \hline')

            print('\t' r'\end{tabular}')
            # print(r'\end{center}')
            print(r'\end{table}')

            print('%' + '=' * 100)

# exit()

for ips_dict in ips_dict_list:
    for ipp, ips in ips_dict.items():
        print(ipp)
        print(ips)
        print('-' * 80)

    print('=' * 100)
