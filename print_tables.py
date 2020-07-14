import pickle
import pathlib
import numpy as np
from pso_electric import InverseProblemParameters, InverseProblemSolution, InverseProblem, ProblemDesign


def frexp10(x):
    e = int(np.log10(x))
    m = x / 10 ** e
    if abs(m) < 1:
        m *= 10
        e -= 1

    return m, e


tr_d = {
    'shielding': 'задача экранирования',
    'external cloaking': 'задача внешней маскировки',
    'full cloaking': 'задача полной маскировки'
}


# def print_2_layer_tables():
#     for RMp1 in 0.07, 0.1, 0.15, 0.3, 0.6:
#         with open(
#                 fr'same solutions different Je 2 layer full cloaking\same solutions different Je full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle',
#                 'rb') as f:
#             # with open(fr'updated 2 layer full cloaking\updated full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle',
#             #           'rb') as f:
#             ips_dict_list.append(pickle.load(f))
#
#         print(f'%same solutions different Je 2 layer full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}')
#         # print(f'%different solutions different Je 2 layer full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}')
#         print(r'\begin{table}[H]')
#
#         caption = \
#             tr_d['full cloaking'] + \
#             ', '.join((
#                 fr'\( a = 0.01 \)',
#                 fr'\( b = 0.05 \)',
#                 r'\( M = 2 \)',
#                 fr'\( R_{{M+1}} = {RMp1} \)',
#                 r'\( \varepsilon_{min}^n = 10^{-n} \)',
#                 r'\( n = \overline{1,10} \)',
#                 r'\( \varepsilon_{max} = 3\)'
#             ))
#         print('\t' fr'\caption{{{caption}}}')  # todo label
#
#         print(r'\resizebox{\textwidth}{!}{%')  # ебашит пиздатый размер таблицы
#
#         print('\t' r'\begin{tabular}{ | c | c | c | c | c | c | c | }')
#         print('\t\t' r'\hline')
#
#         print(
#             '\t\t'
#             r'\(\varepsilon_{min}\) & \(\varepsilon_{max}\) & '
#             r'\(\varepsilon_1^{opt}\) & \(\varepsilon_2^{opt}\) & '
#             r'\(J_i(\mathbf{e}^{opt})\) & \(J_e(\mathbf{e}^{opt})\) & \(J(\mathbf{e}^{opt})\)'
#         )
#         print('\t\t' r'\\ \hline')
#
#         for emin, ips in ips_dict_list[-1].items():
#             print(
#                 '\t\t' +
#                 r'\( 10^{{{}}} \) & '.format(frexp10(emin)[1]) +
#                 r'\(3.0\) &' +
#                 # r'\( {:.1f} \times 10^{{{}}} \) & '.format(*frexp10(ips.optimum_shell[0])) +
#                 r'\( 10^{{{}}} \) & '.format(frexp10(ips.optimum_shell[0])[1]) +
#                 r'\( {:.16f} \) & '.format(*frexp10(ips.optimum_shell[1])) +
#                 ' & '.join(
#                     r'\(0.0\)' if i == 0 else r'\({:.3f} \times 10^{{{}}}\)'.format(*frexp10(i))
#                     for i in ips.functionals.values()
#                 )
#             )
#             print('\t\t' r'\\ \hline')
#             s.add(r'{:.16f}'.format(*frexp10(ips.optimum_shell[1])))  # fordebug
#
#         print('\t' r'\end{tabular}')
#
#         print('}')  # ебашит пиздатый размер таблицы
#
#         print(r'\end{table}')
#
#         print('%' + '=' * 100)


def print_tables(problem_design, a_b_pairs, b_lo_b_up_pairs):
    if problem_design.one_parameter:
        root = pathlib.Path('1-parameter', f'{problem_design.scheme} scheme')
        problem_design_comment_part = f'1-parameter {problem_design.scheme} scheme'
        problem_design_label_part = f'one_parameter_{problem_design.scheme}_scheme'
    else:
        root = pathlib.Path('multi-parameter')
        problem_design_comment_part = 'multi-parameter'
        problem_design_label_part = 'multi_parameter'

    for problem in 'shielding', 'external cloaking', 'full cloaking':
        for a, b in a_b_pairs:
            for b_lo, b_up in b_lo_b_up_pairs:
                ip_list = [
                    pickle.loads(
                        (root /
                         f'{problem}' /
                         # f'Ea={ipp.Ea} ei={ipp.ei} ee={ipp.ee} RMp1={ipp.RMp1}' /
                         f'a={a} b={b}' /
                         f'b_lo={b_lo} b_up={b_up}' /
                         f'M={M}.pickle').read_bytes()
                    )
                    for M in range(2, 18, 2)
                ]

                print(f'% {problem_design_comment_part} {problem} a={a} b={b} b_lo={b_lo} b_up={b_up}')
                print(r'\begin{table}[H]')

                if problem_design.one_parameter:
                    problem_design_caption_part = \
                        f'Однопараметрическая {tr_d[problem]}, ' \
                        f'{problem_design.scheme} схема квазичередующегося дизайна: '
                else:
                    problem_design_caption_part = f'Многопараметрическая {tr_d[problem]}: '

                caption = \
                    problem_design_caption_part + \
                    ', '.join((
                        fr'\( a = {a} \)',
                        fr'\( b = {b} \)',
                        fr'\( \varepsilon_{{min}} = {b_lo} \)',
                        fr'\( \varepsilon_{{max}} = {b_up} \)',
                        fr'\( \varepsilon_{{max}} / \varepsilon_{{min}} = {int(b_up / b_lo)} \)'
                    ))
                print('\t' fr'\caption{{{caption}}}')

                label = f'tab:{problem_design_label_part}_{problem}_a_{a}_b_{b}_b_lo_{b_lo}_b_up_{b_up}' \
                    .replace(' ', '_') \
                    .replace('.', '')
                print('\t' fr'\label{{{label}}}')

                print(r'\resizebox{\textwidth}{!}{%')  # ебашит пиздатый размер таблицы

                if problem_design.one_parameter:
                    print('\t' r'\begin{tabular}{ | c | c | c | c | c | c | }')
                    print('\t\t' r'\hline')

                    print(
                        '\t\t'
                        r'\(M\) & \(\varepsilon_1^{opt}\) & \(\varepsilon_M^{opt}\) & '
                        r'\(J_i(\mathbf{e}^{opt})\) & \(J_e(\mathbf{e}^{opt})\) & \(J(\mathbf{e}^{opt})\)'
                    )
                    print('\t\t' r'\\ \hline')

                    for ip in ip_list:
                        print(
                            '\t\t' +
                            r'{} & \({}\) & '.format(  # r'{} & \({:1.7}\) & '.format(
                                ip.parameters.shell_size,
                                ip.solution.optimum_shell[0]
                            ) +
                            r'\({:.16f} \times 10^{{{}}}\) & '.format(*frexp10(ip.solution.optimum_shell[-1])) +
                            ' & '.join(
                                r'\(0.0\)' if i == 0 else r'\({:.3f} \times 10^{{{}}}\)'.format(*frexp10(i))
                                for i in ip.solution.functionals.values()
                            )
                        )
                        print('\t\t' r'\\ \hline')
                else:
                    print('\t' r'\begin{tabular}{ | c | c | c | c | c | c | c | c | c | c | c | c | }')
                    print('\t\t' r'\hline')

                    print(
                        '\t\t'
                        r'\(M\) & '
                        r'\(\varepsilon_1^{opt}\) & \(\varepsilon_2^{opt}\) & \(\varepsilon_3^{opt}\) & '
                        r'\(\varepsilon_4^{opt}\) & \(\varepsilon_5^{opt}\) & \(\varepsilon_6^{opt}\) & '
                        r'\(\varepsilon_7^{opt}\) & \(\varepsilon_8^{opt}\) & '
                        r'\(J_i(\mathbf{e}^{opt})\) & \(J_e(\mathbf{e}^{opt})\) & \(J(\mathbf{e}^{opt})\)'
                    )
                    print('\t\t' r'\\ \hline')

                    print(
                        '\t\t & '
                        r'\(\varepsilon_9^{opt}\) & \(\varepsilon_{10}^{opt}\) & \(\varepsilon_{11}^{opt}\) & '
                        r'\(\varepsilon_{12}^{opt}\) & \(\varepsilon_{13}^{opt}\) & \(\varepsilon_{14}^{opt}\) & '
                        r'\(\varepsilon_{15}^{opt}\) & \(\varepsilon_{16}^{opt}\) & & &'
                    )
                    print('\t\t' r'\\ \hline')

                    for ip in ip_list:
                        slice1 = ip.solution.optimum_shell[:8]
                        print(
                            '\t\t' +
                            f'{ip.parameters.shell_size} & ' +
                            ' & '.join(
                                r'\({:.4f}\)'.format(i) if i > 0 else ''
                                for i in np.pad(slice1, (0, 8 - slice1.size), constant_values=-1)
                            ) + ' & ' +  # ???
                            ' & '.join(
                                r'\(0.0\)' if i == 0 else r'\({:.3f} \times 10^{{{}}}\)'.format(*frexp10(i))
                                for i in ip.solution.functionals.values()
                            )
                        )
                        print('\t\t' r'\\ \hline')

                        if ip.parameters.shell_size > 8:
                            slice2 = ip.solution.optimum_shell[8:]
                            print(
                                '\t\t & ' +
                                ' & '.join(
                                    (r'\({:.4f}\)'.format(i) if i > 0 else '')
                                    for i in np.pad(slice2, (0, 8 - slice2.size), constant_values=-1)
                                ) + ' & ' +  # ???
                                ' & & '
                            )
                            print('\t\t' r'\\ \hline')

                print('\t' r'\end{tabular}')

                print('}')  # ебашит пиздатый размер таблицы

                print(r'\end{table}')

                print('%' + '=' * 100)


if __name__ == '__main__':
    a_b_pairs = (0.03, 0.05),
    b_lo_b_up_pairs = (0.005, 8), (0.005, 30), (0.005, 150), (0.005, 200)

    print_tables(ProblemDesign(False, None), a_b_pairs, b_lo_b_up_pairs)
    print_tables(ProblemDesign(True, 1), a_b_pairs, b_lo_b_up_pairs)
    print_tables(ProblemDesign(True, 2), a_b_pairs, b_lo_b_up_pairs)
