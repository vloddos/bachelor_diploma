import pickle
import numpy as np
from pso_electric import InverseProblemParameters, InverseProblemSolution


def frexp10(x):
    e = int(np.log10(x))
    m = x / 10 ** e
    if abs(m) < 1:
        m *= 10
        e -= 1

    return m, e


tr_d = {
    'shielding': 'Задача экранирования: ',
    'external cloaking': 'Задача внешней маскировки: ',
    'full cloaking': 'Задача полной маскировки: '
}
ips_dict_list = []


def print_2_layer_tables():
    # for RMp1 in 0.07, 0.1, 0.15, 0.3, 0.6:
    for RMp1 in 0.07, 0.15, 0.3, 0.6:
        # with open(fr'2 layer full cloaking\full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle', 'rb') as f:
        with open(fr'SAME full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}.pickle', 'rb') as f:
            ips_dict_list.append(pickle.load(f))

        print(f'%same solutions different Je 2 layer full cloaking M=2 a=0.01 b=0.05 RMp1={RMp1}')
        print(r'\begin{table}[H]')

        caption = \
            tr_d['full cloaking'] + \
            ', '.join((
                fr'\( a = 0.01 \)',
                fr'\( b = 0.05 \)',
                r'\( M = 2 \)',
                fr'\( R_{{M+1}} = {RMp1} \)',
                r'\( \varepsilon_{min}^n = 10^{-n} \)',
                r'\( n = \overline{1,10} \)',
                r'\( \varepsilon_{max} = 10\)'
            ))
        print('\t' fr'\caption{{{caption}}}')  # todo label

        print(r'\resizebox{\textwidth}{!}{%')  # ебашит пиздатый размер таблицы

        print('\t' r'\begin{tabular}{ | c | c | c | c | c | c | c | }')
        print('\t\t' r'\hline')

        print(
            '\t\t'
            r'\(\varepsilon_{min}\) & \(\varepsilon_{max}\) & '
            r'\(\varepsilon_1^{opt}\) & \(\varepsilon_2^{opt}\) & '
            r'\(J_i(\mathbf{e}^{opt})\) & \(J_e(\mathbf{e}^{opt})\) & \(J(\mathbf{e}^{opt})\)'
        )
        print('\t\t' r'\\ \hline')

        for emin, ips in ips_dict_list[-1].items():
            print(
                '\t\t' +
                r'\( 10^{{{}}} \) & '.format(frexp10(emin)[1]) +
                r'\(10.0\) &' +
                # r'\( {:.1f} \times 10^{{{}}} \) & '.format(*frexp10(ips.optimum_shell[0])) +
                r'\( 10^{{{}}} \) & '.format(frexp10(ips.optimum_shell[0])[1]) +
                r'\( {:.15f} \) & '.format(*frexp10(ips.optimum_shell[1])) +
                ' & '.join(
                    r'\(0.0\)' if i == 0 else r'\({:.3f} \times 10^{{{}}}\)'.format(*frexp10(i))
                    for i in ips.functionals.values()
                )
            )
            print('\t\t' r'\\ \hline')

        print('\t' r'\end{tabular}')

        print('}')  # ебашит пиздатый размер таблицы

        print(r'\end{table}')

        print('%' + '=' * 100)


def print_1_parameter_tables():
    # for problem in 'shielding', 'external cloaking', 'full cloaking':
    for problem in 'external cloaking', 'full cloaking':
        for a, b in (0.01, 0.05), (0.03, 0.05), (0.04, 0.05):
            for b_up in 40, 70:
                with open(f'{problem} a={a} b={b} b_up={b_up}.pickle', 'rb') as f:
                    ips_dict_list.append(pickle.load(f))

                print(f'% {problem} a={a} b={b} b_up={b_up}')
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
                        r'{} & \({:1.7}\) & '.format(
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
                print(r'\end{table}')

                print('%' + '=' * 100)


def print_results():
    for ips_dict in ips_dict_list:
        for ipp, ips in ips_dict.items():
            print(ipp)
            print(ips)
            print('-' * 80)

        print('=' * 100)


def print_multi_parameter_tables():
    # for problem in 'external cloaking', 'full cloaking':
    for problem in ('shielding',):
        # for a, b in (0.01, 0.05), (0.03, 0.05), (0.04, 0.05):
        for a, b in ((0.01, 0.05),):
            for b_up in 40, 70:
                # with open(f'{problem} a={a} b={b} b_up={b_up}.pickle', 'rb') as f:
                with open(fr'1 parameter\1 parameter {problem} a={a} b={b} b_up={b_up}.pickle', 'rb') as f:
                    ips_dict_list.append(pickle.load(f))

                print(f'% multi parameter {problem} a={a} b={b} b_up={b_up}')
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

                print(r'\resizebox{\textwidth}{!}{%')  # ебашит пиздатый размер таблицы

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

                for ipp, ips in ips_dict_list[-1].items():
                    slice1 = ips.optimum_shell[:8]
                    print(
                        '\t\t' +
                        f'{ipp.shell_size} & ' +
                        ' & '.join(
                            r'\({:.4f}\)'.format(i) if i > 0 else ''
                            for i in np.pad(slice1, (0, 8 - slice1.size), constant_values=-1)
                        ) + ' & ' +  # ???
                        ' & '.join(
                            r'\(0.0\)' if i == 0 else r'\({:.3f} \times 10^{{{}}}\)'.format(*frexp10(i))
                            for i in ips.functionals.values()
                        )
                    )
                    print('\t\t' r'\\ \hline')

                    if ipp.shell_size > 8:
                        slice2 = ips.optimum_shell[8:]
                        print(
                            '\t\t & ' +
                            # f'{ipp.shell_size} &'
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
    print_2_layer_tables()
