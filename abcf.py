# with open(
#         (f'1-parameter\\{problem_design.scheme} scheme\\'
#         if problem_design.one_parameter
#         else 'multi-parameter\\') +
#         f'''
#
#         # ''',
#         'w'
# ) as f:
#     f.write('4398weghioidsgerlkfd')
import re

a, b, c, d = 1, '34groig', 883874, False
s = f'''
    98234uwrgvfd {a} {b}
    9weigsvfjirgpoo {c}
    {d}
'''
print(s)
print('-' * 50)
print(re.sub(r'\n(\s+)', '', s))

