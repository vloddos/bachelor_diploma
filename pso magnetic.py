import numpy as np
from scipy.sparse import diags

'''
узнать у алекса как дохуя частиц должно быть при количестве слоев m
так как если там получится дохуя невъебенно большая матрица
и решать ее каким то хитровыебанным аналитическим способом не выйдет
и придется юзать какие то числаки
то нужно узнать кароч че и как чтобы знать можно ли их распараллелить 
'''

'''
todo random vector generator by web with n^m vectors
m layers, n points on segment [σmin,σmax], n^m vectors in total
take k random unique vectors if n^m>k
else take all the n^m vectors
'''
def Ji(A0,Ha):
    return A0/Ha


def Je(BMp1,Ha,RM,RMp1):
    return 2*BMp1/Ha*np.sqrt(np.log(RMp1/RM)/(RMp1**4-RM**4))


def J(A0,BMp1,Ha,RM,RMp1):
    return (Ji(A0, Ha)+Je(BMp1, Ha, RM, RMp1))/2


def get_solver_and_functional(Ha,μi,μe,a,b,shell_size,RMp1,problem):
    assert problem in {'shielding','external cloaking','full cloaking'}
    R=np.hstack((np.linspace(a,b,shell_size+1),RMp1))                          #len = M+2


    def solve_direct_problem(μ):                                               #BM+1==B0!!!!!!!!!!!!!!!!!!!
        μ=np.hstack((μi,μ,μe))                                                 #len = M+2
        #todo try to rename c,d to a,b
        c=np.hstack(#todo try use for loop to initialize c
            np.vstack(                                                         #A
                diags((R[:-1]**2,-R[:-2]**2),(0,1)).toarray(),                 #A R
                diags((μ[:-1]*R[:-1]**2,-μ[1:-1]*R[:-2]**2),(0,1)).toarray()   #A μR
            ),
            np.roll(                                                           #B cyclically shifted on 1 column to right
                np.vstack(                                                     #B
                    diags((-1,1),(0,-1),(shell_size+1,shell_size+1)).toarray(),#B -1 1
                    diags((μ[1:],-μ[1:-1]),(0,-1)).toarray()                   #B μ
                ),
                1,
                1
            )
        )
        d=np.hstack((
            np.zeros(shell_size),-Ha*R[-2]**2,
            np.zeros(shell_size),-μ[-1]*Ha*R[-2]**2
        ))
        return np.hsplit(np.linalg.solve(c,d),2)


    def calculate_functional_value(μ):                                         #(μi, μ, μe) len = M+2
        A,B=solve_direct_problem(μ)
        if problem=='shielding':
            return Ji(A[0],Ha)
        elif problem=='external cloaking':
            return Je(B[0],Ha,*R[-2:])
        elif problem=='full cloaking':
            return J(A[0],B[0],Ha,*R[-2:])


    return solve_direct_problem,calculate_functional_value
        
#check all copy/assignment operations
#todo random function with boundaries
#todo type annotations
#todo try to use python 3.8 with := anywhere
#todo try to use coconut python
#todo rename args to independent from task names
def PSO(f,ε,iter_num,swarm_size,shell_size,b_lo,b_up):#iter_num excess???
	def F(v):
		if v.ndim==1:
			return f(v)
		elif v.ndim==2:
			#return np.array([f(i) for i in v])
			return np.apply_along_axis(f,1,v)


	b_diff=b_up-b_lo
	x=np.random.rand(swarm_size,shell_size)*b_diff+b_lo
	p=x.copy()
	g=p[F(p).argmin()].copy()
	v=np.random.rand(swarm_size,shell_size)*2*abs(b_diff)-abs(b_diff)

	for _ in range(iter_num):
		rp,rg=np.split(                                                        #like vsplit by default
			np.random.rand(2*swarm_size,shell_size),
			2
		)
		v=ω*v+φp*rp*(p-x)+φg*rg*(g-x)                                          #g-vector minus x-matrix works correctly
		x+=v
		
		c=F(x)<F(p)
		p[c]=x[c].copy()                                                       #when c.any()==False works correctly
		
		#todo try reduce code
		fp=F(p)
		if (fp.min()<F(g)).any():
			g=p[fp.argmin()].copy()

		if F(g)<ε:
			break

	return g