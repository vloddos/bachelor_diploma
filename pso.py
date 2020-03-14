import numpy as np

'''
узнать у алекса как дохуя частиц должно быть при количестве слоев m
так как если там получится дохуя невъебенно большая матрица
и решать ее каким то хитровыебанным аналитическим способом не выйдет
и придется юзать какие то числаки
то нужно узнать кароч че и как чтобы знать можно ли их распараллелить 
'''

#todo check all copy/assignment operations
#todo random function with boundaries
#todo type annotations
#todo try to use python 3.8 with := anywhere
def PSO(f,swarm_size,shell_size,σmin,σmax,ε):
	def F(v):
		if v.ndim==1:
			return f(v)
		elif v.ndim==2:
			return np.array([f(i) for i in v])


	σdiff=σmax-σmin
	x=np.random.rand(swarm_size,shell_size)*σdiff+σmin
	p=x.copy()
	g=p[F(p).argmin()].copy()
	v=np.random.rand(swarm_size,shell_size)*2*abs(σdiff)-abs(σdiff)

	i=0
	while F(g)>=ε and i<100:
		rp,rg=np.split(#like vsplit by default?
			np.random.rand(2*swarm_size,shell_size),
			2
		)
		v=ω*v+φp*rp*(p-x)+φg*rg*(g-x)#g vector minus x matrix works correctly
		x+=v
		
		c=F(x)<F(p)
		p[c]=x[c].copy()#когда все false???(вроде работает)
		
		#todo try reduce code
		fp=F(p)
		if fp.min()<F(g):#когда все false???
			g=p[fp.argmin()].copy()

		i+=1

	return g