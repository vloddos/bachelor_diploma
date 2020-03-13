import numpy as np


def f(v):
	if v.ndim==1:
		return sum(i*i for i in v)
	elif v.ndim==2:
		return np.array([f(i) for i in v])


def PSO(swarm_size,shell_size,σmin,σmax,ε):
	σdiff=σmax-σmin
	x=np.random.rand(swarm_size,shell_size)*σdiff+σmin
	p=x.copy()
	g=p[f(p).argmin()].copy()
	v=np.random.rand(swarm_size,shell_size)*2*abs(σdiff)-abs(σdiff)

	i=0
	while f(g)>=eps and i<100:
		rp,rg=np.split(
			np.random.rand(2*swarm_size,shell_size),
			2
		)
		v=ω*v+φp*rp*(p-x)+φg*rg*(g-x)
		x+=v
		
		c=f(x)<f(p)
		p[c]=x[c].copy()
		
		fp=f(p)
		if fp.min()<f(g):
			g=p[fp.argmin()].copy()

		i+=1