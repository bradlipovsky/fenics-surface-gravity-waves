'''
Fenics script to do a harmonic analysis of surface gravity waves.
'''
from dolfin import *
import dolfin
from numpy import sqrt,tanh,pi
from time import perf_counter

t0 = perf_counter()
lmbda,mu = 8e9,3e9
rho = 910
rhof = 1010

Wx = 1000
Hw = 500
Hi = 400
Hc = Hw - (rho/rhof) * Hi
Wx = 1000
xf = Wx/4

gravity = 9.8
k = 2*pi/100
A = 1
omega = sqrt(gravity*k*tanh(k*Hw))



print(f'The phase velocity is {omega/k} m/s')
print(f'The wave length is {2*pi/k} m')
print(f'The wave period is {2*pi/omega} s')



'''
Read mesh and mark domains/boundaries
'''
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
	infile.read(mesh)
print("Read mesh")

mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("facet_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
print("Read facets")


mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)

# Define the two domains
domains = MeshFunction("size_t", mesh, mesh.topology().dim())

# Use dS when integrating over the interior boundaries
# Use ds for the exterior boundaries,
# e.g.,  dS(1) for Γs and ds(0) for ∂B
dS = Measure('dS', domain=mesh, subdomain_data=mf,subdomain_id=3)
dX = Measure('dx', domain=mesh, subdomain_data=mf)
ds = Measure('ds', domain=mesh, subdomain_data=mf,subdomain_id=2)



'''
Variational problem
'''
V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Vv = VectorElement("Lagrange", mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, V*Vv)

TRF = TrialFunction(W)
TTF = TestFunction(W)
(p, u) = split(TRF)
(q, v) = split(TTF)

'''
=================
Marked Areas
==================
Water       dX(0)
Ice         dX(1)

=================
Marked Boundaries
=================
Water Surface, 2
Ice--Water Interface, 3
'''

# Define boundary conditions
u0 = Constant(0.0)
wavebc= Expression("A*omega/k*cosh(k*x[1])/sinh(k*H)",
		A=A,g=gravity,omega=omega,k=k,H=Hw,degree=2)

zero = Constant(0.0)
zero_2d = Constant((0.0, 0.0))

def left_boundary(x):
    return near(x[0],0.0) 
def right_ice_boundary(x):
    return near(x[0],Wx) and (x[1] > Hc)
def right_water_boundary(x):
    return near(x[0],Wx) and (x[1] < Hc) 

bcs = [ DirichletBC(W.sub(0), wavebc, left_boundary), 
	DirichletBC(W.sub(1), zero_2d, right_ice_boundary),
	DirichletBC(W.sub(0), zero, right_water_boundary) ]

''' 
Define variational problem
'''
sigma = 2.0*mu*sym(grad(u)) \
	+ lmbda*tr(sym(grad(u)))*Identity(u.geometric_dimension())
n = FacetNormal(mesh)

#Fluid domain
a_f = inner(grad(p), grad(q))*dX(0) - omega**2/gravity*p*q*ds
L_f = zero*q*ds

#Solid domain
a_s = (inner(sigma, grad(v)) - rho*omega**2*inner(u,v))*dX(1)
L_s = inner(zero_2d,v )*ds

#Interface fluid-solid
a_i = (rho*omega**2 * inner(n('+'), u('+')) * q('+')\
	 - omega*rhof*p('+')*inner(n('+'), v('+')))*dS
#a_i = (rho*omega**2 * inner(avg(n), u('+'))*avg(q) \
#	 -omega*rhof* p('+')*inner(avg(n), v('+')))*dS(3)
# L_i = zero*q('-')*dS(3)

#Weak form
a = a_f + a_s + a_i
L = L_f + L_s

''' 
Compute solution
'''
s = Function(W)
A=assemble(a, keep_diagonal=True)
b=assemble(L)

for bc in bcs: bc.apply(A, b)
A.ident_zeros()
s = Function(W)
print(f'Solve starting at t={perf_counter()-t0} s')
solve(A, s.vector(), b)
pp,uu = split(s)
print(f'Solve finished at t={perf_counter()-t0} s')

writevtk = True
if writevtk:
	file = File("fgw.pvd")
	file << s
else:
	# Plot solution
	import matplotlib.pyplot as plt
	fig,ax=plt.subplots(2,1,figsize=(10,10))
	
	plt.subplot(2,1,1)
	dispamp= project( uu[0]**2 + uu[1]**2, FunctionSpace(mesh, 'P', 1))
	c=plot(uu[1]*1e6)
	plt.colorbar(c,orientation="horizontal")
	
	plt.subplot(2,1,2)
	scale = 10
	cc=plot(pp,cmap='seismic',vmin=-scale,vmax=scale)
	plt.colorbar(cc,orientation="horizontal")
	plt.tight_layout()
	plt.savefig('figures/poisson.png')
print(f'All done at t={perf_counter()-t0} s')
