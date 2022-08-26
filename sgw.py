# Fenics script to do a harmonic analysis of surface gravity waves.

from dolfin import *
from numpy import sqrt,tanh,pi

gravity = 9.8
k = 2*pi/100
H = 100
A = 1
omega = sqrt(gravity*k*tanh(k*H))
print(f'The phase velocity is {omega/k} m/s')
print(f'The wave length is {2*pi/k} m')
print(f'The wave period is {2*pi/omega} s')

# Create mesh and define function space
#mesh = UnitSquareMesh(32, 32)
W,Nx,Ny = 500,320,32
mesh = RectangleMesh(Point(0., 0.), Point(W, H), Nx, Ny)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def left_boundary(x):
    return near(x[0],0.0) 

def right_boundary(x):
    return near(x[0],W) 

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)

bottom = Bottom()
top = Top()

# Initialize mesh function for boundary domains
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1,0)
boundaries.set_all(0)
bottom.mark(boundaries, 1)
top.mark(boundaries, 2)
ds = Measure('ds',subdomain_data=boundaries)

# Define boundary conditions
u0 = Constant(0.0)
wavebc= Expression("A*omega/k*cosh(k*x[1])/sinh(k*H)",
		A=A,g=gravity,omega=omega,k=k,H=H,degree=2)

bc = [ DirichletBC(V, wavebc, left_boundary), 
	DirichletBC(V, Constant(0.0), right_boundary) ]
zero = Constant(0.0)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = inner(grad(u), grad(v))*dx - omega**2/gravity*u*v*ds(2)
L = f*v*dx + zero*v*ds(1)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,4))
c=plot(u,cmap='seismic',vmin=-50,vmax=50)
plt.colorbar(c,orientation="horizontal")
plt.savefig('figures/poisson.png')
