# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:45:07 2026

@author: stanh
"""
"Assumptions made for the problem "
"-uniform mesh  hx=hy=h "
"-Diffusion coefficient k cst "
"- V vector is diagonal to the cartesian plane :u=v "
"Density of the fluid rho is cst"
"Note that this code uses vertex centered FVM because we already know the boundary temperatures as they were given as Dirichlet boundary conditions"
"This python code uses First order upwind differencing scheme (UDS1)  for the FVM "
import numpy as np 
import matplotlib.pyplot as plt


#parameters
L=1
k=5.0 #thermal conductivity
rho=1.25 #density of fluid
#velocity compo.
u=30.0
v=30.0

#Number of nodes in x = Number of nodes in y = N 
N=21 
#grid spacing 
h=L/(N-1)
D=k/h
Fx=rho*u
Fy=rho*v

#Grid peclet number , dont calculate for D = 0 
Pe_x = Fx/D
Pe_y = Fy/D
print(f"Pe in x = {Pe_x} , Pe in y= {Pe_y}")
#iteration Number 
iteration =0 
#initialize temperature 
T=np.zeros((N,N))
T[:,0]=4
T[0,:]=1
#initialise iterated temperature 
T_new=T.copy()

#error variables:
epsilon = 1.E-8
numerical_error=1
#Iterative process , error verfication 
while numerical_error > epsilon :
    for i in range(1,N-1):
        for j in range(1,N-1):
            a_E = D + max(0,-Fx)
            a_W = D + max(0,Fx)
            a_N = D + max(0,-Fy)
            a_S = D + max(0,Fy)
            a_P = a_E + a_W + a_N + a_S
            T_new[i,j] = (a_E*T[i,j+1] + a_W*T[i,j-1] + a_N*T[i-1,j] + a_S*T[i+1,j])/a_P
            
    #resetting the numerical error and recalculate
    numerical_error=0
    for i in range(1,N-1):
         for j in range(1,N-1):
             numerical_error=numerical_error+abs(T[i,j]-T_new[i,j])  
    iteration =iteration + 1
    T=T_new.copy()
print(f'Temperature distribution obtained after {iteration} iterations  ')     
# --- Grid Generation ---
x_dom = np.arange(N)*h
y_dom = L - np.arange(N)*h #very important otherwise domain will be inverted
[X, Y]= np.meshgrid(x_dom, y_dom)
# --- Plot temperature field ---       
plt.figure(figsize=(8, 4))

contour = plt.contourf(X, Y, T, levels=45)
plt.colorbar(orientation= 'vertical' , label="Temperature")


#diagonal line to check for false diffusion
diag = np.linspace(0, L, 1000)
plt.plot(
    diag, diag,
    color='black',
    linewidth=1.0,
    linestyle='-',
    zorder=100
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Temperature distribution 2D")
plt.show()     
            

    

 