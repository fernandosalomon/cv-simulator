import math
import numpy as np
import matplotlib.pyplot as plt

#INDEPENDENT VARIABLES

C = 1.0 # [=] mol/L Initial concentration of O
D = 1E-5 # [=] cm^2/s O and R diffusion coefficient
etai = 0.2 # [=] V Initial overpotential
etaf = -0.2 # [=] V Final overpotential
v = 1E-3 # [=] V/s Sweep rate
n = 1.0 # [=] number of electron transfered
alpha = 0.5 # [=] dimensionless charge-transfer coefficient
k0 = 1E-2 # [=] cm/s electrochemical rate constant
kc = 1E-3 # [=] 1/s chemical rate constant
T = 298.15 # [=] K Temperature

#PHYSICAL CONSTANTS

F = 96485 # [=] C/mol Faraday's constant
R = 8.3145 # [=] J/mol.K Ideal gas constant
f = F/(R*T)

#SIMULATION VARIABLES
L = 500 # [=] number of iterations per t_k
DM = 0.45 # [=] model diffusion coefficient

#DERIVED CONSTANTS
tk = (2*((etai-etaf)/v)) # [=] s Characteristic exp. time
Dt = tk/L #[=] s delta time
Dx = math.sqrt((D*Dt)/DM) #[=] cm delta x
j = math.ceil(4.2*(L**0.5))+5 #number of boxes

#REVERSIBILITY PARAMETERS

ktk = kc*tk #dimensionless kinetic parameter
km = ktk/L #normalized dimensionless kinetic parameter
Lambda = k0/((D*f*v)**0.5) #dimensionless reversibility parameter

if(km > 0.1):
   print('k_c*t_k/l equals ', km, ', which exceeds the upper limit of 0.1')

C = C / 1000           # Convert C from mol/L to mol/cm3
k = np.arange((L/2)+1)                # time index vector
t = Dt* k             # time vector
eta1 = etai - (v*t)      # overpotential vector, negative scan
eta2 = etaf + (v*t)      # overpotential vector, positive scan
eta = np.concatenate((eta1, eta2)) # overpotential scan, both directions
Enorm = eta*f          # normalized overpotential
kf = k0*np.exp(-alpha*n*Enorm) # [=] cm/s, fwd rate constant (pg 799)
kb = k0*np.exp((1-alpha)*n*Enorm) # [=] cm/s, rev rate constant (pg 799)

O = C*np.ones((L+1,j))
R = np.zeros((L+1,j))
JO = np.zeros((L+1))

for i1 in range(1,L):
   for i2 in range(2, j-1):
      O[i1+1][i2] = O[i1][i2] + DM*(O[i1][i2+1]+O[i1][i2-1]-2*O[i1][i2])
      R[i1+1][i2] = R[i1][i2] + DM*(R[i1][i2+1]+R[i1][i2-1]-2*R[i1][i2]) - km*R[i1][i2]
    
   JO[i1+1] = ((kf[i1+1]*O[i1+1][2]) - kb[i1+1]*R[i1+1][2]) / (1 + (Dx/D*(kf[i1+1] + kb[i1+1])))
   
   O[i1+1][1] = O[i1+1][2] - JO[i1+1]*(Dx/D)
   R[i1+1][1] = R[i1+1][2] + JO[i1+1]*(Dx/D) - km*R[i1+1][1]

Z = -n*F*1000*JO

if(eta.size > Z.size):
   eta = eta[:eta.size-1]

plt.plot(eta, Z)
plt.show()