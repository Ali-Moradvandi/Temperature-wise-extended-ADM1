# Thermal-Integrated Anaerobic Digestion Model
"""
Written by Sjoerd Heegstra
Edited by Ali Moradvandi
For any correspondence: a.moradvandi@tudelft.nl
"""
#----- Required packages -----# 
import numpy as np
import scipy.integrate
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import math
import warnings

# Ignore DtypeWarnings from pandas' read_csv   
warnings.filterwarnings('ignore', message="^Columns.*")
# Setting timer
start_time = time.time()
#----- -----#

#####################################################################################################
##----- SECTION ONE -----##
## Determine model specification ##

## Equation framework
# answer = input("Do you want to use the ODE (Rosen and Jeppson, 2007), 
#                                        DAE (Rosen and Jeppson, 2007),
#                                        pure-ODE (Thamsiriroj and Murphy, 2010)?
#                                        Please type 'ODE', 'DAE' or 'pure-ODE'. ")


## ADMn1 Version: 
# 0 = original ADM1
# 1 = R1, 2 = R2, and 3 = R3

## gascontrol: gas control can be "steady" or "variable" strategy.

## solvermethod: Setting the solver method for the simulation.

## Temperature inhibition function:
# choose either 'Cardinal' or 'Arrhenius'

answer       = "ODE"
version      = 0
gascontrol   = 'variable'
solvermethod = 'Radau'
tempmodel    = 'Cardinal'

#####################################################################################################
##----- SECTION TWO -----##
## ADMn1 parameters 
# given from the Rosen et al (2006) BSM2 report
# parameters can be adjusted according to the case study.

R            =  0.083145  #bar.M^-1.K^-1        # Universal Gas Constant
T_base       =  298.15    #K                    
p_atm        =  1.013     #bar
T_op_initial =  300.15    #K                    # Initial operational temperature 
                                                # reference temperature for Van 't Hoff equation
                                                # has influence on physico-chemical proces parameters.


# Stoichiometric parameter
# Sum of fractionation must always be 1
f_sI_xc =  0.1     # fraction of soluble inert material in  composite material (X_xc)
f_xI_xc =  0.2     # fraction of particulate inert material in particulate composite material (X_xc)
f_ch_xc =  0.2     # fraction of carbohydrates in particulate composite material (X_xc)
f_pr_xc =  0.2     # fraction of proteins in particulate composite material (X_xc)
f_li_xc =  0.3     # fraction of lipids in particulate composite material (X_xc)

N_xc =  0.0376/14   #kmole N.kg^-1COD     # Nitrogen content of particulate composite material
N_I  =  0.06/14     #kmole N.kg^-1COD     # Nitrogen content of inert material 
N_aa =  0.007       #kmole N.kg^-1COD     # Nitrogen content of amino acids

C_xc =  0.02786     #kmole C.kg^-1COD     # Carbon content of composite material
C_sI =  0.03        #kmole C.kg^-1COD     # Carbon content of  soluble inert material
C_ch =  0.0313      #kmole C.kg^-1COD     # Carbon content of carhohydrates material
C_pr =  0.03        #kmole C.kg^-1COD     # Carbon content of proteins material
C_li =  0.022       #kmole C.kg^-1COD     # Carbon content of lipids material
C_xI =  0.03        #kmole C.kg^-1COD     # Carbon content of particulate inert material
C_su =  0.0313      #kmole C.kg^-1COD     # Carbon content of monosaccharides material
C_aa =  0.03        #kmole C.kg^-1COD     # Carbon content of amino acids

# f_product_substrate = yield (catabolism only) of product on substrate in kgCOD/kgCOD
f_fa_li  =  0.95     #kgCOD/kgCOD         # Yield of LCFA on lipids, based on palmitic 
                                          # triglyceride as lipid and palmitate as LCFA
C_fa     =  0.0217   #kmole C.kg^-1COD    # Carbon content of LCFA
f_h2_su  =  0.19     #kgCOD/kgCOD         # Yield of hydrogen gas on monosaccharides
f_bu_su  =  0.13     #kgCOD/kgCOD         # Yield of butyrate on monosaccharides
f_pro_su =  0.27     #kgCOD/kgCOD         # Yield of propionate on monosaccharides
f_ac_su  =  0.41     #kgCOD/kgCOD         # Yield of acetate on monosaccharides
N_bac    =  0.08/14  #kmole N.kg^-1COD    # Nitrogen content of bacteria
C_bu     =  0.025    #kmole C.kg^-1COD    # Carbon content of butyrate
C_pro    =  0.0268   #kmole C.kg^-1COD    # Carbon content of propionate
C_ac     =  0.0313   #kmole C.kg^-1COD    # Carbon content of aceta
C_bac    =  0.0313   #kmole C.kg^-1COD    # Carbon content of bacteria
Y_su     =  0.1                           # Yield of biomass on monosaccharides
f_h2_aa  =  0.06                          # Yield of hydrogen on amino acids
f_va_aa  =  0.23                          # Yield of valerate on amino acids
f_bu_aa  =  0.26                          # Yield of butyrate on amino acids
f_pro_aa =  0.05                          # Yield of propionate on amino acids
f_ac_aa  =  0.40                          # Yield of acetate on amino acids
C_va     =  0.024    #kmole C.kg^-1COD    # Carbon content of valerate
Y_aa     =  0.08                          # Yield of biomass on amino acids
Y_fa     =  0.06                          # Yield of biomass on LCFA
Y_c4     =  0.06                          # Yield of biomass on other C4
Y_pro    =  0.04                          # Yield of biomass on propionate
C_ch4    =  0.0156   #kmole C.kg^-1COD    # Carbon content of methane
Y_ac     =  0.05                          # Yield of biomass on acetate
Y_h2     =  0.06                          # Yield of biomass on hydrogen gas


# Biochemical parameter
k_dis        =  0.5            #d^-1                              # first order rate constant disintegration
opt_k_hyd_ch =  10             #d^-1                              # first order rate constant hydrolysis of carbohydrates
opt_k_hyd_pr =  10             #d^-1                              # first order rate constant hydrolysis of proteins
opt_k_hyd_li =  10             #d^-1                              # first order rate constant hydrolysis of lipids
K_S_IN       =  10 ** -4       #M #kgCOD_S/ m^3                  # Half inhibitory concentration
opt_k_m_su   =  30             #d^-1                              # Monod maximum specific uptake rate (mu_max/Y)
K_S_su       =  0.5            #kgCOD.m^-3                        # Half saturation value of monosaccharides
pH_UL_aa     =  5.5            # pH inhibition factor (upper level) acetogenesis/acidogenesis
pH_LL_aa     =  4              # pH inhibition factor (upper level) acetogenesis/acidogenesis
opt_k_m_aa   =  50             #kgCOD_S*kgCOD_X^-1 d^-1           # Monod maximum specific uptake rate amino acids (mumax/Y)
K_S_aa       =  0.3            #kgCOD.m^-3                        # Half saturation value Amino Acids
opt_k_m_fa   =  6              #kgCOD_S*kgCOD_X^-1 d^-1           #Monod maximum specific uptake rate LCFA (mumax/Y)
K_S_fa       =  0.4            #kgCOD.m^-3                        # Half saturation value LCFA
K_I_h2_fa    =  5*10**-6       #kgCOD.m^-3                        # Half inhibitory concentration LCFA
opt_k_m_c4   =  20             #kgCOD_S*kgCOD_X^-1 d^-1           # Monod maximum specific uptake rate C4 fatty acids (mumax/Y)
K_S_c4       =  0.2            #kgCOD.m^-3                        # Half saturation value C4
K_I_h2_c4    =  10**-5         #kgCOD.m^-3                        # Half inhibitory concentration C4
opt_k_m_pro  =  13             #kgCOD_S*kgCOD_X^-1 d^-1           # Monod maximum specific uptake rate propionate (mumax/Y)
K_S_pro      =  0.1            #kgCOD.m^-3                        # Half saturation value propionate
K_I_h2_pro   =  3.5*10**-6     #kgCOD.m^-3                        # inhibitory concentration
opt_k_m_ac   =  8              #kgCOD_S*kgCOD_X^-1 d^-1           #Monod maximum specific uptake rate acetate (mumax/Y)
K_S_ac       =  0.15           #kgCOD.m^-3                        # Half saturation value acetate
K_I_nh3      =  0.0018         #M                                 # Half inhibitory concentration Ammonium
pH_UL_ac     =  7              # pH inhibition factor (upper level) aceticlastic methanogenesis 
pH_LL_ac     =  6              # pH inhibition factor (lower level) aceticlastic methanogenesis
opt_k_m_h2   =  35             #d^-1 #kgCOD_S*kgCOD_X^-1 d^-1     # Monod maximum specific uptake rate hydrogen gas (mumax/Y)
K_S_h2       =  7*10**-6       #kgCOD.m^-3                        # Half saturation value Hydrogen gas
pH_UL_h2     =  6              # pH inhibition factor (upper level) hydrogenotrophic methanogenesis
pH_LL_h2     =  5              # pH inhibition factor (lower level) hydrogenotrophic methanogenesis
k_dec_X_su   =  0.02           #d^-1                              # First order decay rate monosaccharides
k_dec_X_aa   =  0.02           #d^-1                              # First order decay rate Amino acids
k_dec_X_fa   =  0.02           #d^-1                              # First order decay rate LCFA
k_dec_X_c4   =  0.02           #d^-1                              # First order decay rate C4
k_dec_X_pro  =  0.02           #d^-1                              # First order decay rate propionate
k_dec_X_ac   =  0.02           #d^-1                              # First order decay rate acetate
k_dec_X_h2   =  0.02           #d^-1                              # First order decay rate hydrogen
## M is kmole m^-3

# Acid-Base equilibrium coefficients (pKa)
# K_w is the ionic product of water, H2O/[H+]+[OH-]
K_a_va  =  10**-4.86     #M  ADM1 value = 1.38 * 10 ^ -5 ## n-HVa/Va-
K_a_bu  =  10**-4.82     #M  #1.5*10^-5 Hbu/Bu-
K_a_pro =  10**-4.88     #M  #1.32*10^-5 HPr/Pr-
K_a_ac  =  10**-4.76     #M  #1.74*10^-5  HAc/Ac-

# K_a_va_temp  = K_a_va  * np.exp((280 / (100 * R)) * (1 / T_base - 1 / T_ad))
# K_a_bu_temp  = K_a_bu  * np.exp((280 / (100 * R)) * (1 / T_base - 1 / T_ad))
# K_a_pro_temp = K_a_pro * np.exp((75  / (100 * R)) * (1 / T_base - 1 / T_ad))
# K_a_ac_temp  = K_a_ac  * np.exp((41  / (100 * R)) * (1 / T_base - 1 / T_ad))

k_A_B_va  =  10**10     #M^-1*d^-1     # Acid base kinetic rate coefficient
k_A_B_bu  =  10**10     #M^-1*d^-1
k_A_B_pro =  10**10     #M^-1*d^-1
k_A_B_ac  =  10**10     #M^-1*d^-1
k_A_B_co2 =  10**10     #M^-1*d^-1     # solubility of CO2 goes up with temperature
k_A_B_IN  =  10**10     #M^-1*d^-1


k_p        = 5*10**4  #m^3.d^-1.bar^-1 # only for BSM2 AD conditions, recalibrate for other AD cases #gas outlet friction
base_k_L_a = 200.0    #d^-1            # gas-liquid transfer coefficient at T_base temperature. 
# k_L_a_temp = 0.56 * T_ad + 27.9 

#Hill inhibition functions based on the hydrogen ion concentration
K_pH_aa =  (10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0))    
nn_aa   =  (3.0 / (pH_UL_aa - pH_LL_aa))                 
K_pH_ac =  (10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0))    
n_ac    =  (3.0 / (pH_UL_ac - pH_LL_ac))
K_pH_h2 =  (10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0))
n_h2    =  (3.0 / (pH_UL_h2 - pH_LL_h2))

#Same for carbon calculation, see Rosen and Jeppson for formulas
# Carbon content of composite material
s_1  =  (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI) 
# Carbon fraction of carbohydrates
s_2  =  (-1 * C_ch + C_su) 
# Carbon fraction of proteins
s_3  =  (-1 * C_pr + C_aa) 
# Carbon fraction of Lipids
s_4  =  (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa) 
# Carbon fraction of sugars, in  monosaccharides, lose sugars or in bacteria
s_5  =  (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)
# Carbon fraction of amino acids in biomass
s_6  =  (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)
# Carbon fraction of LCFA 
s_7  =  (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac) 
# carbon fraction from valerate
s_8  =  (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)
# carbon fraction from butyrate
s_9  =  (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac) 
# carbon fraction from propionate
s_10 =  (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)
# Carbon fraction fromm acetate
s_11 =  (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)
# Methane production and hydrogenotrophich methanogens yield
s_12 =  ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)
# Carbon content of biomass and composite material, biomass growth decreases s_13, biomass decay increases it
s_13 =  (-1 * C_bac + C_xc) 

#####################################################################################################
##----- SECTION THREE -----##
## Heat network paramaeters

T_soil_sub  = 283.15 #[K]     # Soil temperature at the bottom of the digester


Pr    = 0.7                                      # Prandtl number
sigma = 5.67037*10**-8*86400  # [W*m-2*K^-4]     # Stefan-Boltzmann contant   

Rho_air = 1.205               # [kg*m^-3]        # Density of air 
Rho_sub = 1000                # [kg*m^-3]        # Density of substrate 
mu_air  = 1.82*10**-5         # [Pa*s]           # Dynamic viscosity of air

#Thermal conductivity of plain concrete walls 300 mm thick with air space plus brick facing, from metcalf and eddy p. 1526
labda_digester_cover     = 5*86400       # [W*m^-1*K^-1]
#Thermal conductivity of plain concrete walls surrounded by moist earth, from metcalf and eddy p. 1526
labda_digester_dry_walls = 0.6*86400     # [W*m^-1*K^-1] 
#Thermal conductivity of plain concrete floor surrounded by moist earth, from metcalf and eddy p. 1526
labda_digester_wet_walls = 1.2*86400     # [W*m^-1*K^-1]

#Thermal conductivity of substrate
#labda_sub               = 0.605*86400   # [W*m^-1*K^-1]  
#Thermal conductivity of biogas
#labda_biogas            = 0.0206*86400  # [W*m^-1*K^-1]  

# Thermal conductivity of air
labda_air                = 0.026*86400   # [W*m^-1*K^-1]  

Ieta_cover               = 0.75          # [-]     #Absorptivity   

epsilon_cover = 0.75                     # [-]     #Emmissivity of the cover
epsilon_substrate = 0.67                 # [-]     #Emmissivity of the substrate

C_p_sub = 4.179*10**3                    # [J*kg^-3*K^-1]     #Specific heat capacity of the substrate

# Convective heat transfer coefficients
h_cov_gas   = 2.15*86400                 # [W*m^-2*K^-1]  
h_gas_wall  = 2.70*86400                 # [W*m^-2*K^-1]  
h_gas_sub   = 2.20*86400                 # [W*m^-2*K^-1]  
h_sub_wall  = 177.25*86400               # [W*m^-2*K^-1]  
h_sub_floor = 244.15*86400               # [W*m^-2*K^-1]  

# Temperature shock constants
tau       = 15                           # days
s_hg      = 3.5                          # Celcius
sigma_DYN = (-s_hg**2/(2*np.log(0.5)))**0.5

#####################################################################################################
##----- SECTION FOUR -----##
## Operational parameter 

V_liq    = 4300/100                             #[m^3]      # Volume for liquid fraction of the reactor
V_gas    = 300/100                              #[m^3]      # Reactor volume for gas
V_ad     = V_liq + V_gas                        #[m^3]      # Reactor volume
r_out    = (3 * V_ad / (4 * math.pi))**(1/3)    #[m]        # Reactor outside radius
A        = math.pi * (r_out ** 2)               #[m^2]      # Reacor surface
Delta_X  = 3                                    #[m]        # Digester wall thickness 
r_in     = r_out - Delta_X                      #[m]        # Reactor inside radius
q_ad     = 200/100                              #[m^3*d^-1] # Influent flow rate   
T_feed   = 308.15                               #[K]        # Temperature of influent

#####################################################################################################
##----- SECTION FIVE -----##
## Importing weather condition data from csv files
# Load meteorological data and define it in separate arrays for later use
# IMPORATNT: Reference address should be corrected based on the folder that includes data

grounddata1 = (pd.read_csv(r'C:#####SHOULD BE FILLED OUT BY THE USER#####.csv'\
                           , sep=';', skiprows =16)).to_numpy()
airdata1    = (pd.read_csv(r'C:#####SHOULD BE FILLED OUT BY THE USER#####.csv'\
                           , sep=';', skiprows =52)).to_numpy()

# begin on 01-01-2021
# solar irradiation
Q_solar = np.delete(airdata1[:,20], slice(0,43830))*10000                                           # [J/m^2/d]
# mean temperature at 1m dept
T_soil_gas = np.delete(np.delete(grounddata1[:,7], slice(0,58398)),slice(528,1830)) / 10 + 273.15   #[K]
# mean windspeed of a day
u_wind = np.delete(airdata1[:,4], slice(0,43830))/10                                                #[m/s]
# mean air temperature of a day
T_air = np.delete(airdata1[:,11], slice(0,43830))/10  + 273.15                                      #[K] 

## begin on 01-01-2000
# Q_solar = np.delete(airdata1[:,20], slice(0,36159))*10000                                         # [J/m^2/d]
## mean temperature at 1m dept
# T_soil_gas = np.delete(grounddata1[:,7], slice(0,50727))/10 + 273.15                              # [K] 
## mean windspeed of a day
# u_wind = np.delete(airdata1[:,4], slice(0,36159))/10                                              # [m/s] 
## mean air temperature of a day
# T_air = np.delete(airdata1[:,11], slice(0,36159))/10 + 273.15                                     # [K] 

#####################################################################################################
##----- SECTION SIX -----##
## Influent and initial parameters 
# Load influent and initial parameters for later use
# IMPORATNT: Reference address should be corrected based on the folder that includes data or the associated excel files should be corrected.


"Influent and initial concentrations calculated from labwork"
influent_state = pd.read_csv(r'C:#####SHOULD BE FILLED OUT BY THE USER#####\initials_influents_and_weatherdata\continuousW+S.csv', sep=';')
influent_state = influent_state.to_dict()
initial_state = pd.read_csv(r"C:#####SHOULD BE FILLED OUT BY THE USER#####\initials_influents_and_weatherdata\batchinitial.csv")
initial_state = initial_state.to_dict()

#####################################################################################################
##----- SECTION SEVEN -----##
## Heat transfer model
# Function to take new weather data for each new day
# IMPORATNT: this model can be adapted based on the designed heat network.

def setWeather(i): #
    global Q_solar_in, T_soil_gas_in, u_wind_in, T_air_in
    Q_solar_in    = Q_solar[i]
    T_soil_gas_in = T_soil_gas[i]
    u_wind_in     = u_wind[i]
    T_air_in      = T_air[i]

# Heat mass balance equation
def dTdt(t, T_state_zero):
    global Q_CON_Total, Q_ADV_feed_sub, Q_irr, Q_RAD_sky_sub, Q_CON_air_substrate, Q_CON_wall_gas, Q_CON_wall_sub, Q_CON_floor_sub
    T_op          = T_state_zero[0]
    T_adt         = T_state_zero[1]
    
    Q_solar       = weatherinput[0]          # incoming radiation, adjusted for timestep
    T_soil_gas    = weatherinput[1]
    u_wind        = weatherinput[2] 
    T_air         = weatherinput[3] 

    # Convective heat transfer
    Q_ADV_feed_sub = q_ad * C_p_sub * Rho_sub * (T_feed - T_op)
    
    # Solar Radiance
    # Q_solar      = Adding from statistical data
    Q_irr          = Q_solar * A * Ieta_cover
    
    # Convective heat transfer coefficient from air to the cover, dependent on windspeed.
    Reynolds_air     = Rho_air * u_wind * 2 * r_out / mu_air          # reynolds number of air flowing over the cover of the digester.
    Nusselt_air      = 0.037 * Reynolds_air ** (4/5) * Pr ** (1/3)    # Nusselt number of the air/cover interface
    h_cnv_air_cover  = Nusselt_air * labda_air / r_out                # [W * m^-2 * K^-1] forced convection coefficient
    
    # IMPORATNT: Resistance and  overall heat transfer coefficient should be corrected based on the heat network of case study (HERE is from an illustrative example.).
    # Resistance calculations for convection and conduction. For conduction, be critical on the thickness of the conducting layer.
    R_CNV_air_cover    = 1 / h_cnv_air_cover 
    R_CND_cover        = Delta_X / labda_digester_cover
    R_CNV_biogas_cover = 1 / h_cov_gas
    R_CNV_biogas_wall  = 1 / h_gas_wall
    R_CND_wall         = Delta_X / labda_digester_wet_walls    
    R_CNV_sub_biogas   = 1 / h_gas_sub
    R_CNV_sub_wall     = 1 / h_sub_wall
    
    R_CNV_floor_sub    = 1 / h_sub_floor
    R_CND_floor        = Delta_X / labda_digester_wet_walls
    
    # Overall heat transfer coefficient calculation
    U_air              = 1 / ((R_CNV_air_cover + R_CNV_biogas_cover + R_CNV_sub_biogas) + (R_CND_cover))    
    U_wall_biogas      = 1 / ((R_CNV_biogas_wall + R_CNV_sub_biogas) + (R_CND_wall))   
    U_wall_substrate   = 1 / (R_CNV_sub_wall + R_CND_wall)   
    U_floor_substrate  = 1 / (R_CNV_floor_sub + R_CND_floor)
   
    # Calculation of convective and conductive heat transfer
    Q_CON_air_substrate = A * U_air * (T_air - T_op)
    Q_CON_wall_gas      = 0.5 * 4 * A * U_wall_biogas * (T_soil_gas - T_op)           # half of the wall is filled with biogas, 4 walls have contact with soil 
    Q_CON_wall_sub      = 0.5 * 4 * A * U_wall_substrate * (T_soil_sub - T_op)        # half of the wall is filled with substrate, 4 walls have contact with soil  
    Q_CON_floor_sub     = A * U_floor_substrate * (T_soil_sub - T_op)  
    Q_CON_Total         = Q_CON_air_substrate + Q_CON_floor_sub + Q_CON_wall_sub + Q_CON_wall_gas
    
    # Radiative heat transfer
    T_atmosphere = 0.0552 * T_air ** (3/2)
    Q_RAD_sky_sub = sigma * (T_atmosphere ** 4 - T_op ** 4) / (2 / A + 2 * ((1 - epsilon_cover) / (A * epsilon_cover)) + (1 - epsilon_substrate) / (A * epsilon_substrate))
    
    # Energy balance
    dTdt = (Q_ADV_feed_sub + Q_irr + Q_CON_air_substrate + Q_CON_floor_sub + Q_CON_wall_gas + Q_CON_wall_sub + Q_RAD_sky_sub) / (Rho_sub * C_p_sub * V_liq)
    diff_T_adt = (T_op - T_adt) / tau
     
    return dTdt, diff_T_adt #, Q_RAD_sky_sub, Q_irr, Q_ADV_feed_sub

#####################################################################################################
##----- SECTION EIGHT -----##
## Calculate temperature dependents

def settempdependent(T_op):
    global p_gas_h2o, K_H_co2, K_H_ch4, K_H_h2, K_a_co2, K_a_IN, K_w, k_L_a
    
    K_w       =  10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_op))        #[M]           #2.08 * 10 ^ -14
    K_a_co2   =  10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_op))         #[M]           #4.94 * 10 ^ -7 CO2/HCO3
    K_a_IN    =  10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_op))        #[M]           #1.11 * 10 ^ -9 NH4+/NH3
    p_gas_h2o =  0.0313 * np.exp(5290 * (1 / T_base - 1 / T_op))                            #[bar]         #0.0557 water vapour pressure. The 0.0313 constant depends on Temperature (Van 't Hoff equation)
    K_H_co2   =  0.035 * np.exp((-19410 / (100 * R))* (1 / T_base - 1 / T_op))              #[Mliq.bar^-1] #0.0271 Henry's law coefficient for CO2, K_H decreases with increased solubility
    K_H_ch4   =  0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_op))            #[Mliq.bar^-1] #0.00116 Henry's law coefficient for methane
    K_H_h2    =  7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_op))       #[Mliq.bar^-1] #7.38*10^-4 Henry's law coefficient for hydrogen gas
    
    k_L_a     = base_k_L_a * f(T_base)/f(T_op) * (T_base / T_op) ** 5
    
    # k_L_a, see johnny lee article baseline mass-transfer coefficient
    temp_scale  = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60] 
    temp_scale  = [x + 273.15 for x in temp_scale]
    E_rho_sigma = [151.44, 153.55, 155.77, 157.88, 159.15, 159.36, 159.51, 157.45, 153.63, 148.40]   
    f = interp1d(temp_scale, E_rho_sigma, 2) #interpolate the data so any temperature point can be given

    #IMPORTANT: All data points are found by least square analysis: in case of new parameters, first take a look at that function.
    c_k_hyd_ch =  [8.2975, 37.3800, 45.0239]           #[d^-1]                     # first order rate constant hydrolysis of carbohydrates
    c_k_hyd_pr =  [8.2975, 37.3800, 45.0239]           #[d^-1]                     # first order rate constant hydrolysis of proteins
    c_k_hyd_li =  [8.2975, 37.3800, 45.0239]           #[d^-1]                     # first order rate constant hydrolysis of lipids
    c_k_m_su   =  [12.6022, 40.9377, 45.0078]          #[d^-1]                     # Monod maximum specific uptake rate (mu_max/Y)
    c_k_m_aa   =  [12.6022, 40.9377, 45.0078]          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate amino acids (mumax/Y)
    c_k_m_fa   =  [12.6022, 40.9377, 45.0078]          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate LCFA (mumax/Y)
    c_k_m_c4   =  [1.6970, 35.7785, 45.0316]           #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate C4 fatty acids (mumax/Y)
    c_k_m_pro  =  [2.3920, 34.9377, 45.0368]           #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate propionate (mumax/Y)
    c_k_m_ac   =  [8.1387, 37.7067, 45.022]            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate acetate (mumax/Y)
    c_k_m_h2   =  [8.1387, 37.7067, 45.022]            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate hydrogen gas (mumax/Y)

    a_k_hyd_ch =  [1, 0.116, 0.2742, 0.1977]           #[d^-1]                     # first order rate constant hydrolysis of carbohydrates, determined with least square analysis through coordinates found by bergland
    a_k_hyd_pr =  [1, 0.116, 0.2742, 0.1977]           #[d^-1]                     # first order rate constant hydrolysis of proteins
    a_k_hyd_li =  [1, 0.116, 0.2742, 0.1977]           #[d^-1]                     # first order rate constant hydrolysis of lipids
    a_k_m_su   =  [0.2931, 0.2433, 0.00000034476, 1]   #[d^-1]                     # Monod maximum specific uptake rate (mu_max/Y)
    a_k_m_aa   =  [0.2931, 0.2433, 0.00000034476, 1]   #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate amino acids (mumax/Y)
    a_k_m_fa   =  [0.2931, 0.2433, 0.00000034476, 1]   #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate LCFA (mumax/Y)
    a_k_m_c4   =  [1, 0.0812, 0.1728, 0.1981]          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate C4 fatty acids (mumax/Y)
    a_k_m_pro  =  [1, 0.0770, 0.1537, 0.2017]          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate propionate (mumax/Y)
    a_k_m_ac   =  [1, 0.1151, 0.2893, 0.1976]          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate acetate (mumax/Y)
    a_k_m_h2   =  [1, 0.1151, 0.2893, 0.1976]          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate hydrogen gas (mumax/Y)

def cardinal(p, q, T_op):
    T_min = q[0]
    T_opt = q[1]
    T_max = q[2]
    b_opt = p
    T_op = T_op - 273.15
    y = b_opt * (((T_op - T_max) * (T_op - T_min)**2) / ((T_opt - T_min) * ((T_opt - T_min) * (T_op - T_opt) - (T_opt - T_max) * (T_opt + T_min - 2 * T_op))))
    return y

def arrhenius(p, q, T_op):
    k_1 = q[0]
    a_1 = q[1]
    k_2 = q[2]
    a_2 = q[3]
    y = p * k_1 * np.exp(a_1 * (T_op - 303.15)) - k_2 * np.exp(a_2 * (T_op - 303.15))
    return y

def setgrowthconstantscardinal(T_op, F_dyn):
    global k_hyd_ch, k_hyd_pr, k_hyd_li, k_m_su, k_m_aa, k_m_fa, k_m_c4, k_m_pro, k_m_ac, k_m_h2
    k_hyd_ch =  cardinal(opt_k_hyd_ch, c_k_hyd_ch, T_op)        #[d^-1]                     # first order rate constant hydrolysis of carbohydrates
    k_hyd_pr =  cardinal(opt_k_hyd_pr, c_k_hyd_pr, T_op)        #[d^-1]                     # first order rate constant hydrolysis of proteins
    k_hyd_li =  cardinal(opt_k_hyd_li, c_k_hyd_li, T_op)        #[d^-1]                     # first order rate constant hydrolysis of lipids
    k_m_su   =  cardinal(opt_k_m_su, c_k_m_su, T_op)            #[d^-1]                     # Monod maximum specific uptake rate (mu_max/Y)
    k_m_aa   =  cardinal(opt_k_m_aa, c_k_m_aa, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate amino acids (mumax/Y)
    k_m_fa   =  cardinal(opt_k_m_fa, c_k_m_fa, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate LCFA (mumax/Y)
    k_m_c4   =  cardinal(opt_k_m_c4, c_k_m_c4, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate C4 fatty acids (mumax/Y)
    k_m_pro  =  cardinal(opt_k_m_pro, c_k_m_pro, T_op)          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate propionate (mumax/Y)
    k_m_ac   =  F_dyn * cardinal(opt_k_m_ac, c_k_m_ac, T_op)    #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate acetate (mumax/Y)
    k_m_h2   =  F_dyn * cardinal(opt_k_m_h2, c_k_m_h2, T_op)    #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate hydrogen gas (mumax/Y)


def setgrowthconstantsarrhenius(T_op):
    global k_hyd_ch, k_hyd_pr, k_hyd_li, k_m_su, k_m_aa, k_m_fa, k_m_c4, k_m_pro, k_m_ac, k_m_h2
    k_hyd_ch =  arrhenius(opt_k_hyd_ch, a_k_hyd_ch, T_op)        #[d^-1]                     # first order rate constant hydrolysis of carbohydrates
    k_hyd_pr =  arrhenius(opt_k_hyd_pr, a_k_hyd_pr, T_op)        #[d^-1]                     # first order rate constant hydrolysis of proteins
    k_hyd_li =  arrhenius(opt_k_hyd_li, a_k_hyd_li, T_op)        #[d^-1]                     # first order rate constant hydrolysis of lipids
    k_m_su   =  arrhenius(opt_k_m_su, a_k_m_su, T_op)            #[d^-1]                     # Monod maximum specific uptake rate (mu_max/Y)
    k_m_aa   =  arrhenius(opt_k_m_aa, a_k_m_aa, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate amino acids (mumax/Y)
    k_m_fa   =  arrhenius(opt_k_m_fa, a_k_m_fa, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate LCFA (mumax/Y)
    k_m_c4   =  arrhenius(opt_k_m_c4, a_k_m_c4, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate C4 fatty acids (mumax/Y)
    k_m_pro  =  arrhenius(opt_k_m_pro, a_k_m_pro, T_op)          #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate propionate (mumax/Y)
    k_m_ac   =  arrhenius(opt_k_m_ac, a_k_m_ac, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate acetate (mumax/Y)
    k_m_h2   =  arrhenius(opt_k_m_h2, a_k_m_h2, T_op)            #[kgCOD_S*kgCOD_X^-1 d^-1]  # Monod maximum specific uptake rate hydrogen gas (mumax/Y)


#####################################################################################################
##----- SECTION NINE -----##
## Setting initial and influent values
# Functions to set initial and influent values for influent state variables at each simulation step

def setInfluent(i):
    global S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in,S_ch4_in, S_IC_in, S_IN_in, S_I_in,X_xc_in, X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,S_cation_in,S_anion_in
    # variable definition
    # Input values (influent/feed) 
          # s_i are the formulas as presented in Rosen and Jeppson, S_i are the converted units for each simplification
    
    if version == 0:
        S_su_in  = influent_state['S_su'][i]        #[kg COD.m^-3]
        S_aa_in  = influent_state['S_aa'][i]        #[kg COD.m^-3]
        S_fa_in  = influent_state['S_fa'][i]        #[kg COD.m^-3]
        S_va_in  = influent_state['S_va'][i]        #[kg COD.m^-3]
        S_bu_in  = influent_state['S_bu'][i]        #[kg COD.m^-3]
        S_pro_in = influent_state['S_pro'][i]       #[kg COD.m^-3]
        S_ac_in  = influent_state['S_ac'][i]        #[kg COD.m^-3]
        S_h2_in  = influent_state['S_h2'][i]        #[kg COD.m^-3]
        S_ch4_in = influent_state['S_ch4'][i]       #[kg COD.m^-3]
        S_IC_in  = influent_state['S_IC'][i]        #[kmole C.m^-3]
        S_IN_in  = influent_state['S_IN'][i]        #[kmole N.m^-3]
        S_I_in   = influent_state['S_I'][i]         #[kg COD.m^-3]
        X_xc_in  = influent_state['X_xc'][i]        #[kg COD.m^-3]
        X_ch_in  = influent_state['X_ch'][i]        #[kg COD.m^-3]
        X_pr_in  = influent_state['X_pr'][i]        #[kg COD.m^-3]
        X_li_in  = influent_state['X_li'][i]        #[kg COD.m^-3]
        X_su_in  = influent_state['X_su'][i]        #[kg COD.m^-3]
        X_aa_in  = influent_state['X_aa'][i]        #[kg COD.m^-3]
        X_fa_in  = influent_state['X_fa'][i]        #[kg COD.m^-3]
        X_c4_in  = influent_state['X_c4'][i]        #[kg COD.m^-3]
        X_pro_in = influent_state['X_pro'][i]       #[kg COD.m^-3]
        X_ac_in  = influent_state['X_ac'][i]        #[kg COD.m^-3]
        X_h2_in  = influent_state['X_h2'][i]        #[kg COD.m^-3]
        X_I_in   = influent_state['X_I'][i]         #[kg COD.m^-3]
        S_cation_in = influent_state['S_cation'][i] #[kmole.m^-3]
        S_anion_in  = influent_state['S_anion'][i]  #[kmole.m^-3]
    
        
def setInitial(i):
    global S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2 
    # initiate variables (initial values for the reactor state at the initial time (t0)
      # s_i are the formulas as presented in Rosen and Jeppson, S_i are the converted units for each simplification

    if version == 0:
        S_su    = initial_state['S_su'][0]        #[kg COD.m^-3] 
        S_aa    = initial_state['S_aa'][0]        #[kg COD.m^-3]
        S_fa    = initial_state['S_fa'][0]        #[kg COD.m^-3] 
        S_va    = initial_state['S_va'][0]        #[kg COD.m^-3] 
        S_bu    = initial_state['S_bu'][0]        #[kg COD.m^-3] 
        S_pro   = initial_state['S_pro'][0]       #[kg COD.m^-3] 
        S_ac    = initial_state['S_ac'][0]        #[kg COD.m^-3] 
        S_h2    = initial_state['S_h2'][0]        #[kg COD.m^-3] 
        S_ch4   = initial_state['S_ch4'][0]       #[kg COD.m^-3] 
        S_IC    = initial_state['S_IC'][0]        #[kmole C.m^-3]
        S_IN    = initial_state['S_IN'][0]        #[kmole N.m^-3] 
        S_I     = initial_state['S_I'][0]         #[kg COD.m^-3] 
        
        X_xc    = initial_state['X_xc'][0]        #[kg COD.m^-3] 
        X_ch    = initial_state['X_ch'][0]        #[kg COD.m^-3] 
        X_pr    = initial_state['X_pr'][0]        #[kg COD.m^-3] 
        X_li    = initial_state['X_li'][0]        #[kg COD.m^-3] 
        X_su    = initial_state['X_su'][0]        #[kg COD.m^-3] 
        X_aa    = initial_state['X_aa'][0]        #[kg COD.m^-3] 
        X_fa    = initial_state['X_fa'][0]        #[kg COD.m^-3] 
        X_c4    = initial_state['X_c4'][0]        #[kg COD.m^-3] 
        X_pro   = initial_state['X_pro'][0]       #[kg COD.m^-3] 
        X_ac    = initial_state['X_ac'][0]        #[kg COD.m^-3] 
        X_h2    = initial_state['X_h2'][0]        #[kg COD.m^-3] 
        X_I     = initial_state['X_I'][0]         #[kg COD.m^-3] 
        
        S_cation = initial_state['S_cation'][0]   #[kmole.m^-3] 
        S_anion  = initial_state['S_anion'][0]    #[kmole.m^-3] 
        
        S_H_ion    = initial_state['S_H_ion'][0] #kmole H.m^-3
        S_va_ion   = initial_state['S_va_ion'][0] #kg COD.m^-3 valerate
        S_bu_ion   = initial_state['S_bu_ion'][0] #kg COD.m^-3 butyrate
        S_pro_ion  = initial_state['S_pro_ion'][0] #kg COD.m^-3 propionate
        S_ac_ion   = initial_state['S_ac_ion'][0] #kg COD.m^-3 acetate
        S_hco3_ion = initial_state['S_hco3_ion'][0] #kmole C.m^-3 bicarbonate
        S_nh3      = initial_state['S_nh3'][0] #kmole N.m^-3 ammonia
        S_nh4_ion  = 0.0041 #kmole N.m^-3 the initial value is from Rosen et al (2006) BSM2 report and it is calculated further down and does not need to be initiated
        S_co2      = 0.14 #kmole C.m^-3 the initial value is from Rosen et al (2006) BSM2 report and it is calculated further down and does not need to be initiated
        S_gas_h2   = initial_state['S_gas_h2'][0] #kg COD.m^-3 hydrogen concentration in gas phase
        S_gas_ch4  = initial_state['S_gas_ch4'][0] #kg COD.m^-3 methane concentration in gas phase
        S_gas_co2  = initial_state['S_gas_co2'][0]#kmole C.m^-3 carbon dioxide concentration in gas phas

   
T_state_zero = np.array([T_op_initial, T_op_initial])
setInfluent(0) #setting the influent for the initial time (t0) to be ready for the start of the simulation
setInitial(0)
pH = - np.log10(S_H_ion)
state_zero = [S_su,
              S_aa,
              S_fa,
              S_va,
              S_bu,
              S_pro,
              S_ac,
              S_h2,
              S_ch4,
              S_IC,
              S_IN,
              S_I,
              X_xc,
              X_ch,
              X_pr,
              X_li,
              X_su,
              X_aa,
              X_fa,
              X_c4,
              X_pro,
              X_ac,
              X_h2,
              X_I,
              S_cation,
              S_anion,
              S_H_ion,
              S_va_ion,
              S_bu_ion,
              S_pro_ion,
              S_ac_ion,
              S_hco3_ion,
              S_co2,
              S_nh3,
              S_nh4_ion,
              S_gas_h2,
              S_gas_ch4,
              S_gas_co2]

state_input = [S_su_in,
              S_aa_in,
              S_fa_in,
              S_va_in,
              S_bu_in,
              S_pro_in,
              S_ac_in,
              S_h2_in,
              S_ch4_in,
              S_IC_in,
              S_IN_in,
              S_I_in,
              X_xc_in,
              X_ch_in,
              X_pr_in,
              X_li_in,
              X_su_in,
              X_aa_in,
              X_fa_in,
              X_c4_in,
              X_pro_in,
              X_ac_in,
              X_h2_in,
              X_I_in,
              S_cation_in,
              S_anion_in]

#####################################################################################################
##----- SECTION TEN -----##
#Anaerobic digestion model

def ADM1_ODE(t, state_zero):
  global S_nh4_ion, S_co2, p_gas, q_gas, q_ch4, Rho_T_8, Rho_T_9, Rho_T_10
  
  T_op        = T_state_zero[0]
  
  S_su        = state_zero[0]
  S_aa        = state_zero[1]
  S_fa        = state_zero[2]
  S_va        = state_zero[3]
  S_bu        = state_zero[4]
  S_pro       = state_zero[5]
  S_ac        = state_zero[6]
  S_h2        = state_zero[7]
  S_ch4       = state_zero[8]
  S_IC        = state_zero[9]
  S_IN        = state_zero[10]
  S_I         = state_zero[11]
  X_xc        = state_zero[12]
  X_ch        = state_zero[13]
  X_pr        = state_zero[14]
  X_li        = state_zero[15]
  X_su        = state_zero[16]
  X_aa        = state_zero[17]
  X_fa        = state_zero[18]
  X_c4        = state_zero[19]
  X_pro       = state_zero[20]
  X_ac        = state_zero[21]
  X_h2        = state_zero[22]
  X_I         = state_zero[23]
  S_cation    = state_zero[24]
  S_anion     = state_zero[25]
  S_H_ion     = state_zero[26]
  S_va_ion    = state_zero[27]
  S_bu_ion    = state_zero[28]
  S_pro_ion   = state_zero[29]
  S_ac_ion    = state_zero[30]
  S_hco3_ion  = state_zero[31]
  S_co2       = state_zero[32]
  S_nh3       = state_zero[33]
  S_nh4_ion   = state_zero[34]
  S_gas_h2    = state_zero[35]
  S_gas_ch4   = state_zero[36]
  S_gas_co2   = state_zero[37]
  S_su_in     = state_input[0]
  S_aa_in     = state_input[1]
  S_fa_in     = state_input[2]
  S_va_in     = state_input[3]
  S_bu_in     = state_input[4]
  S_pro_in    = state_input[5]
  S_ac_in     = state_input[6]
  S_h2_in     = state_input[7]
  S_ch4_in    = state_input[8]
  S_IC_in     = state_input[9]
  S_IN_in     = state_input[10]
  S_I_in      = state_input[11]
  X_xc_in     = state_input[12]
  X_ch_in     = state_input[13]
  X_pr_in     = state_input[14]
  X_li_in     = state_input[15]
  X_su_in     = state_input[16]
  X_aa_in     = state_input[17]
  X_fa_in     = state_input[18]
  X_c4_in     = state_input[19]
  X_pro_in    = state_input[20]
  X_ac_in     = state_input[21]
  X_h2_in     = state_input[22]
  X_I_in      = state_input[23]
  S_cation_in = state_input[24]
  S_anion_in  = state_input[25]

  S_nh4_ion   = S_IN - S_nh3
  S_co2       = S_IC - S_hco3_ion
  
  # Inhibition functions
  I_pH_aa     =  ((K_pH_aa ** nn_aa) / (S_H_ion ** nn_aa + K_pH_aa ** nn_aa)) 
  I_pH_ac     =  ((K_pH_ac ** n_ac) / (S_H_ion ** n_ac + K_pH_ac ** n_ac)) 
  I_pH_h2     =  ((K_pH_h2 ** n_h2) / (S_H_ion ** n_h2 + K_pH_h2 ** n_h2))
  I_IN_lim    =  (1 / (1 + (K_S_IN / S_IN)))                                     # Inorganic Nitrogen, non-competitive free ammonia inhibition
  I_h2_fa     =  (1 / (1 + (S_h2 / K_I_h2_fa)))                                  # Hydrogen inhibtion of LCFA 
  I_h2_c4     =  (1 / (1 + (S_h2 / K_I_h2_c4)))                                  # Hydrogen inhibtion of C4 
  I_h2_pro    =  (1 / (1 + (S_h2 / K_I_h2_pro)))                                 # Hydrogen inhibtion of Propionate 
  I_nh3       =  (1 / (1 + (S_nh3 / K_I_nh3)))                                   # Ammonia inhibition

  I_5         = I_pH_aa * I_IN_lim  
  I_6         = I_5
  I_7         = I_pH_aa * I_IN_lim * I_h2_fa
  I_8         = I_pH_aa * I_IN_lim * I_h2_c4
  I_9         = I_8
  I_10        = I_pH_aa * I_IN_lim * I_h2_pro
  I_11        = I_pH_ac * I_IN_lim * I_nh3
  I_12        = I_pH_h2 * I_IN_lim

  # Biochemical process rates 
  Rho_1       =  (k_dis * X_xc)                                                                     # Disintegration
  Rho_2       =  (k_hyd_ch * X_ch)                                                                  # Hydrolysis of carbohydrates
  Rho_3       =  (k_hyd_pr * X_pr)                                                                  # Hydrolysis of proteins
  Rho_4       =  (k_hyd_li * X_li)                                                                  # Hydrolysis of lipids
  Rho_5       =  k_m_su * S_su / (K_S_su + S_su) * X_su * I_5                                       # Uptake of sugars
  Rho_6       =  (k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6)                                   # Uptake of amino-acids
  Rho_7       =  (k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7)                                   # Uptake of LCFA (long-chain fatty acids)
  Rho_8       =  (k_m_c4 * (S_va / (K_S_c4 + S_va )) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8)  # Uptake of valerate
  Rho_9       =  (k_m_c4 * (S_bu / (K_S_c4 + S_bu )) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9)  # Uptake of butyrate
  Rho_10      =  (k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10)                             # Uptake of propionate
  Rho_11      =  (k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11)                                  # Uptake of acetate
  Rho_12      =  (k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12)                                  # Uptake of hydrogen
  Rho_13      =  (k_dec_X_su * X_su)                                                                # Decay of X_su 
  Rho_14      =  (k_dec_X_aa * X_aa)                                                                # Decay of X_aa
  Rho_15      =  (k_dec_X_fa * X_fa)                                                                # Decay of X_fa
  Rho_16      =  (k_dec_X_c4 * X_c4)                                                                # Decay of X_c4
  Rho_17      =  (k_dec_X_pro * X_pro)                                                              # Decay of X_pro
  Rho_18      =  (k_dec_X_ac * X_ac)                                                                # Decay of X_ac
  Rho_19      =  (k_dec_X_h2 * X_h2)                                                                # Decay of X_h2

  # Acid-base rates 
  Rho_A_4     =  (k_A_B_va * (S_va_ion * (K_a_va + S_H_ion) - K_a_va * S_va))
  Rho_A_5     =  (k_A_B_bu * (S_bu_ion * (K_a_bu + S_H_ion) - K_a_bu * S_bu))
  Rho_A_6     =  (k_A_B_pro * (S_pro_ion * (K_a_pro + S_H_ion) - K_a_pro * S_pro))
  Rho_A_7     =  (k_A_B_ac * (S_ac_ion * (K_a_ac + S_H_ion) - K_a_ac * S_ac))
  Rho_A_10    =  (k_A_B_co2 * (S_hco3_ion * (K_a_co2 + S_H_ion) - K_a_co2 * S_IC))
  Rho_A_11    =  (k_A_B_IN * (S_nh3 * (K_a_IN + S_H_ion) - K_a_IN * S_IN))

  # Gas phase algebraic equations
  # pV = nRT, n = mass (m) / molar mass (M), mass = S in gCOD => p = S * R * T / M

  p_gas_h2  =  (S_gas_h2 * R * T_op / 16)                                                          # Hydrogen gas pressure
  p_gas_ch4 =  (S_gas_ch4 * R * T_op / 64)                                                         # Methane pressure
  p_gas_co2 =  (S_gas_co2 * R * T_op)                                                              # Carbon dioxide gas pressure
  p_gas_h2o =  0.0313 * np.exp(5290 * (1 / T_base - 1 / T_op)) 
  p_gas     =  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)                                      # Total gas pressure


  # Gas transfer rates based on Henry's law
  #k_L_a = the mass transfer coefficient, dependend on the gas-liquid exchange surface
 
  Rho_T_8   =  (k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2))  
  Rho_T_9   =  (k_L_a * (S_ch4 - 64 * K_H_ch4 * p_gas_ch4))
  Rho_T_10  =  (k_L_a * (S_co2 - K_H_co2 * p_gas_co2))
  q_gas     =  (k_p * (p_gas- p_atm))                                                               #Total gasflow, k_p = gas friction constant
  if q_gas < 0:    q_gas = 0                                                                        #Bugfix 

  q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow

  # Differential equaitons
  # V_liq = volume of the water in the reactor
  # q_ad = flow rate of influent 
  # S_xx_in = influent, S_xx 
  
  diff_S_su = q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5                                                                  # eq1 Change in monosaccharides, gains from hydrolysis of carbohydrates and lipids (not the LCFA part), are taken up as sugars by biomass
  diff_S_aa = q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6                                                                                          # eq2 Change in Amino Acids, gain by hydrolysis of proteins, loss by biomass growth
  diff_S_fa = q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7                                                                              # eq3 Change in LCFA, gain by hydrolysis of lipids, loss by biomass growth
  diff_S_va = q_ad / V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8                                                                   # eq4 Change in valerate, gain by acidogenesis of amino acids (not the ones taken up by biomass), loss by uptake of valerate by biomass
  diff_S_bu = q_ad / V_liq * (S_bu_in - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9                                    # eq5 Change in butyrate concentration, gain by acidogenesis of monosaccharides, amino acids, loss by cell growth
  diff_S_pro = q_ad / V_liq * (S_pro_in - S_pro) + (1 - Y_su) * f_pro_su * Rho_5 + (1 - Y_aa) * f_pro_aa * Rho_6 + (1 - Y_c4) * 0.54 * Rho_8 - Rho_10  # eq6 change in propionate, gain by acidogenesis of monosaccharides, amino acids and C4, loss by biomass uptake
  diff_S_ac = q_ad / V_liq * (S_ac_in - S_ac) + (1 - Y_su) * f_ac_su * Rho_5 + (1 - Y_aa) * f_ac_aa * Rho_6 + (1 - Y_fa) * 0.7 * Rho_7 + (1 - Y_c4) * 0.31 * Rho_8 + (1 - Y_c4) * 0.8 * Rho_9 + (1 - Y_pro) * 0.57 * Rho_10 - Rho_11  # eq7 # Change in acetate, gain by acidogenesis of monosaccharides, amino acids, LCFA, C4/butyrate, valerate, propionate, minus uptake of acetate by biomass

  if answer == "DAE":
    diff_S_h2 = 0 #eq 8 change in hydrogen gas  
  else:
    diff_S_h2 = q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8
     
  diff_S_ch4 = q_ad / V_liq * (S_ch4_in - S_ch4) + (1 - Y_ac) * Rho_11 + (1 - Y_h2) * Rho_12 - Rho_T_9  # eq9 change in Methane concentration in liquid. gains by methanogenesis of acetate and hydrogen, loss by methane liquid to gas transfer


  # Change in Inorganic Carbon, increases due to yield of biomass
  s_1  =  (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI)                  # Carbon content of composite material
  s_2  =  (-1 * C_ch + C_su)                                                                                                # Carbon fraction of carbohydrates 
  s_3  =  (-1 * C_pr + C_aa)                                                                                                # Carbon fraction of proteins
  s_4  =  (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa)                                                               # Carbon fraction of Lipids
  s_5  =  (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)                    # Carbon fraction of sugars, in  monosaccharides, lose sugars or in bacteria
  s_6  =  (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)   # Carbon fraction of amino acids in biomass
  s_7  =  (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac)                                                              # Carbon fraction of LCFA 
  s_8  =  (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)                                 # Carbon fraction from valerate
  s_9  =  (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac)                                                              # Carbon fraction from butyrate
  s_10 =  (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)                                                          # Carbon fraction from propionate
  s_11 =  (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)                                                                   # Carbon fraction fromm acetate
  s_12 =  ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)                                                                               # Methane production and hydrogenotrophich methanogens yield?
  s_13 =  (-1 * C_bac + C_xc)                                                                                               # Carbon content of biomass and composite material, biomass growth decreases s_13, biomass decay increases it

  #Total conversion from inorganic to organic
  Sigma =  (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))
  
  # Change in inorganic carbon, loss by conversion from inorganic to organic and from CO2 gasification
  diff_S_IC = q_ad / V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10

  # Change in inorganic nitrogen 
  diff_S_IN = q_ad / V_liq * (S_IN_in - S_IN) + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I-f_pr_xc * N_aa) * Rho_1 - Y_su * N_bac * Rho_5 + (N_aa - Y_aa * N_bac) * Rho_6 - Y_fa * N_bac * Rho_7 - Y_c4 * N_bac * Rho_8 - Y_c4 * N_bac * Rho_9 - Y_pro * N_bac * Rho_10 - Y_ac * N_bac * Rho_11 - Y_h2 * N_bac * Rho_12 + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19) # eq11 

  # Change in inorganic material
  diff_S_I = q_ad / V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12

  diff_X_xc  = q_ad / V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19           # eq13   Change in composite material (dead biomass)
  diff_X_ch  = q_ad / V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2                                                        # eq14   Change in carbohydrates in biomass
  diff_X_pr  = q_ad / V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3                                                        # eq15   Change in proteins in biomass
  diff_X_li  = q_ad / V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4                                                        # eq16   Change in Lipids in biomass # HERE THERE IS A MISTAKE IN THE ORIGINAL PYAD
  diff_X_su  = q_ad / V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13                                                          # eq17   Change in monosaccharides in biomass
  diff_X_aa  = q_ad / V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14                                                          # eq18   Change in amino acids in biomass
  diff_X_fa  = q_ad / V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15                                                          # eq19   Change in LCFA in biomass
  diff_X_c4  = q_ad / V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16                                           # eq20   Change in Butyrate in biomass
  diff_X_pro = q_ad / V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17                                                      # eq21   Change in propionate in biomass
  diff_X_ac  = q_ad / V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18                                                         # eq22   Change in acetate in biomass
  diff_X_h2  = q_ad / V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19                                                         # eq23   Change in hydrogenotrophic methanogens
  diff_X_I   = q_ad / V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1                                                                  # eq24   Change in inert material 

  diff_S_cation = q_ad / V_liq * (S_cation_in - S_cation)  # eq25 Change in cations
  diff_S_anion  = q_ad / V_liq * (S_anion_in - S_anion)  # eq26 change in anions
  
  # Differential equations for ion states, only for ODE framework
  if answer == "ODE":
      diff_S_va_ion   = -Rho_A_4      # eq27 
      diff_S_bu_ion   = -Rho_A_5      # eq28
      diff_S_pro_ion  = -Rho_A_6      # eq29
      diff_S_ac_ion   = -Rho_A_7      # eq30
      diff_S_hco3_ion = -Rho_A_10     # eq31
      diff_S_nh3      = -Rho_A_11     # eq32      
      
  else:    
      diff_S_va_ion   = 0             # eq27 
      diff_S_bu_ion   = 0             # eq28
      diff_S_pro_ion  = 0             # eq29
      diff_S_ac_ion   = 0             # eq30
      diff_S_hco3_ion = 0             # eq31
      diff_S_nh3      = 0             # eq32
      
  diff_S_gas_h2  = (q_gas / V_gas * -1 * S_gas_h2) + (Rho_T_8 * V_liq / V_gas)    # eq33
  diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (Rho_T_9 * V_liq / V_gas)   # eq34
  diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (Rho_T_10 * V_liq / V_gas)  # eq35
  
  if answer == "DAE":
    diff_S_H_ion =  0    
  elif answer == "ODE":
    Theta        = S_cation + S_nh4_ion - S_hco3_ion - S_ac_ion / 64 - S_pro_ion / 112 - S_bu_ion / 160 - S_va_ion / 208 - S_anion
    diff_S_H_ion = -Theta / 2 + 0.5 * np.sqrt(pow(Theta, 2) + 4 * K_w)
  elif answer == "pure-ODE":
    A            = diff_S_anion + diff_S_IN * K_a_IN / (K_a_IN + S_H_ion) + diff_S_IC * K_a_co2 / (K_a_co2 + S_H_ion) + \
                   (1 / 64)  * diff_S_ac * K_a_ac / (K_a_ac + S_H_ion) + (1 / 112) * diff_S_pro * K_a_pro / (K_a_pro + S_H_ion) + \
                   (1 / 160) * diff_S_bu * K_a_bu / (K_a_bu + S_H_ion) + (1 / 208) * diff_S_va * K_a_va / (K_a_va + S_H_ion) - diff_S_IN - diff_S_cation

    B            = 1       + K_a_IN * S_IN  / pow((K_a_IN  + S_H_ion), 2) + \
                             K_a_co2* S_IC  / pow((K_a_co2 + S_H_ion), 2) + \
                 (1 / 64)  * K_a_ac * S_ac  / pow((K_a_ac  + S_H_ion), 2) + \
                 (1 / 112) * K_a_pro* S_pro / pow((K_a_pro + S_H_ion), 2) + \
                 (1 / 160) * K_a_bu * S_bu  / pow((K_a_bu  + S_H_ion), 2) + \
                 (1 / 208) * K_a_va * S_va  / pow((K_a_va  + S_H_ion), 2) + \
                             K_w / pow(S_H_ion, 2)
    SH = A / B
    diff_S_H_ion = SH
    
  diff_S_co2 = 0
  diff_S_nh4_ion = 0 #to keep the output same length as input for ADM1_ODE funcion

  return diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I, diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I, diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion,  diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2,  diff_S_nh3, diff_S_nh4_ion, diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2

#####################################################################################################
##----- SECTION ELEVEN -----##
# Simulation of the model
# Solvers

# Function for integration of ADM1 differential equations
def ADM1simulate(t_step, solvermethod):
    if version == 0:
        r = scipy.integrate.solve_ivp(ADM1_ODE, t_step, state_zero, method= solvermethod)
    return r.y

# Function for integration of heat balance equations
def Tsimulate(t_step):
  r = scipy.integrate.solve_ivp(dTdt, t_step, T_state_zero, method=solvermethod)
  return r.y

# Function for DAE equations
def DAESolve():
  global S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_H_ion, pH, p_gas_h2, S_h2, S_nh4_ion, S_co2, P_gas, q_gas
  
  ##  DAE calculations 
  eps         = 0.0000001
  prevS_H_ion = S_H_ion
  
  #initial values for Newton-Raphson solver parameter
  shdelta     = 1.0
  shgradeq    = 1.0
  S_h2delta   = 1.0
  S_h2gradeq  = 1.0
  tol         = 10 ** (-12)          # solver accuracy tolerance
  maxIter     = 1000                 # maximum number of iterations for solver
  i = 1
  j = 1
  
  ## DAE solver for S_H_ion from Rosen et al. (2006)
  while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
    S_va_ion    = K_a_va * S_va / (K_a_va + S_H_ion)
    S_bu_ion    = K_a_bu * S_bu / (K_a_bu + S_H_ion)
    S_pro_ion   = K_a_pro * S_pro / (K_a_pro + S_H_ion)
    S_ac_ion    = K_a_ac * S_ac / (K_a_ac + S_H_ion)
    S_hco3_ion  = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
    S_nh3       = K_a_IN * S_IN / (K_a_IN + S_H_ion)
    shdelta     = S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 - S_va_ion / 208.0 - K_w / S_H_ion - S_anion
    shgradeq    = 1 + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion)) + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion)) + 1 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion))               + 1 / 112.0 * K_a_pro * S_pro / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion))               + 1 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion))               + 1 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion))               + K_w / (S_H_ion * S_H_ion)
    S_H_ion     = S_H_ion - shdelta / shgradeq
    if S_H_ion <= 0:
        S_H_ion = tol
    i+=1
  
  # pH calculation
  pH = - np.log10(S_H_ion)
  
  #DAE solver for S_h2 
  while ((S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter)):
    I_pH_aa  = (K_pH_aa ** nn_aa) / (prevS_H_ion ** nn_aa + K_pH_aa ** nn_aa)
    I_pH_h2  = (K_pH_h2 ** n_h2) / (prevS_H_ion ** n_h2 + K_pH_h2 ** n_h2)
    I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
    I_h2_fa  = 1 / (1 + (S_h2 / K_I_h2_fa))
    I_h2_c4  = 1 / (1 + (S_h2 / K_I_h2_c4))
    I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))
  
    I_5    = I_pH_aa * I_IN_lim
    I_6    = I_5
    I_7    = I_pH_aa * I_IN_lim * I_h2_fa
    I_8    = I_pH_aa * I_IN_lim * I_h2_c4
    I_9    = I_8
    I_10   = I_pH_aa * I_IN_lim * I_h2_pro
    I_12   = I_pH_h2 * I_IN_lim
   
    Rho_5      = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  
    Rho_6      = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  
    Rho_7      = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7 
    Rho_8      = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va+ 1e-6)) * I_8  
    Rho_9      = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va+ 1e-6)) * I_9  
    Rho_10     = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10  
    Rho_12     = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  
    p_gas_h2   = S_gas_h2 * R * T_state_zero[0] / 16
    Rho_T_8    = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
    S_h2delta  = q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8
    S_h2gradeq = - 1.0 / V_liq * q_ad - 3.0 / 10.0 * (1 - Y_fa) * k_m_fa * S_fa / (K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_fa) * (1 + S_h2 / K_I_h2_fa)) / K_I_h2_fa - 3.0 / 20.0 * (1 - Y_c4) * k_m_c4 * S_va * S_va / (K_S_c4 + S_va) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 1.0 / 5.0 * (1 - Y_c4) * k_m_c4 * S_bu * S_bu / (K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 43.0 / 100.0 * (1 - Y_pro) * k_m_pro * S_pro / (K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_pro ) * (1 + S_h2 / K_I_h2_pro )) / K_I_h2_pro - k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) + k_m_h2 * S_h2 / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) - k_L_a
    S_h2 = S_h2 - S_h2delta / S_h2gradeq
    if S_h2 <= 0:
        S_h2 = tol
    j+=1

# Time array definition and further setup of lists to store results
t = np.linspace(0,365, 365+1) 
#t = influent_state['time']
#t = list(t.values())

#ta = t.values(dtype=list)

# Initiate the cache data frame for storing simulation results
simulate_results = []
columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion", "pH", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"]
simulate_results.append(columns)

gasflow = []
gas_columns = ["q gas", "q ch4"]
gasflow.append(gas_columns)
total_ch4 = 0

T_op_data = []
temp_columns = ["Temperature"]
T_op_data.append(temp_columns)

growthfactors = []
# Initiate cache data frame for storing feedflow values
# initq = {'q_ad' : [170]}
# feedflow = pd.DataFrame(initq)
F_dyn_results = []

t0=0
n=0


# Dynamic simulation loop
# Loop for simlating at each time step and feeding the results to the next time step
for u in t[0:]:
  n+=1
  #take new influent data
  setInfluent(n)

  
  # take next weather data points and assign them
  setWeather(int(t0))
  weatherinput = [Q_solar_in, T_soil_gas_in, u_wind_in, T_air_in]
    
  # span for next time step
  tstep = [t0,u]
    
  # solve and store values for next time step
  T_sim, T_adt_sim = Tsimulate(tstep)
  T_op, T_adt = T_sim[-1], T_adt_sim[-1]

  T_state_zero = [T_op, T_adt]
  T_op_data.append(T_state_zero)
  
  
  F_dyn = math.exp( - ((T_op - T_adt) ** 2 / (2 * sigma_DYN ** 2)))
  F_dyn_results.append(F_dyn)
   
  # Calculate temperature dependent factors
  settempdependent(T_state_zero[0])
  if tempmodel == 'Cardinal':
      setgrowthconstantscardinal(T_state_zero[0], F_dyn)
  elif tempmodel == 'Arrhenius':
      setgrowthconstantsarrhenius(T_state_zero[0], F_dyn)
  
  growthfactors.append(k_hyd_ch)
  state_input = [S_su_in,S_aa_in,S_fa_in,S_va_in,S_bu_in,S_pro_in,S_ac_in,S_h2_in,S_ch4_in,S_IC_in,S_IN_in,S_I_in,X_xc_in,X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,S_cation_in,S_anion_in]
  


  # Solve and store ODE results for next step 
  sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_IC, sim_S_IN, sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac, sim_X_h2, sim_X_I, sim_S_cation, sim_S_anion, sim_S_H_ion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion, sim_S_ac_ion, sim_S_hco3_ion, sim_S_co2, sim_S_nh3, sim_S_nh4_ion, sim_S_gas_h2, sim_S_gas_ch4, sim_S_gas_co2 = ADM1simulate(tstep, solvermethod)

  # Store ODE simulation result states
  S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2 =   sim_S_su[-1], sim_S_aa[-1], sim_S_fa[-1], sim_S_va[-1], sim_S_bu[-1], sim_S_pro[-1], sim_S_ac[-1], sim_S_h2[-1], sim_S_ch4[-1], sim_S_IC[-1], sim_S_IN[-1], sim_S_I[-1], sim_X_xc[-1], sim_X_ch[-1], sim_X_pr[-1], sim_X_li[-1], sim_X_su[-1], sim_X_aa[-1], sim_X_fa[-1], sim_X_c4[-1], sim_X_pro[-1], sim_X_ac[-1], sim_X_h2[-1], sim_X_I[-1], sim_S_cation[-1], sim_S_anion[-1], sim_S_H_ion[-1], sim_S_va_ion[-1], sim_S_bu_ion[-1], sim_S_pro_ion[-1], sim_S_ac_ion[-1], sim_S_hco3_ion[-1], sim_S_co2[-1], sim_S_nh3[-1], sim_S_nh4_ion[-1], sim_S_gas_h2[-1], sim_S_gas_ch4[-1], sim_S_gas_co2[-1]
  
  # Solve DAE states
  if answer == "DAE":
      DAESolve()

  # Algebraic equations 
  # 16 for h2 and 64 for ch4 are temperature dependent, and this value is set for 25 degrees celcius
  
  p_gas_h2o =  0.0313 * np.exp(5290 * (1 / T_base - 1 / T_op))
  p_gas_h2  =  (S_gas_h2 * R * T_state_zero[0] / 16)
  p_gas_ch4 =  (S_gas_ch4 * R * T_state_zero[0] / 64)
  p_gas_co2 =  (S_gas_co2 * R * T_state_zero[0])    
  p_gas    =  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)
  
  if gascontrol == 'variable':
      q_gas =  (k_p * (p_gas- p_atm))
  elif gascontrol == 'steady':
      if version == 0:
          q_gas = (R * T_op / (p_atm - p_gas_h2o)) * V_liq * (Rho_T_8/16 + Rho_T_9/64 + Rho_T_10)
      else:
          (R * T_op / (p_atm - p_gas_h2o)) * V_liq * (Rho_T_9/64 + Rho_T_10)
  if q_gas < 0:    
      q_gas = 0  
      
  q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow
  
  if q_ch4 < 0:
      q_ch4 = 0
    
  gasflow.append([q_gas, q_ch4])   
       
  S_nh4_ion =  (S_IN - S_nh3)
  S_co2 =  (S_IC - S_hco3_ion)
  total_ch4 = total_ch4 + q_ch4 

  #state transfer
  state_zero = [S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2]
  state_zero = np.array(state_zero)
  #T_state_zero = np.array([T_state_zero])
  state_zero[state_zero < 0] = 0 #this makes sure no negative concentrations are ported to the next step or results
  simulate_results.append(state_zero)
  
  t0 = u
  print(t0)
 
# Write the dynamic simulation results to dataframe
simulate_results = pd.DataFrame(simulate_results[1:], columns = simulate_results[0])
gasflow = pd.DataFrame(gasflow[1:], columns = gasflow[0])
# Write the dynamic simulation resutls to csv
phlogarray = -1 * np.log10(simulate_results['pH'])
simulate_results['pH'] = phlogarray
simulate_results.to_csv("dynamic_out.csv", index = False)
print(total_ch4)

print("--- %s seconds ---" % (time.time() - start_time))
