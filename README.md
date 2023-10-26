# Thermal-Integrated-Anaerobic-Digestion-Model
Thermal-Integrated Anaerobic Digestion Model is an integrated anaerobic digestion model with a heat transfer network through a novel mechanstic approach to simulate dynamical temperature flactuation on anaerobic digestionestuarine environments.

# Introduction
Thermal-Integrated Anaerobic Digestion Model encompasses two integral models, i.e ADMn1 and the heat transfer model, which were integrated through a novel mechanstic approach by introducing a dynamical thermal state variable. The developed model consists of the following sections to navigate easily and adapt accordingly. 

  * Section one: Determine model specification;
  * Section two: ADMn1 parameters;
  * Section three: Heat network paramaeters;
  * Section four: Operational parameter;
  * Section five: Importing weather condition data from csv files;
  * Section six: Influent and initial parameters;
  * Section seven: Heat transfer model;
  * Section eight: Calculate temperature dependents;
  * Section nine: Setting initial and influent values;
  * Section ten: Anaerobic digestion model;
  * Section Eleven: Simulation of the model;


# Use of the model
How to use

To begin a simulation, firstly a few steps should be taken beforhands. In the following, you can find everywhere within the main-model.py that should be filled out/changed by the user.
Every section is divided by "######". However, other parts may be changed by user as wishes, but primarily, there is no need to take any action beforehand.

###############################################################################################
- Section one: Determine model specification:
  
   --> Line#44: asnwer = 'ODE' or 'DAE' or 'pure-ODE' --> that determines the equation framework of the model.
  
   --> Line#47: solvermethod = 'Radau' --> can be changed if the user wishes trying different solver
  
   --> Line#48: tempmodel = 'Cardinal' or 'Arrhenius' --> that determines type of tempereture inhibition functions

###############################################################################################
- Section three: Heat network paramaeters:
  
   This section and the parameters and values within it, should be adjusted according to the digester heat network under consideration.
  
   The presented heat network was designed based on the heat network of a buried dome digester with insulation.
  
  --> Line#217: T_soil_sub = 283.15 #[K] --> Soil temperature at the bottom of the digester
  
  --> Line#228: labda_digester_cover = 5*86400 # [W*m^-1*K^-1] --> Thermal conductivity of plain concrete walls 300 mm thick with air space plus brick facing
  
  --> Line#230: labda_digester_dry_walls = 0.6*86400 # [W*m^-1*K^-1] --> Thermal conductivity of plain concrete walls surrounded by moist earth
  
  --> Line#232: labda_digester_wet_walls = 1.2*86400 # [W*m^-1*K^-1] --> Thermal conductivity of plain concrete floor surrounded by moist earth

  --> Line#250: h_cov_gas   = 2.15*86400 # [W*m^-2*K^-1] --> Convective heat transfer between cover and biogas
  
  --> Line#251: h_gas_wall  = 2.70*86400 # [W*m^-2*K^-1] --> Convective heat transfer between biogas and wall
   
  --> Line#252: h_gas_sub   = 2.20*86400 # [W*m^-2*K^-1] --> Convective heat transfer between biogas and substrate
   
  --> Line#253: h_sub_wall  = 177.25*86400 # [W*m^-2*K^-1] --> Convective heat transfer between substrate and wall
   
  --> Line#254: h_sub_floor = 244.15*86400 # [W*m^-2*K^-1] --> Convective heat transfer between substrate and floor

###############################################################################################
- Section four: Operational parameters
   All defined parameters in this section may/should be adjusted by user.

   --> Line#265: V_liq = 4300/100 #[m^3] --> Volume for liquid fraction of the reactor
  
   --> Line#266: V_gas = 300/100 #[m^3] --> Reactor volume for gas
  
   --> Line#267: V_ad = V_liq + V_gas #[m^3] --> Reactor volume
  
   --> Line#268: r_out = (3 * V_ad / (4 * math.pi))**(1/3) #[m] --> Reactor outside radius (This is only applicable for dome digester)
  
   --> Line#269: A = math.pi * (r_out ** 2) #[m^2] --> Reacor surface (can be a value for any other type of digesters)
  
   --> Line#270: Delta_X = 3 #[m] --> Digester wall thickness
  
   --> Line#271: r_in = r_out - Delta_X #[m] --> Reactor inside radius (This is only applicable for dome digesters)
  
   --> Line#272: q_ad = 200/100 #[m^3*d^-1] --> Influent flow rate
   
   --> Line#273: T_feed = 308.15 #[K] --> Temperature of influent

###############################################################################################
- Section five: Importing weather condition data from csv files
  
   This section can be adjusted based on type of meteorological dataset to prepare them to feed to the simulation.

  --> Line#281: grounddata1 = (pd.read_csv(r'C:#####SHOULD BE FILLED OUT BY THE USER#####.csv', sep=';', skiprows =16)).to_numpy()
  
  --> Line#283: airdata1 = (pd.read_csv(r'C:#####SHOULD BE FILLED OUT BY THE USER#####.csv', sep=';', skiprows =52)).to_numpy()


###############################################################################################
- Section six: Influent and initial parameters from csv files
  
  Firstly, the excel files of influent and initial should be updated based on BMP test and substrate characterization.
  
  Then, reference addresses should also be added accordingly.

  --> Line#313: influent_state = pd.read_csv(r'C:#####SHOULD BE FILLED OUT BY THE USER#####\influent.csv', sep=';')
  
  --> Line#315: initial_state = pd.read_csv(r"C:#####SHOULD BE FILLED OUT BY THE USER#####\batchinitial.csv")

###############################################################################################
- Section seven: Heat transfer model
  
  Resistances may/should be updated according to the heat transfer network by user.

  --> Line#356:  R_CNV_air_cover    = 1 / h_cnv_air_cover  --> convective resistance between air and cover
   
  --> Line#357:  R_CND_cover        = Delta_X / labda_digester_cover --> conductive resistance of cover
  
  --> Line#358:  R_CNV_biogas_cover = 1 / h_cov_gas --> convective resistance between biogas and cover
   
  --> Line#359:  R_CNV_biogas_wall  = 1 / h_gas_wall --> convective resistance between biogas and wall
  
  --> Line#360:  R_CND_wall         = Delta_X / labda_digester_wet_walls --> conductive resistance of wall
     
  --> Line#361:  R_CNV_sub_biogas   = 1 / h_gas_sub --> convective resistance between biogas and substrate
  
  --> Line#362:  R_CNV_sub_wall     = 1 / h_sub_wall --> convective resistance between substrate and wall
  
  --> Line#364:  R_CNV_floor_sub    = 1 / h_sub_floor --> convective resistance between floor and substrate
  
  --> Line#365:  R_CND_floor        = Delta_X / labda_digester_wet_walls --> conductive resistance of floor

###############################################################################################
- Section eight: Calculate temperature dependents
  
  Data parameters of Cardinal and Arrhenius (Line#414-434) can be updated based on the results of nonlinear LS identification. 
    
  --

More information
A full account of software functionalities and implemented methods can be found [here](link to paper).
