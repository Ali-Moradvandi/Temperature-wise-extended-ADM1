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


# Use of program
How to use

To begin a simulation, the GUI App needs to be started either by:
Executing the installed PROVER-M application.
Opening the scr-directory and double-clicking on the file PROVER_M.mlapp (requires MATLAB).
Opening the PROVER_M.mlapp in MATLAB (requires MATLAB).
The model input can be set by changing individual parameters or by selecting an existing input text file via the "Input case" dropdown menu or the "Load.."-button.
After setting the desired input parameters, the configuration can be saved as a text file by clicking the "Save as.."-button.
The input can be reset to the last saved version of the selected input case in two ways. While "Reset sediment data" resets the sediment characteristics, "Reset input" resets the entire input configuration.
By clicking the "START"-button, the simulation is initiated.
In the right half of the GUI, a live feed of the cloud propagation through the water column and main parameters is presented.
Input
(including suggested order of magnitude)

Ambient parameters (incl. suggested order of magnitude)

Water depth                       (101 to 102)
Ambient velocities             (-100 to 100)
Ambient densities              (103)
Hopper settings

Disposal volume                (102 to 104)
Hopper draft                      (100 to 101)
Dumping instances            (divides a disposal into n-equal intervals that are disposed consecutively)
Settling

-1 to 1 (For positive values [0-1], the reduction factor determines the fraction of the sediment to be settled. For a reduction factor of -1, the settling is only based on the critical shear stress.)
Coefficients (All coefficients should be selected carefully and within the limits of the given order of magnitude)

Entrainment phase 1          (10-1)
Entrainment phase 2          (10-1)
Mass                                   (100)
Drag phase 1                      (100)
Drag phase 2                      (10-2 to 10-1)
Friction                               (10-2)
Stripping                            (10-3)
Sediment characteristics

Type
Density
Volumetric fraction
Fall velocity
Void ratio
Critical shear stress
Cohesiveness
Allowing the material to be stripped (applies to fine sediments, including fine sand)
Output
Output settings

Output time step interval (sets the interval for the output variables for stripped and settled sediments)
Time step for plotting
'Yes': Uses the output time step interval for updating the live feed of the cloud propagation through the water column within the GUI.
'No': Updates the live feed of the cloud propagation through the water column within the GUI using a preset interval of 25 time steps.
GUI live feed

Time
Cloud radius
Stripping volume
Cloud width
Cloud height
Settling volume
Output file

Cloud parameters as textfile
Stripped and settled sediments as textfile
Cloud and sediment variables as MAT-files
Output files are stored in a directory "output". If the PROVER-M stand alone .exe has been used, the data will be stored under C:\Users\username\AppData\Local\Temp\username\mcrCache9.12\PROVER0\PROVER_M\output. The accessability will be changed with the next update.

Functionality (Flow chart)
An overview of the functionality of the program code in the form of a flow chart can be found [here](link to paper).
Prerequisites
For executing the PROVER.exe on a Windows system, no necessary packages or programs are needed.

Source Code
If a user wants to adapt or change the source code, the scr-directory includes the following files:

PROVER_M.mlapp
A MATLAB App that starts the GUI, which may be used for selecting input parameters, starting simulations and accessing the live feed.
prover_m_main.m
Main program code, where the bookkeeping of the cloud and ambient parameters occurs.
prover_m_rk4.m
This numerical solver function utilizes the Runge-Kutta 4th order method to approximate parameter gradients
prover_m_phase1.m
In this function, the addressed variables in the phase of convective descent are calculated using the conservation equations.
prover_m_phase2.m
In this function, the addressed variables in the phase of dynamic collapse are calculated using the energy concept equations.
For running the source code files, an installed and licened version of MATLAB by Mathworks is needed.

License
GNU GPL License
PROVER-M has been developed in MATLAB and is provided via the Application Compiler provided by MATLAB®. © 1984 - 2022a The MathWorks, Inc.

More information
A full account of software functionalities and implemented methods can be found [here](link to paper).
