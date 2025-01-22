# Ostrich configuration file
ProgramType  DDS
ModelExecutable ./CTSM_run_trial.sh
ObjectiveFunction gcop

OstrichWarmStart no

PreserveBestModel ./call_PreserveBestModel.sh
PreserveModelOutput ./call_PreserveModelOutput.sh
OnObsError	-999

BeginFilePairs    
param_factor.tpl; param_factor.txt
EndFilePairs

#Parameter/DV Specification
BeginParams
#parameter			init	lwr	upr	txInN  txOst 	txOut fmt  
snow_canopy_storage_scalar_mtp  0.95	0.5	2.0	none   none	none  free
EndParams

BeginResponseVars
  #name	  filename			      keyword		line	col	token
  MeanErr      ./trial_stats.txt;	      OST_NULL	         0	1  	 ' '
EndResponseVars 

BeginTiedRespVars
  PosMeanErr 1 MeanErr wsum 1.00
EndTiedRespVars

BeginGCOP
  CostFunction PosMeanErr
  PenaltyFunction APM
EndGCOP

BeginConstraints
# not needed when no constraints, but PenaltyFunction statement above is required
# name     type     penalty    lwr   upr   resp.var
EndConstraints

# Randomsed control added
RandomSeed 1234567890

BeginDDSAlg
PerturbationValue 0.20
MaxIterations 5 #5
#UseRandomParamValues
UseInitialParamValues
EndDDSAlg