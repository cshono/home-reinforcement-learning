# HomeReinforcementLearning

## hv_model 
This model establishes the relationship between CDD/HDD and HVAC_kWh. Upon initial exploration, it was determined that predicting HVAC power at 15-minute resolution is too difficult for a simple linear model that does not account for the thermal capacity of the home. However, a linear model can reasonably predict daily consumption of the home given the cumulative HDD and CDD over each 24-hr period. 
- Current model only includes 2019-09 
- A single linear fit for the entire community (only 40 homes have sufficient data quality) 
- Producing a separate model for each home may be too difficult. No significant difference in the relationship apparent for different homes. 
