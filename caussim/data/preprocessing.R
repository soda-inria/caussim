# Load the data from [ACIC 2016](https://github.com/vdorie/aciccomp/tree/master/2016) thanks to their R package and dump corresponding csvs
# Requirements: you need R and the acic 2016 R package installed
library(aciccomp2016)

load_acic_2016_x = function(){
    X = input_2016
    X
    return(X)
}
load_acic_2016_y = function(parameterNum, seed){
    # loaded x object from
    y_setup = dgp_2016(input_2016, parameterNum, seed)
    y_setup
    return(y_setup)
}