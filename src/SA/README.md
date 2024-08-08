When generating LHS from the default parameter ranges (i.e, those in csv), those factors are used to multiply the real parameters (param nc, surf nc, namelist). Thus, even the so-called default parameters in the csv could be very different from real parameters, the sensitivity analysis is still effective. Note that the perturbation range is still decided based on csv default values. For example, for zbedrock as below, the pertubration range is 0.8 to 1.2.

Default                                                    10.639669
Lower                                                          8.512
Upper                                                         12.768