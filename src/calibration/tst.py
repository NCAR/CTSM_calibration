def update_ctsm_parameters(path_CTSM_case, file_parameter_set):


    file_user_nl_clm = f'{path_CTSM_case}/user_nl_clm'
    df_calibparam = pd.read_csv(file_parameter_set)

    dfi = df_calibparam.loc[df_calibparam['Source'] == 'Namelist']
    if len(dfi) > 0:
        param_names = dfi['Parameter'].values
        param_values = dfi['Value'].values
        update_CTSM_parameter_NamelistFile(param_names, param_values, file_user_nl_clm)

    dfi = df_calibparam.loc[df_calibparam['Source'] == 'Param']
    if len(dfi) > 0:
        outfile_newparam = f'{path_CTSM_case}/updated_parameter.nc'
        param_names = dfi['Parameter'].values
        param_values = dfi['Value'].values
        update_CTSM_parameter_ParamFile(param_names, param_values, path_CTSM_case, outfile_newparam)
        # add to the name list file
        with open(file_user_nl_clm, 'a') as f:
            f.write(f"paramfile='{outfile_newparam}'\n")

    dfi = df_calibparam.loc[df_calibparam['Source'] == 'Surfdata']
    if len(dfi) > 0:
        outfile_newsurfdata = f'{path_CTSM_case}/updated_surfdata.nc'
        param_names = dfi['Parameter'].values
        param_values = dfi['Value'].values
        update_CTSM_parameter_SurfdataFile(param_names, param_values, path_CTSM_case, outfile_newsurfdata)
        # add to the name list file
        with open(file_user_nl_clm, 'a') as f:
            f.write(f"fsurdat='{outfile_newsurfdata}'\n")