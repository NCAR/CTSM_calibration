begin
  ;--- read list of the files ---
  lst = asciiread("list.txt", -1, "string")
  nlst = dimsizes(lst)

  ;--- loop over files ---
  do i = 0, nlst-1
    print(i+" "+lst(i))

    ;--- read file ---
    grd_file = addfile(lst(i), "r")

    ;--- list of variables ---
    lst_var = getfilevarnames(grd_file)

    ;--- check variables ---
    lat_name = ""
    lon_name = ""
    msk_name = ""
    do j = 0, dimsizes(lst_var)-1
      var_type = typeof(grd_file->$lst_var(j)$)
      if (isStrSubset(str_lower(lst_var(j)), "lon") .or. \
          isStrSubset(str_lower(lst_var(j)), "xc")) then
        if (var_type .ne. "integer") then
          lon_name = lst_var(j)
        end if
      end if 
      if (isStrSubset(str_lower(lst_var(j)), "lat") .or. \
          isStrSubset(str_lower(lst_var(j)), "yc")) then
        if (var_type .ne. "integer") then
          lat_name = lst_var(j)
        end if
      end if
      if (isStrSubset(str_lower(lst_var(j)), "mask")) then
        msk_name = lst_var(j)
      end if
    end do

    ;--- read variables ---
    if (.not. str_is_blank(lat_name)) then
      lat = grd_file->$lat_name$
      rank = dimsizes(dimsizes(lat))
    end if
    if (.not. str_is_blank(lon_name)) then
      lon = grd_file->$lon_name$
    end if
    if (.not. str_is_blank(msk_name)) then
      msk = grd_file->$msk_name$
    end if
    print(lat_name+" "+lon_name+" "+msk_name+" "+rank)

    ;--- get date and create output file ---
    date = systemfunc("date -u '+%d%m%y'")
    dumm = str_split(lst(i), ".")
    if (dimsizes(dumm) .le. 2) then
      ofile = str_sub_str(lst(i), ".nc", ".SCRIP."+date+".nc")
    else
      ofile = str_join(dumm(0:dimsizes(dumm)-3), ".")+".SCRIP."+date+".nc"
    end if
    ofile = systemfunc("basename "+ofile)
    delete(dumm)

    ;--- write file name to file that mesh generation script could read ---
    if (i .eq. 0) then
      system("echo '"+ofile+"' > create_mesh.txt")
    else
      system("echo '"+ofile+"' >> create_mesh.txt")
    end if

    ;--- generate SCRIP file ---
    opt = True
    opt@Debug = False
    opt@Testit = False
    opt@ForceOverwrite = True
    opt@PrintTimings = True
    opt@NetCDFType = "netcdf4"
    opt@Title = "input file: "+lst(i)
    if (.not. str_is_blank(msk_name)) then
      if (min(msk) .ne. max(msk)) then
        opt@GridMask = msk
        print("mask min = "+sprintf("%8.3f", min(msk))+" max = "+sprintf("%8.3f", max(msk)))
      end if
    end if

    ;--- fix for single point data ---
    dims = dimsizes(lat)
    if (dims(0) .eq. 1 .and. dims(1) .eq. 1) then
      print(lat)
      print(lon)
      rectilinear_to_SCRIP(ofile, lat(0,0), lon(0,0), opt)
    else
      if (rank .eq. 1) then
        rectilinear_to_SCRIP(ofile, lat, lon, opt)
      end if
      if (rank .eq. 2) then
        curvilinear_to_SCRIP(ofile, lat, lon, opt)
      end if
    end if

    ;--- add area to SCRIP file ---
    scripFile = addfile(ofile, "w")

    grid_size = dimsizes(scripFile->grid_center_lat)
    grid_area = new(grid_size,double)
    grid_area!0 = "grid_size"

    do j = 0, grid_size-1
      temp_tlat = (/ scripFile->grid_corner_lat(j,2), \
                     scripFile->grid_corner_lat(j,1), \
                     scripFile->grid_corner_lat(j,0), \
                     scripFile->grid_corner_lat(j,3) /)
      temp_tlon = (/ scripFile->grid_corner_lon(j,2), \
                     scripFile->grid_corner_lon(j,1), \
                     scripFile->grid_corner_lon(j,0), \
                     scripFile->grid_corner_lon(j,3) /)
      grid_area(j) = area_poly_sphere(temp_tlat, temp_tlon, 1)
    end do

    scripFile->grid_area = (/ grid_area /)

    print("area min = "+sprintf("%8.3f", min(grid_area))+" max = "+sprintf("%8.3f", max(grid_area)))
    print("corner lon min = "+sprintf("%8.3f", min(scripFile->grid_corner_lon))+" max = "+sprintf("%8.3f", max(scripFile->grid_corner_lon)))
    print("corner lat min = "+sprintf("%8.3f", min(scripFile->grid_corner_lat))+" max = "+sprintf("%8.3f", max(scripFile->grid_corner_lat)))

    ;--- delete temporary variables ---
    delete([/ lst_var, opt, ofile, grid_area /])
    if (isdefined("lon")) then
      delete(lon)
    end if
    if (isdefined("lat")) then
      delete(lat)
    end if
    if (isdefined("msk")) then
      delete(msk)
    end if
  end do

  ;--- submit job for mesh generation (it reads create_mesh.txt for the SCRIP files) ---
  system("qsub create_mesh.job")
end