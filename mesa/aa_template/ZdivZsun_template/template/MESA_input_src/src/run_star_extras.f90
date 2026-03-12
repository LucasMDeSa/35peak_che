! ***********************************************************************
!
!   Copyright (C) 2012  Bill Paxton
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful,
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************

module run_star_extras

  use star_lib
  use star_def
  use const_def
  use math_lib
  use chem_def
  use num_lib
  use binary_def

  implicit none

contains

  include 'Fuller_AM/Fuller_AM_transport.inc'

  subroutine extras_controls(id, ierr)
    integer, intent(in) :: id
    integer, intent(out) :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return

    s% extras_startup => extras_startup
    s% extras_start_step => extras_start_step
    s% extras_check_model => extras_check_model
    s% extras_finish_step => extras_finish_step
    s% extras_after_evolve => extras_after_evolve
    s% how_many_extra_history_columns => how_many_extra_history_columns
    s% data_for_extra_history_columns => data_for_extra_history_columns
    s% how_many_extra_profile_columns => how_many_extra_profile_columns
    s% data_for_extra_profile_columns => data_for_extra_profile_columns

    s% how_many_extra_history_header_items => how_many_extra_history_header_items
    s% data_for_extra_history_header_items => data_for_extra_history_header_items
    s% how_many_extra_profile_header_items => how_many_extra_profile_header_items
    s% data_for_extra_profile_header_items => data_for_extra_profile_header_items

    ! if use_other_am_mixing = .true.
    s% other_am_mixing => TSF_Fuller_lu22
    s% how_many_other_mesh_fcns => how_many_other_mesh_fcns
    s% other_mesh_fcn_data => resolve_gradients

    ! lucas
    s% other_wind => che_wind

  end subroutine extras_controls

  subroutine how_many_other_mesh_fcns(id, n)
    integer, intent(in) :: id
    integer, intent(out) :: n
    n = 3
  end subroutine how_many_other_mesh_fcns

  subroutine resolve_gradients( &
       id, nfcns, names, gval_is_xa_function, vals1, ierr)
    use const_def
    integer, intent(in) :: id
    integer, intent(in) :: nfcns
    character (len=*) :: names(:)
    logical, intent(out) :: gval_is_xa_function(:) ! (nfcns)
    real(dp), pointer :: vals1(:)
    real(dp), pointer :: vals(:,:)
    integer, intent(out) :: ierr
    integer :: k, j
    real(dp) :: weight, gradT_across_kj, gradmu_across_kj
    real(dp) :: delta_grada_across_kj, delta_gradr_across_kj
    type (star_info), pointer :: s
    gval_is_xa_function(1:nfcns) = .false.
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) then
       call mesa_error(__FILE__,__LINE__,"problem other_mesh_function resolve_gradients")
    end if

    vals(1:s%nz,1:nfcns) => vals1(1:s%nz*nfcns)

    ! initialize to null
    vals(1:s%nz, 1) = 0.0d0

    names(1) = "nabla_T_across_hp"
    names(2) = "nabla_mu_across_hp"
    names(3) = "delta_grada_across_hp"
    ! names(4) = "delta_gradr_across_hp"

    weight = s% x_ctrl(3) ! weight set in inlist1

    ! aim: resolve all these gradients interpolating across hp
    do k = 1, s%nz-1
       j = k+1
       do while (abs(s%r(k)-s%r(j)) <= s% scale_height(k) .and. j<= s%nz-1)
          j = j + 1 ! move one deeper
       end do
       ! now j is the index of the cell below cell k at distance
       ! equal to the local pressure scale heigh at cell k
       gradT_across_kj = (s% xh(s% i_lnT, k) - s%xh(s% i_lnT, j))/(s% lnPeos(k) - s%lnPeos(j))
       gradmu_across_kj = (log(s% mu(k)) - log(s%mu(j)))/(s% lnPeos(k) - s%lnPeos(j))
       delta_grada_across_kj = abs(s% grada(k) - s%grada(j))
       delta_gradr_across_kj = abs(s% gradr(k) - s%gradr(j))
       vals(k, 1) = weight * gradT_across_kj
       vals(k, 2) = weight * gradmu_across_kj
       vals(k, 3) = weight * delta_grada_across_kj
       ! vals(k, 4) = weight * delta_gradr_across_kj
    end do

    ! enforce a floor: gradients are un-resolvable in 1D!
    ! vals(1:s%nz, 1:nfcns) = max(vals(1:s%nz, 1:nfcns), 1d-4)
    ! print *, "pre floor", minval(vals(1:s%nz, 1:nfcns)), maxval(vals(1:s%nz, 1:nfcns))
    ! where (abs(vals(1:s%nz, 1:nfcns)) < 1.0d-4) vals(1:s%nz, 1:nfcns) = sign(1.0d-4, vals(1:s%nz, 1:nfcns))
    ! where (abs(vals(1:s%nz, 1:nfcns)) > 5.0d3) vals(1:s%nz, 1:nfcns) = sign(5.0d3, vals(1:s%nz, 1:nfcns))
    ! print *, "post floor", minval(vals(1:s%nz, 1:nfcns)), maxval(vals(1:s%nz, 1:nfcns))
  end subroutine resolve_gradients

  integer function get_index_no_v(id, ierr) result(k_no_v)
    integer, intent(in) :: id
    integer, intent(out) :: ierr
    real(dp) :: t_sound_cross
    integer :: k, k0
    type (star_info), pointer :: s
    ! initialize at surface
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) then
       call mesa_error(__FILE__,__LINE__,"problem finding index for v expunging")
    end if
    ! initialize indexes at surface
    k_no_v = 1
    k0 = 1

    ! hack to turn this off from inlist
    if (s% x_ctrl(1)<0) return


    ! if O depletion, start at CO core mass
    if ((s%lxtra(3)) .and. (minval(s% entropy(1:s%nz)) < s%x_ctrl(2))) &
         k0 = min(minloc(abs(s%m(1:s%nz)/Msun - s%co_core_mass), dim=1), &
                  minloc(abs(s% entropy(1:s%nz)- s%x_ctrl(2)), dim=1))

    ! ! if past silicon depletion, owerwrite and start from Si core
    ! if (s% lxtra(4) .eqv. .true.) then
    !    k = k0 ! from previous inner boundary (surface or CO core)
    !    do while ((s% xa(s% net_iso(isi28), k) <= s% min_boundary_fraction) .and. &
    !         ((s% xa(s% net_iso(io16), k) >= 0.1d0)) .and. k<s%nz)
    !       k = k + 1 ! one cell deep
    !    end do
    !    ! if not reached center, we are at si core outer boundary
    !    if (k < s% nz) k0 = k ! update
    ! end if

    ! find point in sonic contact with core edge
    t_sound_cross = 0.0d0
    k_no_v=k0 ! start at core edge
    do while ((t_sound_cross < s%x_ctrl(1) * s%dt) .and. (k_no_v >1))
       t_sound_cross = t_sound_cross + (s%r(k_no_v-1)-s%r(k_no_v))/s%csound(k_no_v)
       k_no_v = k_no_v - 1 ! loop outward
    end do
  end function get_index_no_v

  subroutine extras_startup(id, restart, ierr)
    integer, intent(in) :: id
    logical, intent(in) :: restart
    integer, intent(out) :: ierr
    type (star_info), pointer :: s
    integer :: k
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
  end subroutine extras_startup

  integer function extras_start_step(id)
    integer, intent(in) :: id
    integer :: k_no_v, ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    extras_start_step = 0

    if (s% lxtra(1) .eqv. .false.) then
       if (log10(exp(maxval(s% xh(s% i_lnT, 1:s% nz)))) > 8.95d0) then
          write(*,*) "Found layer with logT>8.95"
          s% lxtra(1) = .true.
       end if
    end if

    if (s% lxtra(6) .eqv. .false.) then
      if (s% xa(s% net_iso(ih1), s%nz) <= 1d-6) then
         write(*,*) "Hydrogen depletion!"
         s% lxtra(6) = .true.
      end if
    end if

    if (s% lxtra(5) .eqv. .false.) then
      if ((s% xa(s% net_iso(ihe4), s%nz) <= 5d-3) .and. &
          (s% xa(s% net_iso(ih1), s%nz) <= 0.5d0)) then
         write(*,*) "Helium depletion!"
         s% lxtra(5) = .true.
      end if
    end if

    if (s% lxtra(2) .eqv. .false.) then
      if ((s% xa(s% net_iso(ic12), s%nz) <= 1d-3) .and. &
           (s% xa(s% net_iso(ihe4), s%nz) <= 5d-3) .and. &
           (s% xa(s% net_iso(ih1), s%nz) <= 0.5d0)) then
         write(*,*) "Carbon depletion!"
         s% lxtra(2) = .true.
      end if
    end if

    if (s% lxtra(3) .eqv. .false.) then
       if ((s% xa(s% net_iso(io16), s% nz) <= 0.1d0) .and. &
            (s% xa(s% net_iso(ic12), s%nz) <= 1d-3) .and. &
            (s% xa(s% net_iso(ihe4), s%nz) <= 5d-3) .and. &
            (s% xa(s% net_iso(ih1), s%nz) <= 0.5d0)) then
          write(*,*) "Oxygen depletion!"
          s% lxtra(3) = .true.
          ! get some more terminal output
          s% num_trace_history_values = 2
          s% trace_history_value_name(1) = 'non_fe_core_infall'
          s% trace_history_value_name(2) = 'rel_E_err'
       end if
    end if

    if (s% lxtra(4) .eqv. .false.) then
       if ((s% xa(s% net_iso(isi28), s% nz) <= 5d-3) .and. &
            (s% xa(s% net_iso(io16), s%nz) <= 5d-3) .and. &
            (s% xa(s% net_iso(ic12), s%nz) <= 5d-3) .and. &
            (s% xa(s% net_iso(ihe4), s%nz) <= 0.2d0) .and. &
            (s% xa(s% net_iso(ih1), s%nz) <= 0.5d0)) then
          write(*,*) "Silicon depletion!"
          s% lxtra(4) = .true.
          ! get some more terminal output
          s% num_trace_history_values = 4
          s% trace_history_value_name(1) = 'Fe_core'
          s% trace_history_value_name(2) = 'fe_core_infall'
          s% trace_history_value_name(3) = 'non_fe_core_infall'
          s% trace_history_value_name(4) = 'rel_E_err'
          ! force saving profiles
          s% write_profiles_flag = .true.
          s% profile_interval = 1
       end if
    end if
    k_no_v = get_index_no_v(id, ierr)
    if (k_no_v > 1) then
       ! set new velocity_q_upper_boundx for current timestep
       s% velocity_q_upper_bound = s%q(k_no_v)
    else
       ! reset dummy limit
       s% velocity_q_upper_bound = 1d99
    end if
  end function extras_start_step

  ! returns either keep_going, retry, backup, or terminate.
  integer function extras_check_model(id)
    integer, intent(in) :: id
    integer :: ierr, k
    real(dp) :: dlnL_dt, dlnTeff_dt, dlnR_dt
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    extras_check_model = keep_going

    ! nasty numerical spikes occur at constant R
    ! check for large dL/dt and/or dTeff/dt at small dR/dt
    ! and retry if you find one
    dlnL_dt = (log(s%xh(s%i_lum, 1)) - log(s%xh_old(s%i_lum, 1)))/s%dt
    dlnTeff_dt = (s%xh(s%i_lnT, 1) - s%xh_old(s%i_lnT, 1))/s%dt
    dlnR_dt =  (s%xh(s%i_lnR, 1) - s%xh_old(s%i_lnR, 1))/s%dt

    ! the control below should be ~0 because of the black body relation
    s% xtra(1) = (dlnL_dt - 4*dlnTeff_dt)/dlnR_dt - 2.0d0

    ! ! if max logT>8.95 and deviation from BB dev larger than x_ctrl(4) retry
    ! if (s%lxtra(1) .and. abs(s%xtra(1)) >= s%x_ctrl(4)) then
    !        print *, "test",  s%xtra(1), "large!RETRY"
    !        extras_check_model = retry
    ! end if

    if (extras_check_model == terminate) s% termination_code = t_extras_check_model
  end function extras_check_model

  integer function how_many_extra_history_columns(id)
    integer, intent(in) :: id
    integer :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    how_many_extra_history_columns = 4
  end function how_many_extra_history_columns

  subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
    integer, intent(in) :: id, n
    character (len=maxlen_history_column_name) :: names(n)
    real(dp) :: vals(n)
    integer, intent(out) :: ierr
    type (star_info), pointer :: s
    integer :: k
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return

    ! find location above which we damp v
    k = get_index_no_v(id, ierr)
    names(1) = 'k_no_v_above'
    vals(1) = k
    names(2) = 'q_no_v_above'
    vals(2) = s%q(k)
    names(3) = 'tau_no_v_above'
    vals(3) = s%tau(k)
    names(4) = 'BB_numerical_deviation'
    vals(4) = s%xtra(1)

  end subroutine data_for_extra_history_columns

  integer function how_many_extra_profile_columns(id)
    integer, intent(in) :: id
    integer :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    how_many_extra_profile_columns = 0
  end function how_many_extra_profile_columns

  subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
    integer, intent(in) :: id, n, nz
    character (len=maxlen_profile_column_name) :: names(n)
    real(dp) :: vals(nz,n)
    integer, intent(out) :: ierr
    type (star_info), pointer :: s
    integer :: k
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
  end subroutine data_for_extra_profile_columns


  integer function how_many_extra_history_header_items(id)
    integer, intent(in) :: id
    integer :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    how_many_extra_history_header_items = 0
  end function how_many_extra_history_header_items


  subroutine data_for_extra_history_header_items(id, n, names, vals, ierr)
    integer, intent(in) :: id, n
    character (len=maxlen_history_column_name) :: names(n)
    real(dp) :: vals(n)
    type(star_info), pointer :: s
    integer, intent(out) :: ierr
    ierr = 0
    call star_ptr(id,s,ierr)
    if(ierr/=0) return
  end subroutine data_for_extra_history_header_items


  integer function how_many_extra_profile_header_items(id)
    integer, intent(in) :: id
    integer :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    how_many_extra_profile_header_items = 0
  end function how_many_extra_profile_header_items


  subroutine data_for_extra_profile_header_items(id, n, names, vals, ierr)
    integer, intent(in) :: id, n
    character (len=maxlen_profile_column_name) :: names(n)
    real(dp) :: vals(n)
    type(star_info), pointer :: s
    integer, intent(out) :: ierr
    ierr = 0
    call star_ptr(id,s,ierr)
    if(ierr/=0) return
  end subroutine data_for_extra_profile_header_items


  ! returns either keep_going or terminate.
  ! note: cannot request retry or backup; extras_check_model can do that.
  integer function extras_finish_step(id)
    integer, intent(in) :: id
    real(dp) :: m_infall, dt_div_kh, Ysurf_div_Ycntr, Xcntr, omega, omega_crit
    integer :: ierr, k, k_sonic
    character (len=200) :: fname
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
    extras_finish_step = keep_going

    dt_div_kh = s% dt / (s% kh_timescale)
    Xcntr = s% xa(s% net_iso(ih1), s% nz)
    Ysurf_div_Ycntr = s% xa(s% net_iso(ihe4), 1) &
                    / s% xa(s% net_iso(ihe4), s% nz)
    omega = s% omega_avg_surf
    omega_crit = s% omega_crit_avg_surf

    if ((s% lxtra(1) .eqv. .true.) .and. &
         (s% lxtra(11) .eqv. .false.)) then
       s% lxtra(11) = .true. ! avoid getting in here again
       print *, "save first timestep for max(T)>10^8.95K"
       write(fname, fmt="(a15)") 'CHE_logT895.mod'
       call star_write_model(id, fname, ierr)
       write(fname, fmt="(a17)") 'CHE_logT895.photo'
       call star_write_photo(id, trim(s%photo_directory)//'/'//trim(fname), ierr)
       write(fname, fmt="(a16)") 'CHE_logT895.data'
       call star_write_profile_info(id, trim(s%log_directory)//'/'//trim(fname), ierr)
    end if

    if ((s% lxtra(6) .eqv. .true.) .and. &
        (s% lxtra(16) .eqv. .false.)) then
       s% lxtra(16) = .true. ! avoid getting in here again
       print *, "save first timestep post H core depletion"
       write(fname, fmt="(a10)") 'H_depl.mod'
       call star_write_model(id, fname, ierr)
       write(fname, fmt="(a12)") 'H_depl.photo'
       call star_write_photo(id, trim(s%photo_directory)//'/'//trim(fname), ierr)
       write(fname, fmt="(a11)") 'H_depl.data'
       call star_write_profile_info(id, trim(s%log_directory)//'/'//trim(fname), ierr)
    end if
    
    if ((s% lxtra(5) .eqv. .true.) .and. &
        (s% lxtra(15) .eqv. .false.)) then
       s% lxtra(15) = .true. ! avoid getting in here again
       print *, "save first timestep post He core depletion"
       write(fname, fmt="(a11)") 'He_depl.mod'
       call star_write_model(id, fname, ierr)
       write(fname, fmt="(a13)") 'He_depl.photo'
       call star_write_photo(id, trim(s%photo_directory)//'/'//trim(fname), ierr)
       write(fname, fmt="(a12)") 'He_depl.data'
       call star_write_profile_info(id, trim(s%log_directory)//'/'//trim(fname), ierr)
    end if

    if ((s% lxtra(2) .eqv. .true.) .and. &
        (s% lxtra(12) .eqv. .false.)) then
       s% lxtra(12) = .true. ! avoid getting in here again
       print *, "save first timestep post C core depletion"
       write(fname, fmt="(a10)") 'C_depl.mod'
       call star_write_model(id, fname, ierr)
       write(fname, fmt="(a12)") 'C_depl.photo'
       call star_write_photo(id, trim(s%photo_directory)//'/'//trim(fname), ierr)
       write(fname, fmt="(a11)") 'C_depl.data'
       call star_write_profile_info(id, trim(s%log_directory)//'/'//trim(fname), ierr)
    end if

    if ((s% lxtra(3) .eqv. .true.) .and. &
        (s% lxtra(13) .eqv. .false.)) then
       s% lxtra(13) = .true. ! avoid getting in here again
       print *, "save first timestep post O core depletion"
       write(fname, fmt="(a10)") 'O_depl.mod'
       call star_write_model(id, fname, ierr)
       write(fname, fmt="(a12)") 'O_depl.photo'
       call star_write_photo(id, trim(s%photo_directory)//'/'//trim(fname), ierr)
       write(fname, fmt="(a11)") 'O_depl.data'
       call star_write_profile_info(id, trim(s%log_directory)//'/'//trim(fname), ierr)
       ! save more output
       s% photo_interval = 100
       s% pg% pgstar_interval = 10
       s% terminal_interval = 1
       s% write_header_frequency = 1
       s% profile_interval = 1
    end if

    if ((s% lxtra(4) .eqv. .true.) .and. &
        (s% lxtra(14) .eqv. .false.)) then
       s% lxtra(14) = .true. ! avoid getting in here again
       print *, "save first timestep post Si core depletion"
       write(fname, fmt="(a11)") 'Si_depl.mod'
       call star_write_model(id, fname, ierr)
       write(fname, fmt="(a13)") 'Si_depl.photo'
       call star_write_photo(id, trim(s%photo_directory)//'/'//trim(fname), ierr)
       write(fname, fmt="(a12)") 'Si_depl.data'
       call star_write_profile_info(id, trim(s%log_directory)//'/'//trim(fname), ierr)
       ! save more output
       s% photo_interval = 50
       s% pg% pgstar_interval = 1
       s% terminal_interval = 1
       s% write_header_frequency = 1
       s% profile_interval = 1
       s% pg% Profile_Panels5_file_flag = .true.
    end if

    ! avoid diverging models running for too long
    if (dt_div_kh .le. s% x_ctrl(4)) then
      write(*, '(/,a,/, 99e20.10)') &
               'stop because dt_div_kh <= dt_div_kh_minimum', &
               dt_div_kh, s% x_ctrl(4)
      extras_finish_step = terminate
    end if

    ! avoid evolving non-CHE systems
    if ((Ysurf_div_Ycntr < 0.7) .and. &
       (Xcntr > 1d-7)) then
      write(*, '(/,a,/, 99e20.10)') &
               'stop because Ysurf/Ycntr < 0.7', &
               Ysurf_div_Ycntr, 0.7
      extras_finish_step = terminate
    end if

    ! avoid models critically rotating at ZAMS in SSE
    ! they will crash in SSE
    if ((omega > 0.) .and. &
        (s% lxtra(20) .eqv. .false.))then
       s% lxtra(20) = .true. ! avoid getting in here again
       if (omega >= omega_crit) then
          write(*, *) 'stop because first omega > omega_crit', &
                      omega, omega_crit
          extras_finish_step = terminate
       end if
    end if

    ! post Si depletion, check velocity
    if ((s% lxtra(14) .eqv. .true.) .and. &
        (s% fe_core_mass > 0.0d0)) then
       k = s%nz
       do while (s% m(k) <= s% fe_core_mass * Msun)
          k = k-1 ! loop outwards
       end do
       ! k is now the outer index of the fe core
       if (maxval(abs(s%v(k:s%nz))) >= s% fe_core_infall_limit) then
          s% termination_code = t_fe_core_infall_limit
          write(*, '(/,a,/, 99e20.10)') &
               'stop because fe_core_infall > fe_core_infall_limit', &
               s% fe_core_infall, s% fe_core_infall_limit
          print *, "treshold v used", maxval(abs(s%v(k:s%nz)))
          extras_finish_step = terminate
       end if
    end if

  end function extras_finish_step


  subroutine extras_after_evolve(id, ierr)
    integer, intent(in) :: id
    integer, intent(out) :: ierr
    type (star_info), pointer :: s
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return
  end subroutine extras_after_evolve

  
  subroutine che_wind(id, Lsurf, Msurf, Rsurf, Tsurf, X, Y, Z, w, ierr)
    use star_def
    include 'formats'
    type (star_info), pointer :: s
    integer, intent(in) :: id
    real(dp), intent(in) :: Lsurf, Msurf, Rsurf, Tsurf, X, Y, Z ! surface values (cgs)
    ! NOTE: surface is outermost cell. not necessarily at photosphere.
    ! NOTE: don't assume that vars are set at this point.
    ! so if you want values other than those given as args,
    ! you should use values from s% xh(:,:) and s% xa(:,:) only.
    ! rather than things like s% Teff or s% lnT(:) which have not been set yet.
    real(dp), intent(out) :: w ! wind in units of Msun/year (value is >= 0)
    integer, intent(out) :: ierr
    
    character(len=strlen) :: wind_scheme, vms_wind_transition_scheme
    real(dp) :: X0, dX, beta_extra, gamma_extra, mdot_r_extra, &
                Z_div_Zsun, G_e, vink2001_w, bjorklund2023_w, &
                krticka2024_w, vink2011_w, vink2017_w, sander2023_w, &
                he_poor_thin_w, he_poor_thick_w, he_rich_w, he_poor_w, &
                G_switch, M_switch, he_poor_a
    real(dp), parameter :: Zsun = 0.017d0

    logical, parameter :: dbg = .false.
    
    ierr = 0
    call star_ptr(id, s, ierr)
    if (ierr /= 0) return

    wind_scheme = s% x_character_ctrl(1)
    vms_wind_transition_scheme = s% x_character_ctrl(2)
    X0 = s% x_ctrl(5)
    dX = s% x_ctrl(6)
    beta_extra = s% x_ctrl(7)
    gamma_extra = s% x_ctrl(8)
    mdot_r_extra = s% x_ctrl(9)
    Z_div_Zsun = Z/Zsun
    G_e = exp10(-4.813d0) * (1+X) * (Lsurf/Lsun)/(Msurf/Msun)

    vink2001_w = 0d0
    bjorklund2023_w = 0d0
    krticka2024_w = 0d0
    vink2011_w = 0d0
    vink2017_w = 0d0
    sander2023_w = 0d0
    w = 0d0

    ! No wind mass loss before ZAMS
    if (s% lxtra(21) .eqv. .false.) then
      if (s% L_nuc_burn_total/(Lsurf/Lsun) >= s% Lnuc_div_L_zams_limit) then
        ! do not check after ZAMS
        ! avoid turning winds off after core H exh
        s% lxtra(21) = .true. 
      else
        return
      end if
    end if   

    ! He-poor thin winds
    if (wind_scheme == 'vink') then
      call eval_Vink2001_wind(vink2001_w)
      he_poor_thin_w = vink2001_w
      if (dbg) write(*,*) 'Vink2001_wind', he_poor_thin_w
    else if (wind_scheme == 'bjorklund') then
      call eval_Bjorklund2023_wind(bjorklund2023_w)
      he_poor_thin_w = bjorklund2023_w
      if (dbg) write(*,*) 'Bjorklund2023_wind', he_poor_thin_w
    else if (wind_scheme == 'krticka') then
      call eval_Krticka2024_wind(krticka2024_w)
      he_poor_thin_w = krticka2024_w
      if (dbg) write(*,*) 'Krticka2024_wind', he_poor_thin_w
    else if (wind_scheme == 'bjorklund+krticka') then
      call eval_Bjorklund2023_wind(bjorklund2023_w)
      call eval_Krticka2024_wind(krticka2024_w)
      he_poor_thin_w = min(bjorklund2023_w, krticka2024_w)
      if (dbg) write(*,*) 'Bjorklund2023_wind, Krticka2024_wind', bjorklund2023_w, krticka2024_w
      if (dbg) write(*,*) 'min(Bjorklund2023_wind, Krticka2024_wind)', he_poor_thin_w
    else
      ierr = -1
      if (dbg) write(*,*) 'unknown name for wind scheme: ' // trim(wind_scheme)
      if (dbg) call mesa_error(__FILE__,__LINE__,'debug: bad value for wind scheme')
      return
    end if  
    
    ! He-poor thick winds
    ! once G_switch is crossed, stop computing it
    if (s% lxtra(22) .eqv. .false.) then
      call eval_G_switch(G_switch)
    else
      G_switch = s% xtra(2)
    end if

    ! check if G_switch has been crossed
    if (G_e >= G_switch) then
      if (s% lxtra(22) .eqv. .false.) then
        s% lxtra(22) = .true. ! avoid getting in here again
        s% xtra(2) = G_switch
        s% xtra(3) = Msurf ! store M_switch
      end if
      M_switch = s% xtra(3)
    end if

    ! compute thick winds only after G_switch checks
    call eval_Vink2011_wind(vink2011_w)
    he_poor_thick_w = vink2011_w
    if (dbg) write(*,*) 'Vink2011_wind', vink2011_w

    if (G_e < G_switch) then
      he_poor_w = he_poor_thin_w
    else
      he_poor_w = he_poor_thick_W
    end if

    ! He-rich winds
    call eval_Vink2017_wind(vink2017_w) ! thin winds
    call eval_Sander2023_wind(sander2023_w) ! thick winds
    if (dbg) write(*,*) 'Vink2017_wind, Sander2023_wind', vink2017_w, sander2023_w

    he_rich_w = max(vink2017_w, sander2023_w) &
                * pow(Msurf/Msun/30d0, beta_extra) &
                * pow(Z_div_Zsun, gamma_extra) &
                * mdot_r_extra
    if (dbg) write(*,*) 'he_rich_w', he_rich_w
        
    if (X > X0) then
      he_poor_a = 1d0
    else if (X > X0 - dX) then
      he_poor_a = (X - (X0 - dX)) / dX
    else
      he_poor_a = 0d0
    end if
   
    ! total winds
    if (dbg) write(*,*) 'G_e, G_switch', G_e, G_switch
    if (dbg) write(*,*) 'he_poor_w', he_poor_w
    if (dbg) write(*,*) 'he_poor_a', he_poor_a
    w = he_poor_a * he_poor_w + (1d0-he_poor_a) * he_rich_w

    ierr = 0
    
    contains

    subroutine eval_Krticka2024_wind(w)
      real(dp), intent(inout) :: w
      real(dp) :: log_Z_div_Zsun, log_lin_term, gaussian_mix_term, log_mdot

      log_Z_div_Zsun = log10(Z_div_Zsun)
      log_lin_term = - 13.82d0 &
                     + 0.358d0 * log_Z_div_Zsun &
                     + (1.52d0 - 0.11d0 * log_Z_div_Zsun) * log10(Lsurf/Lsun/1d6)
      gaussian_mix_term = + (1d0 + 0.73d0 * log_Z_div_Zsun) &
                            * exp(-pow2(Tsurf-1.416d4))/pow2(3.58d3) &
                          + 3.84d0 & 
                            * exp(-pow2(Tsurf-3.79d4)/pow2(5.65d4))
      log_mdot = log_lin_term + 13.82d0 * log10(gaussian_mix_term)
      w = exp10(log_mdot)

    end subroutine eval_Krticka2024_wind


    subroutine eval_Bjorklund2023_wind(w)
      ! Bjorklund et al. (2023)
      ! Thin winds from MS and early post-MS O stars
      real(dp), intent(inout) :: w
      real(dp) :: Meff, log_mdot

      Meff = Msurf * (1d0-G_e)
      log_mdot = - 5.52d0 &
                 + 2.39d0 * log10(Lsurf/Lsun/1d6) &
                 - 1.48d0 * log10(Meff/Msun/45d0) &
                 + 2.12d0 * log10(Tsurf/4.5d4) & 
                 + (0.75d0 - 1.87d0 * log10(Tsurf/4.5d4)) * log10(Z_div_Zsun)
      w = exp10(log_mdot)

    end subroutine eval_Bjorklund2023_wind

    subroutine eval_G_switch(G_switch)
      ! Fits over G_switch calculated according to Sabhabhit et al. (2023)
      ! For fixed Teff = 45 kK and Xsurf = 0.7
      real(dp), intent(inout) :: G_switch

      if (vms_wind_transition_scheme == 'vink') then
        G_switch = 0.243d0 * pow(Z_div_Zsun, -0.541d0)
      else if (vms_wind_transition_scheme == 'krticka') then
        G_switch = 0.585d0 * pow(Z_div_Zsun, -0.016d0)
      else if (vms_wind_transition_scheme == 'bjorklund') then
        G_switch = 0.838d0 * pow(Z_div_Zsun, -0.099d0)
      else if (vms_wind_transition_scheme == 'bjorklund+krticka') then
        if (krticka2024_w <= bjorklund2023_w) then
          G_switch = 0.585d0 * pow(Z_div_Zsun, -0.016d0)
        else
          G_switch = 0.838d0 * pow(Z_div_Zsun, -0.099d0)
        end if
      else
        ierr = -1
        if (dbg) write(*,*) 'unknown name for wind scheme: ' // trim(wind_scheme)
        if (dbg) call mesa_error(__FILE__,__LINE__,'debug: bad value for wind scheme')
        return
      end if

    end subroutine eval_G_switch

    subroutine eval_Vink2011_wind(w)
      ! Vink et al. (2011)
      ! Very massive star/thick Main Sequence winds
      real(dp), intent(inout) :: w
      real(dp) :: v01_w

      v01_w = 0d0
      if (dbg) write(*,*) 'vink2011 v01w', v01_w
      call eval_Vink2001_wind(v01_w)
      if (dbg) write(*,*) 'vink2011 v01w', v01_w

      if (G_e < G_switch) then
        w = v01_w
        if (dbg) write(*,*) 'thin ms winds', v01_w
      else
        if (dbg) write(*,*) 'thick ms winds ge gswitch', G_e, G_switch
        w = v01_w * pow(Msurf/M_switch, 0.78d0) * pow(G_e/G_switch, 4.77d0)
      end if

    end subroutine eval_Vink2011_wind

    subroutine eval_Sander2023_wind(w)
      ! Sanders & Vink (2020) (base)
      ! Sanders et al. (2023) (temperature correction)
      real(dp), intent(inout) :: w
      real(dp) :: logz, alpha, l_0, mdot_10, log_w, log_power_term

      ! winds for Teff = 141 kK from Sanders & Vink (2020)
      ! recipe: eq 14
      ! parameters: eqs 18-20
      logz = log10(Z/Zsun)
      alpha = 0.32d0*logz + 1.4d0
      l_0 = exp10(-0.87d0*logz + 5.06d0)  ! Lsun
      mdot_10 = exp10(-0.75d0*logz - 4.06d0)  ! Msun yr-1

      if (Lsurf/Lsun .lt. l_0) then
        log_power_term = 0d0
      else
        log_power_term = pow(log10(Lsurf/Lsun/l_0), alpha)
      end if

      w = mdot_10 * log_power_term * pow(Lsurf/Lsun/l_0/10d0, 0.75d0)
      ! temperature correction
      ! eq 18 of Sanders et al. (2023)
      if ((Tsurf .gt. 1d5) .and. (Lsurf/Lsun .gt. l_0)) then
        log_w = log10(w) - 6*log10(Tsurf/1.41d5)
        w = exp10(log_w)
      end if

    end subroutine eval_Sander2023_wind

    subroutine eval_Vink2017_wind(w)
      ! Vink (2017)
      ! Winds for stripped stars, i.e., "lower-mass He stars"
      ! Thin winds; not adequate for classical WR thick winds
      real(dp), intent(inout) :: w
      real(dp) :: log_mdot

      ! equation 1
      log_mdot = - 13.3d0 &
                 + 1.36d0 * log10(Lsurf/Lsun) &
                 + 0.61d0 * log10(Z/Zsun)
      w = exp10(log_mdot)
    end subroutine eval_Vink2017_wind

    subroutine eval_Vink2001_wind(w)
      ! Vink, de Koter & Lamers (2001)
      ! Winds for MS O/B stars
      real(dp), intent(inout) :: w
      real(dp) :: a, dT, Teff_jump, vinf_div_vesc, log_mdot, mdot1, mdot2

      ! use Vink et al 2001, eqns 14 and 15 to set "jump" temperature
      Teff_jump = 1d3*(61.2d0 + 2.59d0*(-13.636d0 + 0.889d0*log10(Z_div_Zsun)))
      
      if (Tsurf > 27.5d3) then
        a = 1d0
      else if (Tsurf < 22.5d3) then
        a = 0d0
      else
        ! the dT parameter sets the rate at which the winds transition
        ! from the cool to the hot regime around the bi-stability jump
        dT = 100d0
        if (Tsurf > Teff_jump + dT) then
          a = 1d0
        else if (Tsurf < Teff_jump - dT) then
          a = 0d0
        else
          a = (Tsurf - (Teff_jump-dT)) / (2*dT)
        end if
      end if

      if (a > 0) then ! eval hot side wind (eq 24)
        vinf_div_vesc = 2.6d0 ! Galactic value
        ! metallicity rescaling based on Leitherer+92
        vinf_div_vesc = vinf_div_vesc * pow(Z_div_Zsun, 0.13d0)
        log_mdot = - 6.697d0 &
                   + 2.194d0 * log10(Lsurf/Lsun/1d5) &
                   - 1.313d0 * log10(Msurf/Msun/3d1) & 
                   - 1.226d0 * log10(vinf_div_vesc/2d0) &
                   + 0.933d0 * log10(Tsurf/4d4) &
                   - 10.92d0 * pow2(log10(Tsurf/4d4)) &
                   + 0.85d0 * log10(Z/Zsun)
        mdot1 = exp10(log_mdot)
      else
        mdot1 = 0d0
      end if

      if (a < 1) then ! eval cool side wind (eq 25)
        vinf_div_vesc = 1.3d0 ! Galactic value
        ! metallicity rescaling based on Leitherer+92
        vinf_div_vesc = vinf_div_vesc * pow(Z_div_Zsun, 0.13d0)
        log_mdot = - 6.688d0 &
                   + 2.210d0 * log10(Lsurf/Lsun/1d5) &
                   - 1.339d0 * log10(Msurf/Msun/3d1) &
                   - 1.601d0 * log10(vinf_div_vesc/2) &
                   + 1.07d0 * log10(Tsurf/2d4) &
                   + 0.85d0 * log10(Z/Zsun)
        mdot2 = exp10(log_mdot)
      else
        mdot2 = 0d0
      end if

      w = a*mdot1 + (1 - a)*mdot2   
    end subroutine eval_Vink2001_wind

 end subroutine che_wind

end module run_star_extras
