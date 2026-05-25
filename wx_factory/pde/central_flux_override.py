"""
Drop-in central-flux Riemann override for PDEEulerCartesian.

Purpose
-------
Disable the dissipation that the AUSM Riemann solver bakes into the interface
flux, so the entire entropy-stable / entropy-conserving pipeline (ESAV
artificial viscosity + Crank-Nicolson + relaxation post-processor) is the
*only* source of numerical dissipation.

The Riemann flux for any consistent two-point flux can be written as
    f*(qL, qR) = 1/2 (f(qL) + f(qR))  -  D(qL, qR)
where D is the dissipation.  AUSM bundles f and D together, so to "switch off"
dissipation we replace the call to the AUSM kernel with the central flux
    f_central(qL, qR) = 1/2 (f(qL) + f(qR))
computed from the *physical* Euler flux at each interface solution point.

Usage
-----
After constructing PDEEulerCartesian, do

    from central_flux_override import patch_pde_with_central_flux
    patch_pde_with_central_flux(self.rhs.full.pde)   # or wherever the PDE lives

or, more permanently, subclass:

    class PDEEulerCartesianCentral(PDEEulerCartesian):
        riemann_fluxes = central_flux_riemann

Notes
-----
- This is a *non-dissipative* interface flux.  It is unstable on its own; the
  ESAV term must provide dissipation, otherwise the scheme will blow up on
  anything but very smooth, very short-time runs.
- Test case 100 (smooth sine density wave, uniform velocity) is the right
  case for verification.  Test 102 (Riemann problem) will fail with this
  flux + insufficient ESAV.
- Array layout assumed:
      q_itf_x1.shape = (4, num_elem_x2, num_elem_x1, 2*num_solpts)
      last axis: [0:num_solpts]   = WEST face of the element
                 [num_solpts:2*]  = EAST face of the element
  Same for x3 with WEST -> DOWN, EAST -> UP.  This matches both interface.cpp
  and PDEEulerCartesian.entropy_average.
"""

import numpy

from common.definitions import (
    cpd, cvd, p0, Rd,
    idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,
)


def _physical_flux_x1(q_face):
    """Physical Euler flux in the x1 direction, evaluated at a face.

    q_face shape: (4, ...) with components (rho, rho*u, rho*w, rho*theta).
    Returns f shape: (4, ...).
    """
    try:
        xp = q_face.__array_namespace__()
    except AttributeError:
        xp = numpy

    gamma = cpd / cvd

    rho   = q_face[idx_2d_rho]
    rhou  = q_face[idx_2d_rho_u]
    rhow  = q_face[idx_2d_rho_w]
    rhotheta = q_face[idx_2d_rho_theta]

    u = rhou / rho
    w = rhow / rho
    p = p0 * (Rd * rhotheta / p0) ** gamma

    f = xp.empty_like(q_face)
    f[idx_2d_rho]       = rhou
    f[idx_2d_rho_u]     = rhou * u + p
    f[idx_2d_rho_w]     = rhou * w
    f[idx_2d_rho_theta] = rhotheta * u
    return f


def _physical_flux_x3(q_face):
    """Physical Euler flux in the x3 direction."""
    try:
        xp = q_face.__array_namespace__()
    except AttributeError:
        xp = numpy

    gamma = cpd / cvd

    rho   = q_face[idx_2d_rho]
    rhou  = q_face[idx_2d_rho_u]
    rhow  = q_face[idx_2d_rho_w]
    rhotheta = q_face[idx_2d_rho_theta]

    u = rhou / rho
    w = rhow / rho
    p = p0 * (Rd * rhotheta / p0) ** gamma

    f = xp.empty_like(q_face)
    f[idx_2d_rho]       = rhow
    f[idx_2d_rho_u]     = rhow * u
    f[idx_2d_rho_w]     = rhow * w + p
    f[idx_2d_rho_theta] = rhotheta * w
    return f


def central_flux_riemann(
    self,
    q_itf_x1, q_itf_x2, q_itf_x3,
    flux_itf_x1, flux_itf_x2, flux_itf_x3,
):
    """Replacement for PDEEulerCartesian.riemann_fluxes that uses the central
    (dissipation-free) flux instead of AUSM.

    Bound as a method (first arg `self` is the PDE instance).
    """
    del q_itf_x2, flux_itf_x2  # 2D: x2 unused

    num_solpts = self.geometry.num_solpts
    case = self.config.case_number
    periodic_x1 = case in (100, 101, 102)
    periodic_x3 = case in (100, 101, 102)

    try:
        xp = q_itf_x1.__array_namespace__()
    except AttributeError:
        xp = numpy

    # Index slices for "west half" and "east half" of an element along its
    # last axis (length 2*num_solpts).
    W = slice(0, num_solpts)            # west / down face
    E = slice(num_solpts, 2 * num_solpts)  # east / up   face

    # ---------------------------------------------------------------------
    # x1-direction: interface between element j (its EAST face) and j+1
    # (its WEST face).  Compute physical fluxes at both, then average.
    # ---------------------------------------------------------------------
    # Pointwise physical flux on every face of every element (cheap):
    f_x1_face = _physical_flux_x1(q_itf_x1)   # shape: (4, nez, nex, 2*nsp)

    # Left/right face values at each *interior* x1 interface:
    fL = f_x1_face[:, :, :-1, E]   # east face of element j (left of interface)
    fR = f_x1_face[:, :,  1:, W]   # west face of element j+1 (right of interface)
    f_central = 0.5 * (fL + fR)

    # Write the central flux into both sides of every interior interface.
    # Single-valued at the interface: same value stored on both adjacent faces.
    flux_itf_x1[:, :, :-1, E] = f_central
    flux_itf_x1[:, :,  1:, W] = f_central

    # Boundaries along x1
    if periodic_x1:
        # Element -1 east face <-> element 0 west face
        fL_b = f_x1_face[:, :, -1, E]
        fR_b = f_x1_face[:, :,  0, W]
        f_b  = 0.5 * (fL_b + fR_b)
        flux_itf_x1[:, :, -1, E] = f_b
        flux_itf_x1[:, :,  0, W] = f_b
    else:
        # Reflective wall: f* equals the *interior* physical flux with the
        # pressure terms only (velocity contribution cancels because u_wall=0
        # for the mirror state).  Simplest consistent choice: keep the
        # interior physical flux as the "boundary flux" so wall-normal
        # momentum receives only the pressure contribution.  For an inviscid
        # wall with the no-slip mirror states used in entropy_average, the
        # central average of (q, q_mirror) gives:
        #   - rho, rho*theta flux components: u_avg = 0 -> zero advective flux
        #   - rho*u, rho*w flux components: only pressure remains
        # We compute this exactly via the average.
        q_left_b  = q_itf_x1[:, :,  0, W].copy()
        q_right_b = q_itf_x1[:, :, -1, E].copy()
        # Mirror velocity at the wall (matches entropy_average BC)
        q_left_mirror  = q_left_b.copy()
        q_left_mirror[idx_2d_rho_u:idx_2d_rho_w+1] *= -1
        q_right_mirror = q_right_b.copy()
        q_right_mirror[idx_2d_rho_u:idx_2d_rho_w+1] *= -1
        flux_itf_x1[:, :,  0, W] = 0.5 * (
            _physical_flux_x1(q_left_b)  + _physical_flux_x1(q_left_mirror)
        )
        flux_itf_x1[:, :, -1, E] = 0.5 * (
            _physical_flux_x1(q_right_b) + _physical_flux_x1(q_right_mirror)
        )

    # ---------------------------------------------------------------------
    # x3-direction: interface between element i (its UP face) and i+1
    # (its DOWN face).
    # ---------------------------------------------------------------------
    f_x3_face = _physical_flux_x3(q_itf_x3)

    fD = f_x3_face[:, :-1, :, E]   # up   face of element i   (below interface)
    fU = f_x3_face[:,  1:, :, W]   # down face of element i+1 (above interface)
    f_central = 0.5 * (fD + fU)

    flux_itf_x3[:, :-1, :, E] = f_central
    flux_itf_x3[:,  1:, :, W] = f_central

    if periodic_x3:
        fD_b = f_x3_face[:, -1, :, E]
        fU_b = f_x3_face[:,  0, :, W]
        f_b  = 0.5 * (fD_b + fU_b)
        flux_itf_x3[:, -1, :, E] = f_b
        flux_itf_x3[:,  0, :, W] = f_b
    else:
        q_down_b = q_itf_x3[:,  0, :, W].copy()
        q_up_b   = q_itf_x3[:, -1, :, E].copy()
        q_down_mirror = q_down_b.copy()
        q_down_mirror[idx_2d_rho_u:idx_2d_rho_w+1] *= -1
        q_up_mirror = q_up_b.copy()
        q_up_mirror[idx_2d_rho_u:idx_2d_rho_w+1] *= -1
        flux_itf_x3[:,  0, :, W] = 0.5 * (
            _physical_flux_x3(q_down_b) + _physical_flux_x3(q_down_mirror)
        )
        flux_itf_x3[:, -1, :, E] = 0.5 * (
            _physical_flux_x3(q_up_b)   + _physical_flux_x3(q_up_mirror)
        )


def patch_pde_with_central_flux(pde):
    """Monkey-patch a PDEEulerCartesian instance to use the central flux."""
    import types
    pde.riemann_fluxes = types.MethodType(central_flux_riemann, pde)
    print("[central_flux_override] PDE Riemann flux replaced with central (dissipation-free) flux.")
