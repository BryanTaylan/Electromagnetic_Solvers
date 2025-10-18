import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

DX      = 20e-9          # grid step  
PAD     = 5.0e-6         # padding around core 
CORE_W  = 450e-9         # Si width
CORE_T  = 220e-9         # Si thickness 
AA_EPS  = True           # anti-aliased eps (sub-pixel fill) to reduce staircasing bias
VERBOSE = True
SAVE    = True

LAM_LIST  = [ 1310e-9, 1550e-9 ]                 
LAM_SWEEP = np.linspace( 1300e-9, 1600e-9, 13 )  

K_TE   = 10
K_TM   = 24
SIGMA_NEFF_TE = 2.60
SIGMA_NEFF_TM = 1.65     

TRACK_K = 5              # number of guided branches to track in sweep
TRACK_OVERLAP_TOL = 0.92 # min field overlap to accept a match

# Propagation test params
L_PROP  = 1e-3           # propagation length  
NZ_PROP = 200            # samples along z for the overlap curve

eps0 = 8.8541878128e-12
mu0  = 4e-7 * np.pi
c0   = 1 / np.sqrt(eps0 * mu0)

# Computes the refractive index n of SiO2 (fused silica) via a 3-term Sellmeier model.
# lam: Free-space wavelength in meters (scalar or array-like).
# verbose: If True, prints the intermediate n^2 value(s).
# Formula: n^2(lambda) = 1 + Σ_k B_k lambda^2 / (lambda^2 − C_k), with lambda in micrometers.
# Coeffs: B1=0.6961663, C1=(0.0684043)^2; B2=0.4079426, C2=(0.1162414)^2; B3=0.8974794, C3=(9.896161)^2.
# Accuracy: ~0.21–3.7 micrometers 
# Returns: n_SiO2 at lambda (unitless)
def n_SiO2(lam, verbose = False ):
    lam_um = lam * 1e6
    lam2 = lam_um**2
    B1, C1 = 0.6961663, 0.0684043**2
    B2, C2 = 0.4079426, 0.1162414**2
    B3, C3 = 0.8974794, 9.896161**2
    n2 = 1 + B1 * lam2 /( lam2 - C1 ) + B2 * lam2 / ( lam2 - C2 ) + B3 * lam2 / ( lam2 - C3 ) 
    if verbose: print( f"SiO2: n^2={n2}" )
    return np.sqrt( n2 )

_SI_LAM_UM = np.array( [ 1.30, 1.31, 1.40, 1.50, 1.55, 1.60 ] )
_SI_N      = np.array( [ 3.486, 3.482, 3.478, 3.474, 3.473, 3.471 ] )

# Interpolates the refractive index n of crystalline silicon (Si) from reference anchors.
# lam: Free-space wavelength in meters (scalar or 1D array-like).
# verbose: If True, prints the interpolated value(s).
# Anchors in micrometers: [1.30,1.31,1.40,1.50,1.55,1.60] to n=[3.486,3.482,3.478,3.474,3.473,3.471]
# Behavior: Uses np.interp on lam*1e6; clamps outside the tabulated range to endpoints.
# Returns: n_Si at lambda, float or array matching lam.
def n_Si( lam, verbose = False ):
    val = np.interp(lam * 1e6, _SI_LAM_UM, _SI_N)
    if verbose: print( f"Si: n={val}" )
    return val

# Builds a 2D, cell-centered Cartesian grid for the waveguide cross section.
# dx: Grid spacing in meters (uniform in x and y).
# pad: Padding distance (each side) around the core in meters.
# w, t: Core width (x-extent) and thickness (y-extent) in meters.
# verbose: If True, prints grid sizes and physical extents in microns.
# Domain: Lx = w + 2*pad, Ly = t + 2*pad. Ensures nx, ny are even.
# Coordinates: x, y are cell-center coordinates centered at the core.
# Returns: x (nx,), y (ny,), nx, ny.def make_grid(dx, pad, w, t, verbose=False):
    Lx = w + 2 * pad
    Ly = t + 2 * pad
    nx = int( np.ceil( Lx / dx ) )
    ny = int( np.ceil( Ly / dx ) )
    nx += nx % 2
    ny += ny % 2
    x = ( np.arange( nx ) - nx / 2 + 0.5 ) * dx
    y = ( np.arange( ny ) - ny / 2 + 0.5 ) * dx
    if verbose: 
      print( f"nx={nx}, ny={ny}, Lx={Lx*1e6:.2f}, Ly={Ly*1e6:.2f}" )
    return x, y, nx, ny

# Helper for anti-aliased fill: 1D overlap fraction of a pixel with a centered segment.
# centers: 1D array of pixel centers (meters) along an axis.
# dx: Pixel size (meters).
# half_span: Half the extent of the centered segment to fill (meters). segment is [-half_span, +half_span].
# Returns: ovl_frac (same shape as centers), each in [0,1].
def _overlap_fraction_1d( centers, dx, half_span ):
    left  = centers - 0.5 * dx
    right = centers + 0.5 * dx
    ovl = np.clip( np.minimum( right,  half_span ) - np.maximum( left, - half_span ), 0.0, dx )
    return ovl / dx

# Builds relative-permittivity/permeability maps with optional anti-aliased (AA) core fill.
# nx, ny: Grid sizes in x, y.
# dx: Grid spacing (meters).
# w, t: Core width/thickness (meters).
# n_core, n_clad: Refractive indices for core/cladding (unitless).
# verbose: If True, prints ε range and AA flag.
# antialias: If True, uses sub-pixel area fractions (reduces staircasing bias). core_mask := (fraction >= 0.5).
# Returns: eps (ny×nx, epsilon_r), mu ( ny x nx, mu_r = 1 ), core_mask (ny x nx bool).
def build_eps_mu(nx, ny, dx, w, t, n_core, n_clad, verbose = False, antialias = True ):
    x = ( np.arange( nx ) - nx/2 + 0.5 ) * dx
    y = ( np.arange( ny ) - ny/2 + 0.5 ) * dx
    if antialias:
        fx = _overlap_fraction_1d( x, dx, w / 2 )
        fy = _overlap_fraction_1d( y, dx, t / 2 )
        frac = np.outer( fy, fx )
        eps_core, eps_clad = n_core**2, n_clad**2
        eps = eps_clad + frac*( eps_core - eps_clad )
        core_mask = frac >= 0.5
    else:
        X, Y = np.meshgrid( x, y, indexing = 'xy' )
        core_mask = ( np.abs( X ) <= w/2) & ( np.abs( Y ) <= t/2 )
        eps = ( n_clad**2 ) * np.ones( ( ny, nx ), float )
        eps[ core_mask ] = ( n_core**2 )
    mu  = np.ones_like( eps, float)  # nonmagnetic
    if verbose:
        print( f"eps range: [{eps.min():.3f}, {eps.max():.3f}]  (AA={antialias} ) " )
    return eps, mu, core_mask

# Builds a boolean mask for interior cells (Dirichlet boundaries removed).
# nx, ny: Grid sizes.
# verbose: If True, prints interior vs total counts.
# Boundary rows/cols are False; interior (1..ny-2, 1..nx-2) are True.
# Returns: mask (ny×nx bool).
def interior_mask( nx, ny, verbose = False ):
    m = np.ones( ( ny, nx ), bool )
    m[ 0 , : ] = m[ -1 , : ] = m[ : , 0 ] = m [ : , -1 ] = False
    if verbose: 
      print( f"interior {m.sum()} / total {nx*ny}" )
    return m

# Builds a 1D backward-difference sparse matrix (first-order) on a uniform grid.
# n: Number of samples.
# dx: Grid spacing (meters).
# Definition: (Df)_i = (f_i − f_{i−1})/dx for i≥1; row 0 is zero (one-sided).
# Returns: D (n×n CSR).
def D1_back( n, dx ):
    data=list()
    rows=list()
    cols=list()
    invdx = 1.0 / dx
    for i in range( 1 , n ):
        rows += [ i, i ]
        cols += [ i, i - 1 ]
        data += [ invdx, -invdx ]
    return sp.csr_matrix( ( data, ( rows,cols ) ), shape = ( n , n ) )

# Lifts a 1D operator A to a 2D operator via Kronecker products (row-major/C-flattening).
# A: Square sparse matrix along an axis.
# nx, ny: Grid sizes.
# axis: 'x' to act along x. 'y' to act along y.
# Returns: CSR sparse matrix of size (nx*ny)×(nx*ny).
def kron_I( A, nx, ny, axis):
    if axis=='x':
        return sp.kron( sp.eye( ny, format = "csr" ), A, format = "csr" )
    else:
        return sp.kron( A, sp.eye( nx , format = "csr" ), format = "csr" )

# Builds vector FDFD operator blocks on interior DOFs for isotropic media (Et-formulation).
# eps, mu: Relative permittivity/permeability maps (ny×nx).
# dx: Grid spacing (meters).
# mask: Interior mask (ny×nx bool). DOFs kept where mask==True.
# Discretization: First-order backward differences; 2D ops via Kronecker; C-order flattening.
# Returns: Txx, Tyy, Tyx, eps_d — sparse blocks restricted to interior DOFs.
def build_vector_operator_blocks( eps, mu, dx, mask ):
    ny, nx = eps.shape
    keep = mask.ravel( order = "C" )
    N = nx*ny
    eps_d    = sp.diags( eps.ravel(), 0, shape = ( N , N ), format = "csr" )[ keep ][ : , keep ]
    inv_mu_d = sp.diags( ( 1.0 / mu ).ravel(), 0, shape = ( N , N ), format = "csr" )[ keep ] [: ,keep ]

    Dx_full = kron_I( D1_back( nx, dx ), nx, ny, "x" )
    Dy_full = kron_I( D1_back( ny, dx ), nx, ny, "y" )
    Dx = Dx_full[ : , keep ][ keep, : ]
    Dy = Dy_full[ : , keep ][ keep, : ]

    Tyy = -Dy.T.dot( inv_mu_d.dot( Dy ) )
    Txx = -Dx.T.dot( inv_mu_d.dot( Dx ) )
    Tyx =  Dy.T.dot( inv_mu_d.dot( Dx ) )
    return Txx, Tyy, Tyx, eps_d

# Full-vector eigen solve targeting a neff cluster via shift-invert (identity RHS).
# eps, mu: epsilon_r, mu_r maps (ny x nx ).
# dx: Grid spacing. lam: Wavelength.
# k_modes: Number of eigenpairs around the target to compute.
# neff_sigma: Shift-invert target for effective index. sigma = (neff_sigma * k0 )^2.
# mask: Interior mask (ny x nx bool).
# Solver: scipy.sparse.linalg.eigsh with which="LM", shift-invert at σ, tol=1e-9.
# Returns: neff (k,), Ex_i (Nint×k), Ey_i (Nint×k) — interior DOFs, unsorted w.r.t. Re(neff).
def eigsolve_cluster( eps, mu, dx, lam, k_modes, neff_sigma, mask ):
    k0 = 2*np.pi/lam
    Txx, Tyy, Tyx, eps_d = build_vector_operator_blocks( eps, mu, dx, mask )
    M11 = Tyy + eps_d*( k0**2 )
    M22 = Txx + eps_d*( k0**2 )
    M12 = Tyx
    M   = sp.bmat( [ [ M11, M12 ], [ M12.T, M22 ] ], format = "csr" )

    sigma = ( neff_sigma * k0 )**2
    if VERBOSE:
        print(f"[FV] eigsh at neff≈{neff_sigma:.2f}: sigma={sigma:.3e}, size={M.shape[0]}, k={k_modes}")

    vals, vecs = spla.eigsh( M, k = k_modes, sigma = sigma, which = "LM", tol = 1e-9, maxiter = 30000 )
    beta = np.sqrt(np.maximum( vals, 0.0 ) )
    neff = beta / k0
    Nint = int( M.shape[ 0 ] // 2 )
    Ex_i = vecs[ : Nint, : ]
    Ey_i = vecs[ Nint:, : ]
    return neff, Ex_i, Ey_i

# L2-normalizes a pair of interior field vectors (Ex_i, Ey_i).
# Ex_i, Ey_i: 1D complex arrays of same length (interior DOFs).
# Returns: (Ex_i_n, Ey_i_n) normalized so ||Ex||^2+||Ey||^2=1 (up to 1e-300 guard).
def normalize_pair( Ex_i, Ey_i ):
    s = np.sqrt(np.vdot( Ex_i, Ex_i ) + np.vdot( Ey_i, Ey_i ) ).real + 1e-300
    return Ex_i/s, Ey_i/s

# Computes modal overlap between two (Ex,Ey) pairs using standard inner products.
# a, b: Tuples (Ex, Ey) with matching shapes.
# Returns: |<Ex_a,Ex_b> + <Ey_a,Ey_b>| (non-negative float).
def pair_overlap( a, b ):
    Ex_a, Ey_a = a
    Ex_b, Ey_b = b
    return np.abs( np.vdot( Ex_a, Ex_b ) + np.vdot( Ey_a, Ey_b ) )

# Merges modes by deduplicating near-identical eigenvectors (by overlap) and sorting by Re(neff).
# neff_list: 1D array-like of neff values.
# Ex_list, Ey_list: Lists/arrays of vectors (each entry aligns with neff_list).
# overlap_tol: Threshold above which two modes are considered duplicates.
# Returns: (neff, Ex_i, Ey_i) with duplicates removed; Ex_i/Ey_i are column-stacked matrices.
def merge_modes( neff_list, Ex_list, Ey_list, overlap_tol = 0.985 ):
    pairs = [ normalize_pair( Ex_list[ i ], Ey_list[ i ] ) for i in range( len( neff_list ) ) ]
    modes = list()
    for i, ne in enumerate( neff_list ):
        dup = False
        for (_, p) in modes:
            if pair_overlap(pairs[ i ], p) > overlap_tol:
                dup = True
                break
        if not dup:
            modes.append( ( ne, pairs[ i ] ) )
    modes.sort( key=lambda t: np.real( t[ 0 ] ), reverse = True )
    neff = np.array([ m[ 0 ] for m in modes ], dtype = float )
    Ex_i = np.column_stack( [ m[ 1 ][ 0 ] for m in modes ] ) if modes else np.zeros( ( 0, 0 ), complex )
    Ey_i = np.column_stack( [ m[ 1 ][ 1 ] for m in modes ] ) if modes else np.zeros( ( 0, 0 ), complex )
    return neff, Ex_i, Ey_i

# Embeds an interior-DOF vector into a full (ny×nx) field array using a boolean mask.
# vi: 1D array of interior values (length=mask.sum()).
# mask: (ny×nx) bool array marking interior DOFs.
# nx, ny: Grid sizes.
# Returns: full (ny×nx complex) with interior filled and boundaries zero.
def embed_in_full(vi, mask, nx, ny):
    full = np.zeros( ( ny, nx ), dtype = complex )
    full.ravel( order = "C" )[ mask.ravel( order = "C" ) ] = vi
    return full

# Stacks one modal pair (Ex_i_col, Ey_i_col) into a single vector.
# Ex_i_col, Ey_i_col: 1D arrays of equal length (interior DOFs).
# Returns: concatenated 1D array of length 2*Nint.
def stack_pair( Ex_i_col, Ey_i_col ):
    return np.concatenate( [Ex_i_col, Ey_i_col], axis = 0 )

# Stacks modal matrices for Ex_i and Ey_i vertically.
# Ex_i_mat, Ey_i_mat: (Nint×M) each, columns are modes.
# Returns: (2*Nint × M) stacked matrix V = [Ex; Ey].
def stack_modes( Ex_i_mat, Ey_i_mat ):
    return np.vstack( [ Ex_i_mat, Ey_i_mat])  # (2*Nint) × M

# Computes ∂F/∂x via central differences on a uniform grid (one-sided at boundaries).
# F: 2D array (ny×nx), real or complex.
# dx: Grid spacing (meters).
# Returns: G = dF/dx, same shape/type as F.
def ddx_central( F, dx ):
    G = np.empty_like( F, dtype = complex )
    G[ : , 1 : -1 ] = ( F[ : , 2 : ] - F[ : , : -2 ] ) / ( 2 * dx )
    G[ : , 0 ] = (F[ : , 1 ] - F[ : , 0 ] ) / dx
    G[ : , -1 ] = (F[ : , -1 ] - F[:,-2])/dx
    return G

# Computes dF/dy via central differences on a uniform grid (one-sided at boundaries).
# F: 2D array (ny x nx), real or complex.
# dx: Grid spacing (meters) used for y-step.
# Returns: G = dF/dy, same shape/type as F.
def ddy_central( F, dx ):
    G = np.empty_like(F, dtype=complex)
    G[ 1 : -1 , : ] = (F[2:,:] - F[ : -2 , : ] ) / ( 2 * dx )
    G[ 0 ,: ]   = ( F[ 1 , : ] - F[ 0 , : ] ) / dx
    G[ -1 , : ] = (F[ -1 , : ] - F[ -2, : ] ) / dx
    return G

# Reconstructs Ez and magnetic field components from (Ex, Ey) using Maxwell’s equations.
# Ex, Ey: Transverse electric fields (ny×nx complex).
# eps, mu: Relative permittivity/permeability maps (ny×nx).
# lam: Wavelength (meters). beta: Propagation constant (1/m). dx: Grid spacing (meters).
# Conventions: e^{-iωt}. Faraday’s law: curl E = -i ω μ H.
# Returns: Ez, Hx, Hy, Hz (ny×nx complex).
def reconstruct_fields( Ex, Ey, eps, mu, lam, beta, dx ):
    # Gauss' law → Ez
    dDx = ddx_central( eps * Ex, dx )
    dDy = ddy_central( eps * Ey, dx )
    Ez  = -( dDx + dDy ) / ( 1j * beta * eps + 1e-30 )

    # Faraday (e^{-i omega t}): curl E = - i omega mu H
    omega = 2 * np.pi * c0 / lam
    curlEx = ddy_central( Ez, dx ) - 1j * beta * Ey
    curlEy = 1j * beta * Ex - ddx_central( Ez, dx )
    curlEz = ddx_central( Ey, dx ) - ddy_central( Ex, dx )
    Hx = -curlEx / ( 1j * omega * mu0 * mu )
    Hy = -curlEy / ( 1j * omega * mu0 * mu )
    Hz = -curlEz / ( 1j * omega * mu0 * mu )
    return Ez, Hx, Hy, Hz

# Computes time-averaged electromagnetic energy density per unit volume.
# eps, mu: Relative ε_r, μ_r maps (ny×nx).
# Ex, Ey, Ez, Hx, Hy, Hz: Field components (ny×nx complex).
# Formula: w = 0.5*(ε0 Re{ε_r}|E|^2 + μ0 Re{μ_r}|H|^2).
# Returns: w (ny×nx float) in J/m^3.
def energy_density( eps, mu, Ex, Ey, Ez, Hx, Hy, Hz ):
    E2 = ( np.abs( Ex )**2 + np.abs( Ey )**2 + np.abs( Ez )**2 )
    H2 = ( np.abs( Hx )**2 + np.abs( Hy )**2 + np.abs( Hz )** 2)
    return 0.5 * ( eps0 * np.real( eps ) * E2 + mu0 * np.real( mu )*H2 )

# Computes TE/TM polarization scores inside the core region from transverse fields.
# Ex, Ey: Transverse electric fields (ny×nx complex).
# core_mask: Bool mask (ny×nx) marking core region.
# Returns: (TE_score, TM_score), each in [0,1], using core integrals of |Ex|^2, |Ey|^2.
def core_polarization_scores( Ex, Ey, core_mask ):
    Ex2c = float( np.sum( np.abs( Ex[ core_mask ])**2 ) )
    Ey2c = float( np.sum( np.abs( Ey[ core_mask ])**2 ) )
    denom = Ex2c + Ey2c + 1e-30
    return Ex2c/denom, Ey2c/denom

# Fraction of total EM energy stored in the Ez component.
# eps: Relative eplsilon_r (ny×nx). Ex,Ey,Ez,Hx,Hy,Hz: Fields.
# Returns: scalar in [0,1] = integral(½ epsilon0 Re{epsilon_r}|Ez|^2)/integral(½(ε0 Re{ε_r}|E|^2+mu0|H|^2)).
def ez_fraction( eps, Ex, Ey, Ez, Hx, Hy, Hz ):
    epsr = np.real(eps)
    E2 = np.abs( Ex )**2 + np.abs( Ey )**2 + np.abs( Ez )**2
    H2 = np.abs( Hx )**2 + np.abs( Hy )**2 + np.abs( Hz )**2
    W = 0.5*( eps0 * epsr * E2 + mu0*H2)
    Wez = 0.5*( eps0 * epsr * np.abs(Ez)**2)
    return float( np.sum( Wez ) / ( np.sum( W )+1e-30 ) )

# Diagnostics: normalized residuals for curl E + i omega μ H and ∇·(epsilonE).
# eps, mu: epsilon_r, mu_r. lam: wavelength (m). beta: propagation constant (1/m).
# Ex,Ey,Ez,Hx,Hy,Hz: Field components (ny x nx). dx: grid spacing.
# Returns: (curlE_residual, Gauss_residual) = L2 residual / L2 denominator (dimensionless).
def mode_residuals( eps, mu, lam, beta, Ex, Ey, Ez, Hx, Hy, Hz, dx ):
    omega = 2*np.pi*c0/lam
    # curl E + i omega μ H 
    r_x = ddy_central( Ez, dx ) - 1j * beta * Ey + 1j * omega * mu0 * mu * Hx
    r_y = 1j * beta*Ex - ddx_central( Ez, dx ) + 1j * omega * mu0 * mu * Hy
    r_z = ddx_central( Ey, dx ) - ddy_central( Ex, dx ) + 1j * omega * mu0 * mu * Hz
    curlE_err = np.linalg.norm( r_x ) + np.linalg.norm( r_y ) + np.linalg.norm( r_z )
    curlE_den = ( np.linalg.norm( Ex ) + np.linalg.norm( Ey ) + np.linalg.norm( Ez ) ) + 1e-30

    # div(εE)  
    div_epsE = ddx_central( eps * Ex, dx ) + ddy_central( eps * Ey, dx ) + 1j * beta * eps * Ez
    gauss_err = np.linalg.norm( div_epsE )
    gauss_den = np.linalg.norm( eps * ( np.abs( Ex ) + np.abs( Ey ) + np.abs( Ez ) ) ) + 1e-30
    return curlE_err/curlE_den, gauss_err/gauss_den

# Plots the magnitude |F| on the x–y grid as a heatmap with optional core outline.
# x, y: 1D coordinate arrays (meters) of lengths nx, ny.
# F: 2D complex array (ny×nx); plots |F|.
# title: Figure title and (if SAVE=True) base filename.
# vmax: Optional color scale upper bound; defaults to 99.5th percentile of |F|.
# outline: If True, overlay the core rectangle defined by (w, t) in meters.
# Output: Shows the figure; saves PNG if SAVE.
def plot_component( x, y, F, title, vmax = None, outline = False, w = None, t = None ):
    V = np.abs( F )
    if vmax is None:
        finite = V[ np.isfinite( V ) ]
        vmax = np.percentile( finite, 99.5 ) if finite.size else 1.0
    fig, ax = plt.subplots( figsize = ( 5.4, 4.2 ) )
    pcm = ax.pcolormesh( x*1e6, y*1e6, V, shading = "auto", vmin = 0, vmax = vmax )
    fig.colorbar(pcm, ax = ax, label='magnitude' )
    ax.set_xlabel( 'x [µm]' )   
    ax.set_ylabel( 'y [µm]' )
    ax.set_title( title )
    if outline and w is not None and t is not None:
        x0, x1 = -w / 2*1e6, w/2*1e6
        y0, y1 = -t / 2*1e6, t/2*1e6
        ax.plot( [ x0, x1 ,x1, x0, x0 ] , [ y0, y0, y1, y1, y0 ], "w--", lw = 1.0, alpha = 0.9 )
    fig.tight_layout()
    if SAVE:
        plt.savefig( title.replace(' ','_')+'.png', dpi=600, bbox_inches = 'tight', transparent = True )
    plt.show()

# Plots effective index sweep versus wavelength for multiple tracked branches.
# lams: Sequence of wavelengths (meters).
# arrays: Iterable of arrays (same length as lams) containing n_eff values (can include NaN).
# labels: Legend labels for each curve in 'arrays'.
# title: Figure title; also used as filename if SAVE=True.
def plot_neff_sweep( lams, arrays, labels, title ):
    x_nm = np.array(lams) * 1e9
    plt.figure( figsize =( 6.6, 4.2 ) )
    for arr, lab in zip( arrays, labels ):
        plt.plot(x_nm, arr, marker="o", lw=1.6, label=lab)
    plt.xlabel( 'Wavelength [nm]')
    plt.ylabel( 'n_eff' )
    plt.grid(True, alpha = 0.35 )
    plt.legend()
    plt.title( title )
    plt.tight_layout()
    if SAVE:
        plt.savefig( title.replace(' ','_')+'.png', dpi=600, bbox_inches = 'tight', transparent = True )
    plt.show()

# Plots a self-overlap curve |<E(0), E(z)>| along propagation distance.
# z: 1D array of distances (meters). ov: 1D array of overlaps in [0,1].
# title: Figure title; used as filename if SAVE=True.
# Axes: z is shown in mm.
def plot_overlap(z, ov, title ):
    plt.figure(figsize = ( 6.0, 3.5))
    plt.plot(z*1e3, ov, lw = 1.8)
    plt.ylim(0.0, 1.02)
    plt.xlabel('z [mm]')
    plt.ylabel( 'Self-overlap |<E(0),E(z)>|' )
    plt.grid(True, alpha=0.35); plt.title( title )
    plt.tight_layout()
    if SAVE:
        plt.savefig( title.replace(' ','_')+'.png', dpi=600, bbox_inches = 'tight', transparent = True )
    plt.show()

# Solves two shift-invert clusters (TE- and TM-centered) then merges/deduplicates modes.
# eps, mu: ε_r, μ_r maps. dx: grid spacing (m). lam: wavelength (m). mask: interior mask.
# Uses globals: K_TE, K_TM, SIGMA_NEFF_TE, SIGMA_NEFF_TM.
# Returns: neff_all (dedup & sorted), Ex_i_all (Nint×M), Ey_i_all (Nint×M).
def solve_and_merge( eps, mu, dx, lam, mask ):
    ne1, Ex1, Ey1 = eigsolve_cluster( eps, mu, dx, lam, K_TE, SIGMA_NEFF_TE, mask )
    ne2, Ex2, Ey2 = eigsolve_cluster( eps, mu, dx, lam, K_TM, SIGMA_NEFF_TM, mask )
    ne_all = np.concatenate( [ ne1, ne2 ] )
    Ex_all_i = np.concatenate( [ Ex1, Ex2 ], axis = 1 )
    Ey_all_i = np.concatenate( [ Ey1, Ey2 ], axis = 1 )
    return merge_modes( ne_all, Ex_all_i.T, Ey_all_i.T, overlap_tol = 0.985 )

# Selects TE0/TM0 from guided modes, computes diagnostics and returns candidate info.
# x, y, nx, ny: Grid coords/sizes. lam: wavelength (m).
# eps, mu: ε_r, μ_r maps. mask: interior mask. core: core mask.
# neff, Ex_i, Ey_i: Merged eigen solutions (Nint×M).
# Guided definition: n_clad + 1e-6 < neff < n_core + 1e-6 (using min/max sqrt(eps)).
# TM selection heuristic: prefer strong TM score and margin above n_clad (CUT_MARGIN).
# Returns: te0 (dict), tm0 (dict), cands (sorted list), guided_set (dict with neff, Ex_i, Ey_i).
def pick_te_tm(x, y, nx, ny, lam, eps, mu, mask, core, neff, Ex_i, Ey_i):
    # Filter guided
    ncore, nclad = float( np.sqrt( eps.max() ) ), float( np.sqrt( eps.min() ) )
    guided_mask = (neff < ncore + 1e-6) & (neff > nclad + 1e-6 )
    neff = neff[guided_mask ]
    Ex_i = Ex_i[ :, guided_mask ]
    Ey_i = Ey_i[ :, guided_mask ]

    cands = list()
    for m in range( neff.size ):
        beta = neff[ m ] * ( 2 * np.pi / lam )
        Ex = embed_in_full( Ex_i[ : ,  m] , mask, nx, ny )
        Ey = embed_in_full( Ey_i[ : ,  m ], mask, nx, ny )
        Ez, Hx, Hy, Hz = reconstruct_fields( Ex, Ey, eps, mu, lam, beta, DX )
        TE_sc, TM_sc = core_polarization_scores( Ex, Ey, core )
        Ez_frac = ez_fraction( eps, Ex, Ey, Ez, Hx, Hy, Hz )
        r1, r2 = mode_residuals( eps, mu, lam, beta, Ex, Ey, Ez, Hx, Hy, Hz, DX)
        cands.append( { "neff": float( np.real( neff[ m ] ) ), "TE_sc": TE_sc, "TM_sc": TM_sc, "Ez_frac": Ez_frac, "res": ( r1, r2 ), "fields": ( Ex, Ey, Ez, Hx, Hy, Hz ) } )
    cands.sort(key=lambda d: d["neff"], reverse = True )

    # TE0 = highest-neff guided (expected for this wire)
    te0 = cands[ 0 ]
    remaining = cands[ 1 : ]

    TM_THRESH = 0.70
    CUT_MARGIN = 0.12  # require neff > n_clad + margin
    tm_pool = [ d for d in remaining if d["TM_sc"] >= TM_THRESH and d["neff"] > nclad + CUT_MARGIN ]

    def tm_objective( d ):
        return 0.7*d[ "TM_sc" ] + 0.3 * ( d[ "neff" ] - nclad ) / ( ncore - nclad )

    if tm_pool:
        tm0 = max( tm_pool, key=tm_objective )
    else:
        cand1 = [ d for d in remaining if d[ "neff" ] > nclad + CUT_MARGIN ]
        tm0 = max( cand1, key=lambda d: d[ "TM_sc" ], default = max( remaining, key = lambda d: d[ "TM_sc" ] ) )

    guided_set = { "neff": neff, "Ex_i": Ex_i, "Ey_i": Ey_i }
    return te0, tm0, cands, guided_set

# Tracks modal branches over a wavelength sweep by maximizing overlap with previous step.
# lams: 1D array of wavelengths (meters).
# x, y, nx, ny: Grid coords/sizes (not used directly beyond size context).
# mask: Interior mask.
# Uses globals: DX, AA_EPS, TRACK_K, TRACK_OVERLAP_TOL.
# Returns: tracked (len(lams) x TRACK_K) array with neff per tracked branch (NaN if unmatched).
def track_modes_over_sweep( lams, x, y, nx, ny, mask ):
    tracked = np.full( ( len( lams ), TRACK_K ), np.nan, dtype = float )
    prev_pairs = None
    for li, lam in enumerate( lams ):
        ncore, nclad = n_Si( lam ), n_SiO2( lam )
        eps, mu, _ = build_eps_mu( nx, ny, DX, CORE_W, CORE_T, ncore, nclad, antialias = AA_EPS )
        neff, Ex_i, Ey_i = solve_and_merge( eps, mu, DX, lam, mask )
        guided = (neff < ncore + 1e-6 ) & ( neff > nclad + 1e-6 )
        neff = neff[ guided ]
        Ex_i = Ex_i[ :, guided ]
        Ey_i = Ey_i[ :, guided ]
        pairs = [ normalize_pair( Ex_i[ : , j ], Ey_i[ : , j ] ) for j in range( neff.size ) ]

        if li == 0 or prev_pairs is None:
            order = np.argsort( np.real( neff ) ) [ : : -1 ]
            use = order[ : min( TRACK_K, len( order ) ) ]
            tracked[ li, :len( use ) ] = neff[ use ] 
            prev_pairs = [ pairs[ j ] for j in use ]
        else:
            used = set()
            new_prev = list()
            for k in range( TRACK_K ):
                if k >= len( prev_pairs ): break
                best_j, best_ov = -1, -1.0
                for j in range( len( pairs) ) :
                    if j in used: continue
                    ov = pair_overlap(prev_pairs[k], pairs[j])
                    if ov > best_ov:
                        best_ov, best_j = ov, j
                if best_j >= 0 and best_ov >= TRACK_OVERLAP_TOL:
                    tracked[ li, k ] = neff[ best_j ]
                    used.add( best_j )
                    new_prev.append( pairs[ best_j ] )
                else:
                    tracked[ li, k ] = np.nan
                    new_prev.append( prev_pairs[ k ] )
            prev_pairs = new_prev
    return tracked

# Propagates a chosen interior field in the modal basis over length L and computes self-overlap.
# neff: Modal effective indices (M,). V: stacked modal basis (2*Nint × M) = [Ex; Ey].
# f0: Initial stacked field (2*Nint,). lam: wavelength (m).
# L: Propagation length (m), Nz: number of z-samples.
# Returns: (z, ov) where z is (Nz,) in meters and ov is |<f(0), f(z)>|/(||f(0)||·||f(z)||).
def modal_overlap_propagation( neff, V, f0, lam, L=L_PROP, Nz = NZ_PROP ):
    k0 = 2*np.pi/lam
    beta = neff * k0  
    # least-squares modal coefficients
    a, *_ = np.linalg.lstsq( V, f0, rcond = None )
    f0n = np.linalg.norm( f0 )
    z = np.linspace( 0.0, L, Nz )
    ov = np.empty_like( z, dtype = float )
    for i, zi in enumerate( z ):
        phase = np.exp( 1j * beta * zi)
        fz = V @ (a * phase)
        ov[ i ] = np.abs(np.vdot(f0, fz)) / (f0n * np.linalg.norm(fz))
    return z, ov

# Orchestrates the full workflow:
# 1) Build grid and interior mask.
# 2) For each lambda in LAM_LIST:
#    - Build epsilon_r (AA optional) and mu_r from n_Si(lambda), n_SiO2(lambda).
#    - Solve two clusters (TE/TM targets), merge & deduplicate modes.
#    - Select TE0/TM0 from guided set; reconstruct fields; compute diagnostics (residuals, Ez fraction).
#    - Visualize |Ex|, |Ey|, |Ez| and |E| for TE0 and TM0.
#    - Propagation test: modal expansion of TE0/TM0, self-overlap vs z.
# 3) Sweep LAM_SWEEP:
#    - Track first TRACK_K guided branches by modal overlap; plot n_eff vs λ.
# Prints: grid summary, merged neff list, TE0/TM0 summaries, propagation stats, and tracked sweep.
def run():
    x, y, nx, ny = make_grid( DX, PAD, CORE_W, CORE_T )
    mask = interior_mask( nx, ny )
    print(f"[grid] nx={nx}, ny={ny}, dx={DX*1e9:.0f} nm, PAD={PAD*1e6:.1f} ")

    for lam in LAM_LIST:
        ncore, nclad = n_Si(lam), n_SiO2( lam )
        eps, mu, core = build_eps_mu( nx, ny, DX, CORE_W, CORE_T, ncore, nclad, antialias = AA_EPS )

        # Solve & merge clusters
        neff_all, Ex_i_all, Ey_i_all = solve_and_merge( eps, mu, DX, lam, mask )
        print(f"\nλ={int(round( lam * 1e9 )  )} nm  merged neff (top 12): " + ", ".join( f"{np.real( v ):.4f}" for v in neff_all[ : 12 ] ) )
        print(f"n_core={ncore:.4f}, n_clad={nclad:.4f}")

        te0, tm0, _, guided_set = pick_te_tm(x, y, nx, ny, lam, eps, mu, mask, core, neff_all, Ex_i_all, Ey_i_all)
        print(f"TE0 ~ n_eff={te0['neff']:.4f}  (TE_sc={te0['TE_sc']:.3f}, Ez_frac={te0['Ez_frac']:.3f})" f" residuals: curlE={te0['res'][0]:.2e}, Gauss={te0['res'][1]:.2e}")
        print(f"TM0 ~ n_eff={tm0['neff']:.4f}  (TM_sc={tm0['TM_sc']:.3f}, Ez_frac={tm0['Ez_frac']:.3f})" f"  residuals: curlE={tm0['res'][0]:.2e}, Gauss={tm0['res'][1]:.2e}")

        Ex,Ey,Ez,Hx,Hy,Hz = te0[ "fields" ]
        lam_nm = int( round(lam*1e9 ) ) 
        plot_component( x, y , Ex, f"TE0 |Ex| at {lam_nm} nm (n_eff={te0['neff']:.4f})", outline = True, w = CORE_W, t = CORE_T )
        plot_component(x,y,Ey, f"TE0 |Ey| at {lam_nm} nm (n_eff={te0['neff']:.4f})", outline = True, w = CORE_W, t = CORE_T )
        plot_component(x,y,Ez, f"TE0 |Ez| at {lam_nm} nm (n_eff={te0['neff']:.4f})", outline = True, w = CORE_W, t = CORE_T )
        Emag = np.sqrt(np.abs( Ex )**2 + np.abs( Ey )**2 + np.abs( Ez )**2 )
        plot_component(x,y,Emag, f"TE0 |E| at {lam_nm} nm", outline = True, w = CORE_W, t = CORE_T )

        Ex,Ey,Ez,Hx,Hy,Hz = tm0[ "fields" ]
        plot_component( x,y,Ex, f"TM0 |Ex| at {lam_nm} nm (n_eff={tm0['neff']:.4f})", outline = True, w = CORE_W, t = CORE_T )
        plot_component( x,y,Ey, f"TM0 |Ey| at {lam_nm} nm (n_eff={tm0['neff']:.4f})", outline = True, w = CORE_W, t = CORE_T)
        plot_component( x,y,Ez, f"TM0 |Ez| at {lam_nm} nm (n_eff={tm0['neff']:.4f})", outline = True, w = CORE_W, t = CORE_T )
        Emag = np.sqrt( np.abs( Ex )**2 + np.abs( Ey )**2 + np.abs( Ez )**2 )
        plot_component( x , y, Emag, f"TM0 |E| at {lam_nm} nm", outline = True, w = CORE_W, t = CORE_T )

        neff_g   = guided_set[ "neff" ]
        Exi_g    = guided_set[ "Ex_i" ]
        Eyi_g    = guided_set[ "Ey_i" ]
        V        = stack_modes( Exi_g, Eyi_g )      
        
        idx_te = int( np.argmin( np.abs( neff_g - te0[ "neff" ] ) )  )
        idx_tm = int( np.argmin( np.abs( neff_g - tm0[ "neff" ] ) ) ) 
        f0_te = stack_pair( Exi_g[ : , idx_te ], Eyi_g[ : , idx_te ] )
        f0_tm = stack_pair( Exi_g[:, idx_tm ], Eyi_g[ : , idx_tm ] )

        z_te, ov_te = modal_overlap_propagation( neff_g, V, f0_te, lam, L = L_PROP, Nz = NZ_PROP )
        z_tm, ov_tm = modal_overlap_propagation( neff_g, V, f0_tm, lam, L = L_PROP, Nz = NZ_PROP )

        print( f"[propagation check @ {lam_nm} nm] TE0 overlap: min={ov_te.min():.6f}, mean={ov_te.mean():.6f}" )
        print( f"[propagation check @ {lam_nm} nm] TM0 overlap: min={ov_tm.min():.6f}, mean={ov_tm.mean():.6f}" )

        plot_overlap( z_te, ov_te, f"TE0 self-overlap vs z at {lam_nm} nm" )
        plot_overlap( z_tm, ov_tm, f"TM0 self-overlap vs z at {lam_nm} nm" )

    print("\n 1300–1600 nm (tracked first 5 guided neff)")
    tracked = track_modes_over_sweep(LAM_SWEEP, x, y, nx, ny, mask )
    arrays = [ tracked[:,i] for i in range( TRACK_K) ]
    labels = [ f"branch {i}" for i in range( TRACK_K ) ]
    plot_neff_sweep(LAM_SWEEP, arrays, labels, "First 5 guided modes vs λ (tracked by overlap)")

if __name__ == "__main__":
    run()
