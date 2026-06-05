import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path



OMEGA = 8.0             # starting omega
HIDDEN = 128            # SIREN width # excact numbers not speciified in the paper  ##enough to represent oscillatory solutions of the Helmholtz equation. 
NUM_HIDDEN = 4          # SIREN depth # excact numbers not speciified in the paper  ## Small enough to keep training stable and computationally efficient.
OMEGA0 = 50.0           # first-layer frequency scale for SIREN (excite high-freqs). Directly from SIREN
EPOCHS = 20000          # Adam epochs. I chose 20k epochs because PINNs converge slowly, unlike supervised tasks, the loss here comes from physics residuals, which are noisier and harder to optimize. 
                        # 20k gave a good balance: long enough to reduce the PDE and boundary losses, but not so long that training became inefficient.
LR = 1e-3               # Adam LR. I used 1e-3 because it’s a stable default for Adam — big enough to make progress, but small enough to avoid instability
ETA_MIN_FACTOR = 0.05   # cosine schedule final LR = LR*ETA_MIN_FACTOR
LAMBDA_BC = 50.0         # weight for Sommerfeld BC (keep PDE influential). Chosen by try and error. Chosen because it made the boundary residual and PDE residual roughly the same order of magnitude during training.
N_INTERIOR = 40000      # interior collocation per epoch
N_BOUNDARY = 6000       # boundary collocation per epoch
SRC_FRACTION = 0.4      # fraction of interior pts drawn near source
J_SIGMA = 0.05          # Gaussian source width (smoother forcing). Use 0.05 for pointier
USE_LBFGS = True        # finishing pass
PLOT_N = 201            # 201 includes (0,0)
GRAD_CLIP = 5.0         # gradient clipping
EARLY_STOP_THR = 1e-10   # early stopping threshold

OUTDIR = Path( "pinn_em_results" )
OUTDIR.mkdir( parents = True, exist_ok = True )


torch.set_default_dtype( torch.float64 )
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )


# Initializes the weights and biases for a neural network layer.
# in_dim: The number of input dimensions (size of the input features).
# out_dim: The number of output dimensions (size of the layer's output).
# first: A flag (boolean, default is False). If True, initialization uses a different bound formula for the first layer.
# omega_0: A scaling factor (float, default is 30.0) used in the weight initialization for non-first layers.
# Returns: A tuple (W, b) where W is the initialized weight matrix and b is the initialized bias vector.
def init_weights( in_dim, out_dim, first = False, omega_0 = 30.0 ):
    if first:
        bound = 1.0 / in_dim
    else:
        bound = np.sqrt( 6.0 / in_dim ) / omega_0
    W = torch.empty( out_dim, in_dim, device = device ).uniform_( -bound, bound )
    b = torch.zeros( out_dim, device = device )
    return W, b

# Builds a neural network model using sine activations (SIREN-style architecture).
# input_dim: The number of input features (default is 2).
# hidden_dim: The number of hidden units per layer (default is HIDDEN, a predefined constant).
# output_dim: The number of output features (default is 2).
# num_hidden: The number of hidden layers to include (default is NUM_HIDDEN, a predefined constant).
# omega_0: The frequency scaling factor (float, default is OMEGA0, a predefined constant) used in the first layer initialization.
# Returns: Four lists (weights, biases, activations, omegas) defining the model architecture.
def build_model( input_dim = 2, hidden_dim = HIDDEN, output_dim = 2, num_hidden = NUM_HIDDEN, omega_0 = OMEGA0 ):
    weights = list()
    biases = list()
    activations = list()
    omegas = list()
    W, b = init_weights( input_dim, hidden_dim, first = True, omega_0 = omega_0 )
    weights.append( W )
    biases.append( b )
    activations.append( "sine" )
    omegas.append( float ( omega_0 ) )
    for _ in range( num_hidden ):
        W, b = init_weights( hidden_dim, hidden_dim, first = False, omega_0 = 1.0 )
        weights.append( W )
        biases.append( b )
        activations.append( "sine" )
        omegas.append( 1.0 )
    W, b = init_weights( hidden_dim, output_dim, first = False, omega_0 = 1.0 )
    with torch.no_grad():
        W.mul_( 0.5 )
    weights.append( W )
    biases.append( b )
    activations.append( "linear" )
    omegas.append( 1.0 )
    return weights, biases, activations, omegas

# Performs a forward pass through the neural network model.
# x: The input tensor to the network.
# weights: A list of weight matrices for each layer.
# biases: A list of bias vectors for each layer.
# activations: A list of activation function names ("sine" or "linear") corresponding to each layer.
# omegas: A list of frequency scaling factors (floats) applied when using the sine activation.
# Returns: The output tensor after applying all layers and activations.
def forward( x, weights, biases, activations, omegas ):
    h = x
    for W, b, act, w0 in zip( weights, biases, activations, omegas ):
        h = F.linear( h, W, b )
        if act == "sine":
            h = torch.sin( w0 * h )
    return h 


# Computes the Laplacian of a scalar field u with respect to input coordinates x.
# u: A scalar field (tensor) dependent on x, typically the output of a neural network.
# x: The input tensor with respect to which derivatives are taken (requires gradients).
# Returns: The Laplacian of u (second derivatives with respect to x and y), as a tensor.
def laplacian( u, x ):
    g = torch.autograd.grad( u, x, grad_outputs = torch.ones_like( u ), create_graph = True )[ 0 ]
    ux = g[ : , 0 ]
    uy = g[ : , 1 ]
    uxx = torch.autograd.grad( ux, x, grad_outputs=torch.ones_like( ux ), create_graph = True )[ 0 ][ : , 0 ]
    uyy = torch.autograd.grad( uy, x, grad_outputs=torch.ones_like( uy ), create_graph = True )[ 0 ][ : , 1 ]
    return uxx + uyy

# Defines a spatially varying epsilon field with an optional circular region.
# coords: A tensor of shape (N, 2) containing the (x, y) coordinates of N points.
# circle: Parameters of the circular region (x0, y0, r, eps_in), where (x0, y0) is the center, r is the radius, and eps_in is the epsilon value inside the circle. If None, no circle is applied.
# Returns: A tensor of epsilon values for each coordinate, with eps_in assigned inside the circle and 1.0 elsewhere.
def epsilon_field( coords, circle ):
    eps = torch.ones( coords.shape[ 0 ], dtype = coords.dtype, device = coords.device )
    if circle is not None:
        x0, y0, r, eps_in = circle
        dist2 = ( coords[ : , 0 ] - x0 )**2 + ( coords[ : , 1 ] - y0 )**2
        inside = dist2 <= r**2
        eps[ inside ] = eps_in
    return eps

# Defines a Gaussian source term Jz centered at a given location.
# coords: A tensor of shape (N, 2) containing the (x, y) coordinates of N points.
# sigma: The standard deviation (float, default is J_SIGMA) controlling the spread of the Gaussian.
# center: A tuple (cx, cy) specifying the center of the Gaussian (default is (0.0, 0.0)).
# Returns: A tensor of source values computed as a 2D Gaussian function over the input coordinates.
def Jz( coords, sigma = J_SIGMA, center = (0.0, 0.0 ) ):
    x  = coords[ : , 0 ]
    y =  coords[ : , 1 ]
    cx, cy = center
    return torch.exp( - ( ( x - cx )**2 + ( y - cy )**2) / ( 2.0 * sigma**2 ) ) 


# Samples interior and boundary points for training a PDE model.
# n_interior: Total number of interior points to sample (default is N_INTERIOR).
# n_boundary: Total number of boundary points to sample (default is N_BOUNDARY).
# src_frac: fraction of interior points to place near the source (clustered around the origin) instead of uniformly over the domain.
# src_sigma: Standard deviation (float, default is J_SIGMA) of the Gaussian distribution for source points. Controls how tightly interior points cluster near the source.
# Returns: A tuple (xy_i, xy_b, normals) where: xy_i = tensor of interior points (uniform + Gaussian source points), with gradients enabled. xy_b = tensor of boundary points on the square domain [-1, 1]^2, with gradients enabled. normals = tensor of outward normal vectors corresponding to the boundary points.
def sample_points( n_interior = N_INTERIOR, n_boundary = N_BOUNDARY, src_frac = SRC_FRACTION, src_sigma = J_SIGMA ):
    xy_i = torch.rand( n_interior, 2, device = device ) * 2 - 1
    xy_i.requires_grad_( True )

    m = n_boundary // 4
    t = torch.rand( m, 1, device = device ) * 2 - 1
    bL = torch.cat( [ - torch.ones_like( t ), t ], dim = 1 )   # x=-1
    bR = torch.cat( [ torch.ones_like( t ), t ], dim = 1 )   # x=+1
    bB = torch.cat( [ t, - torch.ones_like( t ) ], dim = 1 )   # y=-1
    bT = torch.cat( [ t,  torch.ones_like( t ) ], dim = 1 )   # y=+1
    xy_b = torch.cat( [ bL, bR, bB, bT ], dim = 0 )

    nL = torch.tensor( [ [ -1.0,  0.0 ] ], device = device ).repeat( m,1 )
    nR = torch.tensor( [ [ 1.0,  0.0 ] ], device = device).repeat( m, 1 )
    nB = torch.tensor( [ [ 0.0, -1.0 ] ], device = device).repeat( m, 1 )
    nT = torch.tensor( [ [ 0.0,  1.0 ] ], device = device).repeat( m, 1 )
    normals = torch.cat( [nL, nR, nB, nT ], dim = 0 )

    xy_b.requires_grad_( True )
    return xy_i, xy_b, normals

# Enforces the plane wave inlet condition on the left boundary (x = -1).
# The incoming plane wave is E_z^inc = e^(ikx), so at x = -1:
# E_r = cos(-k) = cos(k), E_i = sin(-k) = -sin(k)
# We penalize the network for deviating from this at the left edge.

def planewave_bc_loss( Er_b, Ei_b, coords_b, omega ):
    k = omega * np.sqrt(1.0)
    # Only apply to left boundary points where x == -1
    left_mask = ( coords_b[ :, 0] < -0.99 )
    if left_mask.sum() == 0:
        return torch.tensor( 0.0, device = device )
    x_left = coords_b[ left_mask, 0]
    Er_left = Er_b[ left_mask ]
    Ei_left = Ei_b[ left_mask ]

    # Plane wave values at x = -1

    Er_inc = torch.cos( k  *  x_left )
    Ei_inc = torch.sin(k * x_left )

    return ( ( Er_left - Er_inc ).pow(2) + (Ei_left - Ei_inc ).pow(2)).mean()


# Computes the Sommerfeld radiation boundary condition (BC) loss for complex fields.
# E_r_b: Real part of the electric field evaluated at the boundary points.
# E_i_b: Imaginary part of the electric field evaluated at the boundary points.
# coords_b: Tensor of boundary coordinates where the fields are evaluated (requires gradients).
# normals_b: Tensor of outward normal vectors corresponding to the boundary points.
# omega: Angular frequency of the wave (float, default is OMEGA).
# eps_boundary: Permittivity at the boundary (float, default is 1.0).
# Returns: A scalar tensor representing the mean squared Sommerfeld BC residual across all boundary points.
def sommerfeld_bc_loss( E_r_b, E_i_b, coords_b, normals_b, omega = OMEGA, eps_boundary = 1.0 ):
    g_r = torch.autograd.grad( E_r_b, coords_b, grad_outputs = torch.ones_like( E_r_b ), create_graph = True )[ 0 ]
    g_i = torch.autograd.grad( E_i_b, coords_b, grad_outputs=torch.ones_like( E_i_b ), create_graph = True )[ 0 ]
    dnEr = ( g_r * normals_b ).sum( dim = 1 )
    dnEi = ( g_i * normals_b ).sum( dim = 1 )
    k = omega * np.sqrt( eps_boundary )
    bc_r = dnEr + k * E_i_b
    bc_i = dnEi - k * E_r_b
    return ( bc_r.pow( 2 ) + bc_i.pow( 2 ) ).mean()


# Trains the neural network to solve the PDE with boundary conditions using Adam (and optionally LBFGS).
# weights: List of weight matrices of the network (all layers).
# biases: List of bias vectors of the network (all layers).
# activations: List of activation function names ("sine" or "linear") for each layer.
# omegas: List of frequency scaling factors (floats) corresponding to each layer.
# omega: Angular frequency of the wave (float, default is OMEGA).
# epochs: Number of training epochs for the Adam optimizer (default is EPOCHS).
# lr: Learning rate for the Adam optimizer (float, default is LR).
# loss_threshold: Threshold (float, default is EARLY_STOP_THR) for early stopping based on loss.
# lambda_bc: Weight (float, default is LAMBDA_BC) applied to the boundary condition loss term.
# circle: Optional tuple (x0, y0, r, eps_in) defining a circular region for the epsilon field (default is None).
# n_interior: Number of interior sample points per epoch (default is N_INTERIOR).
# n_boundary: Number of boundary sample points per epoch (default is N_BOUNDARY).
# use_lbfgs: Boolean flag (default is USE_LBFGS). If True, runs additional optimization with LBFGS after Adam.
# Returns: None (trains the model in place by updating weights and biases).
def train( weights, biases, activations, omegas, omega = OMEGA, epochs = EPOCHS, lr = LR, loss_threshold = EARLY_STOP_THR, lambda_bc = LAMBDA_BC, circle = None, n_interior = N_INTERIOR, n_boundary = N_BOUNDARY, use_lbfgs = USE_LBFGS):

    # First, I make sure all parameters require gradients so autograd can update them.
    for W in weights: W.requires_grad_( True )
    for b in biases:  b.requires_grad_( True )

    # I set up Adam on all weights and biases with the chosen learning rate.
    opt = torch.optim.Adam( [ *weights, *biases ], lr = lr )

    # I also attach a cosine learning-rate schedule that gradually lowers the LR during training.
    # The minimum LR is lr * ETA_MIN_FACTOR so we never fully stop updating.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR( opt, T_max = epochs, eta_min = lr * ETA_MIN_FACTOR )

    # This is the main training loop over epochs.
    for ep in range( epochs ):
        opt.zero_grad()   #  clear old gradients
      
        # At each epoch, I resample collocation points:
        # interior points for the PDE residual and boundary points to enforce the boundary condition.
        xy_i, xy_b, n_b = sample_points( n_interior=n_interior, n_boundary = n_boundary ) # interior collocation points, boundary collocation points, outward normal vectors at the boundary points

        # I run a forward pass on interior points to get the complex field Ez (split into real/imag).
        out_i = forward( xy_i, weights, biases, activations, omegas )
        Er_i = out_i[ : , 0 ]     # real part of Ez at interior points
        Ei_i = out_i[ : , 1 ]     # imaginary part of Ez at interior points

        # I compute Laplacians of the real and imaginary parts (needed for the Helmholtz residual).
        lap_r = laplacian( Er_i, xy_i )
        lap_i = laplacian( Ei_i, xy_i )

        # I build the spatial permittivity (ε), optionally with a dielectric circle,
        # and I evaluate the source term Jz at the interior points.
        eps_i = epsilon_field( xy_i, circle = circle )
        # J = Jz( xy_i )

        # These are the Helmholtz residuals for real and imaginary parts:
        #   Real: -∇²Er - ε ω² Er = 0
        #   Imag: -∇²Ei - ε ω² Ei + ω J = 0
        res_r = - lap_r - ( eps_i * ( omega**2 ) ) * Er_i
        res_i = - lap_i - ( eps_i * ( omega**2 ) ) * Ei_i # J removal instance here

        # I measure the PDE loss as the mean squared residual over all interior points.
        pde_loss = ( res_r.pow( 2 ) + res_i.pow( 2 ) ).mean()

        # I evaluate the field on boundary points for the radiation condition. Radiation, it ensures only outgoing waves at the boundary
        out_b = forward( xy_b, weights, biases, activations, omegas )
        Er_b = out_b[ : , 0 ]    # real part of Ez at boundary points
        Ei_b = out_b[ : , 1 ]    # imaginary part of Ez at boundary points

       # Compute the boundary condition loss by applying the Sommerfeld radiation condition at the boundary points. 
       # This checks how well the network’s field (Er_b, Ei_b)  satisfies ∂u/∂n = i k u, using the boundary coordinates (xy_b) and outward normals (n_b).
       # The result is a single scalar loss value (bc_loss) that penalizes reflections at the boundary.
        
        #bc_loss = sommerfeld_bc_loss( Er_b, Ei_b, xy_b, n_b, omega, eps_boundary = 1.0 )
        bc_sommerfield = sommerfeld_bc_loss( Er_b, Ei_b, xy_b, n_b, omega, eps_boundary = 1.0 )
        bc_inlet = planewave_bc_loss( Er_b, Ei_b, xy_b, omega )
        bc_loss = bc_sommerfield + bc_inlet


        # Total loss = PDE residual loss (interior physics) + weighted boundary condition loss
        # lambda_bc balances how strongly we enforce the radiation condition relative to the PDE.
        loss = pde_loss + lambda_bc * bc_loss

        # Backpropagate the loss to compute gradients for all network parameters.
        loss.backward()
        # Optional: gradient clipping for stability (prevents exploding gradients in PINNs)
        if GRAD_CLIP is not None:
            #The gradient clipping is applied at each optimization step, right after backprop but before Adam updates the weights. 
            #It continuously keeps gradients under control throughout the whole 20k epochs
            torch.nn.utils.clip_grad_norm_( [ *weights, *biases ], max_norm = GRAD_CLIP )  # This line doesn’t change the optimizer or the learning rate. it just rescales the gradients if they get too large. 
                                                                                          # Adam still does the update, and the cosine scheduler still controls the step size.”

        # Take an optimizer step with Adam, then update learning rate using the cosine scheduler.
        opt.step()
        sched.step()

        # I print a short progress line each epoch (and also if we’re already below the threshold),
        # so I can watch the PDE loss, the boundary loss, and the total loss as training proceeds.
        if ep % 1 == 0 or loss.item() < loss_threshold:
            print( f"Epoch {ep:5d} | PDE {pde_loss.item():.3e} | BC {bc_loss.item():.3e} | Total {loss.item():.3e}" )

        # Early stopping: if the loss is tiny AND we’ve trained at least 300 epochs,
        # I stop Adam early to save time—it’s not worth over-optimizing.
        if loss.item() < loss_threshold and ep > 300:
            print( f"Early stopping (Adam) at epoch {ep} | Loss: {loss.item():.6e}" )
            break

    # Optional second phase: use LBFGS to "polish" the solution after Adam.
    # LBFGS is a quasi-Newton optimizer with a line search; it often sharpens PINN results.
    # The second optimizer is called L-BFGS — Limited-memory Broyden–Fletcher–Goldfarb–Shanno. 
    # Unlike Adam, which is a first-order adaptive method, L-BFGS is a quasi-Newton method that approximates curvature from past gradients. 
   #It usually converges much faster once the network is close to a solution, which is why we use Adam for the bulk of training and then switch to L-BFGS to polish.”
    if use_lbfgs:
        #Here I flatten the weights and biases into one parameter list.  the * unpacks each list so the optimizer can see them all together.”
        params = [ *weights, *biases ]
        # LR is a starting guess; LBFGS does its own line search, max LBFGS iterations, how many past updates to approximate curvature, robust line search for stable steps
        lbfgs = torch.optim.LBFGS( params, lr = 1.0, max_iter = 500, history_size = 50, line_search_fn = "strong_wolfe" )



      
        def closure():
            lbfgs.zero_grad()
            xy_i, xy_b, n_b = sample_points( n_interior = n_interior, n_boundary = n_boundary )
            out_i = forward( xy_i, weights, biases, activations, omegas )
            Er_i = out_i[ : , 0 ]
            Ei_i = out_i[ : , 1 ]
            lap_r = laplacian( Er_i, xy_i) 
            lap_i = laplacian( Ei_i, xy_i )
            eps_i = epsilon_field( xy_i, circle = circle )
            ## J = Jz( xy_i )
            res_r = - lap_r - ( eps_i * ( omega**2 ) ) * Er_i
            res_i = - lap_i - ( eps_i * ( omega**2 ) ) * Ei_i # removed j instance
            pde = ( res_r.pow( 2 ) + res_i.pow( 2 ) ).mean()
            out_b = forward( xy_b, weights, biases, activations, omegas )
            Er_b= out_b[ : , 0 ]
            Ei_b = out_b[ : , 1 ]
            bc_sommerfeld = sommerfeld_bc_loss( Er_b, Ei_b, xy_b, n_b, omega, eps_boundary = 1.0 )
            bc_inlet = planewave_bc_loss( Er_b, Ei_b, xy_b, omega )
            bc = bc_sommerfeld + bc_inlet            
            L = pde + lambda_bc * bc
            L.backward()
            return L

        print("Starting LBFGS finishing pass...")
        lbfgs.step( closure )

    return weights, biases, activations, omegas

# Generates a uniform 2D grid of points in the square domain [-1, 1] × [-1, 1].
# N: Number of points along each axis (int, default is PLOT_N).
# Returns: A tensor of shape (N*N, 2) containing the (x, y) coordinates of the grid points, placed on the given device.
def make_plot_grid( N = PLOT_N ):
    xs = np.linspace( -1, 1, N )
    ys = np.linspace( -1, 1, N )
    gx, gy = np.meshgrid( xs, ys )
    grid = np.stack( [ gx.flatten(), gy.flatten() ], axis = -1 )
    return torch.tensor( grid, device = device )

# Renders field components, permittivity, and source distribution on a 2D grid, and saves the plots to file.
# weights: List of weight matrices of the trained network.
# biases: List of bias vectors of the trained network.
# activations: List of activation function names ("sine" or "linear") for each layer.
# omegas: List of frequency scaling factors (floats) corresponding to each layer.
# circle: Optional tuple (x0, y0, r, eps_in) defining a circular region for the epsilon field.
# fname: Output filename (string) for saving the rendered plot (saved in OUTDIR).
# Returns: None (saves a 2×2 grid of plots showing permittivity, source, Re(E_z), and Im(E_z)).
def render_and_save( weights, biases, activations, omegas, circle, fname):
    with torch.no_grad():
        coords = make_plot_grid( PLOT_N )
        out = forward( coords, weights, biases, activations, omegas )
        Er = out[ : , 0 ].reshape( PLOT_N,PLOT_N ).cpu().numpy()
        Ei = out[ : , 1 ].reshape( PLOT_N,PLOT_N ).cpu().numpy()
        eps = epsilon_field( coords, circle ).reshape( PLOT_N, PLOT_N ).cpu().numpy()
         # J = Jz( coords ).reshape( PLOT_N,PLOT_N ).cpu().numpy()

    vmax = max( np.abs( Er ).max(), np.abs( Ei ).max() )
    vmin = -vmax

    fig = plt.figure( figsize = ( 12,9 ) ) 
    ax = plt.subplot(2,2,1)
    ax.set_title( "Permittivity" )
    im = ax.imshow( eps, extent = [ -1,1,-1,1 ], origin = 'lower', cmap = 'gray' )
    plt.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04 )
    ax = plt.subplot( 2,2,2 )
    ax.set_title( "Plane Wave (inlet)" )
    xs = np.linspace( -1, 1, PLOT_N )
    k_val = OMEGA * np.sqrt( 1.0 )
    pw = np.cos( k_val * xs )
    ax.plot( xs, pw )
    ax.set_xlabel( "x" )
    ax.set_ylabel( "cos(kx)" )
  
    ax = plt.subplot( 2,2,3 )
    ax.set_title( "Real(E_z)" )
    im = ax.imshow( Er, extent = [ -1, 1, -1, 1 ], origin = 'lower', cmap = 'seismic', vmin = vmin, vmax = vmax )
    plt.colorbar( im, ax = ax, fraction = 0.046, pad = 0.04 )
  
    ax = plt.subplot( 2, 2, 4 )
    ax.set_title( "Imag(E_z)" )
    im = ax.imshow( Ei, extent = [ -1 , 1, -1 , 1 ], origin = 'lower', cmap = 'seismic', vmin = vmin, vmax = vmax )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    outpath = OUTDIR / fname
    plt.savefig( outpath, dpi = 200 )
    plt.close( fig )
    print( f"Saved: {outpath}" )


# Main execution function: sets random seeds, trains models for test cases, and evaluates performance.
# No arguments.
# Workflow:
# 1. Prints the current computation device.
# 2. Sets random seeds for reproducibility (torch and numpy).
# 3. Defines two cases:
# - "free_space": homogeneous medium without inclusions.
# - "dielectric_circle": medium with a circular dielectric inclusion (centered at origin).
# 4. For each case:
# - Builds the neural network model.
# - Trains the model with PDE and boundary condition losses.
# - Renders and saves plots of permittivity, source, and fields.
# - Computes and prints the RMS of the PDE residual.
# Returns: None.
def main():
    print( f"Device: {device}" )
    torch.manual_seed( 0 )
    np.random.seed( 0 )
    #cases = [ ( "free_space", None ), ( "dielectric_circle", ( 0.0, 0.0, 0.30, 2.0 ) ), ]
    cases = [ ( "planewave_free_space", None ), ( "planewave_dielectric", ( 0.0, 0.0, 0.30, 2.0 ) ), ]
    for name, circle in cases:
        print(f" Training case: {name} ")
        weights, biases, activations, omegas = build_model( input_dim = 2, hidden_dim = HIDDEN, output_dim = 2, num_hidden = NUM_HIDDEN, omega_0 = OMEGA0 )
        weights, biases, activations, omegas = train( weights, biases, activations, omegas, omega = OMEGA, epochs = EPOCHS, lr = LR, loss_threshold = EARLY_STOP_THR, lambda_bc = LAMBDA_BC, circle = circle, n_interior = N_INTERIOR, n_boundary = N_BOUNDARY, use_lbfgs = USE_LBFGS )
        render_and_save( weights, biases, activations, omegas, circle, fname=f"{name}_omega{OMEGA:g}.png" )

        rms = pde_residual_rms( weights, biases, activations, omegas, omega = OMEGA, circle = circle, N = PLOT_N )
        print( f"PDE residual RMS ({name}): {rms:.3e}" )

# Computes the root mean square (RMS) of the PDE residual on a uniform grid.
# weights: List of weight matrices of the trained network.
# biases: List of bias vectors of the trained network.
# activations: List of activation function names ("sine" or "linear") for each layer.
# omegas: List of frequency scaling factors (floats) corresponding to each layer.
# omega: Angular frequency of the wave (float, default is OMEGA).
# circle: Optional tuple (x0, y0, r, eps_in) defining a circular region for the epsilon field (default is None).
# N: Number of grid points along each axis for residual evaluation (int, default is PLOT_N).
# Returns: The PDE residual RMS as a Python float.
def pde_residual_rms( weights, biases, activations, omegas, omega = OMEGA, circle = None, N = PLOT_N ):
    coords = make_plot_grid( N ).requires_grad_( True )
    out = forward( coords, weights, biases, activations, omegas )
    Er = out[ : , 0 ]
    Ei = out[ : , 1 ]
    lap_r = laplacian( Er, coords )
    lap_i = laplacian( Ei, coords )
    eps = epsilon_field( coords, circle = circle )
    # J = Jz( coords )
    res_r = -lap_r - ( eps * ( omega**2 ) ) * Er
    res_i = -lap_i - ( eps * ( omega**2 ) ) * Ei  # removed instance of J here
    return torch.sqrt( torch.mean( res_r**2 + res_i**2 ) ).item()

if __name__ == "__main__":
    main()
