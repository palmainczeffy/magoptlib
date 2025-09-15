import numpy as np
import cupy as cp
from .sh_gpu import cart2sph_gpu, sph2cart_gpu, compute_sh_matrix_gpu


def fitness_batch_gpu(
    angle_vectors,     # (pop_size, d) CuPy of float angles
    positions_gpu,     # (d,3) CuPy
    init_orientations_gpu,
    l_vals, m_vals,    # CPU arrays length Nm
    shZ_gpu, shX_gpu, shY_gpu,  # (Nm,) CuPy
    pG_gpu,            # (3,) CuPy
    alpha
):
    """
    Compute fitness values and total magnetic fields for a population of Halbach configurations.

    Parameters
    ----------
    angle_vectors : cupy.ndarray of shape (pop_size, d)
        Candidate magnetization angles per individual.
    positions_gpu : cupy.ndarray of shape (d, 3)
        3D positions of the magnets.
    init_orientations_gpu : cupy.ndarray of shape (d, 3, 3)
        Base orientation matrices for each magnet.
    l_vals, m_vals : np.ndarray
        Spherical harmonic degrees and orders.
    shZ_gpu, shX_gpu, shY_gpu : cupy.ndarray
        SH coefficients for each magnet.
    pG_gpu : cupy.ndarray of shape (n_points, 3)
        Points of interest to evaluate the field.
    alpha : float
        Penalty scaling factor (currently unused).

    Returns
    -------
    fitnesses : cupy.ndarray of shape (pop_size,)
        Fitness for each individual.
    B_total : cupy.ndarray of shape (pop_size, n_points, 3)
        Total magnetic field at all PoIs per individual.
    """

    pop_size, d = angle_vectors.shape
    init_orientations = init_orientations_gpu.reshape(1, d, 3, 3).repeat(pop_size, axis=0)


    # Build orientation matrices:
    Rz = cp.zeros((pop_size, d, 3, 3), dtype=cp.float64)
    Rz[:, :, 0, 0] = cp.cos(angle_vectors)
    Rz[:, :, 0, 1] = -cp.sin(angle_vectors)
    Rz[:, :, 1, 0] = cp.sin(angle_vectors)
    Rz[:, :, 1, 1] = cp.cos(angle_vectors)
    Rz[:, :, 2, 2] = 1.0

    orientations = cp.matmul(Rz, init_orientations)  # (pop_size, d, 3, 3)

    pG_d = pG_gpu.shape[0] # for multiple points of interest

    # Broadcast positions and pG:
    pos_exp = positions_gpu.reshape(1, d, 1, 3).repeat(pop_size, axis=0).repeat(pG_d, axis = 2)  # (pop_size, d, 3)
    pG_exp  = pG_gpu.reshape(1,1, pG_d, 3).repeat(pop_size, axis=0).repeat(d, axis=1)  # (pop_size, d, 3)

    # Local coords: R_T @ (pG - pos)
    transl = pG_exp - pos_exp  # (pop_size, d, 3)
    R_T = orientations.transpose(0,1,3,2)  # (pop_size, d, 3, 3)
    transl_unsq = transl.reshape(pop_size, d, pG_d, 3, 1)
    p_L = cp.matmul(R_T.reshape(pop_size, d, 1, 3, 3), transl_unsq)  # (pop_size, d, pG_d, 3, 1)
    p_L = p_L.reshape(pop_size, d, pG_d, 3)

    # Spherical coords of each (pop_size*d) point:
    x = p_L[:, :, :, 0]  # (pop_size, d)
    y = p_L[:, :, :, 1]
    z = p_L[:, :, :, 2]
    r, theta, phi = cart2sph_gpu(x, y, z)  # each (pop_size, d)

    
    Num_p = pop_size * d * pG_d # total number of points

    # Reshape theta, phi and r
    phi_flat   = phi.reshape(Num_p)    
    theta_flat = theta.reshape(Num_p)
    r_flat     = r.reshape(Num_p)
    a_val = 20.0

    # Build SH basis at all points
    Zp_flat, Xp_flat, Yp_flat = compute_sh_matrix_gpu(
        l_vals, m_vals,
        phi_flat, theta_flat,
        a_val, r_flat
    )  

    Nm  = len(m_vals)
    shZ_rep = shZ_gpu.reshape(1, d, 1, Nm)          # (1, d, Nm)
    shZ_rep = shZ_rep.repeat(pop_size, axis=0)      # (pop_size, d, Nm)
    shZ_rep = shZ_rep.repeat(pG_d, axis=2)          # (pop_size, d, pG_d, Nm)
    shZ_flat = shZ_rep.reshape(Num_p, Nm)           # (Np_total, Nm)

    shX_rep = shX_gpu.reshape(1, d, 1, Nm)          # (1, d, Nm)
    shX_rep = shX_rep.repeat(pop_size, axis=0)      # (pop_size, d, Nm)
    shX_rep = shX_rep.repeat(pG_d, axis=2)          # (pop_size, d, pG_d, Nm)
    shX_flat = shX_rep.reshape(Num_p, Nm)           # (Np_total, Nm)

    shY_rep = shY_gpu.reshape(1, d, 1, Nm)          # (1, d, Nm)
    shY_rep = shY_rep.repeat(pop_size, axis=0)      # (pop_size, d, Nm)
    shY_rep = shY_rep.repeat(pG_d, axis=2)          # (pop_size, d, pG_d, Nm)
    shY_flat = shY_rep.reshape(Num_p, Nm)           # (Np_total, Nm)

    # Compute B_r, B_th, B_ph for each (pop_size*d):
    B_r_flat  = cp.sum(Zp_flat * shZ_flat, axis=1)   # (Np_total,)
    B_th_flat = cp.sum(Xp_flat * shX_flat, axis=1)
    B_ph_flat = cp.sum(Yp_flat * shY_flat, axis=1)

    # Reshape back to (pop_size, d):
    B_r  = B_r_flat.reshape(pop_size, d, pG_d)
    B_th = B_th_flat.reshape(pop_size, d, pG_d)
    B_ph = B_ph_flat.reshape(pop_size, d, pG_d)

    B_sph = cp.stack((B_r, B_th, B_ph), axis=3)  # (pop_size, d, Num_p, 3)

    # Convert each magnet’s B_sph to local Cartesian:
    M_all_flat = sph2cart_gpu(theta_flat, phi_flat)  # (pop_size*d, 3, 3)
    M_all = M_all_flat.reshape(pop_size, d,  pG_d, 3, 3)

    B_sph_unsq = B_sph.reshape(pop_size, d, pG_d, 3, 1)  # (pop_size,d,Num_p,3,1)
    M_t = M_all.transpose(0,1, 2, 4,3)                   # (pop_size,d,Num_p,3,3)
    B_cart_local = cp.matmul(M_t, B_sph_unsq)            # (pop_size,d,Num_p,3,1)
    B_cart_local = B_cart_local.reshape(pop_size, d, pG_d, 3) 

    # Rotate to global Cartesian:
    B_cart_local_ = B_cart_local.reshape(pop_size, d, pG_d, 3, 1)
    orientations = orientations.reshape(pop_size, d, 1, 3, 3)
    B_global = cp.matmul(orientations, B_cart_local_).reshape(pop_size, d, pG_d, 3)  

    # Sum over magnets:
    B_total = cp.sum(B_global, axis=1)  # (pop_size, Num_p, 3

    Bx = B_total[:, :, 0]
    By = B_total[:, :, 1]
    Bz = B_total[:, :, 2]

    # Calculate fitnesses based on the magnetic field
    fitnesses = (Bx.max(axis=1) - Bx.min(axis=1)) / Bx.mean(axis=1) 

    return fitnesses, B_total



# ============================================================
# GPU Accelerated Genetic Algorithm
# ============================================================

# Parent selection methods
def best_50_selection(population, population_size, sorted_idx):
    cutoff = population_size // 2
    parent_idx = sorted_idx[-cutoff:]       # indices of upper half
    selection_group    = population[parent_idx]
    parent = selection_group[np.random.randint(len(selection_group))]
    return parent

def tournament_selection(population, fitnesses_cpu, T_size, population_size, max_fitness_wins=True):
    parent_idx = np.random.choice(population_size, T_size, replace=False)  # select T_size unique indices
    tourn_fitnesses = fitnesses_cpu[parent_idx]  # shape (T_size,)

    # Sort indices by fitness
    if max_fitness_wins == True:
        winner_within = np.argmax(tourn_fitnesses) # max fitness wins
    else:
        winner_within = np.argmin(tourn_fitnesses) # min fitness wins

    winner_idx = parent_idx[winner_within]
    parent = population[winner_idx]
    return parent


def roulette_wheel_selection(population, fitnesses_cpu, population_size):
    positive_fitness = fitnesses_cpu + np.min(fitnesses_cpu) 
    total_fitness = np.sum(positive_fitness)
    if total_fitness == 0:
        raise ValueError("Total fitness is zero.")
    probabilities = positive_fitness / total_fitness
    cumulative_probabilities = np.cumsum(probabilities)
    rand = np.random.rand()
    for i, cum_p in enumerate(cumulative_probabilities):
        if rand < cum_p:
            parent_idx = i
            break
    parent = population[parent_idx]
    return parent




def genetic_algorithm_gpu(
    positions_cpu, 
    init_orientations_gpu,
    l_vals_cpu, m_vals_cpu,
    shZ_cpu, shX_cpu, shY_cpu, pG_cpu,
    possible_angles_cpu,
    population_size,
    generations,
    mutation_rate,
    alpha
):
    """
    Runs a GA where each generation’s fitness is evaluated in one GPU batch.
    Returns: best_angles (d,), best_fitness (scalar), Bx,By,Bz at best, fitness_history (generations,).
    """


    # Upload static arrays
    positions_gpu_local     = cp.asarray(positions_cpu, dtype=cp.float64)    # (d,3)
    shZ_gpu_local           = cp.asarray(shZ_cpu, dtype=cp.float64)         # (d, Nm)
    shX_gpu_local           = cp.asarray(shX_cpu, dtype=cp.float64)
    shY_gpu_local           = cp.asarray(shY_cpu, dtype=cp.float64)
    pG_gpu_local            = cp.asarray(pG_cpu, dtype=cp.float64)          # (3,)
    possible_angles_gpu_loc = cp.asarray(possible_angles_cpu, dtype=cp.float64)  # (n,)


    d = positions_cpu.shape[0]       # number of magnets
    n = possible_angles_cpu.size     # number of possible angle choices

    # Initialize population (CPU)
    population = np.random.randint(0, n, size=(population_size, d), dtype=np.int32)
    fitness_history = np.zeros(generations, dtype=np.float64)

    for gen in range(generations):
        # (A) Dispatch GPU‐batch fitness:
        pop_idx_gpu   = cp.asarray(population, dtype=cp.int32)  # (pop_size, d)
        angle_vectors = possible_angles_gpu_loc[pop_idx_gpu]    # (pop_size, d) on GPU


        fitnesses_gpu, B_total_gpu = fitness_batch_gpu(
            angle_vectors,
            positions_gpu_local,
            init_orientations_gpu,
            l_vals_cpu, m_vals_cpu,
            shZ_gpu_local, shX_gpu_local, shY_gpu_local,
            pG_gpu_local, 
            alpha
        )  
        cp.cuda.Stream.null.synchronize()


        fitnesses_cpu = cp.asnumpy(fitnesses_gpu)  # (pop_size,)
        B_total_cpu = cp.asnumpy(B_total_gpu)      # (pop_size, 3)

        # Sort by fitness descending
        sorted_idx = np.argsort(fitnesses_cpu)

        # Compute group sizes
        elitism_frac    = 0.05
        #crossover_frac  = 0.95

        n_elite       = int(population_size * elitism_frac) # size of the elite group
        n_crossover   = int(population_size - n_elite)      # size of the crossover group

        # Elitism: clone the top n_elite individuals
        elite_idx = sorted_idx[:n_elite]                   
        elites    = population[elite_idx].copy()            

        # Parent selection
        children = []
        # if the size of the crossover group is odd, produce one extra child
        n_cross_pairs = n_crossover // 2
        if n_crossover % 2 == 1:
            n_cross_pairs += 1
        for _ in range(n_cross_pairs):
            # Choose parent selection method
            # mom = best_50_selection(population, population_size, sorted_idx)
            # dad = best_50_selection(population, population_size, sorted_idx)
            mom = tournament_selection(population, fitnesses_cpu, T_size=5, population_size=population_size, max_fitness_wins=False)
            dad = tournament_selection(population, fitnesses_cpu, T_size=5, population_size=population_size, max_fitness_wins=False)
            # mom = roulette_wheel_selection(population, fitnesses_cpu, population_size)
            # dad = roulette_wheel_selection(population, fitnesses_cpu, population_size)
            
            # single-point crossover
            cp_pt = np.random.randint(1, d)
            c1 = np.concatenate((mom[:cp_pt], dad[cp_pt:]))
            c2 = np.concatenate((dad[:cp_pt], mom[cp_pt:]))
            mask = np.random.rand(d) < mutation_rate
            c1[mask] = np.random.randint(0, n, size=mask.sum())
            c2[mask] = np.random.randint(0, n, size=mask.sum())
            children.append(c1)
            children.append(c2)
        # If overproduced by one (n_crossover is odd), drop the last
        children = np.array(children[:n_crossover], dtype=np.int32)

        # Check if sizes match 
        population = np.vstack((elites, children))
        assert population.shape[0] == population_size, f"Population size mismatch: {population.shape[0]} != {population_size}, n_elite size {elites.shape[0]}, n_children size {children.shape[0]}"

        best_f = fitnesses_cpu[ sorted_idx[0] ]
        fitness_history[gen] = best_f


    # Final evaluation
    pop_idx_gpu   = cp.asarray(population, dtype=cp.int32)
    angle_vectors = possible_angles_gpu_loc[pop_idx_gpu]
    fitnesses_gpu, B_total_gpu = fitness_batch_gpu(
        angle_vectors,
        positions_gpu_local,
        init_orientations_gpu,
        l_vals_cpu, m_vals_cpu,
        shZ_gpu_local, shX_gpu_local, shY_gpu_local,
        pG_gpu_local, alpha
    )
    cp.cuda.Stream.null.synchronize()
    final_fitnesses_cpu = cp.asnumpy(fitnesses_gpu)
    B_total_cpu = cp.asnumpy(B_total_gpu)  # (pop_size, 3)

    best_i      = np.argmin(final_fitnesses_cpu)
    best_inds   = population[best_i]                       # (d,)
    best_angles = possible_angles_cpu[best_inds]           # (d,)
    best_f      = final_fitnesses_cpu[best_i]
    best_Bx = B_total_cpu[best_i, :, 0]
    best_By = B_total_cpu[best_i, :, 1]
    best_Bz = B_total_cpu[best_i, :, 2]
    print("Best fitness:", best_f)
    print("sum of Bx:", np.abs(best_Bx).sum())
    print("sum of By:", np.abs(best_By).sum())
    print("sum of Bz:", np.abs(best_Bz).sum())
    
    return best_angles, best_f, best_Bx, best_By, best_Bz, fitness_history
