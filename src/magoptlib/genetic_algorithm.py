import numpy as np
import cupy as cp
from .sh_gpu import cart2sph_gpu, sph2cart_gpu, compute_sh_matrix_gpu


def fitness_batch_gpu(
    angle_vectors,     # (pop_size, d) CuPy of float angles
    positions_gpu,     # (d,3) CuPy
    init_orientations_gpu,
    l_vals, m_vals,    # CPU arrays length Nm
    shX_gpu, shY_gpu, shZ_gpu,  # (d, Nm) CuPy
    pG_gpu,            # (3,) CuPy
    objective,
    alpha
):
    """
    Compute fitness values and total magnetic fields for a population of 
    Halbach configurations.

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
    init_orientations = init_orientations_gpu.reshape(1, d, 3, 3)
    init_orientations = init_orientations.repeat(pop_size, axis=0)

    # Build orientation matrices:
    Rz = cp.zeros((pop_size, d, 3, 3), dtype=cp.float64)
    Rz[:, :, 0, 0] = cp.cos(angle_vectors)
    Rz[:, :, 0, 1] = -cp.sin(angle_vectors)
    Rz[:, :, 1, 0] = cp.sin(angle_vectors)
    Rz[:, :, 1, 1] = cp.cos(angle_vectors)
    Rz[:, :, 2, 2] = 1.0

    orientations = cp.matmul(Rz, init_orientations)  # (pop_size, d, 3, 3)

    # pG_d = pG_gpu.shape[0] # for multiple points of interest

    # # Broadcast positions and pG:
    # pos_exp = positions_gpu.reshape(1, d, 1, 3).repeat(pop_size, axis=0)
    # pos_exp = pos_exp.repeat(pG_d, axis=2)  # (pop_size, d, 3)
    # pG_exp = pG_gpu.reshape(1, 1, pG_d, 3).repeat(pop_size, axis=0)
    # pG_exp = pG_exp.repeat(d, axis=1)  # (pop_size, d, 3)
    if pG_gpu.ndim == 1:
        pG_gpu = pG_gpu.reshape(1, 3)
    pG_d = pG_gpu.shape[0]

    # Broadcast positions and PoIs
    pos_exp = positions_gpu.reshape(1, d, 1, 3).repeat(pop_size, axis=0).repeat(pG_d, axis=2)  # (P,d,K,3)
    pG_exp = pG_gpu.reshape(1, 1, pG_d, 3).repeat(pop_size, axis=0).repeat(d, axis=1)        # (P,d,K,3)

    # Local coords: R_T @ (pG - pos)
    transl = pG_exp - pos_exp  # (pop_size, d, 3)
    R_T = orientations.transpose(0, 1, 3, 2)  # (pop_size, d, 3, 3)
    transl_unsq = transl.reshape(pop_size, d, pG_d, 3, 1)
    p_L = cp.matmul(R_T.reshape(pop_size, d, 1, 3, 3), transl_unsq)  # (pop_size, d, pG_d, 3, 1)
    p_L = p_L.reshape(pop_size, d, pG_d, 3)

    # Spherical coords of each (pop_size*d) point:
    x = p_L[:, :, :, 0]  # (pop_size, d)
    y = p_L[:, :, :, 1]
    z = p_L[:, :, :, 2]
    r, theta, phi = cart2sph_gpu(x, y, z)  # each (pop_size, d)

    Num_p = pop_size * d * pG_d  # total number of points

    # Reshape theta, phi and r
    phi_flat = phi.reshape(Num_p)
    theta_flat = theta.reshape(Num_p)
    r_flat = r.reshape(Num_p)
    a_val = 20.0

    # Build SH basis at all points
    Zp_flat, Xp_flat, Yp_flat = compute_sh_matrix_gpu(
        l_vals, m_vals,
        phi_flat, theta_flat,
        a_val, r_flat
    )

    Nm = len(m_vals)
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
    B_r_flat = cp.sum(Zp_flat * shZ_flat, axis=1)   # (Np_total,)
    B_th_flat = cp.sum(Xp_flat * shX_flat, axis=1)
    B_ph_flat = cp.sum(Yp_flat * shY_flat, axis=1)

    # Reshape back to (pop_size, d):
    B_r = B_r_flat.reshape(pop_size, d, pG_d)
    B_th = B_th_flat.reshape(pop_size, d, pG_d)
    B_ph = B_ph_flat.reshape(pop_size, d, pG_d)

    B_sph = cp.stack((B_r, B_th, B_ph), axis=3)  # (pop_size, d, Num_p, 3)

    # Convert each magnet’s B_sph to local Cartesian:
    M_all_flat = sph2cart_gpu(theta_flat, phi_flat)  # (pop_size*d, 3, 3)
    M_all = M_all_flat.reshape(pop_size, d,  pG_d, 3, 3)

    B_sph_unsq = B_sph.reshape(pop_size, d, pG_d, 3, 1)  # (pop_size, d, Num_p, 3, 1)
    M_t = M_all.transpose(0, 1, 2, 4, 3)                 # (pop_size, d, Num_p, 3, 3)
    B_cart_local = cp.matmul(M_t, B_sph_unsq)            # (pop_size, d, Num_p, 3, 1)
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

    if objective == 'homogeneity':
        # Calculate fitnesses based on field homogeneity
        fitnesses = (cp.max(Bx, axis=1) - cp.min(Bx, axis=1)) / (cp.mean(cp.abs(Bx), axis=1) + 1e-12)
    elif objective == 'maxBx':
        fitnesses = cp.mean(Bx, axis=1) \
         - alpha * (cp.mean(cp.abs(By), axis=1) + cp.mean(cp.abs(Bz), axis=1))
    else:
        raise ValueError(f"Unknown objective: '{objective}'")
    return fitnesses, B_total

# ============================================================
# GPU Accelerated Genetic Algorithm
# ============================================================


# Parent selection methods
def best_50_selection(population, population_size, sorted_idx):
    cutoff = max(1, population_size // 2)
    parent_idx = sorted_idx[:cutoff]  # top half is selected
    selection_group = population[parent_idx]
    parent = selection_group[np.random.randint(len(selection_group))]
    return parent


def tournament_selection(population, fitnesses_cpu, T_size, population_size, max_fitness_wins=True):
    parent_idx = np.random.choice(population_size, T_size, replace=False)  # select T_size unique indices
    tourn_fitnesses = fitnesses_cpu[parent_idx]  # shape (T_size,)

    # Sort indices by fitness
    if max_fitness_wins:
        winner_within = np.argmax(tourn_fitnesses)  # max fitness wins
    else:
        winner_within = np.argmin(tourn_fitnesses)  # min fitness wins

    winner_idx = parent_idx[winner_within]
    parent = population[winner_idx]
    return parent


def roulette_wheel_selection(population, fitnesses_cpu, population_size, minimize=True):
    """
    Roulette-wheel selection that supports both minimization and maximization.

    Uses a shift based on min() or max() of the fitnesses so that probabilities
    are well-scaled and nonnegative.
    """
    f = np.asarray(fitnesses_cpu, dtype=np.float64)

    if minimize:
        # We invert the scale so best individuals get largest weights.
        weights = f.max() - f
    else:
        # Shift so all weights are >= 0
        weights = f - f.min()

    # Handle case: all weights zero
    if np.allclose(weights, 0):
        parent_idx = np.random.randint(population_size)
        return population[parent_idx]

    probabilities = weights / weights.sum()
    cumulative_probabilities = np.cumsum(probabilities)

    r = np.random.rand()
    parent_idx = int(np.searchsorted(cumulative_probabilities, r, side="right"))
    if parent_idx >= population_size:
        parent_idx = population_size - 1

    return population[parent_idx]


def ACROMUSE_adaptive(population, fitnesses_cpu, T_size_max, population_size, HPD, HPD_max, minimize):
    ratio = HPD/HPD_max
    ratio = np.clip(ratio, 0.0, 1.0)       # keep it in [0,1]
    T_size = max(1, int(np.ceil(ratio * T_size_max)))

    parent_idx = np.random.choice(population_size, T_size, replace=False)  # select T_size unique indices
    tourn_fitnesses = fitnesses_cpu[parent_idx]  # shape (T_size,)
    if minimize:
        winner_within = np.argmin(tourn_fitnesses)  # min fitness wins
    else:
        winner_within = np.argmax(tourn_fitnesses)  # max fitness wins

    winner_idx = parent_idx[winner_within]
    child_fitness = fitnesses_cpu[winner_idx]
    parent = population[winner_idx]
    return parent, child_fitness


def genetic_algorithm_gpu(
    positions_cpu,
    init_orientations_gpu,
    l_vals_cpu, m_vals_cpu,
    shX_cpu, shY_cpu, shZ_cpu, points_of_interest,
    possible_angles_cpu,
    population_size,
    generations,
    mutation_rate,
    objective,
    parent_selection,
    alpha,
    elitism_frac=0.05
):
    """
    Runs a GA where each generation’s fitness is evaluated in one GPU batch.
    Returns: best_angles (d,), best_fitness (scalar), Bx,By,Bz at best, fitness_history (generations,).
    """

    # Upload static arrays
    positions_gpu_local = cp.asarray(positions_cpu, dtype=cp.float64)  # (d, 3)
    shZ_gpu_local = cp.asarray(shZ_cpu, dtype=cp.float64)         # (d, Nm)
    shX_gpu_local = cp.asarray(shX_cpu, dtype=cp.float64)
    shY_gpu_local = cp.asarray(shY_cpu, dtype=cp.float64)
    pG_gpu_local = cp.asarray(points_of_interest, dtype=cp.float64)   # (3,)
    possible_angles_gpu_loc = cp.asarray(possible_angles_cpu, dtype=cp.float64)  # (n,)
    HPD_values = np.zeros(generations)
    SPD_values = np.zeros(generations)

    if objective == 'maxBx':
        global_best = -1E10
    elif objective == 'homogeneity':
        global_best = 1E10

    d = positions_cpu.shape[0]       # number of magnets
    n = possible_angles_cpu.size     # number of possible angle choices

    # ---- objective direction ----
    minimize = (objective == 'homogeneity')
    maximize = (objective == 'maxBx')

    # Initialize population (CPU)
    population = np.random.randint(0, n, size=(population_size, d), dtype=np.int32)
    fitness_history = np.zeros(generations, dtype=np.float64)
    T_size_max = population_size // 5

    for gen in range(generations):
        # (A) Dispatch GPU‐batch fitness:
        pop_idx_gpu = cp.asarray(population, dtype=cp.int32)  # (pop_size, d)
        # pop_idx_gpu = cp.asarray(population, dtype=cp.int32)  # (pop_size, d)
        angle_vectors = possible_angles_gpu_loc[pop_idx_gpu]    # (pop_size, d) on GPU

        fitnesses_gpu, B_total_gpu = fitness_batch_gpu(
            angle_vectors,
            positions_gpu_local,
            init_orientations_gpu,
            l_vals_cpu, m_vals_cpu,
            shX_gpu_local, shY_gpu_local, shZ_gpu_local,
            pG_gpu_local,
            objective,
            alpha
        )
        cp.cuda.Stream.null.synchronize()

        fitnesses_cpu = cp.asnumpy(fitnesses_gpu)  # (pop_size,)
        B_total_cpu = cp.asnumpy(B_total_gpu)      # (pop_size, 3)

        sorted_idx = np.argsort(fitnesses_cpu) if minimize else np.argsort(-fitnesses_cpu)

        n_elite = 1 if parent_selection == 'ACROMUSE_adaptive' else max(1, int(population_size * elitism_frac))

        n_crossover = int(population_size - n_elite)  # size of the crossover group

        # Elitism: clone the top n_elite individuals
        elite_idx = sorted_idx[:n_elite]
        elites = population[elite_idx].copy()

        # Parent selection
        children = []
        # if the size of the crossover group is odd, produce one extra child
        n_cross_pairs = n_crossover // 2
        if n_crossover % 2 == 1:
            n_cross_pairs += 1
        for _ in range(n_cross_pairs):
            # Choose parent selection method
            if parent_selection == 'best_50':
                mom = best_50_selection(population, population_size, sorted_idx)
                dad = best_50_selection(population, population_size, sorted_idx)
            elif parent_selection == 'tournament':
                mom = tournament_selection(population, fitnesses_cpu, T_size=5, population_size=population_size, max_fitness_wins=not minimize)
                dad = tournament_selection(population, fitnesses_cpu, T_size=5, population_size=population_size, max_fitness_wins=not minimize)
            elif parent_selection == 'roulette':
                mom = roulette_wheel_selection(population, fitnesses_cpu, population_size, minimize=minimize)
                dad = roulette_wheel_selection(population, fitnesses_cpu, population_size, minimize=minimize)

            elif parent_selection == 'ACROMUSE_adaptive':
                w_raw = fitnesses_cpu.copy()  # keep raw fitnesses for later use
                w = w_raw / w_raw.sum()

                B_total_cpu = cp.asnumpy(B_total_gpu)  # (pop_size, 3)
                angle_vectors_cpu = cp.asnumpy(angle_vectors)  # (pop_size, d)

                # best_idx = np.argmax(fitnesses_cpu) # max fitness wins
                best_idx = np.argmin(fitnesses_cpu)  # min fitness wins

                best_f = fitnesses_cpu[best_idx]

                global_best = best_f
                # global_best_angles = angle_vectors_cpu[best_idx]

                new_pop = np.empty((population_size, d), dtype=population.dtype)
                angle_range = np.max(angle_vectors_cpu) - np.min(angle_vectors_cpu)

                elite_idx = np.argmin(fitnesses_cpu) if minimize else np.argmax(fitnesses_cpu)
                elite = population[elite_idx].copy()
                new_pop[0] = elite

                if np.cos(angle_vectors_cpu).any() < -1.0 or np.cos(angle_vectors_cpu).any() > 1.0:
                    print("Warning: cos(angle_vectors) out of [-1, 1] range!")
                    print("cos angles", np.cos(angle_vectors_cpu))
                if np.sin(angle_vectors_cpu).any() < -1.0 or np.sin(angle_vectors_cpu).any() > 1.0:
                    print("sin angles", np.sin(angle_vectors_cpu))
                C = np.sum(np.cos(angle_vectors_cpu), axis=0) * (1/population_size)
                S = np.sum(np.sin(angle_vectors_cpu), axis=0) * (1/population_size)
                if np.abs(C).any() > 1.0 or np.abs(S).any() > 1.0:
                    print("Warning: cos/sin angles out of [-1, 1] range!")
                    print("C:", C, "S:", S)
                R = np.sqrt(C**2 + S**2)  # (d,)
                if np.any(-2 * np.log(R)<0):
                    print("Warning: R is too small, leading to negative std!")
                    print("C:", C)
                    print("S:", S)
                    print("R:", R)
                    print("cos angles", np.cos(angle_vectors_cpu))
                    print("sin angles", np.sin(angle_vectors_cpu))
                G_std = np.sqrt(-2 * np.log(R))  # (d,) std of angles in radians
                SPD = float(np.mean(G_std / angle_range)) 

                w = fitnesses_cpu / fitnesses_cpu.sum()  # (P,)

                C_w = np.sum(w[:, None] * np.cos(angle_vectors_cpu), axis=0)

                S_w = np.sum(w[:, None] * np.sin(angle_vectors_cpu), axis=0)
                R_w = np.sqrt(C_w**2 + S_w**2)
                if np.any(-2 * np.log(R_w)<0):
                    print("Warning: R_w is too small, leading to negative std!")
                    print("R_w:", R_w)

                G_w_std = np.sqrt(-2 * np.log(R_w))  # std of angles in radians

                HPD = float(np.mean(G_w_std / angle_range))  # scalar
                HPD_values[gen] = HPD
                SPD_values[gen] = SPD

                SPD_max = 0.4
                HPD_max = 0.3
                K1 = 0.4
                K2 = 0.8
                K = 0.5
                P_c = (SPD / SPD_max)*(K2-K1) + K1
                P_div = ((SPD_max - SPD)/SPD_max)*K
                f_max = np.max(w_raw)
                f_min = np.min(w_raw)
                for i in range(population_size-1):
                    if np.random.rand() < P_c:
                        # Exploitation: uniform crossover, low mutation
                        mom, _ = ACROMUSE_adaptive(population, w_raw, T_size_max, population_size, HPD, HPD_max, minimize=minimize)
                        dad, _ = ACROMUSE_adaptive(population, w_raw, T_size_max, population_size, HPD, HPD_max, minimize=minimize)
                        # uniform crossover
                        toss = np.random.rand(d)
                        child = np.where(toss < 0.5, mom, dad) 
                        # Low mutation
                        mask = np.random.rand(d) < 0.01
                        child[mask] = np.random.randint(0, n, size=mask.sum())
                        # children.append(child)
                        new_pop[i+1, :] = child.copy()
                    else:
                        # Exploration: random mutation
                        child, child_fitness = ACROMUSE_adaptive(population, w_raw, T_size_max, population_size, HPD, HPD_max, minimize=minimize)
                        P_m = 0.5 * (P_div + (K * (f_max - child_fitness) / (f_max - f_min)))
                        mask = np.random.rand(d) < P_m
                        child[mask] = np.random.randint(0, n, size=mask.sum())
                        new_pop[i+1, :] = child.copy()
                population = new_pop  # Update population with new children

                fitness_history[gen] = global_best
                # fitness_history[gen] = fitnesses_cpue[sorted_idx[0]] ???

                continue 

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
        assert population.shape[0] == population_size, (
            f"Population size mismatch: {population.shape[0]} != {population_size}, "
            f"n_elite size {elites.shape[0]}, n_children size {children.shape[0]}"
        )

        best_f = fitnesses_cpu[sorted_idx[0]]
        fitness_history[gen] = best_f

    # Final evaluation
    pop_idx_gpu = cp.asarray(population, dtype=cp.int32)
    angle_vectors = possible_angles_gpu_loc[pop_idx_gpu]
    fitnesses_gpu, B_total_gpu = fitness_batch_gpu(
        angle_vectors,
        positions_gpu_local,
        init_orientations_gpu,
        l_vals_cpu, m_vals_cpu,
        shX_gpu_local, shY_gpu_local, shZ_gpu_local,
        pG_gpu_local, objective, alpha
    )
    cp.cuda.Stream.null.synchronize()
    final_fitnesses_cpu = cp.asnumpy(fitnesses_gpu)
    B_total_cpu = cp.asnumpy(B_total_gpu)  # (pop_size, 3)

    minimize = (objective == 'homogeneity')
    best_i = int(np.argmin(final_fitnesses_cpu)) if minimize else int(np.argmax(final_fitnesses_cpu))
    best_inds = population[best_i]                       # (d,)
    best_angles = possible_angles_cpu[best_inds]           # (d,)
    best_f = final_fitnesses_cpu[best_i]
    best_Bx = B_total_cpu[best_i, :, 0]
    best_By = B_total_cpu[best_i, :, 1]
    best_Bz = B_total_cpu[best_i, :, 2]
    print("Best fitness:", best_f)
    print("sum of Bx:", np.abs(best_Bx).sum())
    print("sum of By:", np.abs(best_By).sum())
    print("sum of Bz:", np.abs(best_Bz).sum())
    
    return best_angles, best_f, best_Bx, best_By, best_Bz, fitness_history
