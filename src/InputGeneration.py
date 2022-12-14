"""
Generate network input from raw halo and initial condition catalogs.
"""


# Imports
import datetime as dt
import numpy as np
import os
from struct import iter_unpack, unpack_from
import sys

from AbacusCosmos import Halos


def halo_counts(boxid, sim_name="emulator_720box_planck", cell_length=4.,
                min_mass=1.e12, max_mass=1.e14):
    """
    Load halo catalog and generate counts on grid.

    Parameters
    ----------
    boxid - str, ID of box being analyzed with form 00-i for default simulation
    sim_name - str, name of the AbacusCosmos simulation being analyzed
    cell_length - float, length of the grid cells in Mpc/h
    min_mass - float, the minimum halo mass included in the analysis in M_sun/h
    max_mass - float, the maximum halo mass included in the analysis in M_sun/h

    Returns
    -------
    halo_counts - np.array, 3D array of halo counts saved as .npy binary file
    """

    # Initialize variable with path to simulation files
    path_root = ("/home/mj3chapm/scratch/abacus/{}_products/"
                 "{}_{}_products".format(sim_name, sim_name, boxid))

    # Load the halo catalog using AbacusCosmos code
    cat = Halos.make_catalog_from_dir(dirname="{}/{}_{}_rockstar_halos/"
                                              "z0.100".format(path_root,
                                                              sim_name,
                                                              boxid),
                                      load_subsamples=False,
                                      load_pids=False)
    halos = cat.halos
    del cat

    # Filter halos to select the correct mass range
    halos = halos[(halos["m"] >= 1.e12) * (halos["m"] < 1.e14)]

    # Find maximum and minimum position values for defining the box size
    pos_mins = np.empty(3)
    pos_maxs = np.empty(3)
    for i in range(3):
        pos_mins[i] = np.floor(np.min(halos["pos"][:, i]))
        pos_maxs[i] = np.ceil(np.max(halos["pos"][:, i]))
    # Output minimum positions to check if catalog uses origin as lower bound,
    # or is symmetric about the origin
    print("Minimum positions:", pos_mins)

    # Calculate the number of cells and initialize halo count array
    cells_per_dim = np.floor((pos_maxs - pos_mins) / cell_length)
    print("Cells per dimension:", cells_per_dim)
    halo_counts = np.zeros((int(cells_per_dim[0]), int(cells_per_dim[1]),
                            int(cells_per_dim[2])))

    # Loop over the halo catalog, incrementing the halo counts grid
    for i in range(halos.shape[0]):
        # For each dimension subtract the minimum position to shift the
        # position to the range [0, box_length], then floor divide by the
        # cell_length to find the grid index.
        halo_counts[int((halos[i]["pos"][0] - pos_mins[0]) // cell_length),
                    int((halos[i]["pos"][1] - pos_mins[1]) // cell_length),
                    int((halos[i]["pos"][2] - pos_mins[2]) //
                        cell_length)] += 1

    # Save the output as a binary file with .npy formatting
    output_path = ("/home/mj3chapm/phys449/PHYS-449-midterm-project/input/"
                   "halo_counts/{}_{}_logmass-{:d}-{:d}_halo_"
                   "counts".format(sim_name, boxid, int(np.log10(min_mass)),
                                   int(np.log10(max_mass))))
    np.save(output_path, halo_counts)


def generate_halo_counts():
    """Run halo_counts for each simulation box."""
    boxids = ["00-0", "00-1", "00-2", "00-3", "00-4", "00-5", "00-6", "00-7",
              "00-8", "00-9", "00-10", "00-11", "00-12", "00-13", "00-14",
              "00-15"]
    for boxid in boxids:
        halo_counts(boxid)


def particle_counts(boxid, sim_name="emulator_720box_planck", cell_length=4.,
                    PPD=1440, boxsize=720.):
    """
    Load initial conditions and generate counts on grid.

    Parameters
    ----------
    boxid - str, ID of box being analyzed with form 00-i for default simulation
    sim_name - str, name of the AbacusCosmos simulation being analyzed
    cell_length - float, length of the grid cells in Mpc/h
    PPD - int, particles per dimension of the simulation
    boxsize - float, length of the simulation box in Mpc/h

    Returns
    -------
    particle_counts - np.array, 3D array of particle counts saved as .npy file
    """

    # Initialize start time for determining the run time given the large number
    # of particles and initial conditions files
    start_time = dt.datetime.now()

    # Initialize variable with path to simulation files
    path_root = ("/home/mj3chapm/scratch/abacus/{}_products/"
                 "{}_{}_products".format(sim_name, sim_name, boxid))

    # Initialize array for storing the particle counts
    particle_counts = np.zeros((int(boxsize / cell_length),
                                int(boxsize / cell_length),
                                int(boxsize / cell_length)))

    # Initialize variables and output the number of initial conditions files
    ic_dir = ("{}/ic_z0.1".format(path_root))
    N_ic_files = len(os.listdir(ic_dir))
    print("Number of initial conditions files: {}".format(N_ic_files))

    # Loop through the initial conditions files, populating the particle count
    # array for each one
    for i in range(N_ic_files):
        # Record the start time for evaluating the run time per file
        print("Starting IC file {}, elapsed time".format(i),
              dt.datetime.now() - start_time)
        # Open the binary initial conditions file
        with open("{}/ic_{}".format(ic_dir, i), "rb") as file:
            bdata = file.read()

            # Data is stored as c structs beginning with 3 unsigned shorts for
            # the particle indices in the initial grid, followed by 3 floats
            # for the Zel'dovich approximation (ZA) displacement, and then 3
            # floats for the ZA velocity. See the ICFormat description of
            # https://github.com/abacusorg/zeldovich-PLT for details.
            particles = iter_unpack("3h6f", bdata)
            # Loop through each particle in the file
            for part in particles:
                # Initialize list to hold position indices for the particle
                # count grid
                pos = []
                # Loop through each dimension (x, y ,z)
                for j in range(3):
                    # Calculate the position of the particle from the initial
                    # index on the grid (part[j]) converted to a position using
                    # the nubmer of particles in each dimension and the
                    # boxsize, then add the ZA displacement (part[j+3]), and
                    # modulo the boxsize to wrap any objects that have been
                    # translated beyond the edge of the box.
                    x = (part[j] / PPD * boxsize + part[j+3]) % boxsize
                    # Calculate the grid index of the particle by floor
                    # dividing by the cell length, and append to the position
                    # list.
                    pos.append(int(x // cell_length))

                # Increment the particle count grid.
                particle_counts[pos[0], pos[1], pos[2]] += 1.

        """
        # The reference paper uses the cloud in cells window function to
        # populate the grid, which helps to smooth edge cases and help with
        # aliasing when calculating the power spectrum. This code applies that
        # method using the nbodykit package, but was not used because of
        # difficulties installing nbodykit on the cluster.
            particle_data_first = unpack_from("3h6f", bdata, offset=0)
            particle_data_last = unpack_from("3h6f", bdata, offset=-32)
            N_particles = ((particle_data_last[0] -
                            particle_data_first[0]) * 1440**2 +
                           (particle_data_last[1] -
                            particle_data_first[1]) * 1440 +
                           (particle_data_last[2] -
                            particle_data_first[2]) + 1)

        partcat = BinaryCatalog("{}/ic_{}".format(ic_dir, i),
                                [('Index', ('h2', 3)),
                                 ('Position', ('f4', 3)),
                                 ('Velocity', ('f4', 3))], size=N_particles)
        partcat['Position'] = partcat['Position'] + (partcat['Index'] / PPD *
                                                     boxsize)
        mesh = partcat.to_mesh(Nmesh=int(boxsize / cell_length),
                               resampler="cic")

        particle_counts = particle_counts + mesh.compute
        """

    # Save the output as a binary file with .npy formatting
    output_path = ("/home/mj3chapm/phys449/PHYS-449-midterm-project/input/"
                   "particle_counts/{}_{}_particle_"
                   "counts".format(sim_name, boxid))

    # # Obsolete path original particle count generation
    # output_path = ("/home/mj3chapm/phys449/output/data_products/"
    #                "particle_counts/{}_{}_particle_"
    #                "counts".format(sim_name, boxid))

    np.save(output_path, particle_counts)
