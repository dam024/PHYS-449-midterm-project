"""
Generate network input from raw halo and initial condition catalogs.
"""


# Imports
import numpy as np
import os
from struct import iter_unpack
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


def particle_counts(boxid, sim_name="emulator_720box_planck", cell_length=4.):
    """
    Load initial conditions and generate counts on grid.

    Parameters
    ----------
    boxid - str, ID of box being analyzed with form 00-i for default simulation
    sim_name - str, name of the AbacusCosmos simulation being analyzed
    cell_length - float, length of the grid cells in Mpc/h

    Returns
    -------
    particle_counts - np.array, 3D array of particle counts saved as .npy file
    """

    # Initialize variable with path to simulation files
    path_root = ("/home/mj3chapm/scratch/abacus/{}_products/"
                 "{}_{}_products".format(sim_name, sim_name, boxid))

    ic_dir = ("{}/ic_{}_z0.1".format(path_root))
    N_ic_files = len(os.listdir(ic_dir))
    print("Number of IC Files: {}".format(N_ic_files))
    # for i in range(N_ic_files):
    for i in range(1):
        print("Starting IC file {}, elapsed time".format(i),
              dt.datetime.now() - start_time)
        with open("{}/ic_{}".format(ic_dir, i), "rb") as file:
            bdata = file.read()
            particles = iter_unpack("3h6f", bdata)

            
