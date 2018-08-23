__author__ = "Dominik Noeger"
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Dominik Noeger"
__email__ = "dominik-franz-josef.noeger@stud.unileoben.ac.at"

import numpy as np
import functools
import os
from pymatgen.io.vasp import Poscar, Incar
from collections import OrderedDict
from math import sqrt
from random import random, shuffle
from itertools import cycle
from os.path import exists
from docopt import docopt
from pymatgen import periodic_table, Structure
from sympy.solvers.diophantine import diophantine
from sympy import symbols
from pymatgen.io.cif import CifWriter
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from fractions import Fraction
__doc__ = """
Generator for disordered non-collinear magentic momentum vector

Usage:
    magsqs ncl <poscar> [supercell <sx> <sy> <sz>] <lengths> [--plot] [--incar=<incar>] [--cif]
    magsqs cl  <poscar> [supercell <sx> <sy> <sz>] <lengths> [direction <dx> <dy> <dz>] [--incar=<incar>] [--cif] [--imbalanced]
    magsqs fm  <poscar> [supercell <sx> <sy> <sz>] <lengths> [direction <dx> <dy> <dz>] [--incar=<incar>] [--cif]

Commands: 
    ncl         (non-collinear) Distribute spins with random orientation
    cl          (collienar) Distribute spins collinear
    fm          (ferro-magnetic) Distribute spins in the same direction

Mandatory arguments:
    <poscar>        The path to the POSCAR file containing the structural information
    <length>        A list specifying the lengths of the magnetic moment vectors of 
                    the individual species. e.g Ni:2.0,Fe:4.5,Co:3.0 or Fe,Ni. If no
                    number is specified after the species symbol a length of 1.0 will
                    be set for that species. If a species exists in the POSCAR which
                    is not contained in this list the magnetic moment vector length will
                    be set to 0.0. If the list contains a element which is not present 
                    in the POSCAR file a error will be raised.
              
Optional commands:
    supercell       If "supercell" is specified the structure specified in <poscar> will
                    be stacked in different directions, furthermore <sx> <sy> <sz> 
                    arguments have to be present.
    <sx>            Number of repetitions of the <poscar> structure in x-direction
    <sy>            Number of repetitions of the <poscar> structure in y-direction
    <sz>            Number of repetitions of the <poscar> structure in z-direction
    direction       Only available for the commands "cl" and "fm". "direction" parameters
                    determine the orientation of the magnetic momenta
    <dx>            The x component of the magenetic momentum vector
    <dy>            The y component of the magenetic momentum vector
    <dz>            The z component of the magenetic momentum vector
    
Options:
    --incar=<incar> If a INCAR file is specified the MAGMOM line of it will be directly 
                    added to the specified file instead of printing it.
    --plot          Only available for the "ncl" command. Plots all momentum vectors as
                    arrows starting from [0,0,0] in a matplotlib window.
    --cif           Writes the output additionally to a magCIF file which can be opened
                    with VESTA
    --imbalanced    Only available for the "cl" command. If the net momenta of all atoms
                    of one species should not be 0, then choose this option. This might
                    help if one or more species have an odd number of atoms. 

Note:
    - The ouput files written get the extension "_MAG_SQS". However this behaviour can be
      altered by setting the environment variable "MAG_SQS_SUFFIX"
    - The number of decimal places printed in the MAGMOM line can be determined using the
      environment variable "MAG_SQS_PREC". Default is 3.              
"""

PREC_DEFAULT = 3
DEFAULT_SUFFIX = '_MAG_SQS'

#Take care of environment variables
if 'MAG_SQS_PREC' in os.environ:
    try:
        PREC = int(os.environ['MAG_SQS_PREC'])
    except:
        PREC = PREC_DEFAULT
else:
    PREC = PREC_DEFAULT

if 'MAG_SQS_SUFFIX' in os.environ:
    OUTPUT_SUFFIX = os.environ['MAG_SQS_SUFFIX']
else:
    OUTPUT_SUFFIX = DEFAULT_SUFFIX



class Arrow3D(FancyArrowPatch):
    # Arrow patch taken from:
    # https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def file_exists(pth):
    if exists(pth):
        print('WARNING: The file "{0}" does already exist! It will be overwritten!'.format(pth))
    return pth


def unit_vector(vec):
    """
    Calculates a unit vector for a
    :param vec: the input vector (shape: (n,))
    :return: output vector (shape (n,))
    """
    return vec / np.linalg.norm(vec)


def star(f):
    """Wraps a function for parameter unpacking in Python 3

    :param f: the function to wrap
    :return: function wrapper of f
    """
    @functools.wraps(f)
    def f_inner(args):
        return f(*args)

    return f_inner


def map_structure(s, predicate=lambda k: True, apply=lambda k: k):
    """
    Utility method to filter sites in a structure and extract properties:
    Syntax is [apply(site) for site in s.sites if predicate(site)]

    :param s: the structure
    :param predicate: filter expression or function. A pymatgen.core.PeriodicSite object will be passed
    :param apply: expression of function to apply. A pymatgen.core.PeriodicSite object will be passed
    :return: filtered and mapped list of sites
    """
    assert isinstance(s, Structure)
    return [apply(site) for site in s.sites if predicate(site)]

#Utility function to generate a random float between to borders
rand_between = lambda a, b: max([a, b]) - abs(b - a) * random()

#Utility function to remove whitespace from a string
remove_whitespace = lambda fstr: fstr.replace(' ', '').replace('\n', '').replace('\t', '')

#Utility function to create a format wildcard with certain precision
make_format_crumb = lambda i: '{' + str(int(i)) + ':.' + str(int(PREC)) + 'f}'


def allp(s, l, c=()):
    """
    Generates a list of all permutations of a set s of length l

    :param s: The elements in the set to permute
    :param c: growing list. [default: ()]
    :param l: length of the permutations
    :return: a list of all permutations
    """
    result = []
    if len(c) == l:
        return [c]
    else:
        for ss in s:
            r = allp(s, l, c=c + (ss,))
            if r:
                result.extend(r)
        return result


def vector_with_projection(vec):
    """
    Generates a random vector vec', which satisfies the condition k = dot(vec, vec').
    Where -1 < k 0 and ||vec'|| = 1. vec= [d, e, e*h]. Where k and h are chosen randomly.

    :param vec: input vector (shape: (3,))
    :return: output vector vec' (shape: (3,))
    """
    a, b, c = vec
    k_zeros = [-a, a, -sqrt(a ** 2 + b ** 2 + c ** 2), sqrt(a ** 2 + b ** 2 + c ** 2)]
    min_k = min(k_zeros)
    min_k_m = min([f for f in k_zeros if f != min_k])
    k = rand_between(min_k, min_k_m)
    border_zeros = [
        (-b * c + sqrt(-(a - k) * (a + k) * (a ** 2 + b ** 2 + c ** 2 - k ** 2))) / (a ** 2 + c ** 2 - k ** 2),
        -(b * c + sqrt(-(a - k) * (a + k) * (a ** 2 + b ** 2 + c ** 2 - k ** 2))) / (a ** 2 + c ** 2 - k ** 2)]
    h_min, h_max = min(border_zeros), max(border_zeros)
    # d_h_min, d_h_max = [2*a**2*f + 2*b*c + 2*c**2*f - 2*f*k**2 for f in (h_min, h_max)]
    h_args = (h_min, h_max) if 2 * a ** 2 + 2 * c ** 2 - 2 * k ** 2 < 0 else (h_max, abs(h_max) + h_max)
    h = rand_between(*h_args)
    d, e = ((k * (a ** 2 * h ** 2 + a ** 2 + b ** 2 + 2 * b * c * h + c ** 2 * h ** 2) + (b + c * h) *
             (a * sqrt(a ** 2 * h ** 2 + a ** 2 + b ** 2 + 2 * b * c * h + c ** 2 * h ** 2 - h ** 2 * k ** 2 - k ** 2)
              - k * (b + c * h))) / (a * (a ** 2 * h ** 2 + a ** 2 + b ** 2 + 2 * b * c * h + c ** 2 * h ** 2)),
            (-a * sqrt(
                a ** 2 * h ** 2 + a ** 2 + b ** 2 + 2 * b * c * h + c ** 2 * h ** 2 - h ** 2 * k ** 2 - k ** 2) + k * (
                     b + c * h)) / (a ** 2 * h ** 2 + a ** 2 + b ** 2 + 2 * b * c * h + c ** 2 * h ** 2))
    result = np.array([d, e, e * h])
    assert np.isclose(np.linalg.norm(result), 1.0)
    return result


def random_normal_vector(vec):
    """
    Choses a random normal unit vector. Therefore ||vec'|| = 1 and dot(vec, vec') = 0 holds true.
    :param vec: input vector (shape: (3,))
    :return: output vector vec' (shape: (3,))
    """
    a, b, c = vec
    d_borders = [-sqrt((b ** 2 + c ** 2) / (a ** 2 + b ** 2 + c ** 2)),
                 sqrt((b ** 2 + c ** 2) / (a ** 2 + b ** 2 + c ** 2))]
    d_min, d_max = min(d_borders), max(d_borders)
    d_args = (d_min, d_max) if -2 * a ** 2 - 2 * b ** 2 - 2 * c ** 2 < 0 else (d_max, abs(d_max) + d_max)
    d = rand_between(*d_args)
    e, f = (
    (-a * b * d + c * sqrt(-a ** 2 * d ** 2 - b ** 2 * d ** 2 + b ** 2 - c ** 2 * d ** 2 + c ** 2)) / (b ** 2 + c ** 2),
    -(a * c * d + b * sqrt(-a ** 2 * d ** 2 - b ** 2 * d ** 2 + b ** 2 - c ** 2 * d ** 2 + c ** 2)) / (b ** 2 + c ** 2))
    result = np.array([d, e, f])
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert np.isclose(np.dot(vec, result), 0)
    return result


def generate_spins_vecs_ncl(structure, lengths):
    """
    Calculates random magnetic moments with a zero net magnetic moment. Non collinear moments.

    :param structure: input structure (pymatgen.core.Structure)
    :param lengths: dictionary for mapping magnetic moment length to the species e.g {'Ni': 2.1, 'Co': 4.0}
    :return: a dictionary of the for {'Ni': [m1, m2, m3], 'Co' : [m4, m5, m6] } where m_i are arrays of shape (3,)
    """
    sum_vec = np.zeros((3,))
    objective = float('inf')
    #Start with the species which has the shortest length, to ensure the residuum vector can vanish in the end
    lengths = OrderedDict(sorted([(k, v) for k, v in lengths.items()], key=star(lambda k, v: v)))

    #Species with the longest vector. Keep it for the end
    last_species = [k for k, _ in lengths.items()][-1]
    last_species_length = lengths[last_species]
    #Calculate objective functions
    obj = lambda v, o: abs(np.linalg.norm(v) - o)
    magmoms = {k: [] for k, _ in lengths.items()}
    for spec, length in lengths.items():
        number_of_atoms = len(map_structure(structure, predicate=lambda s: s.specie.symbol == spec))

        #If its the last species leave 2 moments, to make the residuum vector vanish
        if spec == last_species:
            number_of_atoms -= 2

        for _ in range(number_of_atoms):
            #Make a initial guess for the magnetic moment
            current_magmom = unit_vector(np.random.uniform(-1, 1, (3,))) * length
            #Keep it if it already reduces the residuum vector
            if obj(current_magmom + sum_vec, last_species_length) > objective:
                #Find a random vector with negative inner product
                current_magmom = vector_with_projection(sum_vec) * length
            sum_vec += current_magmom
            objective = obj(sum_vec, last_species_length)
            magmoms[spec].append(current_magmom)

    #Make the residuum vector disappear
    residuum_length = np.linalg.norm(sum_vec)
    #Angle between random normal vector an the two remaining moments, in the place defined by cross(sum_vec, rand_norm_vec)
    triangle_angle = np.arcsin(residuum_length / (2.0 * last_species_length))
    #Calculate the height of the triangle
    r = last_species_length * np.cos(triangle_angle)

    center = r * random_normal_vector(-sum_vec)
    #Compute the remaining magnetic moments
    residuum_vec_1, residuum_vec_2 = center - sum_vec / 2.0, - center - sum_vec / 2.0
    sum_vec += residuum_vec_1
    sum_vec += residuum_vec_2
    magmoms[last_species].append(residuum_vec_1)
    magmoms[last_species].append(residuum_vec_2)
    return magmoms


def generate_spins_vecs_cl(structure, species, direction=np.array([1, 0, 0]), balanced=True):
    """
    Generates a set of collinear vector. If balanced is True and one of the species has an odd number
    of atoms an error will be raised. In imbalanced mode (balanced = False) the function tries to find a solution
    how the lengths of the spins can be arranged to result in a zero net vector, thus also also structures with
    odd numbers of atoms for a species could be handled. If no solution is found the function falls back to balanced
    mode. If the spins are not integer number they are converted to fractions to find integer solutions for the under
    yling diopahntine equation. The precision of the fracational representation is controlled by the PREC flag. However
    notice that with increasing PREC flag it is more likley that no solution is found
    :param structure: input structure (pymatgen.core.Structure)
    :param species: lengths: dictionary for mapping magnetic moment length to the species e.g {'Ni': 2.1, 'Co': 4.0}
    :param direction: the direction of the spins
    :param balanced: if balanced is True, the spins of each species cancel out each other
    :return: a dictionary of the for {'Ni': [m1, m2, m3], 'Co' : [m4, m5, m6] } where m_i are arrays of shape (3,)
    """
    magmoms = {}
    no_solution = False
    #Make a list of species, their length and asssign a sympy symbol
    remaining_atoms = {spec: (
    length, len(map_structure(structure, predicate=lambda s: s.specie.symbol == spec)), symbols('x{0}'.format(i)))
                       for i, (spec, length) in enumerate(species.items())}

    #Utility functions to find the specie which corresponds to symbol
    find_specie = lambda symbl: [spec for spec, (l, rma, sym) in remaining_atoms.items() if sym.name == symbl.name][0]
    #Utility function to find how many atoms are occupied by the species which corresponds to the symbol symbol
    find_remaining_atoms = lambda symbl: [rma for spec, (l, rma, sym) in remaining_atoms.items() if sym.name == symbl.name][0]

    if not balanced:
        #Imbalanced version, check if all length have integer value
        if not all([float(l_).is_integer() for l_, rma_, sym_ in remaining_atoms.values()]):
            print('INFO: Not all spins have integer values. Trying to find number.')
            #Wrap to moment lengths to fractions
            fractions = [(Fraction(l_).limit_denominator(10**PREC), sym_) for l_, rma_, sym_ in remaining_atoms.values()]
            numerator, denominator = 11,1
            #Find common denominator
            for f_ , sym_ in fractions:
                numerator *= f_.numerator
                denominator *= f_.denominator
            #Remap the coefficients to integer values
            coeffs = [(l_ * denominator, sym_) for l_, rma_, sym_ in remaining_atoms.values()]
            assert all([c_.is_integer() for c_, sym_ in coeffs])
        else:
            coeffs = [(l_, sym_) for l_, rma_, sym_ in remaining_atoms.values()]
            denominator = 1
        #Build diophantine equations
        eq = sum([c_ * sym_ for c_, sym_ in coeffs])

        solution = diophantine(eq)
        if len(solution) < 1:
            no_solution = True

        # Unpack solution
        solution = [(sol, symbols('x{0}'.format(i))) for i, sol in enumerate(next(iter(solution)))]
        #Find depending variables
        dependent_variables = set([s for expr, _ in solution for s in expr.free_symbols])
        #Find all permutations of small solutions with t != 0 with lentgh equal to the number of depndent variables
        sets = [dict(zip(dependent_variables, tset)) for tset in allp([-1, 1, 2, -2], len(dependent_variables))]
        # find minimum solution
        #Remap solutions and evaluate them for each parameter set
        results = [(i, tdict, [(find_specie(sym_),
                                find_remaining_atoms(sym_),
                                int(sol_.subs(tdict).evalf()),
                                find_remaining_atoms(sym_) > abs(int(sol_.subs(tdict).evalf())) and
                                (find_remaining_atoms(sym_) - abs(int(sol_.subs(tdict).evalf()))) % 2 == 0 , sym_)
                               for sol_, sym_ in solution]) for i, tdict in enumerate(sets)]
        #Find all solutions which can balanced
        results = [(i, tdict, set_) for i, (_, tdict, set_) in enumerate(results) if all([valid for _, _, _, valid, _ in set_])]
        if len(results) == 0 or no_solution:
            print('WARNING: No appropriate diophantine solution found. Checking if atoms can be balanced')
            min_set = [(spec_, ram_, 0, True, sym_) for spec_, (l_, ram_, sym_) in remaining_atoms.items()]
            if not all([ ram_ %2 == 0 for _, ram_, _, _, _ in min_set]):
                raise ValueError('Could not find diophantine solution. Also balanced mode ist not possible. Try a bigger supercell')
            else:
                print('INFO: Resuming in balanced mode')
        else:
        # Find minmal solution where max(rma - sol_i)
            min_index, min_params, min_set, _ = max(
                [(i, tdict, tset, functools.reduce(lambda x, y: x * y, [abs(n_) for spec_, rma_, n_, _, sym_ in tset]))
                 for i, tdict, tset in results], key=star(lambda i, td, s, pd: pd))
    else:
        #Balanced mode
        min_set = [(spec_, ram_, 0, True, sym_) for spec_, (l_, ram_, sym_) in remaining_atoms.items()]
        if not all([ram_ % 2 == 0 for _, ram_, _, _, _ in min_set]):
            raise ValueError('Balanced mode ist not possible. Try a bigger supercell')

    #Build magnetic moments
    for spec, atoms, diff, _, _ in min_set:
        balanced_spins = int((atoms - abs(diff)) / 2.0)
        spins = [1, -1] * balanced_spins + [int(np.sign(diff))]*abs(diff)
        shuffle(spins)
        l, _ , sym = remaining_atoms[spec]
        magmoms[spec] = [l*s*direction for s in spins]
        remaining_atoms[spec] = (l, 0, sym)

    #Add zero vector for each atom of the unmagnetic species
    remaining_species = [s for s, (_, rma_, _) in remaining_atoms.items() if rma_ > 0]
    if len(remaining_species) > 0:
        for spec in remaining_species:
            _, atoms, _ = remaining_atoms[spec]
            magmoms[spec] = [np.zeros((3,)) for _ in range(atoms)]

    return magmoms


def generate_spins_vecs_fm(structure, lengths, direction=np.array([1, 0, 0])):
    """
    Generates magnetic moments of specified length all within the same direction
    :param structure: input structure (pymatgen.core.Structure)
    :param species: lengths: dictionary for mapping magnetic moment length to the species e.g {'Ni': 2.1, 'Co': 4.0}
    :param direction: the direction of the spins
    :return: a dictionary of the for {'Ni': [m1, m2, m3], 'Co' : [m4, m5, m6] } where m_i are arrays of shape (3,)
    """
    magmoms = {spec_ : [direction*l_ for _ in range(len(map_structure(structure, predicate=lambda s: s.specie.symbol==spec_)))] for spec_, l_ in lengths.items()}
    return magmoms


def parse_lengths(lengthstr, sep=','):
    """
    Parser function for the command line arguments
    :param lengthstr: the argument from sys.argv
    :param sep: separator for the charact
    :return: a dictionary of the form {'Ni': 3.0, 'Co': 4.0}
    """
    crumbs = [remove_whitespace(crumb) for crumb in lengthstr.split(sep) if remove_whitespace(crumb) != '']
    lengths = {}
    for crumb in crumbs:
        if ':' in crumb:
            try:
                element, l = [remove_whitespace(c) for c in crumb.split(':') if remove_whitespace(c) != '']
            except:
                raise ValueError('Could not parse value "{0}"'.format(crumbs))
        else:
            element = crumb
            l = 1.0
            print('INFO: No length specified for element "{0}" resuming with 1.0'.format(element))
        try:
            periodic_table.Element(element)
        except:
            raise ValueError('Unknown element "{0}"'.format(element))
        else:
            try:
                l = float(l)
            except:
                raise ValueError('Could not parse number string "{0}"'.format(l))
            else:
                lengths[element] = l

    return lengths


def plot_magmoms(magmoms):
    """
    Plots the generated magnetic moments generated from the functions generate_spins_vecs_ncl, generate_spins_vecs_cl
    and generate_spins_vecs_fm in a matplotlib plot with arrow starting from [0, 0, 0]
    :param magmoms: a dictionary of the for {'Ni': [m1, m2, m3], 'Co' : [m4, m5, m6] } where m_i are arrays of shape (3,)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cycle('rgbkmo')
    min_x, max_x, min_y, max_y, min_z, max_z = [0, 1] * 3
    for spec, data in magmoms.items():
        c = next(colors)
        for mag_x, mag_y, mag_z in data:
            if mag_x < min_x:
                min_x = mag_x
            if mag_y < min_y:
                min_y = mag_y
            if mag_z < min_z:
                min_z = mag_z
            if mag_x > max_x:
                max_x = mag_x
            if mag_y > max_y:
                max_y = mag_y
            if mag_z > max_z:
                max_z = mag_z
            a = Arrow3D([0.0, mag_x], [0.0, mag_y], [0.0, mag_z], lw=1, color=c, arrowstyle='-|>', mutation_scale=7)
            ax.add_artist(a)
    plt.title('Magnetic moments')
    ax.set_xlim((min_x, max_x))
    ax.set_ylim((min_y, max_y))
    ax.set_zlim((min_z, max_z))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    plt.show()


def main():
    """
    logicMain function. Does the cli
    """

    options = docopt(__doc__)

    poscar_path = options['<poscar>']
    if not exists(poscar_path):
        raise IOError('POSCAR file "{0}" does not exist.'.format(poscar_path))
    else:
        try:
            structure = Poscar.from_file(poscar_path).structure
            species = set(map_structure(structure, apply=lambda s: s.specie.symbol))
        except:
            raise IOError('Could not parse POSCAR file "{0}"'.format(poscar_path))

    if options['--incar']:
        incar_path = options['--incar']
        if not exists(incar_path):
            raise IOError('INCAR file "{0}" does not exist.'.format(incar_path))
        else:
            try:
                incar = Incar.from_file(incar_path)
            except:
                raise IOError('Could not parse INCAR file "{0}"'.format(incar_path))

    if options['supercell']:
        try:
            sx, sy, sz = [int(options['<s{0}>'.format(k)]) for k in 'xyz']
        except:
            raise ValueError('Could not parse supercell parameters')
        else:
            structure.make_supercell([sx, sy, sz])

    if options['direction']:
        try:
            dx, dy, dz = [int(options['<d{0}>'.format(k)]) for k in 'xyz']
        except:
            raise ValueError('Could not parse direction')
        else:
            cl_direction = unit_vector(np.array([dx, dy, dz]))
    else:
        cl_direction = unit_vector(np.array([1.0, 0.0, 0.0]))

    lengths = parse_lengths(options['<lengths>'])
    if not all([k in species for k in lengths.keys()]):
        missing_elements = set(lengths.keys()) - species
        raise ValueError('The following element{0} {1} not specified in "{2}": {3}'.format(
            's' if len(missing_elements) > 1 else '',
            'are' if len(missing_elements) > 1 else 'is',
            poscar_path,
            ', '.join(list(missing_elements))
        ))
    for missing_element in species - set(lengths.keys()):
        lengths[missing_element] = 0.0

    if options['ncl']:
        magmoms = generate_spins_vecs_ncl(structure, lengths)
        # Check if net magnetic moment is really zero
        sum_vec = np.array([array for _, v in magmoms.items() for array in v])
        assert np.isclose(np.sum(sum_vec, axis=0), np.zeros((3,))).all()

    if options['cl']:
        if options['--imbalanced']:
            balanced = False
        else:
            balanced = True
        magmoms = generate_spins_vecs_cl(structure, lengths, direction=cl_direction, balanced=balanced)
        #Check if net magnetic moment is really zero
        sum_vec = np.array([array for _, v in magmoms.items() for array in v])
        assert np.isclose(np.sum(sum_vec, axis=0), np.zeros((3,))).all()

    if options['fm']:
        magmoms = generate_spins_vecs_fm(structure, lengths, direction=cl_direction)

    frac_coords_list = []
    species_list = []
    magmom_str = []
    magmoms_site_property = {'magmom': []}
    for spec, magm in magmoms.items():
        #Assemble the species list and the fractional coords in the same order as the generated magnetic moments
        current_frac_coords = map_structure(structure,
                                            predicate=lambda site: site.specie.symbol == spec,
                                            apply=lambda site: site.frac_coords)
        species_list.extend([spec] * len(current_frac_coords))
        frac_coords_list.extend(current_frac_coords)
        #If direction is given calculate the projection of the moment onto the direction which gives length and
        #orientation
        if options['direction'] or options['ncl']:
            magmom_str.extend(
                [' '.join([make_format_crumb(i) for i in range(3)]).format(mag_x, mag_y, mag_z) for mag_x, mag_y, mag_z in
                 magm])

            magmoms_site_property['magmom'].extend([mag_x, mag_y, mag_z] for mag_x, mag_y, mag_z in magm)
        else:
            magmom_str.extend([ make_format_crumb(0).format(np.dot(cl_direction, np.array([mag_x, mag_y, mag_z]))) for mag_x, mag_y, mag_z in magm])
            magmoms_site_property['magmom'].extend([np.dot(cl_direction, np.array([mag_x, mag_y, mag_z])) for mag_x, mag_y, mag_z in magm])

    #Create the new structure
    new_structure = Structure(structure.lattice, species_list, frac_coords_list, site_properties=magmoms_site_property)
    magmom_str = '  '.join(magmom_str)

    #Write the new Posar file
    Poscar(new_structure).write_file(file_exists('{0}{1}'.format(poscar_path, OUTPUT_SUFFIX)))
    if options['--incar']:
        incar['MAGMOM'] = magmom_str
        incar.write_file(file_exists('INCAR{0}'.format(OUTPUT_SUFFIX)))
    else:
        #Print the magnetic moment vector to the terminal
        print('MAGMOM = {0}'.format(magmom_str))

    if options['--plot']:
        plot_magmoms(magmoms)

    if options['--cif']:
        cif_writer = CifWriter(new_structure, write_magmoms=True)
        cif_writer.write_file(file_exists('{0}{1}.cif'.format(poscar_path, OUTPUT_SUFFIX)))

if __name__ == '__main__':
    main()
