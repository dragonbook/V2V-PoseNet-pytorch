import sys
import numpy as np
import pyigl as igl


def read_mesh_eigen(filename):
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.read_triangle_mesh(filename, V, F)

    stl = filename[-3:] == 'stl'
    if stl:
        # Remove duplicated vertices
        SV = igl.eigen.MatrixXd()
        SVI = igl.eigen.MatrixXi()
        SVJ = igl.eigen.MatrixXi()
        SF = igl.eigen.MatrixXi()

        igl.remove_duplicate_vertices(V, F, 1e-7, SV, SVI, SVJ, SF)

        V = SV
        F = SF
    
    return V, F


def write_mesh_eigen(filename, V, F):
    file_type = filename[-3:]
    if file_type == 'obj':
        igl.writeOBJ(filename, V, F)
    elif file_type == 'ply':
        igl.writePLY(filename, V, F)
    else:
        print('Cannot save {} file, only support obj or ply file'.format(file_type))
