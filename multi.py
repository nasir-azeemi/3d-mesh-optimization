from matplotlib.colors import BoundaryNorm
import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay, delaunay_plot_2d,ConvexHull
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os.path
import pandas as pd

def load_mesh(filename):
    assert os.path.isfile(filename), "File does not exist."
    mesh = o3d.io.read_triangle_mesh(filename)
    return mesh

def get_vertices(mesh):
    assert type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh, 'type of input mesh is incorrect'
    return np.asarray(mesh.vertices)

def get_triangles(mesh):
    assert type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh, 'type of input mesh is incorrect'
    return np.asarray(mesh.triangles)

def get_vertex_normals(mesh):
    assert type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh, 'type of input mesh is incorrect'
    return np.asarray(mesh.vertex_normals)

def draw_mesh(mesh):
    assert (type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh or 
            type(mesh) == o3d.cpu.pybind.geometry.PointCloud), 'type of mesh is incorrect'
    o3d.visualization.draw(mesh)

def copy_mesh(mesh):
    assert type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh, 'type of mesh is incorrect'
    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(get_vertices(mesh))
    mesh2.triangles = o3d.utility.Vector3iVector(get_triangles(mesh))
    mesh2.vertex_normals = o3d.utility.Vector3dVector(get_vertex_normals(mesh))
    return mesh2

def get_cartesian_coordinates_from_mesh(mesh):
    assert type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh, 'type of mesh is incorrect'
    return get_vertices(mesh)

def convert_cartesian_to_cylindrical(coordinates):
    assert type(coordinates) == np.ndarray and coordinates.shape[1] == 3, "The input must be a numpy array with dimensions (X,3)"
    # Calculate r
    r = np.sqrt(np.square(coordinates[:,0]) + np.square(coordinates[:,1]))
    # Calculate theta,
    theta = np.arctan2(coordinates[:,1],coordinates[:,0])
    # Extract z value
    z = coordinates[:,2]
    # Create [[r,theta,z],[r,theta,z],[r,theta,z]]
    coordinates = np.stack((r,theta,z),axis=-1)

    return coordinates

def get_xyz_from_cylindrical(coordinates):
    # Extract rho, theta, z : x => theta, y => z, z => rho
    assert type(coordinates) == np.ndarray and coordinates.shape[1] == 3, f"The input must be a numpy array with dimensions (X,3)."
    return np.append(coordinates[:,1:], coordinates[:,0:1], axis=1)

# Global Variables
POP_SIZE = 5
GENERATIONS = 1

def init_pop(coord_lst):

    population = []

    for i in range(POP_SIZE):
        
        individual = []
        
        for j in range(len(coord_lst)):

            individual.append(np.random.choice([0, 1]))

        # print(individual)

        population.append(individual)
    
    return population

def find_boundries(coordinates,points_per_boundary):
    f = pd.DataFrame(coordinates[:,:2],columns=['x','y'])
    left = f.sort_values(by = ["x","y"],ascending=[True,True])
    right = f.sort_values(by = ["x","y"],ascending=[False,False])
    top = f.sort_values(by = ["y","x"],ascending=[False,True])
    bottom = f.sort_values(by = ["y","x"],ascending=[True,False])
    left_start = left.index[0]
    right_start = right.index[0]
    top_start = top.index[0]
    bottom_start = bottom.index[0]
    # print(left_start,right_start,top_start,bottom_start)
    
    lims = [right_start,top_start,bottom_start]
    left_limit = min([list(left.index).index(lim) for lim in lims])
    
    lims = [left_start,top_start,bottom_start]
    right_limit = min([list(right.index).index(lim) for lim in lims])
    
    lims = [left_start,right_start,bottom_start]
    top_limit = min([list(top.index).index(lim) for lim in lims])
    
    lims = [left_start,right_start,top_start]
    bottom_limit = min([list(bottom.index).index(lim) for lim in lims])
    left = left[:left_limit].to_numpy()
    left = left[left[:,0]<left[0,0] +0.25]
    right = right[:right_limit].to_numpy()
    
    right = right[right[:,0]>right[0,0] -0.25]
    top = top[:top_limit].to_numpy()
    
    top = top[top[:,1] >top[0,1]-10]
    bottom = bottom[:bottom_limit].to_numpy()
    
    bottom = bottom[bottom[:,1]<bottom[0,1] +10]
    # print(left,right,top,bottom)
    borders = [left[0],right[0],top[0],bottom[0]]
    a=left[np.random.choice(left.shape[0],points_per_boundary, replace=False)]
    b= right[np.random.choice(right.shape[0],points_per_boundary, replace=False)]
    c=top[np.random.choice(top.shape[0],points_per_boundary, replace=False)]
    d= bottom[np.random.choice(bottom.shape[0],points_per_boundary, replace=False)]
    return a,b,c,d, borders

def get_delaunay(coordinates, N = -1):
    # Randomly choose N points from the available to make Delaunay triangles
    # If N is -1, it will take all points
    # ! IMPORTANT: GIVE X,Y COORDINATES NOT XYZ Coordinates
    assert type(coordinates) == np.ndarray and coordinates.shape[1] == 2, "The input must be a numpy array with dimensions (X,2)"
    
    return Delaunay(coordinates)

def get_z_from_xy(xyz, xy): 
    ''' Using the x,y values determine the z values of the particular x,y pair '''

    xy_l = xyz[:, :2]
    # If the element exists within our original list of poits, we use that else we approximate 
    if xy in xy_l: 
        return xyz[np.where(xy_l == xy)[0][0]][2]
      
    else:
        # Distance between the our point and points in the list
        dist = np.linalg.norm(xy-xy_l, axis=1)
        # Get indices of the 5 minimum distances
        ind = dist.argsort()[:5]
        # 
        d = 1/dist[ind]
        norm_d = d/np.sum(d)
        z = np.sum(np.dot(xyz[ind,2], norm_d))
        
        return z

def get_z_i_from_triangle(x,y,points_of_triangle):
    # This function needs the value of x,y that we need to find z_i of,
    # To compute this we need to know the points of the Delaunay triangle it belongs to 
    # ! As a numpy array: [[x1,y1,z1][x2,y2,z2],[x3,y3,z3]]
    assert type(points_of_triangle) == np.ndarray and points_of_triangle.shape[0] == 3 and points_of_triangle.shape[1] == 3, f"The input must be a numpy array, with 3 points of the triangle which are of form [x,y,z], {points_of_triangle},{points_of_triangle.shape}"
    p1,p2,p3 = points_of_triangle
    # We find the equation of the plane ax+by+cz+d = 0, and insert the values to find the z values
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    z_i = (d - a * x - b * y) / c
    return z_i

def give_triangle_coords_with_xy(x, y, xyz, tri):

    tri_i = Delaunay.find_simplex(tri, [x,y])
    vert_i = tri.simplices[tri_i] 
    verts = tri.points[vert_i]

    z = np.array([[get_z_from_xy(xyz, [a,b]) for a,b in verts]])
    return np.concatenate((verts,z.T),axis=1)

def dist_xy(x,y, xyz, tri):
    
    trianglecoordinates = give_triangle_coords_with_xy(x, y, xyz, tri)
    return np.linalg.norm(get_z_from_xy(xyz, [x,y]) - get_z_i_from_triangle(x,y,trianglecoordinates))

def triangle_error(trianglecoords, xyz, tri):

    error = sum([np.square(dist_xy(i[0], i[1], xyz, tri)) for i in trianglecoords])

    return error

def convert_cylindrical_to_cart(cylindrical):
    x = cylindrical[:, 0:1] * np.cos(cylindrical[:, 1:2])
    y = cylindrical[:, 0:1] * np.sin(cylindrical[:, 1:2])
    z = cylindrical[:, 2:]
    return np.hstack((x, y, z))

def get_cylindrical_from_xyz(xyz):
    return np.append(xyz[:, 2:], xyz[:, 0:2], axis=1)

def make_np_array_set(np_array):
    '''
    Make a numpy array into a set
    '''
    return set(tuple(map(tuple, np_array)))

def calc_fitness(coordinates, population):

    fitness_lst = []
    
    for indiv in population:

        print("Indiv 1")
        
        indiv_coords = []

        for i in range(len(indiv)):
            
            if indiv[i] == 1:
                indiv_coords.append(coordinates[i])
        
        indiv_coords = np.array(indiv_coords)

        # plt.plot(indiv_coords[:,0], indiv_coords[:,1],'o')
        
        n = len(indiv_coords)*0.15
        lamb = 0.2

        # Find boundaries
        l,r,u,d,corners = find_boundries(indiv_coords, int(np.sqrt(n)//2))
        boundaries = np.vstack((l,r,u,d))

        not_selected = make_np_array_set(indiv_coords[:, :2]).difference(make_np_array_set(boundaries))
        points_r_coordinates = int(n - 4* np.ceil(np.sqrt(n)/2))
        # Find random coordinates excluding boundary coordinates.
        not_selected = np.array(list(not_selected))
        r_coordinates = not_selected[np.random.choice(len(not_selected),points_r_coordinates,replace=False)]

        # print(r_coordinates.shape)
        n_coordinates = np.vstack((r_coordinates[:, :2], l, r, u, d))

        # Use Delaunay Trianglation to calculate total error.
        error_total = 0

        tri = get_delaunay(n_coordinates)
        t_error = np.array([])

        for t_i in tri.simplices[:]:
            verts = tri.points[t_i]
            z = np.array([[get_z_from_xy(indiv_coords, [a,b]) for a,b in verts]])

            t_error = np.append(t_error, triangle_error(np.concatenate((verts,z.T),axis=1), indiv_coords, tri))
        
        error_total = np.sum(t_error)
        
        fitness_lst.append(error_total)

    return fitness_lst

def choose_parents(fitness_lst):
    
    a = np.random.choice(len(fitness_lst), 2)
    
    return a[0], a[1]

def crossover(parent1, parent2):

    cut = np.random.choice(len(parent1), 2)

    # print(cut)

    off1 = []
    off2 = []

    for i in range(0, cut[0]):
        off1.append(parent1[i])

    for i in range(cut[0], len(parent2)):
        off1.append(parent2[i])

    for i in range(0, cut[1]):
        off2.append(parent1[i])

    for i in range(cut[1], len(parent2)):
        off2.append(parent2[i])

    return off1, off2

def del_lowest(fitness_lst):

    lst = np.array(fitness_lst)

    ind = lst.argsort()[::-1][:2]

    return ind[0]

def mutate(population):

    for i in range(len(population)):

        for j in range(len(population[i])):

            if np.random.choice([0, 1]) == 1:

                if np.random.choice([0, 1]) == 1:
                    # print('o', population[i][j])
                    population[i][j] = abs(1-population[i][j])
                    # print('n', population[i][j])

    return population

def main():

    # Load Mesh and get coords.
    mesh = o3d.io.read_triangle_mesh('laurana.ply')
    new_mesh = copy_mesh(mesh)
    coordinates = get_cartesian_coordinates_from_mesh(new_mesh)
    cylindrical = convert_cartesian_to_cylindrical(coordinates)
    coordinates = get_xyz_from_cylindrical(cylindrical)

    # Init population
    population = np.array(init_pop(coordinates))

    # Calculate initial fitness of population.
    fitness_lst = []
    # fitness_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    fitness_lst = calc_fitness(coordinates, population)

    print(fitness_lst)

    # EA starts here
    for gen in range(GENERATIONS):

        print("Generation: ", gen+1)

        c = del_lowest(fitness_lst)

        # print(c)

        # print(population.shape)

        population = np.delete(population, c, axis = 0)
        fitness_lst.pop(c)

        
        c = del_lowest(fitness_lst)

        # print(c)

        population = np.delete(population, c, axis = 0)
        fitness_lst.pop(c)
        
        # print(population.shape)

        a, b = choose_parents(fitness_lst)

        # print(a, b)

        person1 = population[a]
        # print(person1)
        person2 = population[b]
        # print(person2)

        off1, off2 = crossover(person1, person2)
        
        # print(population.shape)
        population = np.append(population, [off1], axis = 0)
        population = np.append(population, [off2], axis = 0)
        # print(population.shape)

        population = mutate(population)

        fitness_lst = calc_fitness(coordinates, population)

        print(fitness_lst)

    lst = np.array(fitness_lst)

    sol = lst.argsort()[0]

    print("Highest Fitness: ", fitness_lst[sol])

    simp_mesh = population[sol]

    indiv_coords = []

    for i in range(len(simp_mesh)):
        
        if simp_mesh[i] == 1:
            indiv_coords.append(coordinates[i])
    
    indiv_coords = np.array(indiv_coords)

    print(indiv_coords)
    print(indiv_coords.shape)

    # indiv_coords = coordinates

    # new_tri = get_delaunay(new_coords[:,:2])
    new_cylinder = get_cylindrical_from_xyz(indiv_coords)
    new_cartes = convert_cylindrical_to_cart(new_cylinder)
    # new_tri = get_delaunay(new_cartes[:, :2])
    
    new_mesh = o3d.geometry.PointCloud()
    new_mesh.points = o3d.utility.Vector3dVector(new_cartes)

    draw_mesh(new_mesh)

if __name__ == "__main__":
    main()
