from math import exp
from numpy.core.fromnumeric import sort
from numpy.random import rand
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

# def get_vertex_normals(mesh):
#     return np.asarray(mesh.vertex_normals)

def draw_mesh(mesh):
    assert type(mesh)== o3d.cpu.pybind.geometry.TriangleMesh, 'type of mesh is incorrect'
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
    assert type(coordinates) == np.ndarray and coordinates.shape[1] == 3, "The input must be a numpy array with dimensions (X,3)"
    return np.append(coordinates[:,1:], coordinates[:,0:1], axis=1)

def plot_delaunay(delaunay):
    assert type(delaunay) == Delaunay, "Please provide parameter Delaunay."
    _ = delaunay_plot_2d(delaunay)
    plt.show()


def get_z_from_xy(xyz, xy): 
    xy_l = xyz[:, :2]

    if xy in xy_l: 
        # print("here")
        return xyz[np.where((xyz[:, :2] == xy).all(axis=1)),2][0]
    
    else:
        dist = np.linalg.norm(xy-xy_l, axis=1)

        ind = dist.argsort()[:5]
        d = 1/dist[ind]
        norm_d = d/np.sum(d)
        
        z = np.sum(np.dot(xyz[ind,2], norm_d))
        
        return z
    

def get_z_i_from_triangle(x,y,points_of_triangle):
    # This function needs the value of x,y that we need to find z_i of,
    # To compute this we need to know the points of the Delaunay triangle it belongs to 
    # ! As a numpy array: [[x1,y1,z1][x2,y2,z2],[x3,y3,z3]]
    assert type(points_of_triangle) == np.ndarray and coordinates.shape[0] == 3 and coordinates.shape[1] == 3, "The input must be a numpy array, with 3 points of the triangle which are of form [x,y,z]"
    p1,p2,p3 = points_of_triangle
    # We find the equation of the plane ax+by+cz+d = 0, and insert the values to find the z values
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    z_i = (d - a * x - b * y) / c
    return z_i


def find_boundaries(coordinates,points_per_boundary):
    '''
    Get the points for each boundary in cylindrical
    '''
    f = pd.DataFrame(coordinates[:,:2],columns=['x','y'])
    left = f.sort_values(by = ["x","y"],ascending=[True,True])
    right = f.sort_values(by = ["x","y"],ascending=[False,False])
    top = f.sort_values(by = ["y","x"],ascending=[False,True])
    bottom = f.sort_values(by = ["y","x"],ascending=[True,False])
    left_start = left.index[0]
    right_start = right.index[0]
    top_start = top.index[0]
    bottom_start = bottom.index[0]
    
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


def get_mapped_values(array,x_map,y_map):
    '''
    Return values after applying x map and y map. Give cylindrical get gridpoints
    '''
    return np.array([(x_map[x],y_map[y]) for x,y in array])

def make_np_array_tuple(np_array):
    '''
    Make A numpy array a tuple
    '''
    return tuple(map(tuple, np_array))

def make_np_array_set(np_array):
    '''
    Make a numpy array into a set
    '''
    return set(tuple(map(tuple, np_array)))



def get_individual(n,x_map,y_map,coordinates,coordinates_set):
    '''
    Gets an individual's chromosome
    Uses x_map to convert x values into gridpoints and y_map respectively
    coordinates has x,y,z values
    coordinates_set is used to get the points that are not on boundary
    '''
    # Get values on boundaries
    for_each_boundary = int(np.ceil(np.sqrt(n)/2))
    l,r,t,b,borders = find_boundaries(coordinates,for_each_boundary)
    # Map the values so that we have them in grid form
    l = get_mapped_values(l,x_map,y_map)
    r = get_mapped_values(r,x_map,y_map)
    t = get_mapped_values(t,x_map,y_map)
    b = get_mapped_values(b,x_map,y_map)
    borders = get_mapped_values(borders,x_map,y_map)
    
    left = make_np_array_set(l)
    right = make_np_array_set(r)
    top = make_np_array_set(t)
    bottom = make_np_array_set(b)

    not_selected = set(coordinates_set).difference(top.union(left,right,bottom))
    num_points_left_to_choose = n - 4* for_each_boundary
    
    not_selected = np.array(list(not_selected))
    # Get the non selected values 
    selected_from_rest = not_selected[np.random.choice(len(not_selected),num_points_left_to_choose,replace=False)]
    # Merge them all up in one list
    all_points_selected = np.concatenate((l,r,t,b,selected_from_rest),axis=0)
    return all_points_selected


def sign (p1,p2,p3):
    # I honestly don't know what this does but this seems to work
    p1x,p1y = p1
    p2x,p2y = p2
    p3x,p3y = p3
    
    return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

def PointInTriangle (point, a,b,c):
    # https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
    d1 = sign(point, a, b)
    d2 = sign(point, b, c)
    d3 = sign(point, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def get_points_in_triangle(vertices,coordinates_set):
    '''
    Gets the vertices of the triangle in grid form and extracts those that 
    fall within the triangle
    '''
    a,b,c = vertices
    xs = a[0],b[0],c[0]
    ys = a[1],b[1],c[1]
    potential = []
    # ? Do you want me to do all of this in a list comprehension
    for x in range(int(min(xs)),int(max(xs))+1):
        for y in range(int(min(xs)),int(max(xs))+1):
            if (x,y) in coordinates_set:
                potential.append((x,y))
    return [point for point in potential if PointInTriangle(point,a,b,c)]

def get_z(xyz,xy):
    xy_l = xyz[:, :2]

    if xy in xy_l: 
        # print(xyz[np.where((xyz[:, :2] == xy).all(axis=1)),2][0])
        return xyz[np.where((xyz[:, :2] == xy).all(axis=1)),2][0][0]
    else:
        raise Exception
def get_error_for_point(vertices_of_triangle,point,coordinates):
    '''
    By using the vertices of the triangle (gridpoint form) and the point (gridpoint form)
    It extracts the relevant z value and using interpolation finds the relevant z value
    '''
    z = point[2]
    p1,p2,p3 = vertices_of_triangle
    # Get x,y,z in our coordinate system "cylindrical"
    try:
        p1 = np.array([p1[0],p1[1],get_z(coordinates,p1)])
        p2 = np.array([p2[0],p2[1],get_z(coordinates,p2)])
        p3 = np.array([p3[0],p3[1],get_z(coordinates,p3)])
    except:
        raise Exception

    # print(p1,p2,p3)
    # print(point)
    # We find the equation of the plane ax+by+cz+d = 0, and insert the values to find the z values
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    z_i = (d - a * point[0]- b * point[1]) / c
    # Return error
    # print(np.abs(z-z_i))
    return np.abs(z-z_i)

def get_error_for_triangle(vertices,coordinates_map,coordinates_set,rev_x_map,rev_y_map):
    '''vertices_of_triangle,point,coordinates
    First we find all the points in the triangle and then for each point we find the error.
    NOTE: I think I made this function redundant 
    '''
    points = get_points_in_triangle(vertices,coordinates_set)
    
    return sum([get_error_for_point(vertices,point,coordinates_map,rev_x_map,rev_y_map) for point in points])

def get_triangulation(vertices,rev_x_map,rev_y_map):
    vertices = np.array([np.array([rev_x_map[x],rev_y_map[y]]) for x,y in vertices])
    return Delaunay(vertices)


def get_error(delaunay,coordinates):
    '''vertices_of_triangle,point,coordinates
    We go over each point, and if we find it in a triangle
    we find the relevant error
    '''
    error = 0 
    for coordinate in coordinates:
        triangle = delaunay.find_simplex(coordinate[:2])
        if triangle !=-1:
            error+=get_error_for_point(delaunay.points[delaunay.simplices[triangle]],coordinate,coordinates)
    return error

def sortPopulation_on_error(population,coordinates,rev_x_map,rev_y_map):
    return sorted(population,key = lambda x: get_error(get_triangulation(x,rev_x_map,rev_y_map),coordinates))

def choose_best(population,how_many):
    return population[:how_many]

def choose_worst(population,how_many):
    return population[-how_many:]

def mutate_individual(person,coordinates_set,prob_mutation,boundaries,coordinate_map,rev_x_map,rev_y_map):
    '''
    The purpose of this function is to take an individual and mutate all its element (with a probability) if they are not one of the boundary points
    '''
    # ? Seems a bit problematic uwu 😪 
    from math import dist
    coordinates = list(coordinates_map)
    for i,vertex in enumerate(person):
        coord = rev_x_map[vertex[0]],rev_y_map[vertex[1]],coordinate_map[(vertex[0],vertex[1])]
        if np.random.random() < prob_mutation and vertex not in boundaries:
            coordinates.sort(key= lambda x: dist((rev_x_map[x[0]],rev_y_map[x[1]],coordinate_map[(x[0],x[1])]),coord))
            person[i] = coordinates[np.random.randint(0,4)]
    return person


def crossover(person1,person2,coordinates_set,prob_mutation,x_max,boundaries):
    '''
    The purpose of this is to facilitate OAX crossover
    '''
    lim = x_max
    border = np.random.randint(0,lim+1)
    # Take the relevant elements from each parent.
    child1 = np.append(person1[person1[:,0]>=border],person2[person2[:,0]<border],axis = 0)
    child2 = np.append(person2[person2[:,0]>=border],person1[person1[:,0]<border],axis = 0)
    # If the child has less elements then add some which are not already removed
    if len(child1) < len(person1):
        chil = make_np_array_set(child1)
        can_add = np.array(list(coordinates_set.difference(chil)))
        chil = can_add[np.random.choice(can_add.shape[0],len(person1)-len(child1),replace=False)]
        # print(child1)
        # print(chil)
        
        child1 = np.vstack((child1,chil))
        
    if len(child2) < len(person1):
        chil = make_np_array_set(child2)
        can_add = np.array(list(coordinates_set.difference(chil)))
        chil = can_add[np.random.choice(can_add.shape[0],len(person1)-len(child2),replace=False)]
        # print(child2)
        # print(chil)
        
        child2 = np.vstack((child2,chil))
    # If child has more elements discard some. 
    if len(child1)>len(person1):
        child1 = child1[np.random.choice(child1.shape[0],len(person1),replace=False)]
        # If boundary is missing add that
        for i,boundary in enumerate(boundaries):
            if boundary not in child1:
                child1[i] = boundary
     
    if len(child2)>len(person1):
        child2 = child2[np.random.choice(child2.shape[0],len(person1),replace=False)]
        for i,boundary in enumerate(boundaries):
            if boundary not in child2:
                child2[i] = boundary
    return child1,child2

def get_individual_from_points(points,rev_x_map,rev_y_map,coordinate_map):
    # theta,z,r= reverse_x_map[x],reverse_y_map[y],coordinates_map[x,y]
    # x,y,z = r * np.cos(theta),r*np.sin(theta),z
    return np.array([np.array([coordinates_map[x,y]* np.cos(reverse_x_map[x]),coordinates_map[x,y]* np.sin(reverse_x_map[x]),reverse_y_map[y]]) for x,y in points])
pop_size = 16
n = 3774
num_generations = 5
new_mesh = o3d.io.read_triangle_mesh('laurana.ply')
coordinates = get_cartesian_coordinates_from_mesh(new_mesh)
cylindrical = convert_cartesian_to_cylindrical(coordinates)
coordinates = get_xyz_from_cylindrical(cylindrical)
# ! Getting grid world
x_values = np.sort(np.unique(coordinates[:,0]))
y_values = np.sort(np.unique(coordinates[:,1]))
x_max = len(x_values) -1
y_max = len(y_values) -1

y_max = y_values[-1]
x_map = {x_value: i for i,x_value in enumerate(x_values)}
reverse_x_map = {i: x_value for i,x_value in enumerate(x_values)}
y_map = {y_value: i for i,y_value in enumerate(y_values)}
reverse_y_map = {i: y_value for i,y_value in enumerate(y_values)}
from random import choice
coordinates_map = {(x_map[x],y_map[y]):z for x,y,z in coordinates}
coordinates_set = set(coordinates_map.keys())
population = [get_individual(n,x_map,y_map,coordinates,coordinates_set) for _ in range(pop_size)]
borders = get_mapped_values(find_boundaries(coordinates,1)[4],x_map,y_map)
remove = int(pop_size*0.5)
coordinates_set = set(list(coordinates_map.keys()))
for i in range(num_generations):
    print(f"generation {i}")
    population = sortPopulation_on_error(population,coordinates,reverse_x_map,reverse_y_map)
    # print(get_error(Delaunay(population[0]),coordinates_default_map,coordinates_set,reverse_x_map,reverse_y_map))
    new_mesh = o3d.geometry.PointCloud()
    new_mesh.points = o3d.utility.Vector3dVector(get_individual_from_points(population[0],reverse_x_map,reverse_y_map,coordinates_map))
    o3d.visualization.draw(new_mesh)
    parents = population[1:remove]

    population = population[:len(population)-remove]
    print("crossover time")
    for i in range(remove//2):
        a,b = crossover(choice(parents),choice(parents),coordinates_set,0.03,x_max,borders)
        population.append(a)
        population.append(b)
    population = [population[0]]+[mutate_individual(person,coordinates_set,0.05,borders,coordinates_map,reverse_x_map,reverse_y_map) for person in population[1:]]
# ! Identify best fitness
# ! Plot that
    # print(get_error(Delaunay(population[0]),coordinates_map,reverse_x_map,reverse_y_map))
    # sortPopulation_on_error(population,coordinates_map,x_map,y_map)