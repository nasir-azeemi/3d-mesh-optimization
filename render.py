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

def make_point_cloud(points):
    mesh = o3d.geometry.PointCloud()
    mesh.points = o3d.utility.Vector3dVector(points)
    return mesh

def make_triangle_mesh(vertices, triangles):
    # Create a new Triangular Mesh
    mesh = o3d.geometry.TriangleMesh()
    # Add vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Add triangles
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

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

def get_delaunay(coordinates, N = -1):
    # Randomly choose N points from the available to make Delaunay triangles
    # If N is -1, it will take all points
    # ! IMPORTANT: GIVE X,Y COORDINATES NOT XYZ Coordinates
    assert type(coordinates) == np.ndarray and coordinates.shape[1] == 2, "The input must be a numpy array with dimensions (X,2)"
    
    return Delaunay(coordinates)

def plot_delaunay(delaunay):
    assert type(delaunay) == Delaunay, "Please provide parameter Delaunay."
    _ = delaunay_plot_2d(delaunay)
    plt.show()

# The delaunay functions are just to illustrate the idea, there needs to be functionality to make sure 
# to add functionality to add vertices from corners
def get_delaunay_from_file(filename,N=-1):
    mesh = load_mesh(filename)
    coordinates = get_cartesian_coordinates_from_mesh(mesh)
    cylindrical = convert_cartesian_to_cylindrical(coordinates)
    coordinates = get_xyz_from_cylindrical(cylindrical)
    delaunay = get_delaunay(coordinates[:, :2],N)
    
    return delaunay

def get_z_from_xy(xyz, xy): 
    ''' Using the x,y values determine the z values of the particular x,y pair '''

    xy_l = xyz[:, :2]
    # If the element exists within our original list of poits, we use that else we approximate 
    if xy in xy_l: 
        return xyz[np.where(xy_l == xy)[0][0]][2]
      
    else:
        print("here")
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
    # print(points_of_triangle)
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
    # I need things in this form. Where {x1,y1},{x2,y2},{x3,y3} are sides of the triangle
    
    trianglecoordinates = give_triangle_coords_with_xy(x, y, xyz, tri)
    return np.linalg.norm(get_z_from_xy(xyz, [x,y]) - get_z_i_from_triangle(x,y,trianglecoordinates))

def triangle_error(trianglecoords, xyz, tri):

    error = sum([np.square(dist_xy(i[0], i[1], xyz, tri)) for i in trianglecoords])

    return error

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

def convert_cylindrical_to_cart(cylindrical):
    x = cylindrical[:, 0:1] * np.cos(cylindrical[:, 1:2])
    y = cylindrical[:, 0:1] * np.sin(cylindrical[:, 1:2])
    z = cylindrical[:, 2:]
    return np.hstack((x, y, z))

def get_cylindrical_from_xyz(xyz):
    return np.append(xyz[:, 2:], xyz[:, 0:2], axis=1)


def plot_fitness(fitness,filename):
    # fitness = [449.48, 553.57, 696.783, 870.133, 1000.4, 1309.1]
    xs = np.linspace(1,len(fitness),len(fitness))
    plt.plot(xs,fitness, color='orange')
    plt.xlabel('Iteration No.')
    plt.ylabel('Error')
    plt.title('Error over Iterations')
    plt.savefig(filename+".png")

def make_np_array_set(np_array):
    '''
    Make a numpy array into a set
    '''
    return set(tuple(map(tuple, np_array)))
def make_np_array_tuple(np_array):
    '''
    Make A numpy array a tuple
    '''
    return tuple(map(tuple, np_array))



    
def main():
    #  Get in our desired coordinate system
    mesh = o3d.io.read_triangle_mesh('laurana.ply')
    new_mesh = copy_mesh(mesh)
    coordinates = get_cartesian_coordinates_from_mesh(new_mesh)
    cylindrical = convert_cartesian_to_cylindrical(coordinates)
    coordinates = get_xyz_from_cylindrical(cylindrical)

    n = int(len(coordinates)*0.15)
    print("n:", n)
    lamb = 0.2
    # Find boundaries
    l,r,u,d,corners = find_boundries(coordinates, int(np.ceil(np.sqrt(n)/2)))
    
    #  List of indices
    indx=[]
    # Waise I implemented this stuff in genetic.py. I can paste it
    boundaries = np.vstack((l,r,u,d))
    # Coordinates minus boundary points
    not_selected = make_np_array_set(coordinates[:, :2]).difference(make_np_array_set(boundaries))
    # no of n coordinates needed except boundary
    points_r_coordinates = int(n - 4* np.ceil(np.sqrt(n)/2))
    
    not_selected = np.array(list(not_selected))
    # Find random coordinates excluding boundary coordinates.
    r_coordinates = not_selected[np.random.choice(len(not_selected),points_r_coordinates,replace=False)]
    # all coordinates are combined to form n of coordinates
    n_coordinates = np.vstack((r_coordinates[:, :2], l, r, u, d))
    
    # total error of one iteration
    error_total = 0
    itera = 0
    error_history = []
    # Loop continues till error becomes stable
    while(True):
        print("Iteration: ", itera)
        itera += 1

        # Create Delaunay Triangle using selected coords.
        tri = get_delaunay(n_coordinates)
        # stores error of individual triangles
        t_error = np.array([])
        # finds error of each triangle and add them to the t_error
        for t_i in tri.simplices[:n]:
            
            verts = tri.points[t_i]
            z = np.array([[get_z_from_xy(coordinates, [a,b]) for a,b in verts]])
            t_error = np.append(t_error, triangle_error(np.concatenate((verts,z.T),axis=1), coordinates, tri))
        # checks if error is stable for termination
        old_error = error_total
        error_total = np.sum(t_error)
        error_history.append(error_total)
        
        print("error:", error_total)
        
        if old_error != 0 and np.absolute((error_total - old_error)/old_error) < 0.05:
            break 

        # Selecting and Removing vertices of min error triangles
        count = 0
        a = 0
        filtered_coordinate = n_coordinates
        to_del = set()
        # run till specified vertices are removed
        while (count < lamb * n) :
            # get vertices coordinates of the triangle after sorting on error
            ind = t_error.argsort()[a]
            tri_max = tri.simplices[ind]
            coord_max = tri.points[tri_max]
            
            points_to_choose_from = []

            # Check which points are already removed
            for point in coord_max:
                if tuple(point) not in to_del:
                    points_to_choose_from.append(tuple(point))
        
            if points_to_choose_from == []:
                pass
        
            else:
                to_del.add(tuple(points_to_choose_from[np.random.randint(0,len(points_to_choose_from))]))
                count += 1
        
                filtered_coordinate = set(tuple(map(tuple, filtered_coordinate)))
                
                filtered_coordinate = list(filtered_coordinate.difference(to_del))
            a +=1
        
        # Adding vertices on the centroids of the triangles with greater error
        count = 0
        a = 0
        to_add = set()
        while (count < lamb * n) :
            # get coordinates of the triangle by sorting the error list and picking highest error
            ind_rev = t_error.argsort()[::-1][a]
            tri_min = tri.simplices[ind_rev]
            coord_min = tri.points[tri_min]
            
            # centroid points are calculated
            centroid = np.sum(coord_min)/3
            j = 0
            # centroid is estimated on the grid points
            dist = np.linalg.norm(centroid - not_selected[:, :2], axis=1)
            while True:
                e = dist.argsort()[j]
                centroid1 = not_selected[e, 0], not_selected[e, 1]

                if centroid1 not in filtered_coordinate:
                    break
                j+=1
            
            # check if the centroid has already been added
            if centroid1 not in to_add:
                to_add.add(centroid1)
                filtered_coordinate.append(centroid1)
                count += 1
            else:
                pass
            a += 1
        
        # coordinates after this generation
        filtered_coordinate = np.array(filtered_coordinate)

        n_coordinates = filtered_coordinate

    # combine xy with z to form 3d data
    z = np.array([[get_z_from_xy(coordinates, i) for i in n_coordinates]])
    new_coords = np.concatenate((n_coordinates,z.T),axis=1)
    
    # Revert back to the starting coordinates
    new_cylinder = get_cylindrical_from_xyz(new_coords)
    new_cartes = convert_cylindrical_to_cart(new_cylinder)
    new_tri = get_delaunay(new_cartes[:, :2])
    # generate point cloud data
    new_mesh = o3d.geometry.PointCloud()
    new_mesh.points = o3d.utility.Vector3dVector(new_cartes)
    
    
    # o3d.visualization.draw(new_mesh)  

    # new_mesh1 = make_point_cloud(new_cartes)
    # new_mesh1.estimate_normals()

    # # estimate radius for rolling ball
    # distances = new_mesh1.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist 
    # # newer_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(new_mesh1,
    # #        o3d.utility.DoubleVector([radius, radius * 2]))
    # newer_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(new_mesh1,
    #        depth=10)
    # tri_mesh = trimesh.Trimesh(np.asarray(newer_mesh.vertices), np.asarray(newer_mesh.triangles),
    #                       vertex_normals=np.asarray(newer_mesh.vertex_normals))
    # print(new_tri.simplices)
    # new_mesh = make_triangle_mesh(new_cartes, new_tri.simplices)


    # plot the 3d mesh data
    draw_mesh(new_mesh)
    # plot graph for error
    plot_fitness(error_history, "error")
    
    # print(type(newer_mesh))
    # o3d.visualization.draw(newer_mesh) 
    # plot_delaunay(new_tri)
    # make_mesh(new_tri.simplices)
    # plt.plot(n_coordinates[:,0], n_coordinates[:,1], 'ro')
    # plt.show()



if __name__ == "__main__":
    main()