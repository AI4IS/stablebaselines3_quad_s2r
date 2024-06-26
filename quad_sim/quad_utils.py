import numpy as np
import numpy.random as nr
from numpy.linalg import norm
from copy import deepcopy

# dict pretty printing
def print_dic(dic, indent=""):
    for key, item in dic.items():
        if isinstance(item, dict):
            print(indent, key + ":")
            print_dic(item, indent=indent+"  ")
        else:
            print(indent, key + ":", item)

# walk dictionary
def walk_dict(node, call):
    for key, item in node.items():
        if isinstance(item, dict):
            walk_dict(item, call)
        else:
            node[key] = call(key, item)

def walk_2dict(node1, node2, call):
    for key, item in node1.items():
        if isinstance(item, dict):
            walk_2dict(item, node2[key], call)
        else:
            node1[key], node2[key] = call(key, item, node2[key])

# numpy's cross is really slow for some reason
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

# returns (normalized vector, original norm)
def normalize(x):
    n = norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

def norm2(x):
    return np.sum(x ** 2)

# uniformly sample from the set of all 3D rotation matrices
def rand_uniform_rot3d():
    randunit = lambda: normalize(np.random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left, _ = normalize(cross(up, fwd))
    # import pdb; pdb.set_trace()
    up = cross(fwd, left)
    rot = np.column_stack([fwd, left, up])
    return rot

# shorter way to construct a numpy array
def npa(*args):
    return np.array(args)

def clamp_norm(x, maxnorm):
    n = np.linalg.norm(x)
    return x if n <= maxnorm else (maxnorm / n) * x

# project a vector into the x-y plane and normalize it.
def to_xyhat(vec):
    v = deepcopy(vec)
    v[2] = 0
    v, _ = normalize(v)
    return v

def log_error(err_str, ):
    with open("/tmp/sac/errors.txt", "a") as myfile:
        myfile.write(err_str)
        # myfile.write('###############################################')


def quat2R(qw, qx, qy, qz):
    R = \
    [[1.0 - 2*qy**2 - 2*qz**2,         2*qx*qy - 2*qz*qw,         2*qx*qz + 2*qy*qw],
     [      2*qx*qy + 2*qz*qw,   1.0 - 2*qx**2 - 2*qz**2,         2*qy*qz - 2*qx*qw],
     [      2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1.0 - 2*qx**2 - 2*qy**2]]
    return np.array(R)

def qwxyz2R(quat):
    return quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])

def quatXquat(quat, quat_theta):
    ## quat * quat_theta
    noisy_quat = np.zeros(4)
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[3] 
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[2] 
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[1] 
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[0]
    return noisy_quat

def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    R = rot.reshape([3,3])
    w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    w4 = (4.0 * w)
    x = (R[2,1] - R[1,2]) / w4
    y = (R[0,2] - R[2,0]) / w4
    z = (R[1,0] - R[0,1]) / w4
    return np.array([w,x,y,z])

def rot2D(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

def rotZ(theta):
    r = np.eye(4)
    r[:2,:2] = rot2D(theta)
    return r

def randyaw():
    rotz = np.random.uniform(-np.pi, np.pi)
    return rotZ(rotz)[:3,:3]

def exUxe(e,U):
    """
    Cross product approximation
    exUxe = U - (U @ e) * e, where
    Args:
        e[3,1] - norm vector (assumes the same norm vector for all vectors in the batch U)
        U[3,batch_dim] - set of vectors to perform cross product on
    Returns:
        [3,batch_dim] - batch-wise cross product approximation
    """
    return U - (U.T @ rot_z).T * np.repeat(rot_z, U.shape[1], axis=1)

def cross_vec(v1,v2):
    return np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]]) @ v2

def cross_mx4(V1,V2):
    x1 = cross(V1[0,:],V2[0,:])
    x2 = cross(V1[1,:],V2[1,:])
    x3 = cross(V1[2,:],V2[2,:])
    x4 = cross(V1[3,:],V2[3,:])
    return np.array([x1,x2,x3,x4])

def cross_vec_mx4(V1,V2):
    x1 = cross(V1,V2[0,:])
    x2 = cross(V1,V2[1,:])
    x3 = cross(V1,V2[2,:])
    x4 = cross(V1,V2[3,:])
    return np.array([x1,x2,x3,x4])

def dict_update_existing(dic, dic_upd):
    for key in dic_upd.keys():
        if isinstance(dic[key], dict):
            dict_update_existing(dic[key], dic_upd[key])
        else:
            dic[key] = dic_upd[key]

class OUNoise:
    """Ornstein–Uhlenbeck process"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state
