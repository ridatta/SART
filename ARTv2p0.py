import scipy.sparse as sparse
import numpy as np
import time

# Simultaneous Algebraic Tomography (Feb 5 2024)
# Written by R. Datta (MIT PSFC)

class ART_3D:
    def __init__(self,dx,dy,dz): # Initialize with size = (dz,dy,dx)
        self.S = np.ones((dz,dy,dx)) # generate initial 3D guess
        self.s = self.S.flatten() # flattened S
        self.s = np.reshape(self.s,(self.s.shape[0],1))
        self.p = np.array([]).reshape(0,1) # initialize projection vector
        self.W = sparse.csr_matrix(np.array([]).reshape(0,dx*dy*dz)) # initialize weight matrix
        
        
        
    def getW(self,phi,th):
        # Returns the weight matrix W for given detector angles
        # Detector angles:
        # 0 < phi < pi [rad] rotates along x
        # -pi < th < pi [rad] # rotates along z
        
        l, m, n = np.shape(self.S)
        detector_size = max([l,m,n]) # detector is a square of size equal to the largent dimension of the 3D object S
        
        print('Generating weight matrix for phi = %1.2f degrees and th = %1.2f degrees'%(np.rad2deg(phi),np.rad2deg(th)))
        
        if ((np.round(np.rad2deg(phi)) == 0) & (np.round(np.rad2deg(th)) == 0)):
            return self.get_sparse_Wxy() # Use the analytical solution for fast xz projection
        else:

            # Unrotated Detector
            center_x = (n-1)/2 
            center_y = (m-1)/2
            center_z = (l-1)/2
            detector_x = np.array(range(detector_size)) - center_x 
            detector_y = np.array(range(detector_size)) - center_y 
            if (detector_size > n) | (detector_size > m):
                detector_x -= (detector_size-1)//4
                detector_y -= (detector_size-1)//4

            detector_x, detector_y = np.meshgrid(detector_x,detector_y) 
            detector_z = np.zeros_like(detector_x) + (max([l,m,n])-1)/2 + 1*(max([l,m,n])-1) - center_y + 0.5
            detector_pos = np.column_stack((detector_x.flatten(),detector_y.flatten(),detector_z.flatten()))

            # Rotation Matrices
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(phi), -np.sin(phi)],
                           [0, np.sin(phi), np.cos(phi)]])

            Rz = np.array([[np.cos(th), -1*np.sin(th), 0],
                           [np.sin(th), np.cos(th), 0],
                           [0, 0, 1]])

            # Unrotated Detector lines
            line_z = np.arange(detector_z[0,0],detector_z[0,0]-2*(max([l,m,n])+1),step=-0.25)

            if np.round(np.rad2deg(phi)) == 90: # Exploit periodicity for LOS normal to z-axis (phi = 90 degrees)
                    W = sparse.csr_matrix(np.zeros((1,1*m*n))) # Initialize weight matrix
                    for ii in range(detector_size):
                        line_y = detector_y[0,ii] * np.ones_like(line_z)
                        line_x = detector_x[0,ii] * np.ones_like(line_z)
                        line = np.column_stack((line_x,line_y,line_z))

                        # Rotate Line
                        line_rot = np.matmul(Rx,line.T)
                        line_rot = line_rot.T

                        line_rot = np.matmul(Rz,line_rot.T)
                        line_rot = line_rot.T

                        line_rot[:,0] = line_rot[:,0] + center_x
                        line_rot[:,1] = line_rot[:,1] + center_y
                        line_rot[:,2] = line_rot[:,2] + center_z

                        # Convert to Integer values
                        line_rot = np.round(line_rot)
                        line_rot = line_rot.astype(int)

                        # Clip lines to within Object
                        idx = np.argwhere((line_rot[:,0] >= 0)
                                          & (line_rot[:,0] < n)
                                          & (line_rot[:,1] >= 0)
                                          & (line_rot[:,1] < m)
                                          & (line_rot[:,2] >= 0)
                                          & (line_rot[:,2] < l))

                        line_rot = line_rot[np.squeeze(idx)]

                        # Weight matrix
                        w = sparse.lil_matrix(np.zeros(1*m*n))
                        w[:,line_rot[:,0] + n * line_rot[:,1] +  n*m*line_rot[:,2]] = 1
                        W = sparse.vstack([W,sparse.csr_matrix(w)])


                    W = W[1:m+1,:m*n]
                    idx = np.argwhere(W.todense()==1)
                    row_idx = idx[:,0]
                    col_idx = idx[:,1]
                    for ii in range(1,l): # Repeat and concatenate
                        row_idx = np.hstack([row_idx,idx[:,0]+ii*W.shape[0]])
                        col_idx = np.hstack([col_idx,idx[:,1]+ii*W.shape[1]])

                    return sparse.csr_matrix(([1]*row_idx.shape[0],(row_idx,col_idx)), shape=(m*n,m*n*l))


            else: # Line Intersection Algorithm
                W = sparse.csr_matrix(np.zeros((1,l*m*n))) # Initialize weight matrix
                for jj in range(detector_size):
                    for ii in range(detector_size):

                        line_y = detector_y[jj,ii] * np.ones_like(line_z)
                        line_x = detector_x[jj,ii] * np.ones_like(line_z)
                        line = np.column_stack((line_x,line_y,line_z))

                        # Rotate Line
                        line_rot = np.matmul(Rx,line.T)
                        line_rot = line_rot.T

                        line_rot = np.matmul(Rz,line_rot.T)
                        line_rot = line_rot.T

                        line_rot[:,0] = line_rot[:,0] + center_x
                        line_rot[:,1] = line_rot[:,1] + center_y
                        line_rot[:,2] = line_rot[:,2] + center_z

                        # Convert to Integer values
                        line_rot = np.round(line_rot)
                        line_rot = line_rot.astype(int)

                        # Clip lines to within Object
                        idx = np.argwhere((line_rot[:,0] >= 0)
                                          & (line_rot[:,0] < n)
                                          & (line_rot[:,1] >= 0)
                                          & (line_rot[:,1] < m)
                                          & (line_rot[:,2] >= 0)
                                          & (line_rot[:,2] < l))

                        line_rot = line_rot[np.squeeze(idx)]

                        # Weight matrix
                        w = sparse.lil_matrix(np.zeros(l*m*n))
                        w[:,line_rot[:,0] + n * line_rot[:,1] + n*m*line_rot[:,2]] = 1
                        W = sparse.vstack([W,sparse.csr_matrix(w)])

                W = W[1:,:]
                return W
   
        
    
    
    def get_sparse_Wxz(self): # Generate weight matrix for integration along y (Analytical)
        l, m, n = np.shape(self.S)
        n_rows, n_cols = l*n, l*m*n
        
        row_idx = np.repeat(np.array(range(n_rows)), m)
        i, k = row_idx // n, np.mod(row_idx, n)
        j = np.tile(np.array([*range(m)]), row_idx.shape[0] // m)
        col_idx = i*n*m + n*j + k
        W = sparse.csr_matrix(([1]*row_idx.shape[0],(row_idx,col_idx)), shape=(n_rows,n_cols))

        return W
    
    def get_sparse_Wxy(self): # Generate weight matrix for integration along z (Analytical)
        l, m, n = np.shape(self.S)
        n_rows, n_cols = m*n, l*m*n
        
        row_idx = np.repeat(np.array(range(n_rows)), l)
        j, k = row_idx // n, np.mod(row_idx, n)
        i = np.tile(np.array([*range(l)]), row_idx.shape[0] // l)


        col_idx = i*n*m + n*j + k

        W = sparse.csr_matrix(([1]*row_idx.shape[0],(row_idx,col_idx)), shape=(n_rows,n_cols))
        return W
    
    
    def getLoss(self):
        # returns the current loss (RMSE)
        self.p_guess = self.W * self.s
        return np.sqrt(np.sum(np.squeeze((self.p_guess - self.p)**2)) / self.p.size)
    
    def run(self,targets,angles,tol=1e-3,max_iter=1000): # Run the algorithm
        # targets: list containing 2D projection targets as np arrays
        # angles: list containing LOSs as touples (phi,theta), e.g. angles = [(0,0),(90,0),(90,22.5)]
        # corresponding to targets
        t0 = time.time()
        print('Running 3D Art with Object Shape:',self.S.shape)
        
        for ii in range(len(targets)):
            p = targets[ii].flatten()
            p = np.reshape(p,(p.shape[0],1))
            self.p = np.vstack((self.p,p)) # combined target projection vector
            
            W = self.getW(phi=np.deg2rad(angles[ii][0]),th=np.deg2rad(angles[ii][1])) # sparse weight matrix
            self.W = sparse.vstack([self.W,W]) # combined weight matrix
        
        print('Weight matrices generated....')
        
        # construct D and M matrices
        self.D =  sparse.eye(self.s.size,dtype=float) # sparse identity matrix
        
        # Construct M (sparse matrix)
        row_idx = [*range(self.W.shape[0])]
        data = np.array(sum(np.sqrt(self.W.power(2).sum(axis=1)).tolist(),[]))
        data =  1/ self.W.shape[0] * data**(-1)
        self.M = sparse.csr_matrix((data,(row_idx,row_idx)),shape=(len(row_idx),len(row_idx)))
        print('D & M matrices generated....')
        
        # projection of initial guess
        self.p_guess = self.W * self.s
        
        current_loss = self.getLoss()
        print('Running....')
        n_iter = 0
        print('{0:4s}   {1:9s}'.format('iter', 'loss'))
        while ((current_loss >= tol) and (n_iter <= max_iter)):
            self.s = self.s + self.D * (self.W.transpose() * (self.M * (self.p - self.W * self.s)))
            prev_loss = current_loss
            current_loss = self.getLoss()
            
            if abs(prev_loss-current_loss)/prev_loss < 1e-6:
                break
            
            if (n_iter % 100 == 0): print('{0:4d} {1: 3.6f}'.format(n_iter, current_loss))
            n_iter+= 1
        
        # Results
        self.S = np.reshape(self.s,self.S.shape)
        
        # return final 2D projetcions list
        p_out = []
        for ii in range(len(targets)):
            idx = targets[ii].flatten().shape[0]
            p_out.append(self.p_guess[ii*idx:ii*idx+idx].reshape(targets[ii].shape)) 
        
        t1 = time.time()
        
        print('ART_3D terminated after %f s, %d iterations with loss = %f'%(t1-t0,n_iter,current_loss))
        
        return p_out
    
class ART_2D: # 2D ART for integration in y-direction; currently supports only 2 orthognal projections
    
    def __init__(self,guess):
        self.S = guess # generate initial 3D guess
        self.s = self.S.flatten() # flattened S
        self.s = np.reshape(self.s,(self.s.shape[0],1))
    
    
    def get_sparse_Wxz(self):
        l, m, n = np.shape(self.S)
        n_rows, n_cols = l*n, l*m*n
        
        row_idx = np.repeat(np.array(range(n_rows)), m)
        i, k = row_idx // n, np.mod(row_idx, n)
        j = np.tile(np.array([*range(m)]), row_idx.shape[0] // m)
        col_idx = i*n*m + n*j + k
        W = sparse.csr_matrix(([1]*row_idx.shape[0],(row_idx,col_idx)), shape=(n_rows,n_cols))

        return W
    
    def get_sparse_Wxy(self):
        l, m, n = np.shape(self.S)
        n_rows, n_cols = m*n, l*m*n
        
        row_idx = np.repeat(np.array(range(n_rows)), l)
        j, k = row_idx // n, np.mod(row_idx, n)
        i = np.tile(np.array([*range(l)]), row_idx.shape[0] // l)


        col_idx = i*n*m + n*j + k

        W = sparse.csr_matrix(([1]*row_idx.shape[0],(row_idx,col_idx)), shape=(n_rows,n_cols))
        return W
    
    
    def getLoss(self):
        # returns the current loss (RMSE)
        self.p_guess = self.W * self.s
        return np.sqrt(np.sum(np.squeeze((self.p_guess - self.p)**2)) / self.p.size)
    
    def run(self,target,tol=1e-3,max_iter=1000):
        print('Running 2D Art with Object Shape:',self.S.shape)
        t0 = time.time()
        p1 = target.flatten() # target projection vector  1
        p1 = np.reshape(p1,(p1.shape[0],1))
        
        self.p = p1 # target projection vector
        
        # get Weight matrices
        W1 = self.get_sparse_Wxz() # sparse matrices
        self.W = W1 # combined weight matrix
        print('Weight matrices constructed...')
        
        # construct D and M matrices
        self.D =  sparse.eye(self.s.size,dtype=float) # sparse identity matrix
        print('D constructed...')
        
        # Construct M (sparse matrix)
        row_idx = [*range(self.W.shape[0])]
        data = np.array(sum(np.sqrt(self.W.power(2).sum(axis=1)).tolist(),[]))
        data =  1/ self.W.shape[0] * data**(-1)
        self.M = sparse.csr_matrix((data,(row_idx,row_idx)),shape=(len(row_idx),len(row_idx)))
        print('M constructed...')
        
        # projection of initial guess
        self.p_guess = self.W * self.s
        
        current_loss = self.getLoss()
        n_iter = 0
        print('{0:4s}   {1:9s}'.format('iter', 'loss'))
        while ((current_loss >= tol) and (n_iter <= max_iter)):
            self.s = self.s + self.D * (self.W.transpose() * (self.M * (self.p - self.W * self.s)))
            prev_loss = current_loss
            current_loss = self.getLoss()
            
            if abs(prev_loss-current_loss)/prev_loss < 1e-6:
                break
            
            if (n_iter % 100 == 0): print('{0:4d} {1: 3.6f}'.format(n_iter, current_loss))
            n_iter+= 1
        
        proj_x = W1 * self.s # final xy proj.
        self.S = np.reshape(self.s,self.S.shape)
        
        t1 = time.time()
        
        print('ART_3D terminated after %f s, %d iterations with loss = %f'%(t1-t0,n_iter,current_loss))
        
        return np.reshape(proj_x,target.shape)
    


# Optimization based on 3D Gaussian Basis functions; Also used to generate phantoms
from scipy.optimize import minimize

class optimize3D:
    def __init__(self,xx,yy,zz):
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.neval = 0

    def generate3D(self,A,mu,sig):
        # A = [n x 1] Array of amplitudes
        # mu = [n x 3] Array of means
        # sig = [n x 3] Array of standard deviations
        out = np.zeros_like(self.xx)
        for ii in range(A.shape[0]):
            out += A[ii] *  np.exp(-(self.xx-mu[ii,0])**2/(2*sig[ii,0])**2) * np.exp(-(self.yy-mu[ii,1])**2/(2*sig[ii,1])**2) * np.exp(-(self.zz-mu[ii,2])**2/(2*sig[ii,2])**2)
        return out

    def convert_to_list(self,A,mu,sig):
        m = A.shape[0]
        u = np.zeros(7*m,)
        u[0:m] = A
        u[1*m:2*m] = mu[:,0]
        u[2*m:3*m] = mu[:,1]
        u[3*m:4*m] = mu[:,2]
        u[4*m:5*m] = sig[:,0]
        u[5*m:6*m] = sig[:,1]
        u[6*m:7*m] = sig[:,2]
        return u

    def convert_from_list(self,u):
        m = u.shape[0] // 7
        A = u[0:1*m]
        mu = np.vstack([u[1*m:2*m],u[2*m:3*m],u[3*m:4*m]]).T
        sig = np.vstack([u[4*m:5*m],u[5*m:6*m],u[6*m:7*m]]).T
        return A,mu,sig

    def get_msd(self,y_pred, y_true):
        return 1/ (y_pred.size) * np.sum((y_true - y_pred)**2)

    def loss_fn(self,variables):
        pred_3D = self.generate3D(*self.convert_from_list(variables)) # generate prediction
        proj_xy, proj_xz, proj_yz = self.get_projs(pred_3D) # get projections of guess
        return np.sqrt(self.get_msd(self.target_xy,proj_xy)**2 + self.get_msd(self.target_xz,proj_xz)**2) # mean sq. deviation


    def get_projs(self,array_3D):
        proj_xy = array_3D.sum(axis=0)
        proj_xz = array_3D.sum(axis=1)
        proj_yz = array_3D.sum(axis=2)
        return proj_xy, proj_xz, proj_yz
    
    def callback(self,variables):
        self.neval += 1
        if (self.neval % 10 == 0):
            print('{0:4d}   {1: 3.6f}'.format(self.neval, self.loss_fn(variables)))

    def run(self,target_xy,target_xz,*argv):

        self.target_xy = target_xy
        self.target_xz = target_xz

        m = 6 # max no. of modes
        A = np.zeros(m,)
        mu, sig = np.zeros((m,3)), 1e9 * np.ones((m,3))
        
        if (len(argv) == 0):
            # Initial Guess
            A[0] = 2
            mu[0,0] = 0.0; mu[0,1] = 0.0; mu[0,2] = 0.0
            sig[0,0] = 0.1; sig[0,1] = 0.25; sig[0,2] = 1
            A[1] = 2; mu[1,0] = 1.5; sig[1,0] = 0.5; sig[1,1] = 0.5
            A[2] = 2; mu[2,0] = -1.5; sig[2,0] = 0.5; sig[2,1] = 0.5

            guess = self.convert_to_list(A,mu,sig)
        else:
            guess = argv[0]
        
        print('{0:4s}   {1:9s}'.format('iter', 'loss'))
        out = minimize(self.loss_fn, x0 = guess, tol = 1e-3, callback=self.callback,
                       options={'disp': True, 'maxiter': 10000},method='Nelder-Mead')
        return out

    