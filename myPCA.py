import numpy as np
from numpy import linalg as LA

# Vamos a crear una funcion que corre PCA en un data set
def myPCA(X,num_comp):
    nd = np.shape(X)
    n = nd[0]
    d = nd[1]
    # Verificar que el numero de comp. princ. sea menor a la dimension de X
    if (num_comp > d):
        print('Error: numero de componentes principales es mayor a la dimension de X')
    elif (num_comp <= 0):
        print('Error: numero de comp. principales menor o igual a 0')
    else:
        A = np.asmatrix(X)
        # Debemos restar de cada columna la media de los datos
        for i in range(d):
            A[0:,i] = np.asarray(A[0:,i]) - np.mean(A[0:,i])
        
        B = A.transpose()
        #Calculamos la matriz de covarainza
        M = B*A
        # Diagonaliza W son lo val propios ordenados y v la matriz de vect. propios corresp.
        w, v = LA.eig(M)
        # Define la matrix para la reduccion de dim
        T = v[:,0:num_comp]
        # transforma los datos
        T_X = T.transpose()*B
        return {'eigen_val':w, 'eigen_vec':v, 'T':T, 'TX':(T_X.transpose())}
