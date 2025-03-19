import numpy as np
from sklearn.preprocessing import OneHotEncoder 


class LinearRegression:
    def __init__(self):
        self.betas = None

    def update_betas(self, X,y):
        self.betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        bias_term = np.ones((n_samples, 1))
        X = np.c_[bias_term, X]
        self.update_betas(X,y)
        self.y_mean = np.mean(y)
        
    def predict(self, X_new):
        n_samples = X_new.shape[0]
        bias_term = np.ones((n_samples, 1))
        X_new = np.c_[bias_term, X_new]
        prediction = X_new.dot(self.betas)
        return prediction
    
    def score(self, X, y):
        y_pred = self.predict(X)
        SSE = np.sum((y- y_pred) **2)
        SST = np.sum((y - self.y_mean)**2)
        r_squared = 1 - (SSE / SST)
        return r_squared
    
    
class RidgeRegression(LinearRegression):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = alpha
        
    def update_betas(self, X, y):
        identity_matrix = np.identity(X.shape[1])
        identity_matrix[0][0] = 0
        self.betas = np.linalg.inv(X.T.dot(X) + self.alpha * identity_matrix).dot(X.T).dot(y)

class GradientDescentRegressor:
    def __init__(self, learning_rate = .01, epochs = 1000, type = "batch", batch_size = 20, penalty = None, alpha = .1, l1_ratio = .5, random_state= None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.type = type
        self.batch_size = batch_size
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.weights = None
        np.random.seed(random_state)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        bias = np.ones(n_samples)
        X = np.c_[bias, X]
        for epoch in range(self.epochs):
            if self.type == "batch":
                gradient = self._compute_gradient(X, y)
            elif self.type == "mini batch":
                indices = np.random.choice(n_samples, self.batch_size, replace=False)
                gradient = self._compute_gradient(X[[indices]], y[[indices]])
            elif self.type == "stochastic":
                index = np.random.choice(n_samples)
                gradient = self._compute_gradient(X[[index]], y[[index]])
            else:
                raise TypeError("only batch, mini batch and stochastic are supported")

            self.weights -= self.learning_rate * 1 / (2*n_samples) * gradient
        self.y_mean = np.mean(y)
            
            
    def _compute_gradient(self, X, y):
        gradient = -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(self.weights)
        if self.penalty is not None:
            if self.penalty =="l1":
                penalty = self.alpha * np.sign(self.weights)
            elif self.penalty == "l2":
                penalty = 2 * self.alpha * self.weights
            elif self.penalty == "elastic net":
                l1_penalty = self.l1_ratio * self.alpha * np.sign(self.weights)
                l2_penalty = (1 - self.l1_ratio) * self.alpha * self.weights
                penalty = self.alpha * (l1_penalty + l2_penalty)
            else:
                raise ValueError("penalty can be None, l1, l2 or elastic net")
            
            gradient[1:] += penalty[1:]
        return gradient
    
    
    def predict(self, X):
        bias = np.ones(X.shape[0])
        X = np.c_[bias, X]
        return X.dot(self.weights)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sst = np.sum((y - self.y_mean)**2)
        sse = np.sum((y - y_pred)**2) 
        r_square = 1 - (sse / sst)
        return r_square       
                





class LogisticRegression:
    def __init__(self,iterations=1000,learning_rate=.01) :
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.weights=None
        self.bias=0
    def sigmoid(self ,z):
        return 1/(1+np.exp(-z))
        

    def fit(self,X,y):
        m,n=X.shape
        self.weights=np.zeros(n)
        for i in range(self.iterations):
            z=X.dot(self.weights)+self.bias
            y_predicted=self.sigmoid(z)
            dj_dw=-1/m*(X.T.dot(y-y_predicted))
            dj_db=-1/m*np.sum(y-y_predicted)

            self.weights-=self.learning_rate*dj_dw
            self.bias-=self.learning_rate*dj_db    

    def predict_propapility(self,X):
        z=X.dot(self.weights)+self.bias
        return self.sigmoid(z)
    def predict(self,X):
        p=self.predict_propapility(X)
        return np.where(p>=.5,1,0)
    
    def score(self,X,y):
        y_predict=self.predict(X)
        return np.sum(y_predict==y)/len(y)
        
class SoftMaxRegression:
    def __init__(self,eposhs=1000,learning_rate=.1):
        self.eposhs=eposhs
        self.learning_rate=learning_rate
        self.weights=None    #shape n*k
        self.bias=None  #shape k*1
    def _softmax(self,z):
        #z shape (m,k)
        #softmax shape(m,k)
        maxx=np.max(z,keepdims=True,axis=1)
        z-=maxx
        expo=np.exp(z)
        return expo/np.sum(expo,keepdims=True,axis=1)
        
    def _gradient_descent(self,X,y):
       for i in range(self.eposhs):
          z=np.dot(X,self.weights)+self.bias
          m,n=X.shape
          softmax=self._softmax(z)
          dj_dw=1/m*(X.T.dot(softmax-y)) #(n,m) *(m,k)
          dj_db= np.mean((softmax-y),axis=0)#k*1
          self.weights-=self.learning_rate*dj_dw
          self.bias-=self.learning_rate*dj_db
        

    def fit(self,X,y):
        m,n=X.shape
        encoder=OneHotEncoder()
        #we want y to pe 2d array
        y=y.reshape(-1,1)       
        y_encoded=encoder.fit_transform(y).toarray()
        k=len(np.unique(y))
        self.weights=np.zeros((n,k))
        self.bias=np.zeros(k)
        self._gradient_descent(X,y_encoded)


    def pedict_propas(self,X):
        Z=X.dot(self.weights)+self.bias
        return self._softmax(Z)
    
    def predict(self,X):
        p=self.pedict_propas(X)
        return np.argmax(p,axis=1)
    
    def score(self,X,y):
        p=self.predict(X)
        return np.mean(p==y)



    




