#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import math
class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 50):
        '''
        Initializes Parameters of the  Logistic Regression Model
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
  
    

    # def Product(self, weight, X):
    # 	# weight and X are all 1 dimension list
    # 	res=0
    # 	for i in range(len(X)):
    # 		res=res+weight[i]*X[i]
    # 	return res
    
    def calculateGradient(self, weight, X, Y, regLambda):# Assume X[0] is 1, which is illusory attribute for W[0]
        '''
        Computes the gradient of the objective function
        Arguments:
        
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is (d+1)-by-1 dimensional numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an (d+1)-by-1 dimensional numpy matrix
        '''
        Gradient=np.zeros((weight.shape[0],1));# matrix, not list
        penalty=[regLambda*theta_j for theta_j in weight] #list not matrix
        penalty[0]=0 # do not to regularize the theta0 parameter
        list_x=np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
        		list_x[i]=np.dot(weight[:,0],X[i,:])
        list_x=self.sigmoid(list_x)
        for j in range(weight.shape[0]):
        	tmp=np.add(list_x,-Y)
        	Gradient[j]=np.dot(tmp[:,0],X[:,j])+penalty[j]

   #      for i in range(X.shape[0]):
   #      	#X[i][0]=1 # the illusory X[i][0] are set to be 1
   #      	tmp=1/(1+math.exp(-np.dot(weight[:,0],X[i,:])))-Y[i]
   #      	for j in range(X.shape[1]):
			# #self.Product([i for i in weight[:][0]],[i for i in X[i][:]])
   #      		Gradient[j]=Gradient[j]+tmp*X[i][j]
   #      for j in range(X.shape[1]):
   #      		Gradient[j]=Gradient[j]+penalty[j]
        return Gradient

    def sigmoid(self, Z):
        '''
        Computes the Sigmoid Function  
        Arguments:
            A n-by-1 dimensional numpy matrix
        Returns:
            A n-by-1 dimensional numpy matrix
       
        '''
        Z=np.exp(-Z)
        for i in range(Z.shape[0]):
        	Z[i]=1/(1.0+Z[i])
        
        return Z

    def update_weight(self,X,Y,weight):
        '''
        Updates the weight vector.
        Arguments:
            X is a n-by-(d+1) numpy matrix
            Y is an n-by-1 dimensional numpy matrix
            weight is a d+1-by-1 dimensional numpy matrix
        Returns:
            updated weight vector : (d+1)-by-1 dimensional numpy matrix
        '''
        new_weight=np.zeros(weight.shape)
        gradient=self.calculateGradient(weight, X, Y, self.regLambda)
        new_weight=np.add(weight,-self.alpha*gradient)
        return new_weight
    
    def check_conv(self, weight, new_weight, epsilon):
        '''
        Convergence Based on Tolerance Values
        Arguments:
            weight is a (d+1)-by-1 dimensional numpy matrix
            new_weights is a (d+1)-by-1 dimensional numpy matrix
            epsilon is the Tolerance value we check against
        Return : 
            True if the weights have converged, otherwise False

        '''
        gap=0
        gap_list=np.add(weight,-new_weight)
        gap=np.linalg.norm(gap_list)
        #print("gap is: %f\n"%gap)
        return gap <= epsilon
        
    def train(self,X,Y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            Y is an n-by-1 dimensional numpy matrix
        Return:
            Updated Weights Vector: (d+1)-by-1 dimensional numpy matrix
        '''
        # Read Data
        n,d = X.shape
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        self.weight = self.new_weight = np.zeros((d+1,1))
        # Calculate weight


        #self.new_weight=self.update_weight(X,Y,self.weight)#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        iters=0
        while iters<self.maxNumIters:
        	self.new_weight=self.update_weight(X,Y,self.weight)
        	if self.check_conv(self.weight,self.new_weight,self.epsilon):
        		print("break up")
        		break;
        	self.weight=self.new_weight
        	iters+=1
        	#print("%d\n" %iters)
        
        
        return self.new_weight

    def predict_label(self, X,weight):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
            weight is a d+1-by-1 dimensional matrix
        Returns:
            an n-by-1 dimensional matrix of the predictions 0 or 1
        '''
        #data
        n=X.shape[0]
        #Add 1's column
        X = np.c_[np.ones((n,1)), X]
        result=np.zeros((X.shape[0],1))
        tmp=0
        for i in range(X.shape[0]):
        	tmp=np.dot(X[i,:],weight[:,0])
        	if tmp<=0:
        		result[i]=0
        	else:
        		result[i]=1
        
        return result
    
    def calculateAccuracy (self, Y_predict, Y_test):
        '''
        Computes the Accuracy of the model
        Arguments:
            Y_predict is a n-by-1 dimensional matrix (Predicted Labels)
            Y_test is a n-by-1 dimensional matrix (True Labels )
        Returns:
            Scalar value for accuracy in the range of 0 - 100 %
        '''
        acc=0
        for i in range(Y_predict.shape[0]):
        	if Y_predict[i]==Y_test[i]:
        		acc+=1

        return (acc*100)/(0.0+Y_test.shape[0])
    
        