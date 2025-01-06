# Housing price Dataset#

#import numpy as np
import pandas as pd
from autograd import numpy as np
import autograd as grad
from mlrefined_libraries import math_optimization_library as moptl 
from mlrefined_libraries import superlearn_library as slib
import matplotlib.pyplot as plt
from matplotlib import gridspec

#  Reading dataset from csv file  #

Input_file = np.array(pd.read_csv('mlrefined_datasets/superlearn_datasets/boston_housing.csv'))
print(Input_file.shape)
print(type(Input_file))

x = np.array([Input_file[0],Input_file[1],Input_file[2],Input_file[3],Input_file[4],Input_file[5],Input_file[6],Input_file[7],Input_file[8],Input_file[9],Input_file[10],Input_file[11],Input_file[12]]);
#print(x[-1])
#print(Input_file[-1])

# Normalizing the dataset with Standard_normalization process

def standard_normalizer(x):
    #for i in range(np.len)
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_standard_normalizer = (x-x_mean)/x_std;
    #print(x_standard_normalizer)
    return(x_standard_normalizer);
print(standard_normalizer(x).shape)

x_p = standard_normalizer(x)

print("Input x_p after applying standard normalization function",x_p);

# Prediction of y_p values for the given data set
# Here we are using perceptor to find the y_p value by calculating accuracy values for the given point belongs to either +1 or -1 class respectively.
# i.e. with the help of sign function and magnitude of the weight vector we can define distance of the point from it's projection
# once, we have distance between the point x_p and it's projection x_p' we can find the accuracy score of the point by using sigmoid function.
# Based on accuracy scores of the points we have assigned the data points to positive class or negative class
# If the point belongs to positive side of the class the accuracy score should be greater than equal to 0.5 and negative if it is less than 0.5
# Later we can assign these values to output array y_p to use them to find cost functions.
# After that, we can use least squares cost function and least absolute deviation function to find the cost values.
# Lastly, we can send these cost values to run gradient descent to find minimum values of the cost function.

#w = np.array([25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0,25.0]) #[:,np.axis]
#w = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
w = np.array([2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0])
#w = np.array([5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0]) #[:,np.axis]
#w = np.array([15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0,15.0]) #[:,np.axis]


''' Definitions for Model function, sine function and magnitude of weights vector, 
Accuracy score functions to find predictions for output values y_p '''

def model(x,w):
    x_model = w[0]+np.dot(x.T,w[1:]);
    return(x_model);

def sine_function(w):
    x_sine_function = np.sin(model(x_p,w));
    return(x_sine_function);

'''def soft_max(w):
    x_softmax = np.log(1+np.exp(model(x,w)));
    return(x_softmax);    '''

def magnitude(w):
    w_mag = np.sqrt(w[1]**2+w[2]**2+w[3]**2+w[4]**2+w[5]**2+w[6]**2+w[7]**2+w[8]**2+w[9]**2+w[10]**2+w[11]**2+w[12]**2);
    print(w_mag)
    return(w_mag);
''' finding accuracy score to assign the value to +1 if the score is >=0.5
     and assigning the value to -1 if the score is < 0.5'''

def accuracy_score(w):
    distance = sine_function(w)/magnitude(w);
    acc_score = 1/(1+np.exp(-distance));
    #acc1_score =soft_max(w)/magnitude(w);
    #print(acc1_score);
    return(acc_score);
accuracy = accuracy_score(w);
y_p = []
y_p = accuracy;
#print(np.round(accuracy,2))
#print(accuracy.shape)
#print(np.size(accuracy))

''' Logic to assign values for y_p either +1 or -1 as mentioned in the above comment'''
i = 0;
count_negative =0
count_positive =0
for i in range(np.size(accuracy)):
    if(accuracy[i]<= 0.50):
        y_p[i] = -1;
        count_negative +=1;
    elif(accuracy[i]>0.50):
        y_p[i] = +1;
        count_positive +=1;
#print(np.size(y_p))

''' Least square cost function definition and logic'''
def least_square_function(w):
    
    model_value = np.sum((model(x_p,w)-y_p)**2);
    #print(model_value)
    least_square_cost = model_value/float(np.size(y_p));
    return least_square_cost

least_square_cost_function = least_square_function(w);

''' Least Absoulte deviation function definition and logic'''

def least_absolute_deviation(w):
    model_val = np.sum(np.absolute(model(x_p,w)-y_p));
    least_absolute_dev = model_val/float(np.size(y_p));
    return(least_absolute_dev);



#least_absolute_deviation_cost_function = least_absolute_deviation(w);

#print(least_square_cost_function,least_absolute_deviation_cost_function)

#print(accuracy_score(w))
#Performing Gradient Descent to minimize the cost functions

grad_optimizer = moptl.optimizers;

g1 = least_square_function;
g2 = least_absolute_deviation;



# Set-1 Comparision

alpha1 = 10**-5;
alpha2 = 10**-2.5;
alpha3 = 10**-1.5;

# Set-2 Comparision

#alpha1 = 10**-1;
#alpha2 = 10**-2;
#alpha3 = 10**-3;


max_iterations = 2000;

least_squares_cost_weight_history1,least_squares_cost_history1 = grad_optimizer.gradient_descent(g1,alpha1,max_iterations,w);
least_squares_cost_weight_history2,least_squares_cost_history2 = grad_optimizer.gradient_descent(g1,alpha2,max_iterations,w);
least_squares_cost_weight_history3,least_squares_cost_history3 = grad_optimizer.gradient_descent(g1,alpha3,max_iterations,w);


least_absolute_deviation_weight_history1,least_absolute_deviation_cost_history1 = grad_optimizer.gradient_descent(g2,alpha1,max_iterations,w);
least_absolute_deviation_weight_history2,least_absolute_deviation_cost_history2 = grad_optimizer.gradient_descent(g2,alpha2,max_iterations,w);
least_absolute_deviation_weight_history3,least_absolute_deviation_cost_history3 = grad_optimizer.gradient_descent(g2,alpha3,max_iterations,w);

cost_history1 =[least_squares_cost_history1,least_squares_cost_history2,least_squares_cost_history3];
cost_history2 = [least_absolute_deviation_cost_history1,least_absolute_deviation_cost_history2,least_absolute_deviation_cost_history3];

# plotting cost histories for both Least squares cost and Least absolute deviation curves
#cost_history1 = [least_squares_cost_history2];
#cost_history2 = [least_absolute_deviation_cost_history2]
''' Plotting cost histories function definition'''

def plot_cost_histories(histories,start,**kwargs):
        # plotting colors
        colors = ['k', 'magenta', 'aqua', 'blueviolet', 'chocolate']
        
        # initialize figure
        fig = plt.figure(figsize = (10,3))

        # create subplot with 1 panel
        gs = gridspec.GridSpec(1, 1) 
        ax = plt.subplot(gs[0]); 
        
        # any labels to add?        
        labels = [' ',' ',' ',' ']
        #print(len(labels))
        if 'labels' in kwargs:
            labels = kwargs['labels']
            
        # plot points on cost function plot too?
        points = False
        if 'points' in kwargs:
            points = kwargs['points']
        history = histories[0]+histories[1]+histories[2]
        # run through input histories, plotting each beginning at 'start' iteration
        #for c in range(len(histories)):
        for c in range(len(labels)):
            if c==3:
                print(" ")
            else:    
                history = histories[c]
            #print(type(history))
            #print(history)
            label = []
            if c == 0:
                label = labels[0]
            elif c == 1:
                label = labels[1]
            elif c == 2:
                label = labels[2]    
            else:
                label = labels[3]   

            # check if a label exists, if so add it to the plot
            if np.size(label) == 0:
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 2*(0.8)**(c),color = colors[c]) 
            else:               
                ax.plot(np.arange(start,len(history),1),history[start:],linewidth = 2*(0.8)**(c),color = colors[c],label = label) 
                
            # check if points should be plotted for visualization purposes
            if points == True:
                ax.scatter(np.arange(start,len(history),1),history[start:],s = 90,color = colors[c],edgecolor = 'w',linewidth = 2,zorder = 3) 


        # clean up panel
        xlabel = 'step $k$'
        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
        ylabel = r'$g\left(\mathbf{w}^k\right)$'
        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        ax.set_xlabel(xlabel,fontsize = 14)
        ax.set_ylabel(ylabel,fontsize = 14,rotation = 0,labelpad = 25)
        if np.size(label) > 0:
            anchor = (1,1)
            if 'anchor' in kwargs:
                anchor = kwargs['anchor']
            plt.legend(loc='upper right', bbox_to_anchor=anchor)
            #leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.set_xlim([start - 0.5,len(history) - 0.5])
        
       # fig.tight_layout()
        plt.show()
''' Calling plot function to show the diagrams for cost functions of both least square cost and least absolute deviation functions'''
print("Comparision of Least squares cost vs Least Absolute Deviation")
print("-------------------------------------------------------------")
print("Least squares squares cost performs better while using Set-1 step length(alpha) values change to Set-2 in the code from previous code window to see results")
print("Least squares Absolute performs better while using Set-2 step length(alpha) values change to Set-1 in the code from previous code window to see results")
print("--------------------------------------------------------------")
plot_cost_histories(cost_history1,0,labels=[r'Least squares cost function',r'alpha=10^-5',r'alpha=10^-2.5',r'alpha=10^-1.5']);

plot_cost_histories(cost_history2,0,labels=[r'Least Absolute Deviation',r'alpha=10^-5',r'alpha=10^-2.5',r'alpha=10^-1.5']);
