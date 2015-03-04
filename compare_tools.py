__author__ = 'augusto'
import math
import numpy as np
import random as rn
import pandas as pd

#Normalized Root Mean Squared Error
def nrmse(real,prediction):
    __check_array_errors__(real,prediction)
    if np.max(real)-np.min(real)==0:
        raise ValueError("The difference between the real max and min can't be 0")

    square_difference=0
    instances=len(real)

    for i in range(instances):
        square_difference+=math.pow((prediction[i]-real[i]),2)

    square_difference=math.sqrt(square_difference/float(instances))
    normalized_error=square_difference/float((np.max(real)-np.min(real)))
    return normalized_error

#Mean Absolute Error
def mae(real,prediction):
    __check_array_errors__(real,prediction)
    instances=len(real)
    sum_difference=0

    for i in range(instances):
        sum_difference+=math.fabs(prediction[i]-real[i])

    mean_absolute_error=sum_difference/float(instances)
    return mean_absolute_error


def compare_random(real,prediction,times=100):
    __check_array_errors__(real,prediction)
    instances=len(real)
    results={}
    real_pred_key="real-prediction"
    real_rand_key="real-random"
    pred_rand_key="prediction-random"
    nrmse_key="nrmse"
    mae_key="mae"
    results[real_pred_key]={nrmse_key:float(0),mae_key:float(0)}
    results[real_rand_key]={nrmse_key:float(0),mae_key:float(0)}
    results[pred_rand_key]={nrmse_key:float(0),mae_key:float(0)}
    #The differences between a real and prediction are fixed
    results[real_pred_key][nrmse_key]=nrmse(real,prediction)
    results[real_pred_key][mae_key]=mae(real,prediction)

    for run in range(times):
        random=[]
        for i in range(instances):
            random.append(rn.random())
        results[real_rand_key][nrmse_key]+=nrmse(real,random)
        results[real_rand_key][mae_key]+=mae(real,random)

        results[pred_rand_key][nrmse_key]+=nrmse(prediction,random)
        results[pred_rand_key][mae_key]+=mae(prediction,random)

    results[real_rand_key][nrmse_key]/=float(times)
    results[real_rand_key][mae_key]/=float(times)
    results[pred_rand_key][nrmse_key]/=float(times)
    results[pred_rand_key][mae_key]/=float(times)
    return results

def compare_files(file1,file2,delim_whitespace=True,header=None,times=100):
    data_set_1=pd.read_csv(file1,header=header,delim_whitespace=delim_whitespace)
    data_set_2=pd.read_csv(file2,header=header,delim_whitespace=delim_whitespace)
    __check_pdmatrix_errors_(data_set_1,data_set_2)
    rows=data_set_1.values.shape[0]
    cols=data_set_1.values.shape[1]
    prediction_1=[]
    prediction_2=[]
    for r in range(rows):
        for c in range(cols):
            if data_set_1.values[r,c]!=data_set_2.values[r,c]:
                #We store the values when they are different
                prediction_1.append(data_set_1.values[r,c])
                prediction_2.append(data_set_2.values[r,c])

    print "Results"
    errors=compare_random(prediction_1,prediction_2,times)
    print errors

def __check_array_errors__(real,prediction):
    if len(real)!=len(prediction):
        raise ValueError.message("Dimensions of arrays must be the same")
    if len(real)==0:
        raise ZeroDivisionError
def __check_pdmatrix_errors_(data1,data2):
    if data1.values.shape[0]!=data2.values.shape[0]:
        raise ValueError.message("Dimensions of matrix must be the same")
    if data1.values.shape[1]!=data2.values.shape[1]:
        raise ValueError.message("Dimensions of matrix must be the same")