from libpgm_static.lgbayesiannetwork import LGBayesianNetwork
from libpgm_static.pgmlearner import PGMLearner
import pandas as pd
import random as rn
import json
class bayesian_fill:
    def __init__(self,random_seed=None,min_iterations=10,max_iterations=50,bins=15,pvalparam=0.050000000000000003,indegree=1):
        self.random_seed=random_seed
        self.max_iterations=max_iterations
        self.min_iterations=min_iterations
        self.bins=bins
        self.__pvalparam__=pvalparam
        self.__current_pval__=pvalparam
        self.__indegree__=indegree
        self.__max_retry__=20
    def fill_missing_data(self,filename_in,filename_out,header=None,float_format='%.5f',delim_whitespace=True):
        #You can set a seed to get always the same random numbers sequence
        if self.random_seed==None:
            rn.seed()
        else:
            rn.seed(self.random_seed)
        #Read the data
        data=pd.read_csv(filename_in,header=header,delim_whitespace=delim_whitespace)
        #Fill missing values with random
        processed_data=[]
        rows=data.values.shape[0]
        cols=data.values.shape[1]
        dict=None
        for r in range(rows):
            dict={}
            for c in range(cols):
                if data.values[r,c]>0:
                    dict[c]=data.values[r,c]
                else:
                    dict[c]=rn.random()
            processed_data.append(dict)

        #Main loop to find the best network
        best_likelihood=[]
        best_lgbn=None
        for i in range(self.max_iterations):
            print "Iteration: "+str(i)

            #Fits a structure for the data
            learner = PGMLearner()
            self.__current_pval__=self.__pvalparam__
            while self.__current_pval__<1:
                try:
                    skel=learner.lg_constraint_estimatestruct(processed_data, pvalparam=self.__current_pval__, bins=self.bins, indegree=self.__indegree__)
                    break
                except AssertionError:
                    self.__current_pval__+=(1-self.__pvalparam__)/float(self.__max_retry__)
                    print 'There is a cycle, the new pval is '+str(self.__current_pval__)
            if self.__current_pval__>1:
                raise IOError("The columns in the input file create cycles, try setting a pval closer to 1 or trying another algorithm")
            #print json.dumps(skel.E, indent=2)#Prints the edges between nodes in json format

            #Fits the best parameters for current structure
            learner = PGMLearner()
            result = learner.lg_mle_estimateparams(skel, processed_data)
            #print json.dumps(result.Vdata, indent=2)#Prints the bayesian networks in json format

            #Loads the bayesian network with the data
            lgbn = LGBayesianNetwork(skel, result)

            #We check if the current likelihood is better than before
            current_likelihood = lgbn.loglikelihood(processed_data)
            current_is_best = True
            for l in range(len(best_likelihood)):
                if best_likelihood[l]>current_likelihood[l]:
                    current_is_best=False
            if (current_is_best and i>self.min_iterations):
                best_likelihood=current_likelihood
                best_lgbn=lgbn
                print "Better bayesian network found"
            #We fill missing data using the current bayesian network
            dict=None
            for r in range(rows):
                dict={}
                missing_cols=[]
                #We check each column to recover all the data available for that row
                for c in range(cols):
                    if data.values[r,c]>0:
                        dict[c]=data.values[r,c]
                    else:
                        missing_cols.append(c)
                #We get a random sample from the bayesian network given all the available data and store it in the processed data
                if len(missing_cols)>0:
                    dict=lgbn.randomsample(1,dict)[0]
                    for c in missing_cols:
                        processed_data[r][c]=dict[c]

        #We store the best network
        self.__best_likelihood__=best_likelihood
        self.__best_lgbn__=best_lgbn
        #We fill missing data using our best bayesian network
        dict=None
        for r in range(rows):
            dict={}
            missing_cols=[]
            #We check each column to recover all the data available for that row
            for c in range(cols):
                if data.values[r,c]>0:
                    dict[c]=data.values[r,c]
                else:
                    missing_cols.append(c)
            #We get a random sample from the bayesian network given all the available data and store it in the processed data
            if len(missing_cols)>0:
                dict=best_lgbn.randomsample(1,dict)[0]
                for c in missing_cols:
                    processed_data[r][c]=dict[c]

        #We write in a file the data with missing values filled
        filled_data=pd.DataFrame(processed_data)
        if delim_whitespace:
            filled_data.to_csv(filename_out," ",header=header, float_format=float_format,index=False)
        else:
            filled_data.to_csv(filename_out,",",header=header, float_format=float_format,index=False)

    def json_network(self):
        print json.dumps(self.__best_lgbn__.Vdata, indent=2)#Prints the bayesian networks in json format