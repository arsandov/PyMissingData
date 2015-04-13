# import packages
try:
    import copy
except ImportError:
    raise ImportError, "copy is not installed on your system."

try:
    import numpy as np
except ImportError:
    raise ImportError, "numpy is not installed on your system."

try: 
    from scipy.stats import norm
except ImportError:
    raise ImportError, "scipy is not installed on your system."

try: 
    from libpgm.pgmlearner import PGMLearner
    from libpgm.graphskeleton import GraphSkeleton
except ImportError:
    raise ImportError, "libpgm is not installed on your system."

#----------------------------------------------------------------------------------------------------------------
try: 
    from multiprocessing import Pool
except ImportError:
    raise ImportError, "multiprocessing is not installed on your system."

try: 
    import itertools
except ImportError:
    raise ImportError, "itertools is not installed on your system."

try: 
    import os
except ImportError:
    raise ImportError, "os is not installed on your system."


#===================================================
# run in parallel
def fill_missing_data(filename_in, filename_out, temp_folder=".",\
                          num_threads=4, num_resets=10, num_graphs=10,\
                          pvalparam=0.05, nbins=10, indegree=1,\
                          delimiter=" ", float_format='%.5f',\
                          missing_threshold=-90,\
                          max_EM_iter=10, EM_tol=10**-4):
    
    # input arguments for parallel workers
    array_iter = list(itertools.product(range(num_resets),\
                                            [filename_in], [temp_folder],\
                                            [pvalparam], [nbins], [indegree],\
                                            [num_graphs],\
                                            [delimiter], [missing_threshold],\
                                            [max_EM_iter], [EM_tol]))

    # initialize parallel pool
    pool = Pool(num_threads)
    pool.map(multi_run_wrapper, array_iter)
    
#----------------------------------------------------------------------------------------------------------------
    # find the global best result
    # loop over all the parallel batches
    for t1 in range(num_resets):
    
        # restore result of each batch
        temp = np.load(temp_folder + "/temp_results_" + str(t1) + ".npz")
    
        # check if it is better
        if not ('best_log_likelihood' in locals()) \
                or temp["best_log_likelihood"] > best_log_likelihood:
            best_log_likelihood = temp["best_log_likelihood"]
            filled_data = temp["filled_data"]
            best_BN_Vdata = temp["best_BN_Vdata"]

#---------------------------------------------------------------------------------------------------------------
    # remove temporary files
    os.system("rm -rf " + temp_folder + "/temp_results_*")

    # print the global best result into a text file
    np.savetxt(filename_out, filled_data, delimiter=delimiter, fmt=float_format)

    # return results
    return filled_data, best_log_likelihood, best_BN_Vdata


#===================================================
# parallel wrapper
def multi_run_wrapper(array_iter_args):
    
    # initialize machine
    bf = bayesian_fill(array_iter_args[1],\
                           pvalparam=array_iter_args[3],\
                           nbins=array_iter_args[4],\
                           indegree=array_iter_args[5],\
                           delimiter=array_iter_args[7],\
                           missing_threshold=array_iter_args[8],\
                           max_EM_iter=array_iter_args[9],\
                           EM_tol=array_iter_args[10])
    bf.bayesian_fill_run(array_iter_args[6])
    
    # save the best result of this batch
    np.savez(array_iter_args[2] + "/temp_results_" + str(array_iter_args[0]) + ".npz",\
                 best_log_likelihood = bf.best_log_likelihood, filled_data = bf.filled_data,\
                 best_BN_Vdata=bf.best_BN_Vdata)


#====================================================
# define class
class bayesian_fill:
    
    # initialize class
    def __init__(self, filename_data,\
                     pvalparam=0.05, nbins=10, indegree=1,\
                     delimiter=" ",\
                     missing_threshold=-90,\
                     max_EM_iter=10, EM_tol=10**-4):

        self.data_original = np.loadtxt(filename_data, delimiter=delimiter)
        self.filled_data = None
        self.best_BN_Vdata = None

        self.rows = self.data_original.shape[0]
        self.cols = self.data_original.shape[1]
        self.best_log_likelihood = -float('Inf')

        self.pvalparam = pvalparam
        self.nbins = nbins
        self.indegree = indegree

        self.missing_threshold = missing_threshold
        
        self.max_EM_iter = max_EM_iter
        self.EM_tol = EM_tol


#====================================================
    # filling up missing data
    def bayesian_fill_run(self, num_graphs):

        # initialize missing data
        original_data = []
        estimated_data = []
        missing_index = []

        # for each data point
        for r in range(self.rows):
            dict_run = {}
            dict_original = {}
            missing_checker = False
    
#-------------------------------------------------------------------------------------------------------------
            # loop over all features
            for c in range(self.cols):
                if self.data_original[r,c] > self.missing_threshold:
                    dict_run[c] = self.data_original[r,c]
                    dict_original[c] = self.data_original[r,c]

                # input missing data from the marginal distribution
                else:
                    dict_run[c] = []
            
                    while not dict_run[c]:
                        ind_draw = np.random.randint(0,self.rows)
                
                        if self.data_original[ind_draw,c] > self.missing_threshold:
                            dict_run[c] = self.data_original[ind_draw,c]

#-------------------------------------------------------------------------------------------------------------
                    # remember which data point has missing data
                    missing_checker = True
        
            # remember which data point has missing data
            if missing_checker:
                missing_index.append(r)

            # append data point dictionary into dictionary sets
            estimated_data.append(dict_run)
            original_data.append(dict_original)

#-----------------------------------------------------------------------------------------------------------
        # learn the structure
        learner = PGMLearner()
        skel = learner.lg_constraint_estimatestruct(estimated_data,\
                                                        pvalparam=self.pvalparam,\
                                                        bins=self.nbins, indegree=self.indegree)

        # the number of possible Bayesian Networks for this structure
        n2 = 2**len([x for x in skel.E_undirected if len(x) == 3])

#-------------------------------------------------------------------------------------------------------------
        # run different possible Bayesian Networks
        for p1 in range(num_graphs):

            # initiate non-cyclic indicator
            ind_non_cyclic = False

            # find a particular non-cyclic Bayesian Network
            while (not ind_non_cyclic):
                x = np.random.randint(0,n2)
                
                # instantiate a graph
                lg_skeleton = GraphSkeleton()
                lg_skeleton.V = skel.V[:]
                new_edges = []

                # randomly choose the edges directions
                for e in skel.E_undirected:
                    if len(e) == 2: 
                        new_edges.append(e)
                        continue
                    new_edge = e[:2]
                    if x%2 == 0:
                        new_edge.reverse()
                    new_edges.append(new_edge)
                    x = x >> 1
                lg_skeleton.E = new_edges

                # check if the graph is acyclic:
                try:
                    lg_skeleton.toporder()
                    ind_non_cyclic= True
                except:
                    continue
      
#------------------------------------------------------------------------------------------------------------
            # given the Bayesian Network, initialize the EM algorithm
            ll_best = -float('Inf')
            num_iter = 0
            converge = False
     
            ## EM algorithm ##
            while not converge and num_iter < self.max_EM_iter:

                # increase the counter
                num_iter += 1
        
#------------------------------------------------------------------------------------------------------------
                ## maximization step ##
                # find the best fitting model
                lgbn = learner.lg_mle_estimateparams(lg_skeleton, estimated_data)

                # calculate total log likelihood
                ll_new = self.calculate_total_log_likelihood(lgbn, original_data)

                # due to missing data, the final few EM steps could oscillate
                # we do not consider oscillations
                if ll_new > ll_best:
                    ll_best_new = ll_new
                else:
                    ll_best_new = ll_best

                # check convergence
                if np.abs((ll_best_new - ll_best)/ll_best_new) < self.EM_tol:
                    converge = True

                # update likelihood
                ll_best = ll_best_new
        
#-------------------------------------------------------------------------------------------------------------
                ## expectation step ##
                # estimate missing data
                for ind_missing in missing_index:
                    dict_run = copy.copy(original_data[ind_missing])
                    dict_run = lgbn.randomsample(1,dict_run)[0]
                    estimated_data[ind_missing] = dict_run

#-------------------------------------------------------------------------------------------------------------
            # check if the current Bayesian Network is better
            if ll_best > self.best_log_likelihood:
                self.best_log_likelihood = ll_best
                self.filled_data = np.array([[estimated_data[i][j]\
                                                  for j in range(self.cols)] for i in range(self.rows)])
                self.best_BN_Vdata = copy.copy(lgbn.Vdata)


#===================================================
    # calculate total log likelihood
    def calculate_total_log_likelihood(self, lgbn_choice, data_dict):
    
        # initialize total log likelihood
        ll_total = 0
    
        # loop over the data set
        for o2 in range(len(data_dict)):
            ll_total += self.calculate_log_likelihood(lgbn_choice, data_dict[o2])

        # return total log likelihood
        return ll_total


#===================================================
    # calculate the log likelihood of one data point
    def calculate_log_likelihood(self, lgbn_choice, dict_choice):
    
        # initialize log likelihood
        ll_point = 0
    
        # loop over all evidences
        for o1 in dict_choice.keys():
            
            # evaluate CPD
            total_mean, total_var = self.calculate_mean_var(o1, lgbn_choice, dict_choice)

            # evaluate likelihood
            ll_temp = norm.pdf(dict_choice[o1], scale=np.sqrt(total_var), loc=total_mean)
            if ll_temp > 0:
                ll_point += np.log(ll_temp)
            else:
                ll_point += -float('Inf')

        # return log likelihood
        return ll_point


#===================================================
    # calculate mean and variance of CPD, given evidences
    def calculate_mean_var(self, node_choice, lgbn_choice, dict_choice):

        # initialize mean and variance
        total_mean = copy.copy(lgbn_choice.Vdata[node_choice]['mean_base'])
        total_var = copy.copy(lgbn_choice.Vdata[node_choice]['variance'])

        # parents in consideration, duplicate the list
        beta_parents_run = list(lgbn_choice.Vdata[node_choice]['parents'])
        beta_scale_run = list(lgbn_choice.Vdata[node_choice]['mean_scal'])

#--------------------------------------------------------------------------------------------------------------
        # eliminate parent by parent
        while beta_parents_run:
    
            # pop a parent
            beta_parents_pop = beta_parents_run.pop()
            beta_scale_pop = beta_scale_run.pop()
        
            # check if there is an evidence
            if beta_parents_pop in dict_choice:
                total_mean += beta_scale_pop*dict_choice[beta_parents_pop]
            
            # if not proceed to the next loop
            else:
                temp_results = self.calculate_mean_var(beta_parents_pop,\
                                                      lgbn_choice, dict_choice)
                total_mean += beta_scale_pop*temp_results[0]
                total_var += (beta_scale_pop**2)*temp_results[1]
            
#----------------------------------------------------------------------------------------------------------------
        # return CPD properties
        return total_mean, total_var
