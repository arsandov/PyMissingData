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
# run bayesian_fill in parallel
def fill_missing_data_parallel(filename_in, filename_out, num_threads=4, \
                                   num_trials=20, temp_folder=".",\
                                   pvalparam=0.05, nbins=10, indegree=0):
    
    # input arguments for the parallel workers
    array_iter = list(itertools.product(range(num_trials), [filename_in], [temp_folder],\
                                            [pvalparam], [nbins], [indegree]))

    # initiate parallel pool
    pool = Pool(num_threads)
    pool.map(multi_run_wrapper, array_iter)
    
#----------------------------------------------------------------------------------------------------------------
    # find the global best result
    best_log_likelihood = -float('Inf')

    # loop over all the parallel batches
    for t1 in range(num_trials):
    
        # restore result of each batch
        temp = np.load(temp_folder + "/temp_results_" + str(t1) + ".npz")
    
        # check if it is better
        if temp["best_log_likelihood"] > best_log_likelihood:
            best_log_likelihood = temp["best_log_likelihood"]
            filled_data = temp["filled_data"]

#---------------------------------------------------------------------------------------------------------------
    # remove temporary files
    os.system("rm -rf " + temp_folder + "/temp_results_*")

    # print the global best result into a text file
    np.savetxt(filename_out, filled_data, fmt='%.5f', delimiter=" ")

    # return results
    return filled_data, best_log_likelihood


#===================================================
# parallel wrapper
def multi_run_wrapper(array_iter_args):
    
    # initiate machine
    bf = bayesian_fill(pvalparam=array_iter_args[3], nbins=array_iter_args[4],\
                           indegree=array_iter_args[5], verbose=False)
    bf.fill_missing_data(array_iter_args[1], num_trials=1)
    
    # save the best result of this batch
    np.savez(array_iter_args[2] + "/temp_results_" + str(array_iter_args[0]) + ".npz",\
             best_log_likelihood = bf.best_log_likelihood, filled_data = bf.filled_data)


#====================================================
# define class
class bayesian_fill:
    
    # initiate class
    def __init__(self,\
                     pvalparam=0.05, nbins=10, indegree=0,\
                     missing_threshold=-90,\
                     max_iter=20, EM_tol=10**-4,\
                     verbose=True):
                
        self.pvalparam = pvalparam
        self.nbins = nbins
        self.indegree = indegree
        
        self.missing_threshold = missing_threshold
        
        self.max_iter = max_iter
        self.EM_tol = EM_tol

        self.verbose = verbose


#====================================================
    # filling up missing data
    def fill_missing_data(self, filename_in, filename_out=None, num_trials=20,\
                              delimiter=" ", float_format='%.5f'):
        
        # restore catalog
        self.data_original = np.loadtxt(filename_in, delimiter=delimiter)
    
        # initiate global best result
        self.best_log_likelihood = -float('Inf')
        
#----------------------------------------------------------------------------------------------------------------
        # run different initializations
        for p1 in range(num_trials):

            # print number of the current trial
            if self.verbose:
                print "Number of trial : ", p1+1

            # reshuffling the nodes
            order_array = np.arange(len(self.data_original[0,:]))
            np.random.shuffle(order_array)

            # run linear Gaussian + EM
            [filled_data_trial, best_log_likelihood_trial] = self.lg_estimatebn_EM(order_array)

#---------------------------------------------------------------------------------------------------------------
            # check if this trial is better
            if best_log_likelihood_trial > self.best_log_likelihood:
                self.best_log_likelihood = best_log_likelihood_trial
                self.filled_data = filled_data_trial
                
            # print an empty line
            if self.verbose:
                print "\n"

#---------------------------------------------------------------------------------------------------------------
        # save the final result to a text file
        if filename_out:
            np.savetxt(filename_out, self.filled_data, fmt=float_format, delimiter=delimiter)

 
#====================================================
    # linear Gaussian with EM algorithm
    def lg_estimatebn_EM(self, k_order):

        # the catalog
        data = np.copy(self.data_original)
        
        # number of data points and features
        rows = data.shape[0]
        cols = data.shape[1]

        # shuffle the features ordering
        data = data[:,k_order]

#--------------------------------------------------------------------------------------------------------------
        # initiate the EM algorithm
        ll_best = -float('Inf')
        num_iter = 0
        converge = False

        original_data = []
        estimated_data = []
        missing_index = []

#-------------------------------------------------------------------------------------------------------------
        # make dictionary sets for EM
        for r in range(rows):

            # for each data point
            dict_run = {}
            dict_original = {}
            missing_checker = False
    
#-------------------------------------------------------------------------------------------------------------
            # loop over all features
            for c in range(cols):
                if data[r,c] > self.missing_threshold:
                    dict_run[c] = data[r,c]
                    dict_original[c] = data[r,c]

                # input missing data from the marginal distribution
                else:
                    dict_run[c] = []
            
                    while not dict_run[c]:
                        ind_draw = np.random.randint(0,rows)
                
                        if data[ind_draw,c] > self.missing_threshold:
                            dict_run[c] = data[ind_draw,c]

#-------------------------------------------------------------------------------------------------------------
                    # remember which data point has missing data
                    missing_checker = True
        
            # remember which data point has missing data
            if missing_checker:
                missing_index.append(r)

            # append data point dictionary into dictionary sets
            estimated_data.append(dict_run)
            original_data.append(dict_original)
    
#------------------------------------------------------------------------------------------------------------
        ## EM algorithm ##
        while not converge and num_iter < self.max_iter:
    
            # increase the counter
            num_iter += 1
        
            ## the maximization step ##
            # find the best fitting model
            learner = PGMLearner()
            lgbn = learner.lg_estimatebn(estimated_data, \
                                             pvalparam=self.pvalparam,\
                                             bins=self.nbins, indegree=self.indegree)

#-------------------------------------------------------------------------------------------------------------
            # calculate the total log likelihood
            ll_new = self.calculate_total_log_likelihood(lgbn, original_data)

            # due to missing data, the final few EM steps could oscillate
            # we do not consider the oscillations
            if ll_new > ll_best:
                ll_best_new = ll_new
            else:
                ll_best_new = ll_best

#-------------------------------------------------------------------------------------------------------------  
            # check convergence
            if np.abs((ll_best_new - ll_best)/ll_best_new) < self.EM_tol:
                converge = True

            # update the likelihood
            ll_best = ll_best_new
        
            # print current EM result
            if self.verbose:
                print 'EM iteration ', num_iter, ', log likelihood = ', ll_best
        
#-------------------------------------------------------------------------------------------------------------
            ## the expectation step ##
            # estimate the missing data
            for ind_missing in missing_index:
    
                dict_run = copy.copy(original_data[ind_missing])
                dict_run = lgbn.randomsample(1,dict_run)[0]
            
                estimated_data[ind_missing] = dict_run

#-------------------------------------------------------------------------------------------------------------
        # restore the original features ordering
        k_order_argsort = np.argsort(k_order)
        filled_data = np.array([[estimated_data[i][k_order_argsort[j]]\
                                     for j in range(cols)] for i in range(rows)])
    
        # return results
        return filled_data, ll_best


#===================================================
    # calculate total log likelihood
    def calculate_total_log_likelihood(self, lgbn_choice, data_dict):
    
        # initiate total log likelihood
        ll_total = 0
    
        # loop over the data set
        for o2 in range(len(data_dict)):
            ll_total += self.calculate_log_likelihood(lgbn_choice, data_dict[o2])

        # return total log likelihood
        return ll_total


#===================================================
    # calculate the log likelihood of one data point
    def calculate_log_likelihood(self, lgbn_choice, dict_choice):
    
        # initiate log likelihood
        ll_point = 0
    
        # loop over all evidences
        for o1 in dict_choice.keys():
            
            # evaluate the CPD
            total_mean, total_var = self.calculate_mean_var(o1, lgbn_choice, dict_choice)

            # evaluate the likelihood
            ll_point += np.log(norm.pdf(dict_choice[o1], \
                                        scale=np.sqrt(total_var), loc=total_mean))

        # return log likelihood
        return ll_point


#===================================================
    # calculate mean and variance of CPD, given evidences
    def calculate_mean_var(self, node_choice, lgbn_choice, dict_choice):

        # initiate mean and variance
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
        # return results
        return total_mean, total_var
