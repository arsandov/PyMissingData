import PyMissingData as pmd
filename_input="Training_Features.txt"
filename_removed="tf_removed.txt"
filename_output="tf_filled.txt"
pmd.compare_tools.random_delete(filename_input,filename_removed)
bf= pmd.bayesian_learner.bayesian_fill(min_iterations=5,max_iterations=80,pvalparam=0.3)
bf.fill_missing_data(filename_removed,filename_output)
bf.json_network()
pmd.compare_tools.compare_predictions(filename_removed,filename_input,filename_output)

