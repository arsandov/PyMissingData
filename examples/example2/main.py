import PyMissingData as pmd
pmd.compare_tools.random_delete("Training_Features.txt","tf_removed.csv")
bf= pmd.bayesian_learner.bayesian_fill(min_iterations=5,max_iterations=80,pvalparam=0.3)
bf.fill_missing_data("tf_removed.txt","tf_filled.txt")
bf.json_network()
pmd.compare_tools.compare_predictions("tf_removed.txt","Training_Features.txt","tf_filled.txt")

