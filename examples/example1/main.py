import PyMissingData as pmd

bf= pmd.bayesian_learner.bayesian_fill(min_iterations=8,max_iterations=80,pvalparam=0.05)
bf.fill_missing_data("data_examples.txt","data_examples_filled.txt")
bf.json_network()

pmd.compare_tools.random_fill("data_examples.txt","random_filled_as.txt")

pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_1.txt","data_examples_filled.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_2.txt","data_examples_filled.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_15_iter_1.txt","data_examples_filled.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_15_iter_2.txt","data_examples_filled.txt")

print "========================================"
print "Differences between the original outputs"
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_1.txt","filled_by_original_algorithm_10_iter_2.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_15_iter_1.txt","filled_by_original_algorithm_15_iter_2.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_2.txt","filled_by_original_algorithm_15_iter_2.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_2.txt","filled_by_original_algorithm_15_iter_1.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_1.txt","filled_by_original_algorithm_15_iter_2.txt")
pmd.compare_tools.compare_predictions("data_examples.txt","filled_by_original_algorithm_10_iter_1.txt","filled_by_original_algorithm_15_iter_1.txt")

