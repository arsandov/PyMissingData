import compare_tools
import bayesian_learner

bf= bayesian_learner.bayesian_fill(min_iterations=8,max_iterations=80)
#bf=bayesian_learner.bayesian_fill(min_iterations=5,max_iterations=10)
bf.fill_missing_data("data_examples.txt","filled_by_now.txt")
bf.json_network()

compare_tools.random_fill("data_examples.txt","random_filled_as.txt")

compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_1.txt","filled_by_now.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_2.txt","filled_by_now.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_15_1.txt","filled_by_now.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_15_2.txt","filled_by_now.txt")

print "Controls"
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_1.txt","filled_by_karim_10_2.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_15_1.txt","filled_by_karim_15_2.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_2.txt","filled_by_karim_15_2.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_2.txt","filled_by_karim_15_1.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_1.txt","filled_by_karim_15_2.txt")
compare_tools.compare_predictions("data_examples.txt","filled_by_karim_10_1.txt","filled_by_karim_15_1.txt")

