#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classification of novels by thematic subgenre, based on temporal tagging.

@author: Ulrike Henny-Krahmer

Created in January 2021.
"""

import pandas as pd
import numpy as np
import glob
import re
from os.path import join
from os import rename
import plotly.graph_objects as go
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


################## FUNCTIONS ################


def scale_feature_set():
	"""
	For use with SVM: scale the feature sets to [0,1]
	"""
	print("scaling feature set...")
	
	
	df = pd.read_csv(feat_path, index_col=0)
		
	# scale the features
	scaler = MinMaxScaler()
	new_data = scaler.fit_transform(df.to_numpy())
	new_df = pd.DataFrame(index=df.index, columns=df.columns, data=new_data)
	
	# store the scaled feature set in a new file
	new_path = feat_path[:-4] + "_MinMax.csv"
	new_df.to_csv(new_path)
	
	print("done")
	


def select_metadata(wdir, md_file, subgenre_sets, outpath):
	"""
	select metadata for specific subgenre constellations to analyze
	save the metadata subsets
	
	Arguments:
	wdir (str): path to the working directory
	md_file (str): relative path to the metadata file
	subgenre_sets (list): list of dicts describing which subgenre constellations to choose, e.g. [{"level": "subgenre-current", "class 1": "novela romántica", "class 2": "other"}]
	outpath (str): relative path to the output directory for the metadata selection files
	"""
	print("select metadata...")
	
	
	for sb_set in subgenre_sets:
		md = pd.read_csv(join(wdir, md_file), index_col=0)
		
		level = sb_set["level"]
		class1 = sb_set["class 1"]
		class2 = sb_set["class 2"]
		
		print("selecting metadata for " + level + ", " + class1 + " vs. " + class2, "...")
		
		
		# get the instances of the first subgenre
		sub1 = md.loc[md[level] == class1]
		
		# get the instances of the second subgenre
		if class2 == "other":
			md.loc[np.logical_not(md[level].isin([class1, "unknown"])),level]  = "other"
			
		sub2 = md.loc[md[level] == class2]
			
		
		# is one class bigger than the other? if yes, undersample (select random samples from the bigger class)
		num_sub1 = len(sub1)
		num_sub2 = len(sub2)
		
		print("Size of class 1: " + str(len(sub1)))
		print("Size of class 2: " + str(len(sub2)))
		
		# repeat the sampling process 10 times
		for i in range(10):
			if num_sub1 > num_sub2:
				sub1_sampled = sub1.sample(n=num_sub2)
			elif num_sub2 > num_sub1:
				sub2_sampled = sub2.sample(n=num_sub1)
				
			# create new metadata frame with selected entries
			if num_sub1 > num_sub2:
				new_md = sub1_sampled.append(sub2)
			elif num_sub2 > num_sub1:
				new_md = sub2_sampled.append(sub1)
			# sort by idno
			new_md = new_md.sort_values(by="idno")
			# store new metadata selection
			outfile = "metadata_" + level + "_" + re.sub(r"\s", r"_", class1) + "_" + re.sub(r"\s", r"_", class2) + "_" + str(i) + ".csv"
			new_md.to_csv(join(wdir, outpath, outfile))
			
		
	print("done")
	


def select_data(wdir, md_inpath, feature_inpath, sb_set, rep):
	"""
	prepare data for classifier as X (np data array), y (labels)
	returns X, y, idnos
	
	Arguments:
	wdir (str): path to the working directory
	md_inpath (str): relative path to the directory containing selected metadata for subgenre constellations
	feature_inpath (str): relative path to the file containing the feature set
	sb_set (dict): dictionary describing the subgenre constellation to analyze, e.g. {"level": "subgenre-current", "class 1": "novela romántica", "class 2": "other"}
	rep (int): number of the data selection repetition to use
	"""	
	# which type of subgenre is analyzed?
	level = sb_set["level"]
	class1 = re.sub(r"\s", r"_", sb_set["class 1"])
	class2 = re.sub(r"\s", r"_", sb_set["class 2"])
	
	# load the metadata file corresponding to the selected subgenre constellation and feature set
	md_path = join(wdir, md_inpath, "metadata_" + level + "_" + class1 + "_" + class2 + "_" + str(rep) + ".csv")
	md = pd.read_csv(md_path, index_col=0)
	
	# prepare the data to return
	# labels
	y = md[level]
		
	data = pd.read_csv(join(wdir, feature_inpath), index_col=0)
	X = data.loc[md.index].to_numpy()
	
	idnos = md.index
		
	return X,y,idnos


def do_grid_search(X,y):
	"""
	Do a grid search for the SVM classifier and different settings of C.
	
	Arguments:
	X (nparray): data to use
	y (list): labels to use
	"""
	clf = svm.SVC(kernel="linear")
	param_grid = [{"C": [1,10,100,1000]}]
	
	grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
	grid_search.fit(X,y)
	results = grid_search.cv_results_
	results = pd.DataFrame.from_dict(results)
	return results


def parameter_study():
	"""
	test different subgenre constellations
	do grid searches for SVM with the C parameter to see which value of C works best
	"""
	
	print("running parameter study...")
	
	# chosen subgenre constellations
	subgenre_sets = [{"level": "subgenre-theme", "class 1": "novela histórica", "class 2": "other"},
	{"level": "subgenre-theme", "class 1": "novela sentimental", "class 2": "novela de costumbres"}]

	# how often should the data selection (with undersampling) be repeated?
	repetitions = 10

	# select metadata for subgenre constellations
	select_metadata(wdir, "../../conha19/metadata.csv", subgenre_sets, "data_selection/preliminary/")
	
	# data frames for results
	fr_svm = pd.DataFrame()

	for sb_set in subgenre_sets:
		for rep in range(repetitions):
						
			# do grid searches for the classifier
			X, y, idnos = select_data(wdir, "data_selection/preliminary/", join("features", feat_filename), sb_set, rep)
		
			results = do_grid_search(X,y)
			
			results["subgenre_level"] = sb_set["level"]
			results["class1"] = sb_set["class 1"]
			results["class2"] = sb_set["class 2"]
			results["repetition"] = rep
			
			fr_svm = fr_svm.append(results, sort=False)
		
	# store results
	fr_svm.to_csv(join(outdir, "grid-searches-SVM.csv"))
	
	print("done")


def get_rank1_groups(df, param):
	"""
	Check the results of the parameter study and keep only rows with rank 1.
	Return these rows grouped by the different values of the selected parameter.
	
	Arguments:
	df (DataFrame): data frame containing the parameter study results
	param (str): which parameter to evaluate (e.g. "C")
	"""
	# keep only rows with rank_test_score = 1
	df_1 = df.loc[df["rank_test_score"] == 1]
	# group these by the values of the selected parameter
	df_grouped = df_1.groupby(by=param).size().reset_index(name="counts").sort_values(by="counts", ascending=False)
	return df_grouped


def get_rank1_counts(df, param, param_value):
	"""
	Get the number of times a specific parameter value reached rank 1
	
	Arguments:
	df (DataFrame): data frame containing the parameter study results
	param (str): which parameter to evaluate (e.g. "C")
	param_value (str/int): which parameter value to look for
	"""
	# get the rows which have this parameter value
	rows_param_value = df.loc[df[param] == param_value]
	if rows_param_value.empty == False:
		rows_param_value = rows_param_value["counts"].values[0]
	else:
		rows_param_value = 0
	return rows_param_value



def evaluate_parameter_study(grid_search_results):
	"""
	count how often each parameter value is on rank 1 for the test score
	
	Argument:
	grid_search_results (str): path to the file with the grid search results
	"""
	print("evaluating parameter study...")
	
	param = "param_C"
	
	# load results
	df = pd.read_csv(grid_search_results)
	df_grouped = get_rank1_groups(df, param)
	
	'''
	# inspect results:
	print("general:")
	print(df.shape)
	print(df_grouped)
	'''
	# get the different parameter values
	param_values = sorted(df_grouped[param].tolist())
	
	
	# create a grouped bar chart showing how often which parameter value reached rank 1 in the different feature sets
	
	fig = go.Figure()
	
	# add bars for each parameter value
	for p_val in param_values:
		y = [get_rank1_counts(df_grouped, param, p_val)]
		
		fig.add_trace(go.Bar(name=str(p_val), x=["all"], y=y))
	
	if param == "param_C":
		xtitle = "feature type / C"
		
	fig.update_layout(autosize=False, width=500, height=500, title="Grid search results for SVM", barmode="group",legend_font=dict(size=14))
	fig.update_xaxes(title=xtitle,tickfont=dict(size=14))
	fig.update_yaxes(title="frequency of rank 1")

	fig.write_image(join(wdir, outdir, "ranks1_" + param + ".png")) # scale=2 (increase physical resolution)
	fig.write_html(join(wdir, outdir, "ranks1_" + param + ".html")) # include_plotlyjs="cdn" (don't include whole plotly library)

	print("done")


def get_feature_names():
	"""
	Return the names of the features.
	"""
	feat_fr = pd.read_csv(feat_path, index_col=0)
	feat_list = list(feat_fr.columns)
	return feat_list


def get_estimator():
	"""
	Get an instance of the chosen classifier, setting the parameters that were determined in the preliminary parameter study.
	"""
	C = 100
	clf = svm.SVC(kernel="linear", C=C)
		
	return clf
	
	
def get_scores(estimator, X, y, class1, cv):
	"""
	Get the cross validation scores for the chosen classifier and data
	
	Arguments:
	estimator (object): the classifier
	X (nparray): data
	y (list): labels
	class1 (str): label of the positive class
	cv (int): number of cross validation folds to use
	"""
					
	scoring = {"accuracy": make_scorer(accuracy_score), 
	"precision": make_scorer(precision_score, average="binary", pos_label=class1),
	"recall": make_scorer(recall_score, average="binary", pos_label=class1),
	"f1": make_scorer(f1_score, average="binary", pos_label=class1)}
	scores = cross_validate(estimator, X, y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True)
	
	return scores


def get_score_frame(scores):
	"""
	Convert the dictionary of scores into a data frame.
	
	Arguments:
	scores (dict): dictionary of scores returned from cross validation
	"""
	
	score_frame = pd.DataFrame.from_dict(scores)
	score_frame = score_frame.reset_index()
	score_frame = score_frame.rename(columns={"index":"call"})
	
	return score_frame
	
	
def set_frame_metadata(frame, level, class1, class2, data_rep):
	"""
	Set metadata columns for classification results frame.
	
	Arguments:
	frame (DataFrame): Data frame for the results
	level (str): subgenre level that is analyzed, e.g. "theme"
	class1 (str): the positive class, e.g."novela histórica"
	class2 (str): the negative class, e.g. "other"
	data_rep (int): number of the data repetition
	"""
							
	frame["subgenre_level"] = level
	frame["class1"] = class1
	frame["class2"] = class2
	frame["data_repetition"] = data_rep
	
	return frame
	
	
def store_features(scores, cv, feature_names):
	"""
	Store feature importances for each cv run
	and return a data frame containing all of them
	
	Arguments:
	scores (dict): dictionary of scores returned from cross validation
	cv (int): number of cv volds
	feature_names (list): the names of the features (the topic numbers or the words or ngrams)
	"""
	columns = ["cv_call", "class1_cl", "class2_cl"] + feature_names
	
	feature_frame = pd.DataFrame(columns=columns)
	
	for run in range(cv):
		coef = scores["estimator"][run].coef_.tolist()[0]
		classes = scores["estimator"][run].classes_
		data = [run, classes[0], classes[1]] + coef
		coef = pd.Series(index=columns, data=data)
		
		feature_frame = feature_frame.append(coef, ignore_index=True)
	
	return feature_frame


def store_labels(scores, cv, X, y, idnos):
	"""
	Store true labels and predicted labels for each cv run
	and return a data frame containing all of them.
	
	Arguments:
	scores (dict): dictionary of scores returned from cross validation
	cv (int): number of cv folds
	X (nparray): data
	y (nparray): true labels
	idnos (nparray): identifiers of the data
	"""		
			
	label_frame = pd.DataFrame()
	label_frame["idno"] = idnos
	label_frame["y_true"] = list(y)
	
	for run in range(cv):
		predicted_labels = scores["estimator"][run].predict(X)
		label_frame["y_" + str(run)] = predicted_labels
	
	return label_frame
	
	
	
def run_main_classification():
	"""
	Run the main classification task: classify the novels by thematic subgenre based
	on temporal expression features, with SVM
	"""

	#select_metadata(wdir, "../../conha19/metadata.csv", subgenre_sets, "data_selection/main/")


	print("classify...")

	# how often was the data selection (with undersampling) repeated?
	repetitions = 10
	# number of cv folds
	cv = 10

	# prepare data frame for classification results
	fr_svm = pd.DataFrame()

	# prepare collection of feature importances
	label_columns = ["subgenre_level", "class1", "class2", "data_repetition", "cv_call", "class1_cl", "class2_cl"]
	feature_names = get_feature_names()
	columns = label_columns + feature_names
	features_frame = pd.DataFrame(columns=columns)

	# prepare collection of true and predicted labels
	label_columns = ["subgenre_level", "class1", "class2", "data_repetition", "idno", "y_true"]
	for label_rep in range(repetitions):
		label_columns.append("y_" + str(label_rep))
	label_frame = pd.DataFrame(columns=label_columns)

	for sb_set in subgenre_sets:
		class1 = re.sub(r"\s", r"-", sb_set["class 1"])
		class2 = re.sub(r"\s", r"-", sb_set["class 2"])

		for data_rep in range(repetitions):
		
			# select data corresponding to the chosen features and classifier
			X,y,idnos = select_data(wdir, "data_selection/main/", join("features", feat_filename), sb_set, data_rep)
		
			# get an instance of the classifier with the chosen parameter settings
			estimator = get_estimator()
			
			# run cross validation and collect results
			scores = get_scores(estimator, X, y, sb_set["class 1"], cv)
			score_frame = get_score_frame(scores)
			score_frame = set_frame_metadata(score_frame, sb_set["level"], sb_set["class 1"], sb_set["class 2"], data_rep)
			score_frame = score_frame.drop("estimator", axis=1)
			
			fr_svm = fr_svm.append(score_frame, sort=False, ignore_index=True)
			
			# collect true labels and predicted labels for each cv run
			label_frame_cv = store_labels(scores, cv, X, y, idnos)
			label_frame_cv = set_frame_metadata(label_frame_cv, sb_set["level"], sb_set["class 1"], sb_set["class 2"], data_rep)
			label_frame = label_frame.append(label_frame_cv, sort=False, ignore_index=True)
			
			# collect feature importances
			feature_frame_cv = store_features(scores, cv, feature_names)
			feature_frame_cv = set_frame_metadata(feature_frame_cv, sb_set["level"], sb_set["class 1"], sb_set["class 2"], data_rep)
			features_frame = features_frame.append(feature_frame_cv, sort=False, ignore_index=True)
			
			
	# store classification results
	fr_svm.to_csv(join(outdir, "results_SVM.csv"))

	# store label information
	label_filename = "labels_SVM.csv"
	label_frame.to_csv(join(outdir, label_filename))

	# store feature importances
	features_filename = "features_SVM.csv"
	features_frame.to_csv(join(outdir, features_filename))

	print("done: classification")


def get_results_subgenres(subgenre_1, subgenre_2):
	"""
	Get the top and mean results for a certain subgenre constellation (e.g. "novela histórica" vs. "other",
	given the classifier (e.g. "SVM").
	Returns the following numbers: top accuracy, mean accuracy, standard deviation accuracy,
	top F1, mean F1, std.dev. F1
	
	Arguments:
	subgenre_1 (str): the positive class
	subgenre_2 (str): the negative class 
	"""
	print("get results subgenres mfw...")
	
	accuracy_collected = []
	f1_collected = []
	
	# name of results file e.g. results-SVM-mfw100_4gram_chars_word_tf.csv
	result_file = "results_SVM.csv"
	results = pd.read_csv(join(outdir, result_file), index_col=0)
	
	# select only the results for the subgenre constellation
	results_sub = results.loc[(results["class1"]==subgenre_1)  & (results["class2"]==subgenre_2)]
	
	acc = results_sub["test_accuracy"].tolist()
	for acc_value in acc:
		accuracy_collected.append(acc_value)
		
	f1 = results_sub["test_f1"].tolist()
	for f1_value in f1:
		if f1_value != 0:
			f1_collected.append(f1_value)
			
	top_acc = max(accuracy_collected)
	mean_acc = np.mean(accuracy_collected)
	std_acc = np.std(accuracy_collected)
	
	top_f1 = max(f1_collected)
	mean_f1 = np.mean(f1_collected)
	std_f1 = np.std(f1_collected)
	
	return len(accuracy_collected), top_acc, mean_acc, std_acc, top_f1, mean_f1, std_f1
	
	print("done")



def get_result_table_subgenres():
	"""
	Create an overview of the classification results 
	for all subgenre constellations (all chosen thematic subgenres),
	for a selected classifier and with a fixed parameter constellation.
	Returns a CSV table containing accuracy and f1 scores for each subgenre constellation.
	"""
	print("get result table subgenres...")
	
	# prepare the data frame
	columns = ["class_1", "class_2", "num_runs", "top_acc", "mean_acc", "sd_acc", "top_f1", "mean_f1", "sd_f1"]
	summary_fr = pd.DataFrame(columns=columns)
	
	# get results for each subgenre constellation
	for const in subgenre_sets:
		class1 = const["class 1"]
		class2 = const["class 2"]
		print(class1 + " vs. " + class2)
		
		res = get_results_subgenres(class1, class2)
		data = [class1, class2] + list(res)
		res_ser = pd.Series(index=columns, data=data)
		# append to overall summary
		summary_fr = summary_fr.append(res_ser, ignore_index = True)
	
	# save result summary
	summary_fr.to_csv(join(outdir, "results_subgenres_SVM.csv"))
	
	print("done")
	

def plot_feature_importances(num_top_feat):
	"""
	Plot feature importances.
	The classifier and feature parameters are chosen beforehand.
	One plot is created for each subgenre constellation.
	
	Argument:
	num_top_feat (int): number of top features to include in the plot, e.g. 20
	"""
	print("plot feature importances...")
	
	# get the relevant features file
	feat_data = pd.read_csv(join(outdir,"features_SVM.csv"), index_col=0)
	
	# get results for each subgenre constellation
	for const in subgenre_sets:
		class1 = const["class 1"]
		class2 = const["class 2"]
		print(class1 + " vs. " + class2)
		
		# select the rows that are relevant for this constellation
		feat_rows = feat_data[(feat_data["class1"] == class1) & (feat_data["class2"] == class2)]
		
		# drop metadata columns
		feat_rows = feat_rows.drop(labels=["class1", "class2", "data_repetition", "cv_call", "class1_cl", "class2_cl"], axis=1)
		# get column means
		feat_means = feat_rows.mean(axis=0)
		# sort by absolute values
		#feat_means = feat_means.sort_values(key=abs,ascending=False) # pandas update needed before "key" can be used
		feat_means = feat_means.iloc[(-feat_means.abs()).argsort()]
		# get top values
		feat_means = feat_means.iloc[0:num_top_feat]
		# reorder (for the plot)
		feat_means = feat_means.iloc[feat_means.abs().argsort()]
		
		
		labels = list(feat_means.index)
		values = feat_means
		
		# create a bar chart
		xaxis_title = "feature weights"
		chart_title = "feature importances (" + class1 + " vs. " + class2 + ")"
		
		fig = go.Figure(go.Bar(
		x=values,
		y=labels,
		orientation='h'))
		fig.update_layout(autosize=False, width=600, height=800, title=chart_title)
		fig.update_yaxes(type="category",title="feature",tickfont=dict(size=14),automargin=True)
		fig.update_xaxes(title=xaxis_title)
		
		outfile = "feat_imp_" + re.sub(r"\s",r"_",class1) + "_" + re.sub(r"\s",r"_",class2)

		fig.write_image(join(outdir, outfile + ".png")) # scale=2 (increase physical resolution)
		fig.write_html(join(outdir, outfile + ".html")) # include_plotlyjs="cdn" (don't include whole plotly library)

	print("done")


################## SET SOME PATHS and variables ###################

# path to the metadata file for the corpus
md_path = "/home/ulrike/Git/conha19/metadata.csv"
# path to the working directory for current analyses
wdir = "/home/ulrike/Git/papers/time_for_genre_eadh21/"
# path to the output directory
# data_results_temp_ex
# data_results_mfw
# data_results_combined
outdir = join(wdir, "data_results_combined")
# path to the temporal expression feature set
# temp_ex_features_rel_ext2.csv
# bow_mfw4000_tfidf_MinMax.csv
# temp_ex_bow_combined.csv
feat_filename = "temp_ex_bow_combined.csv"
feat_path = join(wdir, "features", feat_filename)

subgenre_sets = [{"level": "subgenre-theme", "class 1": "novela histórica", "class 2": "other"},
	{"level": "subgenre-theme", "class 1": "novela sentimental", "class 2": "other"},
	{"level": "subgenre-theme", "class 1": "novela de costumbres", "class 2": "other"},
	{"level": "subgenre-theme", "class 1": "novela histórica", "class 2": "novela sentimental"},
	{"level": "subgenre-theme", "class 1": "novela histórica", "class 2": "novela de costumbres"},
	{"level": "subgenre-theme", "class 1": "novela sentimental", "class 2": "novela de costumbres"}]



################## MAIN Part ###################

'''
scale features to range [0,1] for use with SVM
'''
#scale_feature_set()


'''
preliminary parameter study
result: C = 10,100,1000 work equally well, so C=100 is kept
'''
#parameter_study()

#evaluate_parameter_study(join(outdir, "grid-searches-SVM.csv"))


'''
main classification task
'''
#run_main_classification()


'''
analyze the classification results for the different subgenre constellations
'''
#get_result_table_subgenres()


plot_feature_importances(25)

