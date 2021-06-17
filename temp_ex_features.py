#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate features for classification based on temporal tagging.

@author: Ulrike Henny-Krahmer

Created in January 2021.
"""

from os.path import join
from os.path import basename
import pandas as pd
from lxml import etree
import numpy as np
import glob
import re
import plotly.graph_objects as go


################## FUNCTIONS ################

def collect_results_timex3(result_fr, temp_ex_type, xml, idno):
	"""
	Retrieve the temporal expressions of a specific type for a corpus file and save 
	the results in the overall frame for that type
	(e.g. how often the DATE expression "hoy" occurs in that novel).
	
	Arguments:
	result_fr (DataFrame): Data frame in which the results should be stored
	temp_ex_type (str): which type of temporal expression to evaluate: DATE, TIME, DURATION, or SET
	xml (ElementTree): the parsed XML file of the novel
	idno (str): identifier of the novel, e.g. "nh0001"
	"""
	
	results = xml.xpath("//TIMEX3[@type='" + temp_ex_type + "']/text()")
	
	for res in results:
		res = re.sub(r"\s", r"_", res.lower())
		
		if res in result_fr.columns:
			result_fr.loc[idno,res] = result_fr.loc[idno,res] + 1
		else:
			result_fr[res] = 0
			result_fr.loc[idno,res] = 1


def collect_results_main_temp_ex_types():
	"""
	inspect the features: which temporal expressions were found, how many different ones, and how frequent are they?
	create a frame for each of the four basic types: DATE, TIME, DURATION, SET
	"""
	
	print("collect results for DATE, TIME, DURATION, SET...")

	# read existing metadata
	md_table = pd.read_csv(md_path, index_col=0)
	idnos = md_table.index

	DATE_fr = pd.DataFrame(index=idnos)
	TIME_fr = pd.DataFrame(index=idnos)
	DURATION_fr = pd.DataFrame(index=idnos)
	SET_fr = pd.DataFrame(index=idnos)


	# loop through corpus files to get HeidelTime results

	for infile in glob.glob(join(annotated_corpus, "*.xml")):
			
		idno = basename(infile)[0:6]
		xml = etree.parse(infile)
		
		# collect the results for the basic types of temp. expr.
		collect_results_timex3(DATE_fr, "DATE", xml, idno)
		collect_results_timex3(TIME_fr, "TIME", xml, idno)
		collect_results_timex3(DURATION_fr, "DURATION", xml, idno)
		collect_results_timex3(SET_fr, "SET", xml, idno)
		
	# save result frames
	DATE_fr.to_csv(join(outdir, "DATE_expressions.csv"))
	TIME_fr.to_csv(join(outdir, "TIME_expressions.csv"))
	DURATION_fr.to_csv(join(outdir, "DURATION_expressions.csv"))
	SET_fr.to_csv(join(outdir, "SET_expressions.csv"))

	print("saved DATE, TIME, DURATION, SET frames")


 
def evaluate_results_main_temp_ex_types():
	"""
	Evaluate the results for the main temporal expression types in the corpus.
	"""
	print("evaluate results for main types of temp. expr...")
	
	# How many different expressions are there for each main type of temp. expr.?
	# Which ones are the most frequent ones for each type? create lists of the 50 most frequent ones
	# create bar charts for an overview
	main_types = ["DATE", "TIME", "DURATION", "SET"]
	
	for mt in main_types:
		# number of features
		feature_set = pd.read_csv(join(outdir, mt + "_expressions.csv"), index_col=0)
		num_columns = len(feature_set.columns)
		print(mt + " has " + str(num_columns) + " different features")
		"""
		DATE has 2851 different features
		TIME has 86 different features
		DURATION has 1227 different features
		SET has 101 different features
		"""
		
		# 50 most frequent ones
		sums = feature_set.sum(axis=0).sort_values(ascending=False)
		mf50 = sums[0:50]
		mf50.to_csv(join(outdir, mt + "_mf50.csv"),header=False)
		
		# plot 20 mf as bar chart
		mf20 = sums[0:20]
		
		animals=['giraffes', 'orangutans', 'monkeys']

		fig = go.Figure([go.Bar(x=mf20.index, y=list(mf20))])
		fig.update_layout(autosize=False, width=600, height=500, title="Top 20 temporal expressions (" + mt + ")")
		fig.update_xaxes(title="temporal expression",tickfont=dict(size=14),tickangle=270)
		fig.update_yaxes(title="number of occurrences (absolute)")

		outfile = "bar_top20_" + mt
		fig.write_image(join(outdir, outfile + ".png")) # scale=2 (increase physical resolution)
		#fig.write_html(join(outdir, outfile + ".html")) # include_plotlyjs="cdn" (don't include whole plotly library)
		#fig.show()
		
		
	print("done")
	
	

def collect_features_absolute():
	"""
	Collect the overall feature set of temporal expressions (and verb tenses) with absolute values.
	"""
	print("collect features (absolute values)...")
	
	# labels for features
	labels = ["tpx_all", "DATE", "TIME", "DURATION", "SET", "PRESENT_REF", "FUTURE_REF", "PAST_REF", "SP", "SU", "FA", "WI", "WE",
	"MO", "MI", "AF", "EV", "NI",
	"DATE_century", "DATE_decade", "DATE_year", "DATE_month", "DATE_day",
	"DATE_spec_none", "DATE_spec_year", "DATE_spec_year_month", "DATE_spec_month", "DATE_spec_day", "DATE_spec_month_day", "DATE_spec_any", "DATE_spec_full",
	"TIME_daytime", "TIME_hour", "TIME_minute", "TIME_day_spec",
	"DURATION_century", "DURATION_decade", "DURATION_year", "DURATION_month", "DURATION_week", "DURATION_day", "DURATION_daytime", "DURATION_hour", "DURATION_minute",
	"SET_century", "SET_decade", "SET_year", "SET_month", "SET_week", "SET_day", "SET_daytime", "SET_hour", "SET_minute",
	"VERB_present", "VERB_imperfect", "VERB_future", "VERB_past", "VERB_conditional"]

	"""
	explanations:
	
	tpx_all: number of all kinds of temporal expressions (DATE, TIME, DURATION, SET)
	DATE: number of DATE expressions
	TIME: number of TIME expressions
	DURATION: number of DURATION expressions
	SET: number of SET expressions
	PRESENT_REF: temporal expressions that refer to the present in general terms
	FUTURE_REF: temporal expressions that refer to the future in general terms
	PAST_REF: temporal expressions that refer to the past in general terms
	SP: temporal expressions referring to spring or springtime
	SU: temporal expressions referring to summer or summertime
	FA: temporal expressions referring to autumn/fall
	WI: temporal expressions referring to winter or wintertime
	WE: temporal expressions referring to weekends
	MO: temporal expressions referring to the morning
	MI: temporal expressions referring to mid-day
	AF: temporal expressions referring to the afternoon
	EV: temporal expressions referring to the evening
	NI: temporal expressions referring to the night
	DATE_century: DATE expressions that refer to centuries (whether specified or not)
	DATE_decade: DATE expressions that refer to decades (whether specified or not)
	DATE_year: DATE expressions that refer to years (whether specified or not)
	DATE_month: DATE expressions that refer to months (whether specified or not)
	DATE_day: DATE expressions that refer to days (whether specified or not)
	DATE_spec_none: DATE expressions where no value is specified ("last year")
	DATE_spec_year: DATE expressions where just the year is specified ("1877")
	DATE_spec_year_month: DATE expressions where just year and month are specified ("March 1877")
	DATE_spec_month: DATE expressions where just the month is specified ("March")
	DATE_spec_day: DATE expressions where just the day is specified ("on the 2nd") 
	DATE_spec_month_day: DATE expressions where just month and day are specified ("on the 2nd of March")
	DATE_spec_any: DATE expressions where any part is specified
	DATE_spec_full: DATE expressions where all parts (year, month, day) are specified ("on the 2nd of March 1877")
	TIME_daytime: TIME expressions that refer to daytimes (whether specified or not)
	TIME_hour: TIME expressions that refer to hours (whether specified or not)
	TIME_minute: TIME expressions that refer to minutes (whether specified or not)
	TIME_day_spec: TIME expressions where the time of the day (e.g. hour or minute) is specified
	DURATION_century: DURATION expressions that refer to centuries (whether specified or not)
	DURATION_decade: DURATION expressions that refer to decades (whether specified or not)
	DURATION_year: DURATION expressions that refer to years (whether specified or not)
	DURATION_month: DURATION expressions that refer to months (whether specified or not)
	DURATION_week: DURATION expressions that refer to weeks (whether specified or not)
	DURATION_day: DURATION expressions that refer to days (whether specified or not)
	DURATION_daytime: DURATION expressions that refer to daytimes (whether specified or not)
	DURATION_hour: DURATION expressions that refer to hours (whether specified or not)
	DURATION_minute: DURATION expressions that refer to minutes (whether specified or not)
	SET_century: SET expressions that refer to centuries (whether specified or not)
	SET_decade: SET expressions that refer to decades (whether specified or not)
	SET_year: SET expressions that refer to years (whether specified or not)
	SET_month: SET expressions that refer to months (whether specified or not)
	SET_week: SET expressions that refer to weeks (whether specified or not)
	SET_day: SET expressions that refer to days (whether specified or not)
	SET_daytime: SET expressions that refer to daytimes (whether specified or not)
	SET_hour: SET expressions that refer to hours (whether specified or not)
	SET_minute: SET expressions that refer to minutes (whether specified or not)
	VERB_present: verbs in present tense
	VERB_imperfect: verbs in imperfect tense
	VERB_future: verbs in future tense
	VERB_past: verbs in past tense
	VERB_conditional: verbs in conditional tense
	"""

	# read existing metadata
	md_table = pd.read_csv(md_path, index_col=0)
	idnos = md_table.index
	
	# create new data frame for features
	feat_fr = pd.DataFrame(columns=labels)

	# collect values from all the files
	
	for infile in glob.glob(join(annotated_corpus, "*.xml")):
			
		idno = basename(infile)[0:6]
		xml = etree.parse(infile)
		
		tpx_all = len(xml.xpath("//TIMEX3"))
		DATE = len(xml.xpath("//TIMEX3[@type='DATE']"))
		TIME = len(xml.xpath("//TIMEX3[@type='TIME']"))
		DURATION = len(xml.xpath("//TIMEX3[@type='DURATION']"))
		SET = len(xml.xpath("//TIMEX3[@type='SET']"))
		PRESENT_REF = len(xml.xpath("//TIMEX3[@value='PRESENT_REF']"))
		FUTURE_REF = len(xml.xpath("//TIMEX3[@value='FUTURE_REF']"))
		PAST_REF = len(xml.xpath("//TIMEX3[@value='PAST_REF']"))
		SP = len(xml.xpath("//TIMEX3[contains(@value,'SP')]"))
		SU = len(xml.xpath("//TIMEX3[contains(@value,'SU')]"))
		FA = len(xml.xpath("//TIMEX3[contains(@value,'FA')]"))
		WI = len(xml.xpath("//TIMEX3[contains(@value,'WI')]"))
		WE = len(xml.xpath("//TIMEX3[contains(@value,'WE')]"))
		MO = len(xml.xpath("//TIMEX3[contains(@value,'MO')]"))
		MI = len(xml.xpath("//TIMEX3[contains(@value,'MI')]"))
		AF = len(xml.xpath("//TIMEX3[contains(@value,'AF')]"))
		EV = len(xml.xpath("//TIMEX3[contains(@value,'EV')]"))
		NI = len(xml.xpath("//TIMEX3[contains(@value,'NI')]"))
		
		# calculate DATE features which cannot be determined directly with XPath
		date_values = xml.xpath("//TIMEX3[@type='DATE']/@value")
		
		dates_century = []
		dates_decade = []
		dates_year = []
		dates_month = []
		dates_day = []
		
		dates_spec_none = []
		dates_spec_year = []
		dates_spec_year_month = []
		dates_spec_month = []
		dates_spec_day = []
		dates_spec_month_day = []
		dates_spec_any = []
		dates_spec_full = []
		
		for date in date_values:
			if re.match(r"^.{2}$", date):
				dates_century.append(date)
			if re.match(r"^.{3}$", date):
				dates_decade.append(date)
			if re.match(r"^.{4}$", date):
				dates_year.append(date)
			if re.match(r"^.{4}-.{2}$", date):
				dates_month.append(date)
			if re.match(r"^.{4}-.{2}-.{2}$", date):
				dates_day.append(date)
				
			if re.match(r"^\D+$", date):
				dates_spec_none.append(date)
			if re.match(r"^\d{2,4}", date) and not re.match(r"^.{2,4}-\d{2}", date) and not re.match(r"^.{2,4}-.{2}-\d{2}", date):
				dates_spec_year.append(date)
			if re.match(r"^\d{2,4}-\d{2}", date) and not re.match(r"^.{2,4}-.{2}-\d{2}", date):
				dates_spec_year_month.append(date)
			if re.match(r"^.{2,4}-\d{2}", date) and not re.match(r"^\d{2,4}", date) and not re.match(r"^.{2,4}-.{2}-\d{2}", date):
				dates_spec_month.append(date)
			if re.match(r"^.{2,4}-.{2}-\d{2}", date) and not re.match(r"^\d{2,4}", date) and not re.match(r"^.{2,4}-\d{2}", date):
				dates_spec_day.append(date)
			if re.match(r"^.{2,4}-\d{2}-\d{2}", date) and not re.match(r"^\d{2,4}", date):
				dates_spec_month_day.append(date)
			if re.match(r"^\d{2,4}", date) or re.match(r"^.{2,4}-\d{2}", date) or re.match(r"^.{2,4}-.{2}-\d{2}", date):
				dates_spec_any.append(date)
			if re.match(r"^\d{2,4}-\d{2}-\d{2}", date):
				dates_spec_full.append(date)
				
		DATE_century = len(dates_century)
		DATE_decade = len(dates_decade)
		DATE_year = len(dates_year)
		DATE_month = len(dates_month)
		DATE_day = len(dates_day)
		
		DATE_spec_none = len(dates_spec_none)
		DATE_spec_year = len(dates_spec_year)
		DATE_spec_year_month = len(dates_spec_year_month)
		DATE_spec_month = len(dates_spec_month)
		DATE_spec_day = len(dates_spec_day)
		DATE_spec_month_day = len(dates_spec_month_day)
		DATE_spec_any = len(dates_spec_any)
		DATE_spec_full = len(dates_spec_full)
		
		# calculate TIME features which cannot be determined directly with XPath
		time_values = xml.xpath("//TIMEX3[@type='TIME']/@value")
		
		times_daytime = []
		times_hour = []
		times_minute = []
		times_day_spec = []
		
		for time in time_values:
			if re.match(r"^XXXX-XX-XXT(MO|MI|AF|EV|NI)$", time):
				times_daytime.append(time)
			if re.match(r"^XXXX-XX-XXT\d{2}:00$", time):
				times_hour.append(time)
			if re.match(r"^XXXX-XX-XXT\d{2}:([1-9]\d)|(\d[1-9])$", time):
				times_minute.append(time)
			if re.match(r"^XXXX-XX-XXT\d{2}:d{2}$", time):
				times_day_spec.append(time)
		
		TIME_daytime = len(times_daytime)
		TIME_hour = len(times_hour)
		TIME_minute = len(times_minute)
		TIME_day_spec = len(times_day_spec)
		
		# calculate DURATION features which cannot be determined directly with XPath
		duration_values = xml.xpath("//TIMEX3[@type='DURATION']/@value")
		
		durations_century = []
		durations_decade = []
		durations_year = []
		durations_month = []
		durations_week = []
		durations_day = []
		durations_daytime = []
		durations_hour = []
		durations_minute = []
		
		for dur in duration_values:
			if re.match(r"^P[X0-9]*CE$", dur):
				durations_century.append(dur)
			if re.match(r"^P[X0-9]*DE$", dur):
				durations_decade.append(dur)
			if re.match(r"^P[X0-9]*Y$", dur):
				durations_year.append(dur)
			if re.match(r"^P[X0-9]*M$", dur):
				durations_month.append(dur)
			if re.match(r"^P[X0-9]*W$", dur):
				durations_week.append(dur)
			if re.match(r"^P[X0-9]*D$", dur):
				durations_day.append(dur)
			if re.match(r"^PT(MO|MI|AF|EV|NI)$", dur):
				durations_daytime.append(dur)
			if re.match(r"^PT[X0-9]*H$", dur):
				durations_hour.append(dur)
			if re.match(r"^PT[X0-9]*M$", dur):
				durations_minute.append(dur)
		
		DURATION_century = len(durations_century)
		DURATION_decade = len(durations_decade)
		DURATION_year = len(durations_year)
		DURATION_month = len(durations_month)
		DURATION_week = len(durations_week)
		DURATION_day = len(durations_day)
		DURATION_daytime = len(durations_daytime)
		DURATION_hour = len(durations_hour)
		DURATION_minute = len(durations_minute)
		
		
		# calculate SET features which cannot be determined directly with XPath
		set_values = xml.xpath("//TIMEX3[@type='SET']/@value")
		
		sets_century = []
		sets_decade = []
		sets_year = []
		sets_month = []
		sets_week = []
		sets_day = []
		sets_daytime = []
		sets_hour = []
		sets_minute = []
		
		for set_val in set_values:
			if re.match(r"^P.*C$", set_val):
				sets_century.append(set_val)
			if re.match(r"^P.*DE$", set_val):
				sets_decade.append(set_val)
			if re.match(r"^P[X0-9]*Y$", set_val):
				sets_year.append(set_val)
			if re.match(r"^P[X0-9]*M$", set_val):
				sets_month.append(set_val)
			if re.match(r"^P[X0-9]*W$", set_val):
				sets_week.append(set_val)
			if re.match(r"^P[X0-9]*D$", set_val):
				sets_day.append(set_val)
			if re.match(r"^PT(MO|MI|AF|EV|NI)$", set_val):
				sets_daytime.append(set_val)
			if re.match(r"^PT[X0-9]*H$", set_val):
				sets_hour.append(set_val)
			if re.match(r"^PT[X0-9]*M$", set_val):
				sets_minute.append(set_val)
		
		SET_century = len(sets_century)
		SET_decade = len(sets_decade)
		SET_year = len(sets_year)
		SET_month = len(sets_month)
		SET_week = len(sets_week)
		SET_day = len(sets_day)
		SET_daytime = len(sets_daytime)
		SET_hour = len(sets_hour)
		SET_minute = len(sets_minute)
		
		# get verb tense counts
		freeling_file = join(annotated_corpus_ling, idno + ".xml")
		xml_freeling = etree.parse(freeling_file)
		namespaces = {'tei':'http://www.tei-c.org/ns/1.0', 'cligs':'https://cligs.hypotheses.org/ns/cligs'}
		
		VERB_present = len(xml_freeling.xpath("//tei:w[@pos='verb'][@cligs:tense='present']", namespaces=namespaces))
		VERB_imperfect = len(xml_freeling.xpath("//tei:w[@pos='verb'][@cligs:tense='imperfect']", namespaces=namespaces))
		VERB_future = len(xml_freeling.xpath("//tei:w[@pos='verb'][@cligs:tense='future']", namespaces=namespaces))
		VERB_past = len(xml_freeling.xpath("//tei:w[@pos='verb'][@cligs:tense='past']", namespaces=namespaces))
		VERB_conditional = len(xml_freeling.xpath("//tei:w[@pos='verb'][@cligs:tense='conditional']", namespaces=namespaces))
		
		# create new entry for the general feature frame and append it
		new_entry = pd.Series(name=idno, index=labels, data=[tpx_all, DATE, TIME, DURATION, SET, PRESENT_REF, FUTURE_REF, PAST_REF, SP, SU, FA, WI, WE, 
		MO, MI, AF, EV, NI,
		DATE_century, DATE_decade, DATE_year, DATE_month, DATE_day,
		DATE_spec_none, DATE_spec_year, DATE_spec_year_month, DATE_spec_month, DATE_spec_day, DATE_spec_month_day, DATE_spec_any, DATE_spec_full,
		TIME_daytime, TIME_hour, TIME_minute, TIME_day_spec,
		DURATION_century, DURATION_decade, DURATION_year, DURATION_month, DURATION_week, DURATION_day, DURATION_daytime, DURATION_hour, DURATION_minute,
		SET_century, SET_decade, SET_year, SET_month, SET_week, SET_day, SET_daytime, SET_hour, SET_minute,
		VERB_present, VERB_imperfect, VERB_future, VERB_past, VERB_conditional])
		
		feat_fr = feat_fr.append(new_entry)
		
	# drop columns that are only 0 and report which ones these are
	"""
	all zero: 'WE', 'EV', 'TIME_minute', 'TIME_day_spec', 'DURATION_daytime',
       'SET_decade', 'SET_hour', 'SET_minute'
    result: 50 different features
	"""
	zero_cols = feat_fr.columns[(feat_fr == 0).all()]
	feat_fr = feat_fr.drop(labels=zero_cols,axis=1)
	
	# save feature frame
	feat_fr = feat_fr.sort_index()
	feat_fr.to_csv(join(outdir, "temp_ex_features_abs.csv"))
	
	print("done")
	
	
def get_mf_frame(data_fr, num_features):
	"""
	Select the top most frequent features from a data frame
	and return the frame with only these features.
	
	Arguments:
	data_fr (DataFrame): frame holding the feature values
	num_features (int): how many top most features to select
	"""
	sums_fr = data_fr.sum(axis=0).sort_values(ascending=False)
	mf_labels_fr = sums_fr[0:num_features].index
	data_fr = data_fr.loc[:,mf_labels_fr]
	
	return data_fr

	
def add_mf_features(num_features):
	"""
	Add the most frequent temporal expressions of the four basic types (DATE, TIME, DURATION, SET)
	to the overall feature frame.
	
	Argument:
	num_features (int): how many most frequent features to add, e.g. 50
	"""
	print("add mf features to feature frame...")
	
	# get data
	feat_fr = pd.read_csv(join(outdir, "temp_ex_features_abs.csv"), index_col=0)
	DATE_fr = pd.read_csv(join(outdir, "DATE_expressions.csv"), index_col=0)
	TIME_fr = pd.read_csv(join(outdir, "TIME_expressions.csv"), index_col=0)
	DURATION_fr = pd.read_csv(join(outdir, "DURATION_expressions.csv"), index_col=0)
	SET_fr = pd.read_csv(join(outdir, "SET_expressions.csv"), index_col=0)
	
	# select top most frequent ones for DATE, TIME, DURATION, SET
	DATE_fr = get_mf_frame(DATE_fr, num_features)
	TIME_fr = get_mf_frame(TIME_fr, num_features)
	DURATION_fr = get_mf_frame(DURATION_fr, num_features)
	SET_fr = get_mf_frame(SET_fr, num_features)
	
	
	# add these to the general feature frame
	feat_fr = pd.concat([feat_fr, DATE_fr, TIME_fr, DURATION_fr, SET_fr], axis=1)
	feat_fr.to_csv(join(outdir, "temp_ex_features_abs_ext1.csv"))
	
	print("done")
	
	
def convert_features_relative():	
	"""
	Convert absolute feature values to relative ones.
	(Relative to text length in number of tokens)
	"""
	print("convert feature values...")
	
	# get data
	feat_fr = pd.read_csv(join(outdir, "temp_ex_features_abs_ext1.csv"), index_col=0)
	
	# for each text of the corpus: get length and divide values in absolute frame
	for infile in glob.glob(join(annotated_corpus, "*.xml")):
			
		idno = basename(infile)[0:6]
		xml = etree.parse(infile)
		namespaces = {'tei':'http://www.tei-c.org/ns/1.0', 'cligs':'https://cligs.hypotheses.org/ns/cligs'}
		body = xml.xpath("//tei:body//text()",namespaces=namespaces)
		body_string = " ".join(body)
		body_string = re.sub(r"\s+", r" ", body_string)
		tokens = re.split(r"\W+", body_string, flags=re.MULTILINE)
		num_words = len(tokens)
		
		feat_fr.loc[idno] = feat_fr.loc[idno] / num_words
		
	# save feature frame
	feat_fr.to_csv(join(outdir, "temp_ex_features_rel_ext1.csv"))
	
	print("done")
	
	

def add_features_proportional():
	"""
	Add proportional feature values to the frame with relative values,
	for the temporal expressions e.g. how many DATE expressions there are
	in proportion to the overall number of temporal expressions in a text,
	and for the verb tenses e.g. how many verbs in present tense there are
	in proportion to the overall number of verbs in the text.
	
	The proportions are calculated based on the absolute feature values.
	"""
	print("add proportional features...")
	
	# get data
	feat_abs = pd.read_csv(join(outdir, "temp_ex_features_abs_ext1.csv"), index_col=0)
	feat_rel = pd.read_csv(join(outdir, "temp_ex_features_rel_ext1.csv"), index_col=0)
	
	# create additional columns with proportional values
	# verb columns need to be treated differently
	verb_cols = ["VERB_present", "VERB_imperfect", "VERB_future", "VERB_past", "VERB_conditional"]
	
	# get overall values
	tpx_all = feat_abs["tpx_all"]
	VERBs_all = feat_abs["VERB_present"] + feat_abs["VERB_imperfect"] + feat_abs["VERB_future"] + feat_abs["VERB_past"] + feat_abs["VERB_conditional"]
	
	# loop through columns, skip the first one with all temp. expr. values
	for col in feat_rel.columns[1:]:
		new_col_name = col + "_prop"
		
		if col in verb_cols:
			new_col_values = feat_abs[col] / VERBs_all
		else:
			new_col_values = feat_abs[col] / tpx_all
		
		# add column to frame
		feat_rel[new_col_name] = new_col_values
			
	
	# save new frame
	print("number of features: " + str(len(feat_rel.columns))) # 499
	feat_rel.to_csv(join(outdir, "temp_ex_features_rel_ext2.csv"))
	
	print("done")
	
	
def combine_feature_sets(feat_1, feat_2, feat_out):
	"""
	Combine temporal expression and mfw feature sets.
	
	Arguments:
	feat_1 (str): filename of the first feature set
	feat_2 (str): filename of the second feature set
	feat_out (str): filename of the combined output feature set
	"""
	print("combine feature sets...")
	
	feat_1_fr = pd.read_csv(join(outdir, feat_1), index_col=0)
	feat_2_fr = pd.read_csv(join(outdir, feat_2), index_col=0)
	feat_comb = pd.concat([feat_1_fr, feat_2_fr], axis=1)
	
	print(len(feat_1_fr.columns))
	print(len(feat_2_fr.columns))
	print(len(feat_comb.columns))
	
	feat_comb.to_csv(join(outdir, feat_out))
	
	print("done")
	

################## SET SOME PATHS ###################
# path to the TEI files containing TimeML annontations
annotated_corpus = "/home/ulrike/Git/conha19/heideltime/teia"
# path to the TEI files containing general linguistic annotations
annotated_corpus_ling = "/home/ulrike/Git/conha19/annotated"
# path to the metadata file for the corpus
md_path = "/home/ulrike/Git/conha19/metadata.csv"
# path to the working directory for current analyses
wdir = "/home/ulrike/Git/papers/time_for_genre_eadh21/"
# path to the output directory
outdir = join(wdir, "features")



################## MAIN Part ###################

'''
collect how often which temporal expression of the main types DATE, TIME, DURATION, SET occurs in each corpus file
'''
#collect_results_main_temp_ex_types()


'''
how many different expressions are there for each main type of temp. expr.?
which ones are the most frequent ones for each type?
(create word clouds and bar charts for an overview)
'''
#evaluate_results_main_temp_ex_types()


'''
create the overall feature set with absolute values
result: 50 features
'''
#collect_features_absolute()


'''
attach 50mf temp expressions of the four basic types to the feature frame
result: 250 features
'''
#add_mf_features(50)


'''
convert feature counts to relative values (relative to text length)
'''
#convert_features_relative()


'''
add proportional values (e.g. how many DATE expressions in proportion to the total absolute number of temp. expr. in the text)
'''
#add_features_proportional()



'''
combine temporal expression and mfw features
'''
#combine_feature_sets("temp_ex_features_rel_ext2_MinMax.csv", "bow_mfw4000_tfidf_MinMax.csv", "temp_ex_bow_combined.csv")


############## NOTES ################

# which subtypes are there? (@value) inspect the attribute of the TimeML tags
# which ones seem relevant for the analysis of subgenres of the novel?

"""
About @value, see https://www.cs.brandeis.edu/~cs112/cs112-2004/annPS/TimeML12wp.htm:
"The datatypes specified for the value attribute---Duration, Date, Time, WeekDate,  WeekTime, Season, PartOfYear, PaPrFu---
are XML datatypes based on the 2002 TIDES guideline, which extends the ISO 8601 standard for representing dates, times, and durations. 
See the 2002 TIDES guidelines for details about the value attribute, and see the TimeML Schema (www.timeml.org/timeMLdocs/TimeML.xsd) 
for complete definitions of each of these datatypes."

See also: https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/english-timex2-guidelines-v0.1.pdf
especially the chapter on "Fuzzy temporal expressions"

Examples for different values:

DATE:
PRESENT_REF (e.g. "hoy", "ahora", "el día"), PAST_REF (e.g. "reciente"), FUTURE_REF (e.g. "próxima", "pronto", "el futuro")
XXXX-XX-XX (e.g. "domingo", "el lunes", "mañana", "ayer"), XXXX-05 (e.g. "mayo"), XXXX-WI (e.g. "invierno")
XXXX (e.g. "medio año", "el año"), XX ("del Siglo"), 192 ("los veinte" ??), 1830 ("1830"), 1830-WI ("el invierno"), etc.

TIME:
XXXX-XX-XXTAF ("la tarde"), XXXX-XX-XXTNI ("la noche", "anoche"), XXXX-XX-XXTMO ("la mañana", "la madrugada"),
XXXX-XX-XXTNI ("la noche del viernes"),
XXXX-XX-XXT21:00 ("las 9 de la noche"), XXXX-XX-XXT09:00 ("las 9 de la mañana")

DURATION:
PXY ("años", "algunos años"), P5Y ("cinco años"), PT2H ("dos horas", "un par de horas")
PT1M ("un minuto"), P2D ("dos días"), P1CE ("un siglo") etc.
"por tres noches" - not recognized as duration

SET:
P1D ("todos los días", "cada día", "diariamente"), PTNI ("todas las noches"), PTMO ("cada mañana")
P1Y ("anual"), P1C ("el siglo"), PW ("dos veces por semana"), PTAF ("todas las tardes"), P1W ("todas las semanas"), P1M ("mensual")
"""
