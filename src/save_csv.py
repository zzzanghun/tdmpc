import pandas as pd
import os
import datetime
print(os.path.join(os.path.dirname(os.path.realpath(__file__))), "!!!!!!!!!!!!!")

def result_save(cfg):
	now = datetime.datetime.now()
	local_now = now.astimezone()

	result_name_dir = os.path.join(cfg.task, cfg.name_for_result_savem, "{}_{}_{}".format(local_now.month, local_now.day, local_now.hour))
	if not os.path.exists(result_name_dir):
		os.mkdir(result_name_dir)