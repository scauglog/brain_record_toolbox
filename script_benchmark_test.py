import benchmark_walk_classifier as bwc

bench=bwc.Benchmark_Koho(32, 1, 32)
ares = bwc.Analyse_Result(32, 1)

bench.benchmark_simulated_data_from_healthy()
bench.save_result(extra_txt='simulated_v1')

ares.ground_truth.append('real_gnd_truth')
res_dict = ares.import_file('result_simulated_v1.pyObj')
res_dict = ares.success_rate_over_day(res_dict, group_by=4)
ares.plot_over_day(res_dict, 'success_rate', exclude_res=None)

res_dict = ares.success_rate_mean_day(res_dict)
ares.plot_mean(res_dict, 'success_rate_mean')

res_dict = ares.accuracy_over_day(res_dict, group_by=4)
ares.plot_over_day(res_dict, 'accuracy', exclude_res=None)

res_dict = ares.accuracy_mean_day(res_dict)
ares.plot_mean(res_dict, 'accuracy_mean')

bench.change_chan_group_by(128, 4)
bench.benchmark_SCI_data()
bench.save_result(extra_txt='SCI_v1')

del ares.ground_truth[-1]
res_dict = ares.import_file('result_SCI_v1.pyObj')
res_dict = ares.success_rate_over_day(res_dict, group_by=4)
ares.plot_over_day(res_dict, 'success_rate', exclude_res=None)

res_dict = ares.success_rate_mean_day(res_dict)
ares.plot_mean(res_dict, 'success_rate_mean')

res_dict = ares.accuracy_over_day(res_dict, group_by=4)
ares.plot_over_day(res_dict, 'accuracy', exclude_res=None)

res_dict = ares.accuracy_mean_day(res_dict)
ares.plot_mean(res_dict, 'accuracy_mean')
print 'END'