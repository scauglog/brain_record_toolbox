import benchmark_walk_classifier as bwc

bench=bwc.Benchmark_Koho(32, 1, 32)
bench.benchmark_simulated_data_from_healthy()

bench.change_chan_group_by(128, 4)
bench.benchmark_SCI_data()
