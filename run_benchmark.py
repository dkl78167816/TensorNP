'''
@Author: duke
@Date: 2021-05-20 11:06:51
@LastEditTime: 2021-06-01 00:33:15
@LastEditors: duke
@Description: Run benchmarks.
'''
import tensornp

# whether to run banchmarks for specific algorithm.
run = {
    'cp_speed': False,
    'cp_valid': False,
    'hosvd_speed': False,
    'hosvd_valid': False,
    'tucker_speed': False,
    'tucker_valid': False,
    't_svd_speed': False,
    't_svd_valid': False,
    'tt_speed': False,
    'tt_valid': False,
    'tr_speed': False,
    'tr_valid': False
}


run['tr_speed'] = True
run['tr_valid'] = True


if run['cp_speed']:
    import benchmarks.bench_cp
    benchmarks.bench_cp.test_speed_3ord()
    benchmarks.bench_cp.test_speed_8ord()
if run['cp_valid']:
    import benchmarks.bench_cp
    benchmarks.bench_cp.valid_3ord()
    
if run['hosvd_speed']:
    import benchmarks.bench_hosvd
    benchmarks.bench_hosvd.test_speed_3ord()
    benchmarks.bench_hosvd.test_speed_8ord()
if run['hosvd_valid']:
    import benchmarks.bench_hosvd
    benchmarks.bench_hosvd.valid_3ord()

if run['tucker_speed']:
    import benchmarks.bench_tucker
    benchmarks.bench_tucker.test_speed_3ord()
    benchmarks.bench_tucker.test_speed_8ord()
if run['tucker_valid']:
    import benchmarks.bench_tucker
    benchmarks.bench_tucker.valid_3ord()

if run['t_svd_speed']:
    import benchmarks.bench_t_svd
    benchmarks.bench_t_svd.test_speed_3ord()
if run['t_svd_valid']:
    import benchmarks.bench_t_svd
    benchmarks.bench_t_svd.valid_3ord()

if run['tt_speed']:
    import benchmarks.bench_tt
    benchmarks.bench_tt.test_speed_3ord()
    benchmarks.bench_tt.test_speed_8ord()
if run['tt_valid']:
    import benchmarks.bench_tt
    benchmarks.bench_tt.valid_3ord()

if run['tr_speed']:
    import benchmarks.bench_tr
    benchmarks.bench_tr.test_speed_3ord()
    benchmarks.bench_tr.test_speed_8ord()
if run['tr_speed']:
    import benchmarks.bench_tr
    benchmarks.bench_tr.valid_3ord()
