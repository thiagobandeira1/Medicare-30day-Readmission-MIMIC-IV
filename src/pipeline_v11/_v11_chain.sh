#!/bin/bash
# v11 pipeline after feature selection; stops at the first failure
cd "/c/Users/Thiago/Documents/Publication Hospital Research/presentation-update"
PY=/c/Users/Thiago/anaconda3/envs/capstone/python.exe
for s in _v11_phase57_final _v11_extras _v11_stages_and_figures _v11_review_analyses _v11_leak_test _v11_make_fig1 _v11_episode_end_rule _v11_baselines_oof _v11_evaluate _v11_trend_causal _v11_index_stay_audit _v11_subgroup_calibration _v11_leaked_strata _v11_fig4; do
  echo "=== $s $(date +%H:%M:%S)"
  $PY $s.py > _log_$s.txt 2>&1 || { echo "FAILED $s"; tail -5 _log_$s.txt; exit 1; }
done
echo "=== chain done $(date +%H:%M:%S)"
