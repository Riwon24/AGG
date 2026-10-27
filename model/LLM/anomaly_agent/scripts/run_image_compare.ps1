# 이미지 단위(AGG0 11장 vs AGG2 18장, 후보를 만든 AGG1 이미지는 제외) 평가를 후보 두 벌로 차례로 돌린다.
# 먼저 같은 PowerShell 창에서 ANTHROPIC_API_KEY, ANTHROPIC_WORKSPACE_ID 를 설정해 둘 것 (키는 채팅에 붙이지 말 것).
$env:WSLENV = "ANTHROPIC_API_KEY/u:ANTHROPIC_WORKSPACE_ID/u"
$m = "anthropic/claude-sonnet-5-5"
foreach ($run in @(@("agg_img_cand7", "atomic_candidates_agg.json"), @("agg_img_cand2", "atomic_candidates_agg_2x2.json"))) {
  $save = $run[0]; $file = $run[1]
  wsl -d Ubuntu --cd "~/AnomalyAgent" -- bash -c "`$HOME/anomalyagent_venv/bin/python evaluate_agg_v12_target.py --unit image --save_path ./results/$save --atomic_candidates_file $file --attn_implementation sdpa --planner_model_type api --planner_api_model $m --reflector_model_type api --reflector_api_model $m --reasoner_model_type api --reasoner_api_model $m --image_description_model_type api --image_description_api_model $m --atomic_candidate_llm_model_type api --atomic_candidate_llm_api_model $m --concurrent 1"
}
