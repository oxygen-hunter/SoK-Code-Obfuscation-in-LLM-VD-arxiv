# SoK: Code Obfuscation in the Age of LLM-based Vulnerability Detection

This is the repo for paper "SoK: Code Obfuscation in the Age of LLM-based Vulnerability Detection".

---

Main components:

`.\data`: unobfuscated and obfuscated code  

`.\results`: LLM vulnerability detection results

`.\scripts`: scripts for obfuscation, auditing, evaluating and analyzing

`.\obfuscators`: tools to obfuscate code

`.\auditors`: models for LLM vulnerability detection

`.\evaluators`: tools to evaluate detection report

`.\config`: secret info like proxy and OPENROUTER_API_KEY



**!!! Due to their large size, the `data` and `results` are not uploaded here. They are available upon request. Please contact xiao.li@smail.nju.edu.cn**

## dataset

| Dataset      | Lang     | Vuln Type | Count | Avg LOC | Avg Func |
| ------------ | -------- | --------- | ----- | ------- | -------- |
| Smart-bugs   | Solidity | 9         | 128   | 35      | 4        |
| PrimeVul     | C        | 119       | 143   | 256     | 9        |
| ReposVul_cpp | C++      | 35        | 84    | 172     | 10       |
| ReposVul_py  | Python   | 89        | 210   | 223     | 5        |



## models

| **Non-resoning LLM** |         | **Reasoning LLM**    |           | **Param** |
| -------------------- | ------- | -------------------- | --------- | --------- |
| Model                | Abbr.   | Model                | Abbr.     |           |
| **Qwen Series**      |         |                      |           |           |
| Qwen2.5-7B-Inst      | qn-7b   | DS-R1-Dist-Qwen-7B   | r1-qn-7b  | 7B        |
| Qwen2.5-14B-Inst     | qn-14b  | DS-R1-Dist-Qwen-14B  | r1-qn-14b | 14B       |
| Qwen2.5-32B-Inst     | qn-32b  | DS-R1-Dist-Qwen-32B  | r1-qn-32b | 32B       |
| **Llama Series**     |         |                      |           |           |
| Llama-3.1-8B-Inst    | lm-8b   | DS-R1-Dist-Llama-8B  | r1-lm-8b  | 8B        |
| Llama-3.3-70B-Inst   | lm-70b  | DS-R1-Dist-Llama-70B | r1-lm-70b | 70B       |
| **DeepSeek Series**  |         |                      |           |           |
| DeepSeek-V3          | ds-v3   | DeepSeek-R1          | ds-r1     | 671B      |
| **OpenAI Series**    |         |                      |           |           |
| GPT-3.5-turbo        | gpt-3.5 | -                    | -         | -         |
| GPT-4o               | gpt-4o  | -                    | -         | -         |
| -                    | -       | o3-mini              | o3-mini   | -         |



## commands for experiments

## set up environment

1. (for general-purpose LLM) fill your own `proxy` and `OPENROUTER_API_KEY` in `.\config\secret_config.json`
2. (for coding agent) ensure the accessible of your GitHub Copilot CLI and Codex CLI 
3. use conda to set up: `conda env create -f environment.yml`

### obfuscate

> python scripts/obfuscate.py --combo_mode diy-1 --code_path data\smartbugs\original --output_path data\smartbugs --do append
>
> python scripts/obfuscate.py --combo_mode diy-1 --code_path data\ReposVul_cpp\original --output_path data\ReposVul_cpp --do append
>
> python scripts/obfuscate.py --combo_mode diy-1 --code_path data\ReposVul_py\original --output_path data\ReposVul_py --do append
>
> python scripts/obfuscate.py --combo_mode diy-1 --code_path data\PrimeVul_c\original --output_path data\PrimeVul_c --do append

### audit

-- full

> python scripts/audit.py --auditors api --dataset smartbugs --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors api --dataset ReposVul_cpp --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors api --dataset ReposVul_py --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors api --dataset PrimeVul_c --combo_mode diy-1 --do append

---

-- downgrade top 20

> python scripts/audit.py --auditors agent --dataset smartbugs_downgrade_top20 --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors agent --dataset ReposVul_cpp_downgrade_top20 --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors agent --dataset ReposVul_py_downgrade_top20 --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors agent --dataset PrimeVul_c_downgrade_top20 --combo_mode diy-1 --do append

-- upgrade top 20

> python scripts/audit.py --auditors agent --dataset smartbugs_upgrade_top20 --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors agent --dataset ReposVul_cpp_upgrade_top20 --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors agent --dataset ReposVul_py_upgrade_top20 --combo_mode diy-1 --do append
>
> python scripts/audit.py --auditors agent --dataset PrimeVul_c_upgrade_top20 --combo_mode diy-1 --do append

### evaluate

-- full

> python scripts/evaluate.py --dataset smartbugs --auditors all
>
> python scripts/evaluate.py --dataset ReposVul_cpp --auditors all
>
> python scripts/evaluate.py --dataset ReposVul_py --auditors all
>
> python scripts/evaluate.py --dataset PrimeVul_c --auditors all

-- downgrade top 20

> python scripts/evaluate.py --dataset smartbugs_downgrade_top20 --auditors agent
>
> python scripts/evaluate.py --dataset ReposVul_cpp_downgrade_top20 --auditors agent
>
> python scripts/evaluate.py --dataset ReposVul_py_downgrade_top20 --auditors agent
>
> python scripts/evaluate.py --dataset PrimeVul_c_downgrade_top20 --auditors agent

-- upgrade top 20

> python scripts/evaluate.py --dataset smartbugs_upgrade_top20 --auditors agent
>
> python scripts/evaluate.py --dataset ReposVul_cpp_upgrade_top20 --auditors agent
>
> python scripts/evaluate.py --dataset ReposVul_py_upgrade_top20 --auditors agent
>
> python scripts/evaluate.py --dataset PrimeVul_c_upgrade_top20 --auditors agent

## analyze evaluation result

RQ1-RQ5

> python scripts/RQ1-ori-obf-by-dataset.py
>
> python scripts/RQ234_score_shift.py
>
> python scripts/RQ234_shift_analysis.py
>
> python scripts/RQ5_best_20_cases_in_bare_LLMs.py
>
> python scripts/RQ5_copy_large_to_small.py
>
> python scripts/RQ5-ori-obf-by-dataset-for-agent.py
>
> python scripts/RQ5-ori-obf-by-dataset-for-agent-downup.py



