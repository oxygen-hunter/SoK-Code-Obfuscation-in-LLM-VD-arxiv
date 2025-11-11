# SoK: Code Obfuscation in the Age of LLM-based Vulnerability Detection

This is the repo for paper "SoK: Code Obfuscation in the Age of LLM-based Vulnerability Detection".

---

Main components:

`\data`: obfuscated code  

`\results`: LLM vulnerability detection results

`\scripts`: scripts for obfuscation, auditing and evaluating

## commands for experiments

### obfuscate

python scripts/obfuscate.py --combo_mode diy-1 --code_path data\smartbugs\original --output_path data\smartbugs --do append

python scripts/obfuscate.py --combo_mode diy-1 --code_path data\ReposVul_cpp\original --output_path data\ReposVul_cpp --do append

python scripts/obfuscate.py --combo_mode diy-1 --code_path data\ReposVul_py\original --output_path data\ReposVul_py --do append

python scripts/obfuscate.py --combo_mode diy-1 --code_path data\PrimeVul_c\original --output_path data\PrimeVul_c --do append

### audit

python scripts/audit.py --auditors api --dataset smartbugs --combo_mode diy-1 --do append

python scripts/audit.py --auditors api --dataset ReposVul_cpp --combo_mode diy-1 --do append

python scripts/audit.py --auditors api --dataset ReposVul_py --combo_mode diy-1 --do append

python scripts/audit.py --auditors api --dataset PrimeVul_c --combo_mode diy-1 --do append

---

-- downgrade top 20

python scripts/audit.py --auditors agent --dataset smartbugs_downgrade_top20 --combo_mode diy-1 --do append

python scripts/audit.py --auditors agent --dataset ReposVul_cpp_downgrade_top20 --combo_mode diy-1 --do append

python scripts/audit.py --auditors agent --dataset ReposVul_py_downgrade_top20 --combo_mode diy-1 --do append

python scripts/audit.py --auditors agent --dataset PrimeVul_c_downgrade_top20 --combo_mode diy-1 --do append

-- upgrade top 20

python scripts/audit.py --auditors agent --dataset smartbugs_upgrade_top20 --combo_mode diy-1 --do append

python scripts/audit.py --auditors agent --dataset ReposVul_cpp_upgrade_top20 --combo_mode diy-1 --do append

python scripts/audit.py --auditors agent --dataset ReposVul_py_upgrade_top20 --combo_mode diy-1 --do append

python scripts/audit.py --auditors agent --dataset PrimeVul_c_upgrade_top20 --combo_mode diy-1 --do append

### evaluate

python scripts/evaluate.py --dataset test --auditors DeepSeek-R1

python scripts/evaluate.py --dataset smartbugs --auditors all

python scripts/evaluate.py --dataset ReposVul_cpp --auditors all

python scripts/evaluate.py --dataset ReposVul_py --auditors all

python scripts/evaluate.py --dataset PrimeVul_c --auditors all

-- downgrade top 20

python scripts/evaluate.py --dataset smartbugs_downgrade_top20 --auditors agent

python scripts/evaluate.py --dataset ReposVul_cpp_downgrade_top20 --auditors agent

python scripts/evaluate.py --dataset ReposVul_py_downgrade_top20 --auditors agent

python scripts/evaluate.py --dataset PrimeVul_c_downgrade_top20 --auditors agent

-- upgrade top 20

python scripts/evaluate.py --dataset smartbugs_upgrade_top20 --auditors agent

python scripts/evaluate.py --dataset ReposVul_cpp_upgrade_top20 --auditors agent

python scripts/evaluate.py --dataset ReposVul_py_upgrade_top20 --auditors agent

python scripts/evaluate.py --dataset PrimeVul_c_upgrade_top20 --auditors agent


### count blind

python scripts/count_blind.py --dataset smartbugs --judge_mode all

python scripts/count_blind.py --dataset ReposVul_cpp --judge_mode all

python scripts/count_blind.py --dataset ReposVul_py --judge_mode all

python scripts/count_blind.py --dataset PrimeVul_c --judge_mode all

