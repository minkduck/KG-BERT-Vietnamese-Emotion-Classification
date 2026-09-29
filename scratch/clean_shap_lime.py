import json, os, re, sys

sys.stdout.reconfigure(encoding='utf-8')

def cleanup_notebook(nb_path, dataset_name, seed_val, default_sentence, checkpoint_name):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    config_source = [
        "# =========================================================\n",
        "# CONFIGURATION & HYPER-PARAMETERS FOR SHAP/LIME EXPLAINABILITY\n",
        "# =========================================================\n",
        "import os, sys\n",
        "import torch\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "# Add repository root to python path\n",
        "sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))\n",
        "\n",
        "# Configurable experiment settings\n",
        f'DATASET_NAME = "{dataset_name}"\n',
        f'SEED = {seed_val}\n',
        'RDF_PATH = "ontology/ekman_appraisal_ontology.rdf"\n',
        f'CHECKPOINT_PATH = "results/{checkpoint_name}"\n',
        f'EXPLAIN_SENTENCE = "{default_sentence}"\n',
        "\n",
        "# Set random seed for reproducibility\n",
        "def seed_everything(seed=42):\n",
        "    import random\n",
        "    random.seed(seed)\n",
        '    os.environ["PYTHONHASHSEED"] = str(seed)\n',
        "    np.random.seed(seed)\n",
        "    torch.manual_seed(seed)\n",
        "    torch.cuda.manual_seed(seed)\n",
        "    torch.cuda.manual_seed_all(seed)\n",
        "    torch.backends.cudnn.deterministic = True\n",
        "    torch.backends.cudnn.benchmark = False\n",
        "\n",
        "seed_everything(SEED)\n",
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
        'print(f"Configured environment for {DATASET_NAME} (Seed: {SEED}) on device: {device}")\n'
    ]
    nb['cells'][0]['source'] = config_source

    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])

        src = re.sub(r'(!pip\s+install[^\n]*)', r'# \1', src)
        src = re.sub(r'"/kaggle/input/[^"]*\.rdf"', 'RDF_PATH', src)
        src = re.sub(r'"/kaggle/input/datasets/minkduck/uit-vsfc/UIT-VSFC"', '"data/vsfc"', src)
        src = re.sub(r'"/kaggle/input/datasets/minkduck/vsfcekman-1"', '"data/vsfc-ekman"', src)
        src = re.sub(r'"/kaggle/input/datasets/minkduck/uit-vsmec"', '"data/vsmec"', src)
        src = re.sub(r'"/kaggle/input/models/[^"]+\.pth"', 'CHECKPOINT_PATH', src)
        src = re.sub(r"'/kaggle/input/models/[^']+\.pth'", 'CHECKPOINT_PATH', src)

        lines = src.split('\n')
        cell['source'] = [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Cleaned up {nb_path}')

lime_dir = r'e:\MinkDuck\CS\write paper\code github KG+BERT\notebooks\05-shap-lime'
cleanup_notebook(os.path.join(lime_dir, 'vsfc-lime-roc.ipynb'), 'UIT-VSFC', 123, 'giảng viên nhiệt tình trong công tác giảng dạy .', 'phobert_gate_m3_vsfc.pth')
cleanup_notebook(os.path.join(lime_dir, 'vsfcekman-roc-lime (2).ipynb'), 'VSFC-Ekman', 0, 'giảng viên nhiệt tình trong công tác giảng dạy .', 'vsfc_ekman_raw_gate_m3.pth')
cleanup_notebook(os.path.join(lime_dir, 'vsmec-lime-roc (1).ipynb'), 'UIT-VSMEC', 1234, 'mấy ai được như vậy ??', 'vsmec_gate_m3.pth')
