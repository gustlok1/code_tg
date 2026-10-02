import sys
from pathlib import Path

# permite importar dataset_v2.py e transformadores.py (um diretório acima de tests/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
