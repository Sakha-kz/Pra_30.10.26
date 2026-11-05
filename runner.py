"""
Универсальный раннер ноутбуков без внешних тяжелых зависимостей.
Использует стандартный модуль json и Python exec(), перехватывает вывод stdout,
а также сохраняет графики matplotlib в ячейки ipynb как display_data (PNG base64).
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt

# Импортируем коды решений
from process_all_notebooks import LAB_CODES

class PlotCapture:
    """Перехватчик вызовов plt.show() для встраивания картинок в ipynb."""
    def __init__(self):
        self.images_b64 = []
        self._orig_show = plt.show

    def __enter__(self):
        self.images_b64 = []
        captured = self.images_b64
        
        # Обычная функция (не bound method), чтобы не ломать pyplot при смене бэкенда
        def _capture_show(*args, **kwargs):
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=130)
                buf.seek(0)
                b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                captured.append(b64_str)
            plt.close('all')
            
        plt.show = _capture_show
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.show = self._orig_show
        plt.close('all')

def run_notebook_suite():
    nb_dir = Path("notebooks")
    
    for num in range(1, 11):
        nb_path = nb_dir / f"lab_{num:02d}_student.ipynb"
        
        print(f"[Лабораторная {num:02d}/10] Выполнение и обновление: {nb_path.name}")
        
        
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        code_to_run = LAB_CODES[num]
        
        # Находим целевую ячейку кода
        target_idx = None
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code" and ("TODO" in "".join(cell.get("source", [])) or "NotImplementedError" in "".join(cell.get("source", []))):
                target_idx = i
                break
                
        if target_idx is None:
            for i in reversed(range(len(nb["cells"]))):
                if nb["cells"][i].get("cell_type") == "code":
                    target_idx = i
                    break
                    
        print(f"Обновление ячейки #{target_idx}...")
        nb["cells"][target_idx]["source"] = [line + "\n" for line in code_to_run.splitlines()]
        
        # Очищаем все ячейки
        for cell in nb["cells"]:
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
                
        # Выполняем код в изолированном пространстве имен, меняя CWD на папку ноутбука
        orig_cwd = os.getcwd()
        os.chdir(nb_dir)
        
        stdout_buf = io.StringIO()
        outputs = []
        
        try:
            with PlotCapture() as plotter:
                with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
                    exec_globals = {
                        "__name__": "__main__",
                        "__file__": str(nb_path.resolve())
                    }
                    exec(code_to_run, exec_globals)
                    
            text_output = stdout_buf.getvalue()
            if text_output.strip():
                print("STDOUT:")
                print(text_output[:500] + ("..." if len(text_output) > 500 else ""))
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in text_output.splitlines()]
                })
                
            for b64 in plotter.images_b64:
                outputs.append({
                    "data": {
                        "image/png": b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
                
            nb["cells"][target_idx]["outputs"] = outputs
            nb["cells"][target_idx]["execution_count"] = 1
            print(f"Ячейка успешно выполнена! Получено {len(plotter.images_b64)} графиков.")
            
        except Exception as e:
            print(f"ОШИБКА при выполнении: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            os.chdir(orig_cwd)
            
        # Сохраняем ноутбук
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            
        print(f"Ноутбук {nb_path.name} успешно сохранен!")

if __name__ == "__main__":
    run_notebook_suite()
