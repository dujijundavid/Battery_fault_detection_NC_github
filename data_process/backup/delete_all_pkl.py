import os
import subprocess
import shutil

def delete_dir_fast(path: str):
    if not os.path.exists(path):
        return
    # 使用 \\?\ 前缀，避免长路径问题
    p = os.path.normpath(path)
    if p.startswith(r"\\"):          # UNC 路径
        p_long = r"\\?\UNC" + p[1:]
    else:
        p_long = r"\\?\{}".format(p)

    # 优先用系统 rmdir（最快）
    rc = subprocess.call(["cmd", "/c", "rmdir", "/s", "/q", p_long],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc != 0 and os.path.exists(path):
        # 失败时兜底
        shutil.rmtree(path, ignore_errors=True)

if __name__ == "__main__":
    target = r"C:\Users\YIFSHEN\Documents\01_InputRawData\3000_normal_pkl"
    delete_dir_fast(target)
