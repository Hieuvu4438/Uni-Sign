import os
import zipfile
import subprocess
import time
import shutil

# --- CẤU HÌNH ---
ZIP_PATH = r"D:\PROJECTS\Sign Language\CoSign_Dataset.zip"
SERVER = "user@<IP_SERVER>"  # Thay <IP_SERVER> bằng IP/Host của server Linux
REMOTE_DEST = "/home/haipd/Uni-Sign/data/CoSign/"
TEMP_EXTRACT_DIR = r"D:\PROJECTS\Sign Language\temp_extract"
DELAY_SECONDS = 2  # Khoảng nghỉ (giây) giữa mỗi folder để tránh nghẽn mạng

# --- KIỂM SOÁT SỐ LƯỢNG FOLDER ---
MAX_FOLDERS = 2  # Đặt số lượng folder muốn tải (ví dụ: 2). Đặt None nếu muốn tải tất cả.
# SPECIFIC_FOLDERS = ["Ăn", "Bàn"]  # Hoặc nếu muốn chỉ định đích danh tên folder, điền vào đây. Để [] nếu muốn tải theo MAX_FOLDERS.
SPECIFIC_FOLDERS = []

# 1. Tạo thư mục tạm và thư mục đích trên Server
os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)
print("Đang tạo thư mục đích trên server...")
subprocess.run(["ssh", SERVER, f"mkdir -p '{REMOTE_DEST}'"])

# 2. Đọc danh sách thư mục con từ file ZIP
print("Đang đọc danh sách folder trong file ZIP...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_files = zf.namelist()
    
    # Lấy tên các thư mục con cấp 1 (ví dụ: Ăn, Bàn, ...)
    top_folders = set()
    for name in all_files:
        parts = name.strip('/').split('/')
        if len(parts) > 0 and parts[0]:
            top_folders.add(parts[0])
            
    top_folders = sorted(list(top_folders))

    # Lọc danh sách folder theo cấu hình
    if SPECIFIC_FOLDERS:
        top_folders = [f for f in top_folders if f in SPECIFIC_FOLDERS]
    elif MAX_FOLDERS is not None and MAX_FOLDERS > 0:
        top_folders = top_folders[:MAX_FOLDERS]

    total = len(top_folders)
    print(f"Số lượng thư mục con sẽ tải: {total} (Danh sách: {top_folders})")

    # 3. Lặp qua từng thư mục con để giải nén & đẩy lên server
    for idx, folder_name in enumerate(top_folders, 1):
        print(f"\n[{idx}/{total}] Processing folder: '{folder_name}'")
        
        # Chỉ giải nén riêng folder hiện tại
        folder_files = [f for f in all_files if f.startswith(folder_name + '/')]
        zf.extractall(path=TEMP_EXTRACT_DIR, members=folder_files)
        
        local_folder_path = os.path.join(TEMP_EXTRACT_DIR, folder_name)
        
        # Upload folder này lên server qua SCP
        print(f" -> Uploading to server...")
        cmd = ["scp", "-r", local_folder_path, f"{SERVER}:{REMOTE_DEST}"]
        res = subprocess.run(cmd)
        
        if res.returncode == 0:
            print(f" -> SUCCESS: {folder_name}")
        else:
            print(f" -> ERROR: Không thể upload {folder_name}")
            
        # Xóa folder tạm trên máy Windows để tiết kiệm dung lượng
        shutil.rmtree(local_folder_path, ignore_errors=True)
        
        # Nghỉ ngắn giữa các lần upload
        time.sleep(DELAY_SECONDS)

print("\nHoàn tất tải các folder đã chọn lên Server!")
