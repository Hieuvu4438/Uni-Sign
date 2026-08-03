import os
import zipfile
import subprocess
import time
import shutil

# --- CẤU HÌNH ---
ZIP_PATH = r"D:\PROJECTS\Sign Language\CoSign_Dataset.zip"
SERVER = "cosign@171.226.10.153"  # IP / Host của server Linux
REMOTE_DEST = "/home/cosign/Uni-Sign/data/CoSign/"
TEMP_EXTRACT_DIR = r"D:\PROJECTS\Sign Language\temp_extract"
DELAY_SECONDS = 2  # Khoảng nghỉ (giây) giữa mỗi folder để tránh nghẽn mạng

# --- TÙY CHỌN TẢI & BỎ QUA ---
SKIP_EXISTING = True  # True: Bỏ qua folder đã có trên server; False: Đẩy đè/tải lại
MAX_FOLDERS = None    # Đặt số lượng folder muốn tải (ví dụ: 2). Đặt None nếu muốn tải tất cả.
SPECIFIC_FOLDERS = [] # Chỉ định danh sách tên folder (ví dụ: ["Ăn", "Bàn"]). Để [] nếu tải theo MAX_FOLDERS.

# 1. Tạo thư mục tạm và thư mục đích trên Server
os.makedirs(TEMP_EXTRACT_DIR, exist_ok=True)
print("Đang tạo thư mục đích trên server...")
subprocess.run(["ssh", SERVER, f"mkdir -p '{REMOTE_DEST}'"])

# 2. Lấy danh sách các folder đã có sẵn trên Server (phục vụ tính năng SKIP_EXISTING)
existing_remote_folders = set()
if SKIP_EXISTING:
    print("Đang kiểm tra danh sách folder đã có trên server...")
    temp_list_file = os.path.join(TEMP_EXTRACT_DIR, "_remote_folders.txt")
    try:
        with open(temp_list_file, "w", encoding="utf-8") as f_out:
            res = subprocess.run(
                ["ssh", SERVER, f"ls -1 '{REMOTE_DEST}'"],
                stdout=f_out
            )
        if res.returncode == 0 and os.path.exists(temp_list_file):
            with open(temp_list_file, "r", encoding="utf-8") as f_in:
                existing_remote_folders = set(line.strip() for line in f_in if line.strip())
            print(f" -> Đã tìm thấy {len(existing_remote_folders)} folder đã có trên server.")
            if os.path.exists(temp_list_file):
                os.remove(temp_list_file)
        else:
            print(" -> Không thể lấy danh sách folder từ server (sẽ tải bình thường).")
    except Exception as e:
        print(f" -> Không thể lấy danh sách folder từ server: {e}")


# 3. Đọc danh sách thư mục con từ file ZIP
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
    print(f"Số lượng thư mục con xét tải: {total}")

    skipped_count = 0
    uploaded_count = 0

    # 4. Lặp qua từng thư mục con để giải nén & đẩy lên server
    for idx, folder_name in enumerate(top_folders, 1):
        print(f"\n[{idx}/{total}] Processing folder: '{folder_name}'")
        
        # BỎ QUA NẾU FOLDER ĐÃ TỒN TẠI TRÊN SERVER
        if SKIP_EXISTING and folder_name in existing_remote_folders:
            print(f" -> SKIPPED: Thư mục '{folder_name}' đã tồn tại trên Server.")
            skipped_count += 1
            continue

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
            uploaded_count += 1
            existing_remote_folders.add(folder_name)
        else:
            print(f" -> ERROR: Không thể upload {folder_name}")
            
        # Xóa folder tạm trên máy local để tiết kiệm dung lượng
        shutil.rmtree(local_folder_path, ignore_errors=True)
        
        # Nghỉ ngắn giữa các lần upload
        time.sleep(DELAY_SECONDS)

print(f"\nHoàn tất! Thành công: {uploaded_count} folder, Bỏ qua (đã có sẵn): {skipped_count} folder.")

