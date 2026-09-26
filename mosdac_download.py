import paramiko
import os
from getpass import getpass

# ==============================
# MOSDAC SFTP DETAILS
# ==============================

HOST = "download.mosdac.gov.in"
PORT = 22

USERNAME = input("Enter MOSDAC username: ")
PASSWORD = getpass("Enter MOSDAC password: ")

# Remote folder containing your ordered files
REMOTE_FOLDER = "/"

# Local folder where you want to save the files
LOCAL_FOLDER = r"D:\MOSDAC_Data"

# ==============================
# CREATE LOCAL FOLDER
# ==============================

os.makedirs(LOCAL_FOLDER, exist_ok=True)

# ==============================
# CONNECT TO SFTP
# ==============================

print("\nConnecting to MOSDAC SFTP...")

transport = paramiko.Transport((HOST, PORT))

transport.connect(
    username=USERNAME,
    password=PASSWORD
)

sftp = paramiko.SFTPClient.from_transport(transport)

print("Connected successfully!")

# ==============================
# LIST REMOTE FILES
# ==============================

print("\nFiles/folders available on server:")

files = sftp.listdir_attr(REMOTE_FOLDER)

for file in files:
    print(file.filename)

# ==============================
# DOWNLOAD FUNCTION
# ==============================

def download_folder(remote_path, local_path):

    os.makedirs(local_path, exist_ok=True)

    for item in sftp.listdir_attr(remote_path):

        remote_item = remote_path + "/" + item.filename
        local_item = os.path.join(local_path, item.filename)

        # Check whether it is a directory
        if item.st_mode & 0o40000:

            print(f"\nEntering folder: {item.filename}")

            download_folder(
                remote_item,
                local_item
            )

        else:

            print(f"Downloading: {item.filename}")

            sftp.get(
                remote_item,
                local_item
            )

            print("Completed:", item.filename)


# ==============================
# START DOWNLOAD
# ==============================

print("\nStarting download...")

download_folder(
    REMOTE_FOLDER,
    LOCAL_FOLDER
)

# ==============================
# CLOSE CONNECTION
# ==============================

sftp.close()
transport.close()

print("\n================================")
print("DOWNLOAD COMPLETED")
print("================================")