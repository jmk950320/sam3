import base64
import requests

def onedrive_public_download(share_url, output_path):
    # 1. Base64 URL 처리용
    base64_value = base64.b64encode(share_url.encode()).decode()
    base64_value = base64_value.replace("/", "_").replace("+", "-").rstrip("=")

    # 2. OneDrive API: 공유 링크 기반 직접 다운로드
    api_url = f"https://api.onedrive.com/v1.0/shares/u!{base64_value}/root/content"

    print("Downloading from:", api_url)

    r = requests.get(api_url)
    if r.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(r.content)
        print("Saved to", output_path)
    else:
        print("Error:", r.status_code, r.text)

# ---------- 실행 ----------
onedrive_public_download(
    "https://1drv.ms/p/c/e748add40d2bda36/IQBqGzl3Vy1ZS5XptK6YCwBuAdZL58XpR6yq1tlZ3ptCvEE?e=sQhCeD",
    "downloaded.pptx"
)
