
import subprocess
import pymupdf

pdf = "data/raw_pdfs/Original_Document_5_SSP_Post_Eligibility_Document_2526.pdf"
tesseract = r"D:\Engg\7\LLM\tesseract.exe"

with pymupdf.open(pdf) as doc:
    page = doc[8]  # Page 9
    print("Page dimensions:", page.rect)
    pix = page.get_pixmap(dpi=300, alpha=False)
    image_bytes = pix.tobytes("png")

for mode in ["6", "11"]:
    result = subprocess.run(
        [
            tesseract,
            "stdin",
            "stdout",
            "-l",
            "kan+eng",
            "--psm",
            mode,
        ],
        input=image_bytes,
        capture_output=True,
        timeout=120,
    )

    print(f"\n{'=' * 20} PSM {mode} {'=' * 20}")

    if result.returncode == 0:
        print(result.stdout.decode("utf-8", errors="replace"))
    else:
        print(result.stderr.decode("utf-8", errors="replace"))
