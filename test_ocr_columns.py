
import subprocess
import pymupdf

PDF = "data/raw_pdfs/Original_Document_5_SSP_Post_Eligibility_Document_2526.pdf"
TESSERACT = r"D:\Engg\7\LLM\tesseract.exe"

with pymupdf.open(PDF) as doc:
    page = doc[8]  # Page 9
    print("Page dimensions:", page.rect)

    # Render each region separately to prevent OCR from
    # mixing text from adjacent table columns.
    regions = {
        "LEFT / SERIAL + SCHEME": (0, 90, 175, 760),
        "SCHEME": (100, 90, 260, 760),
        "ELIGIBILITY": (225, 90, 435, 760),
        "DOCUMENTS REQUIRED": (400, 90, 612, 760),
    }

    for name, coords in regions.items():
        clip = pymupdf.Rect(*coords)
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            clip=clip,
            alpha=False,
        )

        result = subprocess.run(
            [
                TESSERACT,
                "stdin",
                "stdout",
                "-l",
                "kan+eng",
                "--psm",
                "6",
            ],
            input=pix.tobytes("png"),
            capture_output=True,
            timeout=120,
        )

        print(f"\n{'=' * 15} {name} {'=' * 15}")

        if result.returncode == 0:
            print(result.stdout.decode("utf-8", errors="replace"))
        else:
            print(result.stderr.decode("utf-8", errors="replace"))

# Save a page preview with the proposed crop boundaries.
with pymupdf.open(PDF) as doc:
    page = doc[8]
    preview = page.new_shape()

    regions = {
        "SCHEME": (100, 90, 260, 760),
        "ELIGIBILITY": (225, 90, 435, 760),
        "DOCUMENTS": (400, 90, 612, 760),
    }

    for name, coords in regions.items():
        rect = pymupdf.Rect(*coords)
        preview.draw_rect(rect)
        preview.insert_text(
            rect.tl + (3, 12),
            name,
            fontsize=8,
            color=(1, 0, 0),
        )

    preview.commit()
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
    pix.save("page9_crop_preview.png")

print("Saved page9_crop_preview.png")

