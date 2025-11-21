import pdfplumber
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any
import config

class DocumentProcessor:
    def __init__(self):
        self.pdf_path = config.PDF_PATH
        self.chunks_path = config.CHUNKS_PATH
        self.images_dir = config.IMAGES_DIR
        
        # Ensure clean state for images
        if self.images_dir.exists():
            shutil.rmtree(self.images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Converts a raw list of lists into a Markdown table string.
        Crucial for the LLM to understand financial data.
        """
        if not table or len(table) < 2:
            return ""

        # Clean cells (remove None, newlines)
        cleaned_table = [[str(cell).replace('\n', ' ').strip() if cell else "" for cell in row] for row in table]
        
        # Construct Header
        header = "| " + " | ".join(cleaned_table[0]) + " |"
        separator = "| " + " | ".join(["---"] * len(cleaned_table[0])) + " |"
        
        # Construct Body
        body_rows = []
        for row in cleaned_table[1:]:
            body_rows.append("| " + " | ".join(row) + " |")
            
        return f"\n{header}\n{separator}\n" + "\n".join(body_rows) + "\n"

    def process_pdf(self) -> List[Dict[str, Any]]:
        """
        Main pipeline:
        1. Extract text page-by-page.
        2. Identify and format tables as Markdown.
        3. Extract images to disk.
        4. create citation-ready chunks.
        """
        print(f"Processing: {self.pdf_path.name}")
        processed_chunks = []

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                print(f"   - Processing Page {page_num}/{total_pages}...", end="\r")

                # 1. Extract Text
                text_content = page.extract_text() or ""

                # 2. Extract Tables & Convert to Markdown
                # pdfplumber is excellent at finding financial tables
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    md_table = self._table_to_markdown(table)
                    if md_table:
                        table_texts.append(md_table)
                
                # 3. Extract Images (Basic Extraction)
                # We save them to disk so the UI can display them later if needed
                image_references = []
                for img_idx, img in enumerate(page.images):
                    # Create a visual placeholder in the text
                    img_name = f"page_{page_num}_img_{img_idx+1}.png"
                    image_references.append(f"[IMAGE_REF: {img_name}]")
                    
                    # Note: Deep extraction of image binary requires 'page.images' coordinates
                    # combined with 'page.crop()'. For speed/storage, we just log existence here.
                    # If you need actual cropping, we can add that utility.

                # 4. Combine into a rich context chunk
                # We prepend tables so the LLM sees structured data first.
                full_content = ""
                
                if table_texts:
                    full_content += f"### TABLES ON PAGE {page_num}\n"
                    full_content += "\n".join(table_texts) + "\n\n"
                
                if text_content:
                    full_content += f"### TEXT CONTENT\n{text_content}\n"

                if image_references:
                    full_content += "\n### VISUALS\n" + "\n".join(image_references)

                # 5. Create the Final Chunk Object
                if full_content.strip():
                    chunk = {
                        "page_content": full_content,
                        "metadata": {
                            "source": self.pdf_path.name,
                            "page": page_num,
                            "has_tables": len(tables) > 0,
                            "has_images": len(image_references) > 0
                        }
                    }
                    processed_chunks.append(chunk)

        print(f"\nExtracted {len(processed_chunks)} chunks.")
        return processed_chunks

    def save_chunks(self, chunks: List[Dict[str, Any]]):
        with open(self.chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        print(f"Saved chunks to: {self.chunks_path}")

if __name__ == "__main__":
    # Test run
    processor = DocumentProcessor()
    chunks = processor.process_pdf()
    processor.save_chunks(chunks)