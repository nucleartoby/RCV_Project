import PyMuPDF as fitz
import pdfplumber
import camelot
import pytesseract
from PIL import Image
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    text: str
    tables: List[pd.DataFrame]
    metadata: Dict[str, Any]
    entities: List[Dict[str, Any]]

class FinancialNER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForTokenClassification.from_pretrained("ProsusAI/finbert")
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
    
    def extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        spacy_entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            }
            for ent in doc.ents
        ]

        try:
            finbert_entities = []
            ner_results = self.ner_pipeline(text)
            for entity in ner_results:
                finbert_entities.append({
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "confidence": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "source": "finbert"
                })
        except Exception as e:
            logger.warning(f"FinBERT NER failed: {e}")
            finbert_entities = []

        all_entities = spacy_entities + finbert_entities
        return self._deduplicate_entities(all_entities)
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        entities.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        
        deduplicated = []
        for entity in entities:
            overlap = False
            for existing in deduplicated:
                if self._entities_overlap(entity, existing):
                    overlap = True
                    break
            if not overlap:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _entities_overlap(self, e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
        return not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"])

class DocumentProcessor:
    def __init__(self):
        self.ner = FinancialNER()
    
    def process_pdf(self, file_path: str) -> ExtractedContent:
        try:
            text_content = self._extract_text_pymupdf(file_path)

            tables = self._extract_tables_comprehensive(file_path)

            metadata = self._extract_pdf_metadata(file_path)

            if self._needs_ocr(text_content):
                ocr_text = self._extract_text_ocr(file_path)
                text_content = self._merge_text_content(text_content, ocr_text)

            entities = self.ner.extract_financial_entities(text_content)
            
            return ExtractedContent(
                text=text_content,
                tables=tables,
                metadata=metadata,
                entities=entities
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path}: {e}")
            raise
    
    def _extract_text_pymupdf(self, file_path: str) -> str:
        text_blocks = []
        
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    text_blocks.append(f"[Page {page_num + 1}]\n{text}")
        
        return "\n\n".join(text_blocks)
    
    def _extract_tables_comprehensive(self, file_path: str) -> List[pd.DataFrame]:
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df = self._clean_table(df)
                            if not df.empty:
                                tables.append(df)
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")

        try:
            camelot_tables = camelot.read_pdf(file_path, pages='all')
            for table in camelot_tables:
                if table.accuracy > 80:
                    df = table.df
                    df = self._clean_table(df)
                    if not df.empty:
                        tables.append(df)
        except Exception as e:
            logger.warning(f"Camelot table extraction failed: {e}")
        
        return self._deduplicate_tables(tables)
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how='all').dropna(axis=1, how='all')

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    
    def _deduplicate_tables(self, tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
        unique_tables = []
        for table in tables:
            is_duplicate = False
            for existing in unique_tables:
                if self._tables_similar(table, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _tables_similar(self, df1: pd.DataFrame, df2: pd.DataFrame, threshold: float = 0.8) -> bool:
        if df1.shape != df2.shape:
            return False
        
        # Compare content similarity
        try:
            similarity = (df1.astype(str) == df2.astype(str)).mean().mean()
            return similarity > threshold
        except:
            return False
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = {}
        
        try:
            with fitz.open(file_path) as doc:
                metadata.update(doc.metadata)
                metadata["page_count"] = len(doc)
                metadata["file_size"] = doc.xref_length()
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _needs_ocr(self, text: str, min_text_ratio: float = 0.1) -> bool:
        return len(text.strip()) < 100 or len(text.split()) < 20
    
    def _extract_text_ocr(self, file_path: str) -> str:
        ocr_text = []
        
        try:
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")

                    image = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(image)
                    
                    if text.strip():
                        ocr_text.append(f"[OCR Page {page_num + 1}]\n{text}")
        
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        
        return "\n\n".join(ocr_text)
    
    def _merge_text_content(self, extracted_text: str, ocr_text: str) -> str:
        """Merge extracted text with OCR text"""
        if not extracted_text.strip():
            return ocr_text
        if not ocr_text.strip():
            return extracted_text
        
        return f"{extracted_text}\n\n--- OCR CONTENT ---\n\n{ocr_text}"

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    section_type: Optional[str] = None
    contains_table: bool = False

class IntelligentChunker:
    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.nlp = spacy.load("en_core_web_sm")

        self.section_patterns = {
            'md_a': r'(?i)management.{0,20}discussion.{0,20}analysis',
            'risk_factors': r'(?i)risk\s+factors?',
            'business': r'(?i)^business$|^item\s+1\.\s*business',
            'financial_statements': r'(?i)financial\s+statements?',
            'notes': r'(?i)notes?\s+to\s+.*financial\s+statements?',
            'controls': r'(?i)internal\s+controls?',
            'legal': r'(?i)legal\s+proceedings?'
        }
    
    def intelligent_chunking(self, document: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Financial document-aware chunking"""
        sections = self._identify_sections(document)
        
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._process_section(section, metadata, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return chunks
    
    def _identify_sections(self, document: str) -> List[Dict[str, Any]]:
        """Identify document sections"""
        sections = []
        lines = document.split('\n')
        
        current_section = {
            'title': 'Introduction',
            'content': '',
            'section_type': 'general',
            'start_line': 0
        }
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()

            section_type = self._classify_section(line_stripped)
            
            if section_type and len(current_section['content']) > 100:
                sections.append(current_section)

                current_section = {
                    'title': line_stripped,
                    'content': '',
                    'section_type': section_type,
                    'start_line': i
                }
            else:
                current_section['content'] += line + '\n'

        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _classify_section(self, line: str) -> Optional[str]:
        """Classify section type based on header"""
        for section_type, pattern in self.section_patterns.items():
            if re.search(pattern, line):
                return section_type
        return None
    
    def _process_section(self, section: Dict[str, Any], metadata: Dict[str, Any], start_index: int) -> List[Chunk]:
        """Process a document section into chunks"""
        content = section['content'].strip()
        
        if not content:
            return []

        contains_tables = self._contains_financial_data(content)
        
        if contains_tables:
            return self._table_aware_split(section, metadata, start_index)
        else:
            return self._semantic_split(section, metadata, start_index)
    
    def _contains_financial_data(self, text: str) -> bool:
        """Check if text contains financial tables or data"""
        financial_indicators = [
            r'\$[\d,]+(?:\.\d{2})?',
            r'\b\d+\.?\d*\s*(?:million|billion|thousand)\b',
            r'\b\d{4}\s*(?:revenue|income|profit|loss|assets|liabilities)\b',
            r'(?:Q[1-4]|FY)\s*\d{4}',
            r'\b(?:GAAP|non-GAAP|EBITDA|EPS)\b'
        ]
        
        for pattern in financial_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        table_patterns = [
            r'(\n\s*\|.*\|.*\n){2,}',
            r'(\n.*\t.*\t.*\n){2,}',
            r'(\n\s*\d+\.\d+\s+\d+\.\d+.*\n){2,}'
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _table_aware_split(self, section: Dict[str, Any], metadata: Dict[str, Any], start_index: int) -> List[Chunk]:
        """Split section while preserving table structures"""
        content = section['content']
        chunks = []

        paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        chunk_index = start_index
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                chunk_metadata = {**metadata, **section}
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                    section_type=section['section_type'],
                    contains_table=True
                ))
                
                chunk_index += 1
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk.strip():
            chunk_metadata = {**metadata, **section}
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                chunk_index=chunk_index,
                section_type=section['section_type'],
                contains_table=True
            ))
        
        return chunks
    
    def _semantic_split(self, section: Dict[str, Any], metadata: Dict[str, Any], start_index: int) -> List[Chunk]:
        """Split section using semantic boundaries"""
        content = section['content']

        doc = self.nlp(content)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = start_index
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunk_metadata = {**metadata, **section}
                chunks.append(Chunk(
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    chunk_index=chunk_index,
                    section_type=section['section_type'],
                    contains_table=False
                ))
                
                chunk_index += 1

                overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        if current_chunk.strip():
            chunk_metadata = {**metadata, **section}
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                chunk_index=chunk_index,
                section_type=section['section_type'],
                contains_table=False
            ))
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of current chunk"""
        if len(text) <= overlap_size:
            return text

        sentences = text.split('. ')
        if len(sentences) > 1:
            overlap = '. '.join(sentences[-2:])
            if len(overlap) <= overlap_size:
                return overlap

        return text[-overlap_size:]