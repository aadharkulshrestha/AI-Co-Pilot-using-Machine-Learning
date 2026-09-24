"""
PDF Ingestion Pipeline for Aviation Manuals (FCOM, QRH, FCTM)
Extracts page-level text, parses headings/sections, performs paragraph-aware chunking,
and indexes into ChromaDB with comprehensive metadata and citations.
"""

import os
import re
import sys
import glob
import logging
import argparse
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from dotenv import load_dotenv

from rag.vector_store import (
    add_chunks,
    get_collection,
    is_document_indexed,
    delete_document,
    get_indexed_documents
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag.ingest")

MANUALS_PDF_DIR = os.path.join("data", "manuals", "pdf")
MANUALS_PROCESSED_DIR = os.path.join("data", "manuals", "processed")


def infer_aircraft(filename: str, sample_text: str = "") -> str:
    """Infers aircraft type identifier from filename or initial text."""
    combined = f"{filename} {sample_text}".upper()
    if "787" in combined or "B787" in combined or "DREAMLINER" in combined:
        return "B787"
    if "350" in combined or "A350" in combined:
        return "A350"
    if "777" in combined or "B777" in combined:
        return "B777"
    if "737" in combined or "B737" in combined:
        return "B737"
    if "320" in combined or "A320" in combined:
        return "A320"
    if "330" in combined or "A330" in combined:
        return "A330"
    if "TEST" in combined or "SYNTHETIC" in combined:
        return "DEMO-AIRCRAFT"
    return "GENERAL"


def detect_section_heading(text: str, fallback_section: str = "General Procedures") -> str:
    """
    Identifies section or checklist heading from the text lines.
    Looks for standard aviation manual patterns (e.g. 'SECTION 2', 'ENGINE ABNORMAL', 'HYDRAULIC SYSTEM').
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return fallback_section

    # Check top 6 lines for strong section headers
    heading_patterns = [
        r"^(?:SECTION|CHAPTER|SYSTEM|PART)\s+[0-9A-Z\.-]+.*",
        r"^[A-Z0-9\s\-\/\(\)]{4,50}$", # Short all-caps line
        r".*(?:PROCEDURE|CHECKLIST|LIMITATION|INDICATION|MALFUNCTION|EMERGENCY|ABNORMAL|NORMAL).*",
    ]

    for line in lines[:6]:
        # Ignore common non-informative lines
        if len(line) < 4 or line.lower().startswith("page ") or re.match(r"^\d+$", line):
            continue
        for pat in heading_patterns:
            if re.match(pat, line, re.IGNORECASE) and len(line) < 65:
                # Clean up header
                return line.title() if line.isupper() else line

    return fallback_section


def chunk_text_by_paragraphs(
    text: str,
    target_words: int = 500,
    overlap_words: int = 100
) -> List[str]:
    """
    Splits text using paragraph boundaries to maintain semantic continuity,
    targeting 500-1000 tokens (words) per chunk with 100-150 word overlap.
    """
    if not text.strip():
        return []

    # Split into paragraphs (double newline or line breaks)
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) <= 1:
        # Fallback to single line breaks if no double newlines exist
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_count = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        words = para_clean.split()
        para_len = len(words)

        # If a single paragraph is larger than target_words * 1.5, subdivide it
        if para_len > target_words * 1.5:
            # Sub-split into word windows
            start = 0
            while start < para_len:
                end = min(start + target_words, para_len)
                sub_slice = " ".join(words[start:end])
                chunks.append(sub_slice)
                start += max(1, target_words - overlap_words)
            continue

        if current_count + para_len > target_words and current_chunk:
            # Emit current chunk
            chunk_str = "\n\n".join(current_chunk)
            chunks.append(chunk_str)

            # Keep overlapping tail
            all_words = chunk_str.split()
            overlap_tail = " ".join(all_words[-overlap_words:]) if len(all_words) > overlap_words else chunk_str
            current_chunk = [overlap_tail, para_clean]
            current_count = len(overlap_tail.split()) + para_len
        else:
            current_chunk.append(para_clean)
            current_count += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text and metadata from a PDF file page by page,
    generating structured chunks with document, page, section, and aircraft tags.
    """
    filename = os.path.basename(pdf_path)
    logger.info(f"Processing manual: {filename}")

    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to read PDF '{filename}': {e}")
        return []

    aircraft = infer_aircraft(filename)
    current_section = "General Procedures"
    all_chunks = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting text from {filename} page {page_num}: {e}")
            raw_text = ""

        clean_text = raw_text.strip()
        if not clean_text:
            continue

        # Detect new section if present on this page
        detected_sec = detect_section_heading(clean_text, fallback_section=current_section)
        current_section = detected_sec

        # Chunk this page's text
        page_chunks = chunk_text_by_paragraphs(clean_text, target_words=500, overlap_words=100)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = f"{filename}_p{page_num}_c{chunk_idx + 1}"
            metadata = {
                "document": filename,
                "page": page_num,
                "section": current_section,
                "aircraft": aircraft,
                "chunk_id": chunk_id,
                "total_pages": total_pages,
            }

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })

    logger.info(f"Extracted {len(all_chunks)} chunks from '{filename}' across {total_pages} pages.")
    return all_chunks


def ingest_manuals(
    pdf_dir: str = MANUALS_PDF_DIR,
    rebuild: bool = False,
    force: bool = False
) -> Dict[str, Any]:
    """
    Discovers all PDF manuals in pdf_dir, parses them, and inserts them into ChromaDB.
    Avoids duplicate ingestion unless rebuild or force is True.
    """
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(MANUALS_PROCESSED_DIR, exist_ok=True)

    if rebuild:
        logger.warning("Rebuild flag specified: Resetting ChromaDB collection...")
        get_collection(reset=True)

    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    if not pdf_files:
        logger.warning(f"No PDF manuals found in directory '{pdf_dir}'.")
        return {
            "status": "empty",
            "message": f"No PDF manuals found in {pdf_dir}. Place FCOM/QRH PDFs there or run synthetic demo generator.",
            "documents_indexed": 0,
            "chunks_added": 0,
            "files": []
        }

    indexed_docs = get_indexed_documents()
    total_chunks_added = 0
    processed_files = []

    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)

        if fname in indexed_docs and not rebuild and not force:
            logger.info(f"Skipping '{fname}': already indexed in ChromaDB.")
            processed_files.append({"document": fname, "status": "already_indexed", "chunks": 0})
            continue

        if fname in indexed_docs and force:
            delete_document(fname)

        chunks = process_pdf(pdf_path)
        if chunks:
            added = add_chunks(chunks)
            total_chunks_added += added
            processed_files.append({"document": fname, "status": "indexed", "chunks": added})
        else:
            processed_files.append({"document": fname, "status": "no_text_extracted", "chunks": 0})

    return {
        "status": "success",
        "documents_indexed": len([f for f in processed_files if f["status"] == "indexed"]),
        "chunks_added": total_chunks_added,
        "files": processed_files
    }


def create_synthetic_test_manual(output_dir: str = MANUALS_PDF_DIR) -> str:
    """
    Generates a synthetic test flight manual (TEST_FLIGHT_MANUAL.pdf) with fictional procedures
    for testing and Demo Mode presentations without infringing copyright.
    """
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, "TEST_FLIGHT_MANUAL.pdf")

    # Use pypdf to create a multi-page PDF document
    from pypdf import PdfWriter
    from io import BytesIO

    # Check if reportlab or fpdf exists; if not, we can generate a valid PDF stream
    # or write structured PDF objects. Let's see if we can use simple PDF generation.
    try:
        # Standard PDF stream construction with valid header, catalog, pages, fonts, contents
        pages_content = [
            # Page 1: Engine 2 Vibration
            ("SECTION 7 - ENGINE INDICATIONS & ABNORMALS",
             "BOEING 787 / AIRBUS TEST FLIGHT MANUAL (SYNTHETIC DEMO EDITION)\n\n"
             "7.1 ENGINE VIBRATION INDICATION (N1 / N2 EXCEEDANCE)\n\n"
             "CONDITION: High engine vibration levels indicated on Engine Indication and Crew Alerting System (EICAS) or Electronic Centralized Aircraft Monitor (ECAM).\n"
             "Vibration display illuminates AMBER when broadband vibration index exceeds 4.0 units on either spool.\n\n"
             "PILOT ACTION / CREW RESPONSE:\n"
             "1. Thrust Lever (affected engine): Retard slowly toward IDLE until vibration decreases below 2.5 units.\n"
             "2. If vibration remains above 4.0 units accompanied by abnormal secondary engine indications (high ITT/EGT, abnormal oil pressure, or violent airframe buffet):\n"
             "   - Conclude severe engine mechanical damage exists.\n"
             "   - Execute ENGINE SHUTDOWN CHECKLIST.\n"
             "   - Set Autothrottle / A/THR to DISENGAGE on affected channel.\n"
             "3. If vibration decreases to normal operational range with thrust reduction:\n"
             "   - Maintain reduced thrust for remainder of flight.\n"
             "   - Plan descent and landing at nearest suitable diversion airport.\n"
             "   - Cross-check fuel balancing and bleed air configurations.\n"
             "4. Do NOT attempt engine relight if high vibration was accompanied by abnormal noise or uncommanded yaw."),

            # Page 2: Dual Engine Flameout
            ("SECTION 8 - DUAL ENGINE FAILURE & RELIGHT",
             "8.2 DUAL ENGINE FLAMEOUT / RELIGHT PROCEDURE\n\n"
             "CONDITION: Sudden loss of thrust on all engines at cruising altitude.\n\n"
             "IMMEDIATE MEMORY ITEMS:\n"
             "1. Airspeed: Establish and maintain 270 KIAS optimum glide speed / windmill relight speed.\n"
             "2. APU (Auxiliary Power Unit): START switch to ON. Verify APU generator online within 60 seconds.\n"
             "3. RAT (Ram Air Turbine): Manual deploy handle PULL if electrical emergency bus not automatically powered.\n"
             "4. Engine Start Switches: Set to CONT (Continuous Ignition).\n"
             "5. Thrust Levers: Retard to IDLE to prevent fuel pooling during restart cycle.\n"
             "6. Crew Coordination: Notify ATC via Mayday broadcast on 121.5 MHz; Squawk 7700.\n"
             "7. Altitude: Determine glide distance to nearest suitable runway (approx 2.5 NM per 1,000 ft altitude lost)."),

            # Page 3: Rapid Cabin Depressurization
            ("SECTION 2 - EMERGENCY DESCENT PROCEDURES",
             "2.4 RAPID CABIN DEPRESSURIZATION & EMERGENCY DESCENT\n\n"
             "CONDITION: Cabin altitude exceeds 10,000 feet or CABIN ALTITUDE WARNING alert illuminates.\n\n"
             "CREW IMMEDIATE ACTIONS:\n"
             "1. Oxygen Masks: DON, Set 100% and Emergency Flow.\n"
             "2. Crew Communications: Establish interphone communication through mask microphones.\n"
             "3. Flight Controls: Initiate immediate emergency descent.\n"
             "   - Autopilot: DISENGAGE or select FLCH / OPEN DESCENT.\n"
             "   - Thrust Levers: IDLE.\n"
             "   - Speedbrake: EXTEND fully.\n"
             "   - Target Altitude: Descend immediately to 10,000 feet MSL or Minimum Enroute IFR Altitude (MEA), whichever is higher.\n"
             "4. Passenger Oxygen: Verify passenger oxygen masks deployed (overhead switch OVERRIDE).\n"
             "5. Transponder: Select SQUAWK 7700 on ATC transponder."),

            # Page 4: Hydraulic System Failure
            ("SECTION 13 - HYDRAULIC SYSTEMS",
             "13.3 HYDRAULIC SYSTEM B LEAK / LOW PRESSURE\n\n"
             "CONDITION: Hydraulic System B reservoir quantity below 10% and low pressure warning active.\n\n"
             "OPERATIONAL IMPACT:\n"
             "Normal landing gear extension may be inoperative. Alternate gear extension via gravity drop will be required.\n"
             "Inboard flight spoilers and left thrust reverser may have degraded response times.\n\n"
             "PROCEDURES:\n"
             "1. Hydraulic Pump B: Switch to OFF to prevent pump cavitation and thermal damage.\n"
             "2. Flaps / Slats: Plan approach using Flap 20 or designated abnormal configuration.\n"
             "3. Approach Speed: Compute VREF + 20 knots additive for degraded spoiler braking.\n"
             "4. Landing Distance: Multiply dry runway landing distance by factor of 1.45.\n"
             "5. Nose Wheel Steering: System B loss causes loss of normal nosewheel steering; plan towing off runway after rollout."),

            # Page 5: Windshear Escape Maneuver
            ("SECTION 16 - ADVERSE WEATHER OPERATIONS",
             "16.7 WINDSHEAR WARNING ON APPROACH & TAKEOFF\n\n"
             "CONDITION: Automated 'WINDSHEAR! WINDSHEAR! WINDSHEAR!' audio warning sounds or severe airspeed deviation (>15 kts) during final approach.\n\n"
             "ESCAPE MANEUVER PROCEDURE:\n"
             "1. Thrust Levers: Disconnect Autothrottle and smoothly advance to MAXIMUM TAKEOFF/GO-AROUND THRUST (TOGA).\n"
             "2. Wings: Level wings and rotate smoothly toward initial pitch attitude of 15 degrees nose up.\n"
             "3. Configuration: Do NOT change flap or gear configuration until clear of windshear conditions.\n"
             "4. Flight Director: Follow Flight Director windshear guidance pitch cue if available; respect stick shaker stall margin.\n"
             "5. Positive Rate: When clear of windshear and positive rate of climb established, retract gear and flaps per normal go-around procedure.\n"
             "6. ATC Report: Report windshear encounter, altitude, and airspeed loss to ATC Tower immediately.")
        ]

        # Let's write a standard clean PDF using raw PDF syntax if no external PDF generator is installed
        # A raw PDF 1.4 can be written reliably:
        def build_simple_pdf(pages_data):
            objects = []
            def add_obj(content):
                objects.append(content)
                return len(objects)

            # Font object
            font_id = 3
            # Pages array
            page_ids = []

            # We will generate catalog, pages, font, and individual page objects
            # Page objects: page_id, content_id
            current_id = 4
            page_specs = []
            for title, body in pages_data:
                p_id = current_id
                c_id = current_id + 1
                current_id += 2
                page_specs.append((p_id, c_id, title, body))

            # Root Catalog
            # obj 1: Catalog
            # obj 2: Pages
            # obj 3: Font
            lines = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
            offsets = []

            def write_object(obj_num, data_bytes):
                offsets.append(sum(len(x) for x in lines))
                lines.append(f"{obj_num} 0 obj\n".encode("latin1"))
                lines.append(data_bytes)
                lines.append(b"\nendobj\n")

            # Catalog
            write_object(1, b"<< /Type /Catalog /Pages 2 0 R >>")
            # Pages
            kids_str = " ".join([f"{p[0]} 0 R" for p in page_specs])
            write_object(2, f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_specs)} >>".encode("latin1"))
            # Font
            write_object(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

            for p_id, c_id, title, body in page_specs:
                # Text stream
                text_stream = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
                # Header
                text_stream.append(f"({escape_pdf(title)}) Tj T* T*")
                for line in body.split("\n"):
                    if not line:
                        text_stream.append("T*")
                    else:
                        text_stream.append(f"({escape_pdf(line)}) Tj T*")
                text_stream.append("ET")
                stream_bytes = "\n".join(text_stream).encode("latin1", errors="replace")

                content_obj = f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1") + stream_bytes + b"\nendstream"
                write_object(c_id, content_obj)

                page_obj = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {c_id} 0 R >>".encode("latin1")
                write_object(p_id, page_obj)

            # XRef table
            start_xref = sum(len(x) for x in lines)
            lines.append(f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n".encode("latin1"))
            for off in offsets:
                lines.append(f"{off:010d} 00000 n \n".encode("latin1"))
            lines.append(f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{start_xref}\n%%EOF\n".encode("latin1"))

            return b"".join(lines)

        def escape_pdf(s):
            return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

        pdf_bytes = build_simple_pdf(pages_content)
        with open(dest_path, "wb") as f:
            f.write(pdf_bytes)

        logger.info(f"Generated synthetic test flight manual at: {dest_path}")
        return dest_path

    except Exception as e:
        logger.error(f"Failed to generate synthetic test flight manual: {e}")
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest aviation flight manuals into ChromaDB vector store.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild collection from scratch.")
    parser.add_argument("--force", action="store_true", help="Force re-indexing of all existing manuals.")
    parser.add_argument("--create-demo-manual", action="store_true", help="Generate synthetic test manual for demonstration.")
    parser.add_argument("--dir", type=str, default=MANUALS_PDF_DIR, help="Directory containing manual PDFs.")

    args = parser.parse_args()

    # Check if manual directory exists and has files
    os.makedirs(args.dir, exist_ok=True)
    existing_pdfs = glob.glob(os.path.join(args.dir, "*.pdf"))

    if args.create_demo_manual or not existing_pdfs:
        logger.info("No manuals found or --create-demo-manual passed. Generating synthetic flight manual...")
        create_synthetic_test_manual(args.dir)

    result = ingest_manuals(pdf_dir=args.dir, rebuild=args.rebuild, force=args.force)
    print(f"\nIngestion Summary:")
    print(f"Status: {result['status']}")
    print(f"Documents Indexed: {result['documents_indexed']}")
    print(f"Chunks Added: {result['chunks_added']}")
    for f in result.get("files", []):
        print(f" - {f['document']}: {f['status']} ({f['chunks']} chunks)")
