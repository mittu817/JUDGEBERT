import io
import os
import re
from datetime import datetime
from typing import Optional


class DocumentExporter:
    """
    Exports repaired or verified plain-language legal documents to formatted
    Microsoft Word (.docx) or Unicode vector PDF (.pdf) files.
    Supports English, French diacritics, currencies ($ € £ ₹), and Hindi Devanagari.
    """

    _WINDOWS_FONT_CANDIDATES = {
        "nirmala": r"C:\Windows\Fonts\Nirmala.ttc",
        "arial": r"C:\Windows\Fonts\arial.ttf",
        "arial_bold": r"C:\Windows\Fonts\arialbd.ttf",
        "segoe": r"C:\Windows\Fonts\segoeui.ttf",
        "segoe_bold": r"C:\Windows\Fonts\segoeuib.ttf",
    }

    @staticmethod
    def safe_filename(base_name: str, ext: str) -> str:
        """Sanitizes download filename to safe alphanumeric characters."""
        clean = re.sub(r"[^A-Za-z0-9_\-]+", "_", (base_name or "judgebert_repaired_document").strip())
        clean = clean.strip("_") or "judgebert_repaired_document"
        clean_ext = ext.lstrip(".").lower()
        return f"{clean}.{clean_ext}"

    @classmethod
    def export_docx(
        cls,
        text: str,
        score: Optional[float] = None,
        tier: str = "",
        mode: str = "",
        title: str = "Verified Plain-Language Document",
    ) -> bytes:
        """Generates a clean formatted .docx document containing the repaired/verified text."""
        import docx
        from docx.shared import Pt, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        if not text or not text.strip():
            raise ValueError("Cannot export an empty document.")

        doc = docx.Document()

        # Subtle top page margins
        for section in doc.sections:
            section.top_margin = Inches(0.8)
            section.bottom_margin = Inches(0.8)
            section.left_margin = Inches(0.9)
            section.right_margin = Inches(0.9)

        # Document Header Banner
        h_para = doc.add_paragraph()
        h_para.paragraph_format.space_after = Pt(2)
        h_run = h_para.add_run(title)
        h_run.bold = True
        h_run.font.size = Pt(16)
        h_run.font.color.rgb = RGBColor(15, 23, 42)

        # Certification Metadata Strip
        meta_parts = [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
        if score is not None:
            meta_parts.append(f"JUDGEBERT Meaning Score: {float(score):.1f}/10")
        if tier:
            meta_parts.append(tier)
        if mode:
            meta_parts.append(mode)

        meta_p = doc.add_paragraph()
        meta_p.paragraph_format.space_after = Pt(12)
        meta_run = meta_p.add_run("  |  ".join(meta_parts))
        meta_run.font.size = Pt(9.5)
        meta_run.font.color.rgb = RGBColor(100, 116, 139)

        # Divider paragraph
        div_p = doc.add_paragraph("_" * 72)
        div_p.paragraph_format.space_after = Pt(12)
        for r in div_p.runs:
            r.font.size = Pt(8)
            r.font.color.rgb = RGBColor(203, 213, 225)

        # Document Paragraphs / Clauses
        paragraphs = [ln.strip() for ln in text.replace("\r\n", "\n").splitlines() if ln.strip()]
        for para_txt in paragraphs:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.25
            run = p.add_run(para_txt)
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(30, 41, 59)

        # Footer Stamp
        footer_sec = doc.sections[0].footer
        fp = footer_sec.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr = fp.add_run("Certified by JUDGEBERT Neuro-Symbolic Meaning Preservation Arbiter")
        fr.font.size = Pt(8.5)
        fr.font.color.rgb = RGBColor(148, 163, 184)

        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    @classmethod
    def export_pdf(
        cls,
        text: str,
        score: Optional[float] = None,
        tier: str = "",
        mode: str = "",
        title: str = "Verified Plain-Language Document",
    ) -> bytes:
        """
        Generates a clean vector PDF with full Unicode support (English, French,
        Hindi Devanagari, and currency symbols) using fpdf2.
        """
        from fpdf import FPDF

        if not text or not text.strip():
            raise ValueError("Cannot export an empty document.")

        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text + " " + title + " " + tier))
        nirmala_path = cls._WINDOWS_FONT_CANDIDATES["nirmala"]
        arial_path = cls._WINDOWS_FONT_CANDIDATES["arial"]
        arial_bold_path = cls._WINDOWS_FONT_CANDIDATES["arial_bold"]

        class _JudgeBertPDF(FPDF):
            def __init__(self, footer_font_family: str):
                super().__init__()
                self._footer_font_family = footer_font_family

            def footer(self):
                self.set_y(-14)
                self.set_font(self._footer_font_family, size=8)
                self.set_text_color(148, 163, 184)
                self.cell(
                    0,
                    8,
                    f"Page {self.page_no()}  |  Verified by JUDGEBERT Meaning Preservation Arbiter",
                    align="C",
                )

        # Choose fonts safely based on Unicode script coverage and system availability
        use_unicode_ttf = False
        body_family = "Helvetica"
        bold_family = "Helvetica"

        if has_devanagari and os.path.exists(nirmala_path):
            use_unicode_ttf = True
            body_family = "Nirmala"
            bold_family = "Arial" if os.path.exists(arial_bold_path) else "Nirmala"
        elif os.path.exists(arial_path):
            use_unicode_ttf = True
            body_family = "Arial"
            bold_family = "Arial"
        elif os.path.exists(nirmala_path):
            use_unicode_ttf = True
            body_family = "Nirmala"
            bold_family = "Nirmala"

        pdf = _JudgeBertPDF(footer_font_family=body_family)
        pdf.set_auto_page_break(auto=True, margin=18)

        if use_unicode_ttf:
            if body_family == "Nirmala" or os.path.exists(nirmala_path):
                if body_family == "Nirmala":
                    pdf.add_font("Nirmala", "", nirmala_path)
            if os.path.exists(arial_path) and (body_family == "Arial" or bold_family == "Arial"):
                pdf.add_font("Arial", "", arial_path)
                if os.path.exists(arial_bold_path):
                    pdf.add_font("Arial", "B", arial_bold_path)

        pdf.add_page()
        pdf.set_margins(18, 18, 18)
        pdf.set_y(18)

        # Sanitize text ONLY if running in a headless fallback without TrueType fonts
        def _safe_str(s: str) -> str:
            if use_unicode_ttf:
                return s
            return s.encode("latin-1", errors="replace").decode("latin-1")

        # Title Header
        if bold_family == "Arial" and os.path.exists(arial_bold_path):
            pdf.set_font("Arial", style="B", size=16)
        elif not use_unicode_ttf:
            pdf.set_font("Helvetica", style="B", size=16)
        else:
            pdf.set_font(body_family, size=16)

        pdf.set_text_color(15, 23, 42)
        pdf.multi_cell(w=pdf.epw, h=8, text=_safe_str(title))
        pdf.ln(2)

        # Metadata Certification Callout Box
        meta_parts = [f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"]
        if score is not None:
            meta_parts.append(f"Meaning Score: {float(score):.1f}/10")
        if tier:
            meta_parts.append(tier)
        if mode:
            meta_parts.append(mode)

        pdf.set_fill_color(240, 249, 255)
        pdf.set_draw_color(186, 230, 253)
        pdf.set_font(body_family, size=9)
        pdf.set_text_color(3, 105, 161)
        pdf.multi_cell(
            w=pdf.epw,
            h=6,
            text=_safe_str("  |  ".join(meta_parts)),
            border=1,
            fill=True,
        )
        pdf.ln(6)

        # Clean Body Paragraphs / Clauses
        pdf.set_font(body_family, size=11)
        pdf.set_text_color(30, 41, 59)
        paragraphs = [ln.strip() for ln in text.replace("\r\n", "\n").splitlines() if ln.strip()]
        for p_txt in paragraphs:
            pdf.multi_cell(w=pdf.epw, h=6.5, text=_safe_str(p_txt))
            pdf.ln(3)

        return bytes(pdf.output())
