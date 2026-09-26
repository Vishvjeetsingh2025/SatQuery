"""Generates a one-click downloadable PDF report for any saved analysis."""
import io, base64, time
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

NAVY = colors.HexColor("#1F3864")
LIGHTBG = colors.HexColor("#F2F5FA")


def build_report(record: dict) -> bytes:
    """record: a dict from db.get_history() (prompt, answer, confidence, intent, thumb_b64, observations, ts)."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm, title="SatQuery AI Report")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], textColor=NAVY, fontSize=18)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=NAVY, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("B", parent=styles["Normal"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY)

    story = [Paragraph("SatQuery AI — Analysis Report", title_style), Spacer(1, 4)]
    story.append(Paragraph(time.strftime("Generated on %d %b %Y, %H:%M", time.localtime(record.get("ts", time.time()))),
                            styles["Normal"]))
    story.append(Spacer(1, 12))

    meta = Table([
        ["Query", record.get("prompt", "")],
        ["Detected Intent", record.get("intent", "-")],
        ["Confidence", record.get("confidence", "-")],
    ], colWidths=[35 * mm, 130 * mm])
    meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHTBG),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta)
    story.append(Spacer(1, 10))

    if record.get("thumb_b64"):
        try:
            img_bytes = base64.b64decode(record["thumb_b64"])
            story.append(RLImage(io.BytesIO(img_bytes), width=90 * mm, height=90 * mm, kind="proportional"))
            story.append(Spacer(1, 10))
        except Exception:
            pass

    story.append(Paragraph("Answer", h2))
    answer_html = (record.get("answer") or "").replace("**", "<b>").replace("<b>", "<b>", 1)
    # simple **bold** -> <b> pairing fallback (best-effort, avoids crashing on odd markdown)
    story.append(Paragraph(_safe(record.get("answer", "")), body))

    obs = record.get("observations") or []
    if obs:
        story.append(Paragraph("Evidence / Observations", h2))
        for o in obs:
            story.append(Paragraph("• " + _safe(o), body))

    story.append(Spacer(1, 16))
    story.append(Paragraph("Generated automatically by SatQuery AI (SIH26167) — Team Neural Minds.",
                            ParagraphStyle("F", parent=styles["Normal"], fontSize=8, textColor=colors.grey)))

    doc.build(story)
    return buf.getvalue()


def _safe(t: str) -> str:
    t = (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # restore simple **bold** markdown as <b>
    parts = t.split("**")
    out, bold = "", False
    for i, p in enumerate(parts):
        out += f"<b>{p}</b>" if i % 2 == 1 else p
    return out.replace("\n", "<br/>")
