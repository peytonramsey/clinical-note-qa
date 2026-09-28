from src.retrieve import bm25_rank, gold_section, hybrid_rank, split_sections

NOTE = """Discharge Summary:

Patient Name: [REDACTED]
Age: 64

Hospital Course:
The patient was admitted with community-acquired pneumonia and treated with intravenous ceftriaxone.

Discharge Medications:
Amoxicillin 500 mg three times daily for five days.

Follow-up:
See the primary care physician in one week for a repeat chest X-ray."""


def test_split_on_blank_lines_keeps_headers_with_their_text():
    sections = split_sections(NOTE)
    assert len(sections) == 3
    assert sections[1].startswith("Discharge Medications:") and "Amoxicillin" in sections[1]


def test_short_chunks_are_merged_forward():
    # the title and the 3-line patient block are each under 8 words, so both merge into Hospital Course
    first = split_sections(NOTE)[0]
    assert first.startswith("Discharge Summary:\nPatient Name") and "ceftriaxone" in first


def test_note_without_blank_lines_falls_back_to_lines():
    note = ("Admission Diagnosis: pneumonia confirmed on chest radiograph at arrival\n"
            "Hospital Course: treated with intravenous antibiotics for three days")
    assert len(split_sections(note)) == 2


SECTIONS = ["Allergies: none known",
            "Hospital Course: admitted with pneumonia and treated with intravenous ceftriaxone",
            "Discharge Medications: amoxicillin 500 mg three times daily",
            "Follow-up: primary care physician in one week"]


def test_bm25_ranks_relevant_section_first():
    assert bm25_rank(SECTIONS, "What medications was the patient discharged on?")[0] == 2


def test_hybrid_ranks_relevant_section_first():
    assert hybrid_rank(SECTIONS, "What medications was the patient discharged on?")[0] == 2


def test_bm25_cannot_rank_two_sections():
    # Known rank_bm25 (BM25Okapi) edge case: with 2 documents a term found in 1 of them gets IDF 0,
    # so every score is 0. Harmless in practice: notes with <= TOP_K sections are sent whole.
    two = ["Allergies: none known", "Discharge Medications: amoxicillin 500 mg"]
    assert bm25_rank(two, "What medications was the patient discharged on?") == [0, 1]


def test_gold_section_picks_most_overlapping_section():
    sections = split_sections(NOTE)
    assert gold_section(sections, "Amoxicillin 500 mg three times daily") == 1
