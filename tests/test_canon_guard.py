import unittest
from pathlib import Path

import main
from canon_guard import (
    MISSING_INFORMATION,
    NOT_FOUND_MESSAGE,
    answer_from_sources,
    assess_finding,
    format_finding,
    load_fragments,
    parse_canon_text,
    relevant_canon,
    review_fragments,
)


ROOT = Path(__file__).resolve().parents[1]
CANON_DIR = ROOT / "data" / "knowledge" / "canon"


class StatusParsingTests(unittest.TestCase):
    def test_only_explicit_canon_is_eligible(self):
        fragments = parse_canon_text(
            "Unmarked fact.\n\n[КАНОН]\nApproved fact.\n\n"
            "[ПРЕДЛОЖЕНИЕ]\nProposed fact.\n\n[НЕ РЕШЕНО]\nUnknown fact.\n\n"
            "[СВЕРИТЬ]\nCheck this fact."
        )
        self.assertEqual([f.status for f in fragments], [
            "БЕЗ СТАТУСА", "КАНОН", "ПРЕДЛОЖЕНИЕ", "НЕ РЕШЕНО", "СВЕРИТЬ"
        ])
        self.assertEqual([f.text for f in fragments if f.status == "КАНОН"], ["Approved fact."])

    def test_conflicting_status_tags_are_quarantined(self):
        fragments = parse_canon_text("[КАНОН] [СВЕРИТЬ]\nDo not trust this.")
        self.assertEqual(fragments[0].status, "НЕОПРЕДЕЛЁН")
        self.assertEqual(relevant_canon("trust", fragments), [])

    def test_canon_marker_in_long_heading_is_explicit_canon(self):
        fragment = parse_canon_text("[КАНОН ИЗ РАНЕЕ УТВЕРЖДЁННОГО ЛОРА]\nФакт.")[0]
        self.assertEqual(fragment.status, "КАНОН")

    def test_combined_status_label_is_ambiguous(self):
        fragment = parse_canon_text("[НЕ РЕШЕНО / СВЕРИТЬ]\nНе подтверждать.")[0]
        self.assertEqual(fragment.status, "НЕОПРЕДЕЛЁН")


class EvidenceAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fragments = load_fragments(CANON_DIR)

    def test_answers_are_verbatim_source_fragments(self):
        answer, used = answer_from_sources("Какова длина Ханзо?", self.fragments)
        self.assertTrue(used)
        self.assertTrue(any("3000 метров" in f.text for f in used))
        for fragment in used:
            self.assertIn(fragment.text, answer)
            self.assertEqual(fragment.status, "КАНОН")

    def test_proposals_and_uncertain_fragments_never_enter_answer(self):
        answer, used = answer_from_sources("Что такое Нифельхеймброд?", self.fragments)
        self.assertTrue(all(f.status == "КАНОН" for f in used))
        self.assertNotIn("[ПРЕДЛОЖЕНИЕ", answer)
        self.assertNotIn("[СВЕРИТЬ", answer)
        self.assertNotIn("Нифельхеймброд", "\n".join(f.text for f in used))

    def test_missing_information_uses_required_exact_phrase(self):
        answer, used = answer_from_sources("Как зовут прорицателя?", self.fragments)
        self.assertEqual(answer, NOT_FOUND_MESSAGE)
        self.assertEqual(used, [])
        self.assertEqual(MISSING_INFORMATION, "В утверждённом каноне DENGO эта информация не указана")
        self.assertIn(MISSING_INFORMATION, format_finding("имя", assess_finding([], exhaustive_review=True)))

    def test_sources_have_statuses_and_never_promote_unknown_material(self):
        self.assertTrue(self.fragments)
        self.assertTrue(any(f.status == "КАНОН" for f in self.fragments))
        self.assertTrue(any(f.status == "ПРЕДЛОЖЕНИЕ" for f in self.fragments))
        self.assertTrue(any(f.status == "НЕ РЕШЕНО" for f in self.fragments))
        self.assertTrue(any(f.status == "СВЕРИТЬ" for f in self.fragments))

    def test_chat_route_uses_extractive_answer_without_model_call(self):
        response = main.chat(main.ChatRequest(message="Какова длина Ханзо?"))
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["model"], "extractive-local-guard")
        self.assertIn("3000 метров", response["answer"])

    def test_chat_route_returns_required_unknown_phrase(self):
        response = main.chat(main.ChatRequest(message="Как зовут прорицателя?"))
        self.assertEqual(response["status"], "not_found_by_search")
        self.assertEqual(response["answer"], NOT_FOUND_MESSAGE)


class StageTwoSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fragments = load_fragments(CANON_DIR)

    def test_country_list_is_found_in_country_category(self):
        answer, used = answer_from_sources("Какие государства существуют в мире DENGO?", self.fragments)
        self.assertEqual({f.category for f in used}, {"государства"})
        combined = "\n".join(f.text for f in used)
        for name in ("Золами", "Бурлин", "Балбул", "Реян", "Зовлил"):
            self.assertIn(name, combined)

    def test_ren_kurozaki_profile_is_found_without_other_characters(self):
        answer, used = answer_from_sources("Что известно о Рене Куросаки?", self.fragments)
        self.assertEqual({f.entity for f in used}, {"Рен Куросаки"})
        self.assertTrue(any("Казэкири" in f.text for f in used))
        self.assertNotIn("Брок", answer)
        self.assertNotIn("Арто — ребёнок", answer)

    def test_arto_profile_does_not_include_ren_or_brock_profile_facts(self):
        answer, used = answer_from_sources("Кто такой Арто?", self.fragments)
        self.assertEqual({f.entity for f in used}, {"Арто"})
        self.assertNotIn("Друг Арто, ровесник", answer)
        self.assertNotIn("Брок сможет", answer)

    def test_bare_ren_requests_clarification(self):
        answer, used = answer_from_sources("Что известно о Рене?", self.fragments)
        self.assertIn("Уточните", answer)
        self.assertIn("Рен Куросаки", answer)
        self.assertEqual(used, [])

    def test_compound_dimensions_are_reported_separately(self):
        answer, used = answer_from_sources(
            "Какова длина, ширина, высота и уровень Ханзо?", self.fragments
        )
        for field in ("длина", "ширина", "высота", "уровень"):
            self.assertIn(f"Параметр «{field}»:", answer)
        self.assertIn("около 3000 метров длиной", answer)
        self.assertIn("10-й уровень", answer)
        self.assertIn("Параметр «ширина»: Не найдено поиском", answer)
        self.assertIn("Параметр «высота»: Не найдено поиском", answer)
        self.assertTrue(all(f.entity == "Ханзо" for f in used))

    def test_cause_question_marks_reason_as_not_found_instead_of_claiming_it(self):
        answer, used = answer_from_sources("Что известно о причине Великой войны?", self.fragments)
        self.assertIn("Параметр «причина»: Не найдено поиском", answer)
        self.assertIn("не подтверждает отсутствие сведений во всех материалах", answer)
        self.assertEqual(used, [])

    def test_arto_seals_query_uses_only_arto_section(self):
        answer, used = answer_from_sources("Какие две печати есть у Арто?", self.fragments)
        self.assertEqual({f.entity for f in used}, {"Арто"})
        self.assertTrue(any("Печать №5" in f.text and "Печать №2" in f.text for f in used))
        self.assertNotIn("Казэкири", answer)
        self.assertNotIn("Джи — природная энергия", answer)

    def test_search_failure_does_not_claim_global_absence(self):
        answer, _ = answer_from_sources("Какова ширина Ханзо?", self.fragments)
        self.assertIn("Не найдено поиском", answer)
        self.assertIn("не подтверждает отсутствие сведений во всех материалах", answer)
        self.assertNotIn(MISSING_INFORMATION, answer)

    def test_finding_states_distinguish_search_from_exhaustive_review(self):
        self.assertEqual(assess_finding([]).state, "NOT_FOUND_BY_SEARCH")
        self.assertEqual(assess_finding([], exhaustive_review=True).state, "ABSENT_AFTER_REVIEW")

    def test_local_verify_clause_is_separated_from_canon_clause(self):
        clauses = [f for f in self.fragments if "соответствие каждого имени" in f.text]
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0].status, "СВЕРИТЬ")
        self.assertEqual(clauses[0].source, "seals\\01_seals_and_jhi.txt")
        self.assertEqual((clauses[0].start_line, clauses[0].end_line), (7, 7))
        names = [f for f in self.fragments if "Известные главы" in f.text]
        self.assertEqual(len(names), 1)
        self.assertEqual(names[0].status, "КАНОН")

    def test_review_mode_returns_only_requested_status_with_provenance(self):
        proposals = review_fragments("ПРЕДЛОЖЕНИЕ", self.fragments)
        self.assertTrue(proposals)
        self.assertTrue(all(f.status == "ПРЕДЛОЖЕНИЕ" and f.source and f.start_line for f in proposals))
        self.assertFalse(any(f.status == "КАНОН" for f in proposals))
        self.assertTrue(review_fragments("НЕОПРЕДЕЛЁН", self.fragments))

    def test_categories_cover_all_requested_topics(self):
        self.assertEqual(
            {f.category for f in self.fragments if f.status == "КАНОН"},
            {"персонажи", "государства", "чудовища", "печати", "история", "устройство мира"},
        )

    def test_chat_exposes_review_route_instead_of_mixing_statuses(self):
        response = main.chat(main.ChatRequest(message="Какие сведения ещё не утверждены?"))
        self.assertIn("/review?status=НЕ%20РЕШЕНО", response["answer"])
        self.assertEqual(response["canon_used"], "")
        reviewed = main.review(status="СВЕРИТЬ")
        self.assertTrue(reviewed["count"])
        self.assertTrue(all(item["status"] == "СВЕРИТЬ" for item in reviewed["items"]))


if __name__ == "__main__":
    unittest.main()
