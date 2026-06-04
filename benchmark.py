import json
from router import classify_query

# ─────────────────────────────────────────────────────────────
# Benchmark — Arabic Query Router
#
# Groups:
#   basic_sql    — clear data questions, unambiguous
#   basic_rag    — clear policy questions, unambiguous
#   basic_hybrid — clear person + policy questions
#   advanced_sql — looks like HYBRID but is pure data
#   advanced_rag — mentions departments/people but is pure policy
#   advanced_hybrid — indirect phrasing, multi-entity, implicit references
# ─────────────────────────────────────────────────────────────

benchmark = [

    # ── BASIC SQL ─────────────────────────────────────────────
    # Straightforward data questions with one clear signal.

    ("كم عدد الموظفين في قسم الهندسة؟",                    "SQL"),  # count
    ("ما هو راتب سارة الخالدي؟",                           "SQL"),  # single record lookup
    ("كم عدد الموظفين الذين يتقاضون أكثر من 10000؟",       "SQL"),  # filtered count
    ("ما هي الأقسام الموجودة في الشركة؟",                   "SQL"),  # list of values
    ("أعطني قائمة بجميع الموظفين ورواتبهم",                 "SQL"),  # full data dump
    ("ما هو متوسط راتب موظفي الشركة؟",                     "SQL"),  # aggregation

    # ── BASIC RAG ─────────────────────────────────────────────
    # Straightforward policy questions with one clear signal.

    ("هل يمكن للموظفين العمل عن بُعد؟",                    "RAG"),  # general policy
    ("من يوافق على طلبات العمل عن بُعد؟",                  "RAG"),  # approval process
    ("ما هي سياسة العمل عن بعد لجميع الأقسام؟",            "RAG"),  # full policy overview
    ("هل يحتاج الموظف إلى موافقة مسبقة للعمل عن بعد؟",    "RAG"),  # policy rule
    ("ما هي أيام عمل الشركة الرسمية؟",                     "RAG"),  # work week policy

    # ── BASIC HYBRID ──────────────────────────────────────────
    # Clear person + policy — needs both sources, no ambiguity.

    ("هل يسمح لخالد الراشد العمل عن بعد؟",                 "HYBRID"),  # Khalid → Finance → not eligible
    ("هل يسمح لأحمد المنصوري العمل عن بعد؟",               "HYBRID"),  # Ahmed → HR → 1 day
    ("هل يسمح لسارة الخالدي العمل عن بعد؟",                "HYBRID"),  # Sara → Engineering → 3 days
    ("هل يسمح لعمر فاروق العمل عن بعد؟",                   "HYBRID"),  # Omar → HR → 1 day

    # ── ADVANCED SQL ──────────────────────────────────────────
    # These look like they might need policy but are pure data.
    # Main risk: classifier picks HYBRID because the answer is a name
    # or the question involves a comparison between departments.

    ("من هو الموظف الأعلى راتباً؟",                        "SQL"),  # answer is a name — HYBRID risk
    ("من هو الموظف الأقل راتباً؟",                         "SQL"),  # answer is a name — HYBRID risk
    ("من يعمل في قسم الهندسة؟",                            "SQL"),  # answer is a list of names
    ("من يعمل في قسم الموارد البشرية؟",                    "SQL"),  # answer is a list of names
    ("هل راتب موظفي الهندسة أعلى من موظفي الموارد البشرية؟","SQL"),  # dept comparison, pure data
    ("ما هو أعلى راتب في قسم المالية؟",                    "SQL"),  # dept-specific ranking
    ("كم عدد الموظفين في كل قسم؟",                         "SQL"),  # grouped count, no policy
    ("من هو الموظف الأعلى راتباً في قسم الموارد البشرية؟", "SQL"),  # ranking within dept — HYBRID risk

    # ── ADVANCED RAG ──────────────────────────────────────────
    # These mention departments or scenarios that sound data-related
    # but the answer comes entirely from the policy document.
    # Main risk: classifier picks HYBRID because a department is named.

    ("هل قسم المالية مؤهل للعمل عن بُعد؟",                 "RAG"),  # dept named, but pure policy
    ("كم يوماً يمكن لموظفي الهندسة العمل من المنزل؟",      "RAG"),  # dept named, answer is in policy
    ("ما هي سياسة الحضور لقسم الموارد البشرية؟",           "RAG"),  # dept named, pure policy
    ("هل هناك أقسام لا يُسمح لها بالعمل عن بعد إطلاقاً؟", "RAG"),  # policy comparison, no individual
    ("هل يمكن لموظفي الموارد البشرية العمل من المنزل يوم الأحد؟", "RAG"),  # dept + day, no individual
    ("هل يمكن لموظفي الهندسة العمل من المنزل طوال أيام الأسبوع؟", "RAG"),  # policy limit check, no individual
    ("ما الفرق بين سياسة العمل عن بعد لقسم الهندسة وقسم الموارد البشرية؟", "RAG"),  # policy comparison between depts

    # ── ADVANCED HYBRID ───────────────────────────────────────
    # Indirect phrasing, implicit references, multi-entity,
    # and reverse direction (policy → find who qualifies).
    # These are the hardest cases for both the classifier and blend step.

    # Reverse direction: policy defines the rule, SQL finds who matches
    ("من هم الموظفون الذين يستطيعون العمل من المنزل؟",      "HYBRID"),  # Engineering eligible → find Engineering employees
    ("كم عدد الموظفين المؤهلين للعمل عن بعد؟",             "HYBRID"),  # eligibility rule → count matching employees
    ("من هم الموظفون غير المؤهلين للعمل عن بعد؟",          "HYBRID"),  # Finance not eligible → find Finance employees

    # Indirect reference: person implied, not explicitly named
    ("هل يمكن لموظفة ليلى حسن العمل من المنزل ثلاثة أيام في الأسبوع؟", "HYBRID"),  # Layla → Engineering → exactly 3, true
    ("هل يمكن لسارة الخالدي العمل من المنزل ثلاثة أيام في الأسبوع؟",   "HYBRID"),  # Sara → Engineering → yes, exactly 3

    # Multi-entity: involves two people
    ("هل يملك أحمد المنصوري وعمر فاروق نفس سياسة العمل عن بعد؟", "HYBRID"),  # both HR → same policy

    # Comparative across people
    ("من لديه المزيد من أيام العمل عن بعد، سارة الخالدي أم أحمد المنصوري؟", "HYBRID"),  # Engineering vs HR policy

    # Multi-step: ranking + policy
    ("من هو الموظف الأعلى راتباً في قسم الهندسة وهل يحق له العمل عن بعد؟", "HYBRID"),  # SQL for ranking, RAG for eligibility
]


def run_benchmark():
    results = []
    correct = 0

    groups = [
        "basic_sql", "basic_rag", "basic_hybrid",
        "advanced_sql", "advanced_rag", "advanced_hybrid"
    ]

    group_sizes = [6, 5, 4, 8, 7, 8]
    group_labels = []
    for group, size in zip(groups, group_sizes):
        group_labels.extend([group] * size)

    print("=" * 65)
    print("Running benchmark...")
    print("=" * 65)

    current_group = None
    for i, (query, expected) in enumerate(benchmark):
        group = group_labels[i]

        if group != current_group:
            current_group = group
            print(f"\n── {group.upper()} {'─' * (50 - len(group))}")

        predicted = classify_query(query)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "group": group
        })

        status = "✓" if is_correct else "✗"
        print(f"\n  {status} {query}")
        print(f"     Expected: {expected} | Got: {predicted}")

    total = len(benchmark)

    print("\n" + "=" * 65)
    print(f"OVERALL: {correct}/{total} ({correct/total*100:.1f}%)\n")

    for group in groups:
        subset = [r for r in results if r["group"] == group]
        g_correct = sum(1 for r in subset if r["correct"])
        print(f"  {group:<20} {g_correct}/{len(subset)} ({g_correct/len(subset)*100:.1f}%)")

    print("\nRoute breakdown:")
    for route_name in ["SQL", "RAG", "HYBRID"]:
        subset = [r for r in results if r["expected"] == route_name]
        r_correct = sum(1 for r in subset if r["correct"])
        print(f"  {route_name:<10} {r_correct}/{len(subset)} correct")

    print("=" * 65)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": correct / total * 100,
            "correct": correct,
            "total": total,
            "by_group": {
                group: {
                    "correct": sum(1 for r in results if r["group"] == group and r["correct"]),
                    "total": sum(1 for r in results if r["group"] == group),
                }
                for group in groups
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()