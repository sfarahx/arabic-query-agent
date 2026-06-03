import json
from router import route
from router import classify_query

# ─────────────────────────────────────────────────────────────
# Benchmark query set
# Organised into:
#   - original (kept for regression)
#   - hard (new stress-test cases targeting real failure modes)
#
# Hard case design rationale per category:
#   SQL  : questions whose answer IS a person's name — classifier
#          previously confused these with HYBRID
#   RAG  : questions that mention a department name but ask about
#          policy only — no specific individual involved
#   HYBRID: reverse direction (policy → find who qualifies),
#           multi-step reasoning, indirect phrasing
# ─────────────────────────────────────────────────────────────

benchmark = [

    # ── ORIGINAL (regression) ────────────────────────────────

    # SQL
    ("كم عدد الموظفين في قسم الهندسة؟", "SQL"),
    ("ما هو راتب سارة الخالدي؟", "SQL"),
    ("كم عدد الموظفين الذين يتقاضون أكثر من 10000؟", "SQL"),
    ("ما هي الأقسام الموجودة في الشركة؟", "SQL"),
    ("من هو الموظف الأعلى راتباً؟", "SQL"),

    # RAG
    ("هل يمكن للموظفين العمل عن بُعد؟", "RAG"),
    ("كم يوماً يمكن لموظفي الهندسة العمل من المنزل؟", "RAG"),
    ("من يوافق على طلبات العمل عن بُعد؟", "RAG"),
    ("هل قسم المالية مؤهل للعمل عن بُعد؟", "RAG"),
    ("ما هي سياسة الحضور لقسم الموارد البشرية؟", "RAG"),

    # HYBRID
    ("كم عدد موظفي الهندسة وما هي سياسة عملهم عن بُعد؟", "HYBRID"),
    ("ما هو راتب موظفي المالية وهل يحق لهم العمل عن بُعد؟", "HYBRID"),
    ("كم عدد الموظفين في قسم الموارد البشرية وكم يوماً يجب أن يكونوا في المكتب؟", "HYBRID"),

    # ── HARD: SQL ─────────────────────────────────────────────
    # These return a person's name as the answer.
    # The classifier previously misrouted these to HYBRID.

    ("من يعمل في قسم الهندسة؟", "SQL"),                          # answer = list of names
    ("من هو الموظف الأقل راتباً؟", "SQL"),                        # ranking, answer = a name
    ("أعطني قائمة بجميع الموظفين ورواتبهم", "SQL"),               # pure data dump
    ("ما هو متوسط راتب موظفي الشركة؟", "SQL"),                    # aggregation, no policy involved
    ("هل راتب موظفي الهندسة أعلى من موظفي الموارد البشرية؟", "SQL"), # comparison between two groups, pure data

    # ── HARD: RAG ─────────────────────────────────────────────
    # These mention a department but ask about policy only.
    # No specific individual is referenced — answer comes from
    # the policy document alone, not the database.

    ("هل يحتاج الموظف إلى موافقة مسبقة للعمل عن بعد؟", "RAG"),   # pure policy rule
    ("هل هناك أقسام لا يُسمح لها بالعمل عن بعد إطلاقاً؟", "RAG"),  # policy comparison across departments
    ("هل يمكن لموظفي الموارد البشرية العمل من المنزل يوم الأحد؟", "RAG"),  # dept + day, no individual
    ("ما هي سياسة العمل عن بعد لجميع الأقسام؟", "RAG"),           # full policy overview

    # ── HARD: HYBRID ──────────────────────────────────────────
    # Mix of forward (person → policy) and reverse (policy → person).
    # Also tests indirect phrasing and multi-hop reasoning.

    # Forward: SQL finds person's dept, RAG applies the rule
    ("هل يسمح لأحمد المنصوري العمل عن بعد؟", "HYBRID"),           # Ahmed is HR
    ("هل يسمح لعمر فاروق العمل عن بعد؟", "HYBRID"),               # Omar is HR
    ("هل يمكن لسارة الخالدي العمل من المنزل ثلاثة أيام في الأسبوع؟", "HYBRID"),  # Sara is Engineering → yes, exactly 3

    # Reverse: RAG gives the eligibility rule, SQL finds who qualifies
    ("من هم الموظفون الذين يستطيعون العمل من المنزل؟", "HYBRID"),   # RAG: Engineering eligible → SQL: who is in Engineering
    ("كم عدد الموظفين المؤهلين للعمل عن بعد؟", "HYBRID"),          # RAG: rule → SQL: count matching employees

    # Multi-step: needs SQL for data AND policy for eligibility, combined answer
    ("من هو الموظف الأعلى راتباً في قسم الهندسة وهل يحق له العمل عن بعد؟", "HYBRID"),
]


def run_benchmark():
    results = []
    correct = 0
    original_count = 13  # number of original queries kept for regression tracking

    print("=" * 60)
    print("Running benchmark...")
    print("=" * 60)

    for i, (query, expected) in enumerate(benchmark):
        predicted = classify_query(query)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        tag = "[original]" if i < original_count else "[hard]"
        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "difficulty": "original" if i < original_count else "hard"
        })

        status = "✓" if is_correct else "✗"
        print(f"\n{status} {tag} {query}")
        print(f"   Expected: {expected} | Predicted: {predicted}")

    total = len(benchmark)
    original_results = [r for r in results if r["difficulty"] == "original"]
    hard_results     = [r for r in results if r["difficulty"] == "hard"]

    def accuracy(subset):
        if not subset:
            return 0
        return sum(1 for r in subset if r["correct"]) / len(subset) * 100

    def route_breakdown(subset, route_name):
        total_r  = sum(1 for r in subset if r["expected"] == route_name)
        correct_r = sum(1 for r in subset if r["expected"] == route_name and r["correct"])
        return f"{correct_r}/{total_r}"

    print("\n" + "=" * 60)
    print(f"OVERALL:  {correct}/{total} correct ({accuracy(results):.1f}%)")
    print(f"Original: {sum(r['correct'] for r in original_results)}/{len(original_results)} ({accuracy(original_results):.1f}%)")
    print(f"Hard:     {sum(r['correct'] for r in hard_results)}/{len(hard_results)} ({accuracy(hard_results):.1f}%)")
    print()
    print("Route breakdown (all queries):")
    for route_name in ["SQL", "RAG", "HYBRID"]:
        print(f"  {route_name}:    {route_breakdown(results, route_name)} correct")
    print()
    print("Route breakdown (hard only):")
    for route_name in ["SQL", "RAG", "HYBRID"]:
        print(f"  {route_name}:    {route_breakdown(hard_results, route_name)} correct")
    print("=" * 60)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "overall_accuracy": accuracy(results),
            "original_accuracy": accuracy(original_results),
            "hard_accuracy": accuracy(hard_results),
            "correct": correct,
            "total": total,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print("\nDetailed results saved to benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
    