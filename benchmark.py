import json
from router import route
from router import classify_query

# Labeled queries: (query, expected_route)
benchmark = [
    # SQL queries
    ("كم عدد الموظفين في قسم الهندسة؟", "SQL"),
    ("ما هو راتب سارة الخالدي؟", "SQL"),
    ("كم عدد الموظفين الذين يتقاضون أكثر من 10000؟", "SQL"),
    ("ما هي الأقسام الموجودة في الشركة؟", "SQL"),
    ("من هو الموظف الأعلى راتباً؟", "SQL"),

    # RAG queries
    ("هل يمكن للموظفين العمل عن بُعد؟", "RAG"),
    ("كم يوماً يمكن لموظفي الهندسة العمل من المنزل؟", "RAG"),
    ("من يوافق على طلبات العمل عن بُعد؟", "RAG"),
    ("هل قسم المالية مؤهل للعمل عن بُعد؟", "RAG"),
    ("ما هي سياسة الحضور لقسم الموارد البشرية؟", "RAG"),

    # HYBRID queries
    ("كم عدد موظفي الهندسة وما هي سياسة عملهم عن بُعد؟", "HYBRID"),
    ("ما هو راتب موظفي المالية وهل يحق لهم العمل عن بُعد؟", "HYBRID"),
    ("كم عدد الموظفين في قسم الموارد البشرية وكم يوماً يجب أن يكونوا في المكتب؟", "HYBRID"),
]

def run_benchmark():
    results = []
    correct = 0

    print("=" * 60)
    print("Running benchmark...")
    print("=" * 60)

    for query, expected in benchmark:
        predicted = classify_query(query)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct
        })

        status = "✓" if is_correct else "✗"
        print(f"\n{status} Query: {query}")
        print(f"  Expected: {expected} | Predicted: {predicted}")

    total = len(benchmark)
    accuracy = (correct / total) * 100

    print("\n" + "=" * 60)
    print(f"Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print(f"  SQL:    {sum(1 for r in results if r['expected'] == 'SQL' and r['correct'])}/{sum(1 for r in results if r['expected'] == 'SQL')} correct")
    print(f"  RAG:    {sum(1 for r in results if r['expected'] == 'RAG' and r['correct'])}/{sum(1 for r in results if r['expected'] == 'RAG')} correct")
    print(f"  HYBRID: {sum(1 for r in results if r['expected'] == 'HYBRID' and r['correct'])}/{sum(1 for r in results if r['expected'] == 'HYBRID')} correct")
    print("=" * 60)

    # Save results to file
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, ensure_ascii=False, indent=2)

    print("\nDetailed results saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
    