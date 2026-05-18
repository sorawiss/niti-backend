from app.rag.query import retrieve_laws


test_queries = [
    "นายจ้างต้องจ่ายค่าจ้างเมื่อไหร่",
    "ถ้าลูกจ้างทำงานล่วงเวลาต้องได้รับค่าล่วงเวลาหรือไม่",
    "นายจ้างให้ลูกจ้างทำงานวันหยุดได้หรือไม่",
    "ถ้าเลิกจ้างต้องจ่ายค่าชดเชยไหม",
    "สัญญาระบุว่าไม่มีค่าล่วงเวลาถูกกฎหมายหรือไม่",
]


for query in test_queries:
    print("=" * 80)
    print("Query:", query)

    results = retrieve_laws(query=query, top_k=3)

    for result in results:
        print("Score:", result["score"])
        print("Section:", result["metadata"].get("section"))
        print("Topic:", result["metadata"].get("topic"))
        print("Text:", result["text"][:300])
        print("-" * 50)