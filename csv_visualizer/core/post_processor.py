def apply_post_operations(result: dict, operations: dict | None):
    
    if not operations:
        return result

    categories = result.get("categories")
    values = result.get("values")

    if not categories or not values:
        return result

    # -------- TOP K --------
    if "top_k" in operations:
        k = operations["top_k"]

        combined = list(zip(categories, values))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:k]

        result["categories"] = [c for c, _ in combined]
        result["values"] = [v for _, v in combined]

    # -------- BOTTOM K --------
    if "bottom_k" in operations:
        k = operations["bottom_k"]

        combined = list(zip(categories, values))
        combined.sort(key=lambda x: x[1])
        combined = combined[:k]

        result["categories"] = [c for c, _ in combined]
        result["values"] = [v for _, v in combined]

    return result
