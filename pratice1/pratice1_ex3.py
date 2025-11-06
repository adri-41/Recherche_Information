index = {
    "arabia": {"D4"},
    "casablanca": {"D1"},
    "citizen": {"D0"},
    "godfather": {"D2"},
    "gone": {"D3"},
    "graduate": {"D6"},
    "in": {"D9"},
    "kane": {"D0"},
    "lawrence": {"D4"},
    "list": {"D8"},
    "of": {"D4", "D5"},
    "on": {"D7"},
    "oz": {"D5"},
    "rain": {"D9"},
    "s": {"D8"},
    "schindler": {"D8"},
    "singin": {"D9"},
    "the": {"D2", "D3", "D5", "D6", "D7", "D9"},
    "waterfront": {"D7"},
    "wind": {"D3"},
    "with": {"D3"},
    "wizard": {"D5"}
}


# Fonctions pour les opérateurs booléens
def boolean_and(term1, term2):
    return index.get(term1, set()) & index.get(term2, set())


def boolean_or(term1, term2):
    return index.get(term1, set()) | index.get(term2, set())


def boolean_not(term1, term2):
    return index.get(term1, set()) - index.get(term2, set())


# Test
print("Requête : 'the AND wizard'")
print(boolean_and("the", "wizard"))  # D5

print("\nRequête : 'the OR godfather'")
print(boolean_or("the", "godfather"))  # D2, D3, D5, D6, D7, D9

print("\nRequête : 'the AND NOT wizard'")
print(boolean_not("the", "wizard"))  # D2, D3, D6, D7, D9
