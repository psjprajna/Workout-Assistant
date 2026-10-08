"""Read the logged-sets text ("80x8,80x7") and convert weights to kg.

Jefit stores a whole workout's sets in one string. This module turns that string into
(weight, reps) pairs, drops empty "0x0" placeholders, and converts old lbs weights to kg.
Logs before 2026-01-01 were entered in lbs; later ones are already kg.
"""

from datetime import date

# Logs before this date were typed in lbs; this date and later are already kg.
# Cutoff date for weight conversion rules
KG_CONVERSION_CUTOFF_DATE = date(2026, 1, 1)

def parse_logged_sets(text: str) -> list[tuple[float, int]]:
    """
    Parses a string of logged sets split by commas.
    Each set is expected to be in the format 'weightxreps'.
    Removes empty pieces and drops any (0.0, 0) pairs.
    """
    result = []
    
    # Split on commas and strip whitespace
    pieces = text.split(',')
    
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
            
        # A piece must look like "weightxreps"
        try:
            weight_str, reps_str = piece.split("x")
        except ValueError:
            raise ValueError(f"bad set token: {piece!r}") from None
        
        # Convert values (will automatically raise ValueError if invalid)
        weight = float(weight_str)
        reps = int(reps_str)
        
        # Round the weight to 2 decimals
        weight = round(weight, 2)
        
        # Drop the pair if it is exactly (0.0, 0)
        if weight == 0.0 and reps == 0:
            continue
            
        result.append((weight, reps))
        
    return result

def lbs_to_kg(lbs: float) -> float:
    """
    Converts pounds (lbs) to kilograms (kg) and rounds to 2 decimal places.
    Jefit logs before this date are in lbs.
    """
    return round(lbs * 0.45359237, 2)

def weight_in_kg(weight: float, on_date: date) -> float:
    """
    Converts the weight to kilograms only if the given date is before January 1, 2026.
    Otherwise, returns the weight as-is.
    """
    if on_date < KG_CONVERSION_CUTOFF_DATE:
        return lbs_to_kg(weight)
    return weight
