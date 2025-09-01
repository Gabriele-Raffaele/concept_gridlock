#Base dictionary for single-part expansion
singlepart_expansion = {
    # Agents
    "Ped": "Pedestrian",
    "Car": "Car",
    "Cyc": "Cyclist",
    "Mobike": "Motorbike",
    "MedVeh": "Medium vehicle",
    "LarVeh": "Large vehicle",
    "Bus": "Bus",
    "EmVeh": "Emergency vehicle",
    "TL": "AV traffic light",
    "OthTL": "Other traffic light",

    # Actions
    "Red": "Red traffic light",
    "Amber": "Amber traffic light",
    "Green": "Green traffic light",
    "MovAway": "Move away",
    "MovTow": "Move towards",
    "Mov": "Move",
    "Brake": "Brake",
    "Stop": "Stop",
    "IncatLft": "Indicating left",
    "IncatRht": "Indicating right",
    "HazLit": "Hazards lights on",
    "TurLft": "Turn left",
    "TurRht": "Turn right",
    "Ovtak": "Overtake",
    "Wait2X": "Wait to cross",
    "XingFmLft": "Cross from left",
    "XingFmRht": "Cross from right",
    "Xing": "Crossing",
    "PushObj": "Push object",

    # Locations
    "VehLane": "AV lane",
    "OutgoLane": "Outgoing lane",
    "OutgoCycLane": "Outgoing cycle lane",
    "IncomLane": "Incoming lane",
    "IncomCycLane": "Incoming cycle lane",
    "Pav": "Pavement",
    "LftPav": "Left pavement",
    "RhtPav": "Right pavement",
    "Jun": "Junction",
    "xing": "Crossing location",
    "BusStop": "Bus stop",
    "parking": "Parking",
}

# Base dictionary for multi-part expansions
multipart_expansion = {
    # Agents
    "Ped": "pedestrian",
    "Car": "car",
    "Cyc": "cyclist",
    "Mobike": "motorbike",
    "MedVeh": "medium vehicle",
    "LarVeh": "large vehicle",
    "Bus": "bus",
    "EmVeh": "emergency vehicle",
    "TL": "AV traffic light",
    "OthTL": "other traffic light",

    # Actions
    "Red": "is red",
    "Amber": "is amber",
    "Green": "is green",
    "MovAway": "is moving away",
    "MovTow": "is moving towards",
    "Mov": "is moving",
    "Brake": "is braking",
    "Stop": "has stopped",
    "IncatLft": "is indicating left",
    "IncatRht": "is indicating right",
    "HazLit": "has hazard lights on",
    "TurLft": "is turning left",
    "TurRht": "is turning right",
    "Ovtak": "is overtaking",
    "Wait2X": "is waiting to cross",
    "XingFmLft": "is crossing from the left",
    "XingFmRht": "is crossing from the right",
    "Xing": "is crossing",
    "PushObj": "is pushing an object",

    # Locations
    "VehLane": "in the AV lane",
    "OutgoLane": "in the outgoing lane",
    "OutgoCycLane": "in the outgoing cycle lane",
    "IncomLane": "in the incoming lane",
    "IncomCycLane": "in the incoming cycle lane",
    "Pav": "on the pavement",
    "LftPav": "on the left pavement",
    "RhtPav": "on the right pavement",
    "Jun": "at the junction",
    "xing": "at the crossing",
    "BusStop": "at the bus stop",
    "parking": "in the parking area",
}

def choose_article(word: str) -> str:
    """
    Returns 'A' or 'An' depending on whether the word starts
    with a vowel sound (simple heuristic: a, e, i, o, u).
    """
    if not word:
        return "A"
    return "An" if word[0].lower() in "aeiou" else "A"


def expand_label(diminutive: str) -> str:
    """
    Expands diminutives into natural-language sentences.
    Handles single, duplex (agent+action), and triplet (agent+action+location).
    """
    parts = diminutive.split("-")

    if len(parts) == 1:  # Single element
        return singlepart_expansion.get(parts[0], parts[0])

    elif len(parts) == 2:  # Duplex 
        first_part, second_part = parts
        first_word = multipart_expansion.get(first_part, first_part)
        return f"{choose_article(first_word)} {first_word} that {multipart_expansion.get(second_part, second_part)}"

    elif len(parts) == 3:  # Triplet 
        first_part, second_part, third_part = parts
        first_word = multipart_expansion.get(first_part, first_part)
        return f"{choose_article(first_word)} {first_word} that {multipart_expansion.get(second_part, second_part)}, placed {multipart_expansion.get(third_part, third_part)}"

    else:
        # Fallback for unexpected format
        return " ".join(multipart_expansion.get(p, p) for p in parts)






############################## MAIN ##############################
# -----------------------
# Example usage
# -----------------------

#diminutives = [
#    "Ped",
#    "Car-Brake",
#    "Bus-MovTow-IncomLane",
#    "OthTL-Red",
#    "Ped-Wait2X-LftPav",
#    "Ped-XingFmRht-VehLane"
#]
#for d in diminutives:
#   print(f"{d} -> {expand_label(d)}")
