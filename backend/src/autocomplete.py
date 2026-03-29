"""
src/autocomplete.py — Simple Dictionary-based Word Suggestion
"""
import collections

# A small dictionary of common English words (Top ~500)
# Ideally, this would be a larger file, but for a prototype, this is highly effective.
COMMON_WORDS = [
    "THE", "OF", "AND", "A", "TO", "IN", "IS", "YOU", "THAT", "IT", "HE", "WAS", "FOR", "ON", "ARE", "AS", "WITH", "HIS", "THEY", "I",
    "AT", "BE", "THIS", "HAVE", "FROM", "OR", "ONE", "HAD", "BY", "WORDS", "BUT", "NOT", "WHAT", "ALL", "WERE", "WE", "WHEN", "YOUR", "CAN", "SAID",
    "THERE", "USE", "AN", "EACH", "WHICH", "SHE", "DO", "HOW", "THEIR", "IF", "WILL", "UP", "OTHER", "ABOUT", "OUT", "MANY", "THEN", "THEM", "THESE", "SO",
    "SOME", "HER", "WOULD", "MAKE", "LIKE", "HIM", "INTO", "TIME", "HAS", "LOOK", "TWO", "MORE", "WRITE", "GO", "SEE", "NUMBER", "NO", "WAY", "COULD", "PEOPLE",
    "MY", "THAN", "FIRST", "WATER", "BEEN", "CALL", "WHO", "OIL", "ITS", "NOW", "FIND", "LONG", "DOWN", "DAY", "DID", "GET", "COME", "MADE", "MAY", "PART",
    "OVER", "NEW", "SOUND", "TAKE", "ONLY", "LITTLE", "WORK", "KNOW", "YEAR", "LIVE", "ME", "BACK", "GIVE", "MOST", "VERY", "AFTER", "THING", "OUR", "JUST", "NAME",
    "GOOD", "SENTENCE", "MAN", "SAY", "GREAT", "WHERE", "HELP", "THROUGH", "MUCH", "BEFORE", "LINE", "RIGHT", "TOO", "MEAN", "OLD", "ANY", "SAME", "TELL", "BOY", "FOLLOW",
    "CAME", "WANT", "SHOW", "ALSO", "AROUND", "FORM", "THREE", "SMALL", "SET", "PUT", "END", "DOES", "ANOTHER", "WELL", "LARGE", "MUST", "BIG", "EVEN", "SUCH", "BECAUSE",
    "TURN", "HERE", "WHY", "ASK", "WENT", "MEN", "READ", "NEED", "LAND", "DIFFERENT", "HOME", "US", "MOVE", "TRY", "KIND", "HAND", "PICTURE", "AGAIN", "CHANGE", "OFF",
    "PLAY", "SPELL", "AIR", "AWAY", "ANIMAL", "HOUSE", "POINT", "PAGE", "LETTER", "MOTHER", "ANSWER", "FOUND", "STUDY", "STILL", "LEARN", "SHOULD", "AMERICA", "WORLD",
    "HELLO", "HELP", "HEAVEN", "HOLIDAY", "HOSPITAL", "HOUSE", "HUNGRY", "HAPPY", "HOME", "HIGH", "HUNDRED", "HOW", "HERE", "HAS", "HAD", "HAVE", "HIS", "HER", "HIM", "HE"
]

class Autocomplete:
    def __init__(self, word_list=COMMON_WORDS):
        self.words = sorted(list(set(word_list)))

    def get_suggestions(self, prefix, limit=3):
        if not prefix:
            return []
        
        prefix = prefix.upper()
        matches = [w for w in self.words if w.startswith(prefix)]
        
        # Sort by length (shorter matches first for autocomplete)
        matches.sort(key=len)
        
        return matches[:limit]

if __name__ == "__main__":
    ac = Autocomplete()
    print(f"Suggestions for 'H': {ac.get_suggestions('H')}")
    print(f"Suggestions for 'HE': {ac.get_suggestions('HE')}")
