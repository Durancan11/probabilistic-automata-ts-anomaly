import numpy as np
from scipy.stats import norm

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class ProbabilisticAutomata:
    def __init__(self, window_size, alphabet_size):
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.sax_bins = self._create_sax_bins()
        self.transitions = {}
        self.state_counts = {}
        self.vocabulary = set()

    def _create_sax_bins(self):
        return norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])

    def _value_to_sax(self, value):
        for i, b in enumerate(self.sax_bins):
            if value < b:
                return chr(97 + i)
        return chr(97 + len(self.sax_bins))

    def transform_to_sax(self, time_series):
        return [self._value_to_sax(x) for x in time_series]

    def extract_patterns(self, sax_sequence):
        patterns = []
        for i in range(len(sax_sequence) - self.window_size + 1):
            pattern = "".join(sax_sequence[i:i + self.window_size])
            patterns.append(pattern)
        return patterns

    def fit(self, patterns):
        for i in range(len(patterns) - 1):
            current_state = patterns[i]
            next_state = patterns[i+1]
            
            self.vocabulary.add(current_state)
            self.vocabulary.add(next_state)
            
            if current_state not in self.transitions:
                self.transitions[current_state] = {}
                self.state_counts[current_state] = 0
                
            if next_state not in self.transitions[current_state]:
                self.transitions[current_state][next_state] = 0
                
            self.transitions[current_state][next_state] += 1
            self.state_counts[current_state] += 1

    def get_transition_probability(self, current_state, next_state):
        if current_state not in self.transitions or next_state not in self.transitions[current_state]:
            return 0.0
        return self.transitions[current_state][next_state] / self.state_counts[current_state]

    def handle_unseen_pattern(self, unseen_pattern):
        best_match = None
        min_dist = float('inf')
        for known_pattern in self.vocabulary:
            dist = levenshtein_distance(unseen_pattern, known_pattern)
            if dist < min_dist:
                min_dist = dist
                best_match = known_pattern
        return best_match, min_dist

    def explain_decision(self, time_step, current_state, incoming_pattern, threshold=0.05):
        status = "seen"
        mapped_to = incoming_pattern
        
        if incoming_pattern not in self.vocabulary:
            status = "unseen"
            mapped_to, _ = self.handle_unseen_pattern(incoming_pattern)
            
        probability = self.get_transition_probability(current_state, mapped_to)
        
        decision = "anomaly" if probability < threshold else "normal"
        
        explanation = {
            "time_step": time_step,
            "state": current_state,
            "pattern": incoming_pattern,
            "status": status,
            "mapped_to": mapped_to if status == "unseen" else mapped_to,
            "probability": round(probability, 3),
            "decision": decision
        }
        return explanation