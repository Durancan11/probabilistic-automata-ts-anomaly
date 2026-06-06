import unittest
from src.models.automata import ProbabilisticAutomata

class TestUnseenPattern(unittest.TestCase):
    def setUp(self):
        self.automata = ProbabilisticAutomata(window_size=3, alphabet_size=3)
        dummy_patterns = ["abc", "bca", "cab"]
        self.automata.fit(dummy_patterns)

    def test_unseen_mapping(self):
        mevcut_durum = "abc"
        gelen_oruntu = "abd" 
        
        aciklama = self.automata.explain_decision(
            time_step=1, 
            current_state=mevcut_durum, 
            incoming_pattern=gelen_oruntu, 
            threshold=0.1
        )
        
        self.assertEqual(aciklama["status"], "unseen")
        self.assertEqual(aciklama["mapped_to"], "abc")

if __name__ == '__main__':
    unittest.main()