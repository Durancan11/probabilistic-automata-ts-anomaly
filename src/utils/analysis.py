import pandas as pd
from src.models.automata import ProbabilisticAutomata

def run_parameter_analysis(time_series, window_variations, alphabet_variations):
    results = []
    
    for w in window_variations:
        for a in alphabet_variations:
            automata = ProbabilisticAutomata(window_size=w, alphabet_size=a)
            
            sax_dizisi = automata.transform_to_sax(time_series)
            oruntuler = automata.extract_patterns(sax_dizisi)
            
            automata.fit(oruntuler)
            
            state_sayisi = len(automata.vocabulary)
            
            gecis_sayisi = sum(len(hedefler) for hedefler in automata.transitions.values())
            
            results.append({
                "Window Size": w,
                "Alphabet Size": a,
                "State Sayısı": state_sayisi,
                "Geçiş Yoğunluğu (Edge Count)": gecis_sayisi
            })
            
    df_results = pd.DataFrame(results)
    return df_results