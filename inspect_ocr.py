import pickle
import numpy as np
import os
from collections import Counter, defaultdict

# --- CONFIGURAÇÃO ---
FILE_PATH = "ocr_features_final_production.pkl"

# Mapeamento exato da função detect_scripts() do seu pipeline final:
# 0: Latin, 1: Cyrillic, 2: Greek, 3: Thai, 4: Japanese, 5: Indic (Hindi/Kannada)
SCRIPT_NAMES = ['Latin', 'Cyrillic', 'Greek', 'Thai', 'Japanese', 'Indic']

def get_city_from_path(path):
    """Extrai o nome da cidade do caminho do arquivo."""
    parts = path.replace('\\', '/').split('/')
    
    # Procura 'database' ou 'query' e pega a pasta anterior
    if 'database' in parts:
        idx = parts.index('database')
        return parts[idx-1]
    elif 'query' in parts:
        idx = parts.index('query')
        return parts[idx-1]
    
    # Fallback genérico
    return parts[-3] if len(parts) >= 3 else "unknown"

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Arquivo {FILE_PATH} não encontrado. Verifique se o nome está correto.")
        return

    print(f"📂 Carregando {FILE_PATH}...")
    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ Erro ao abrir o arquivo pickle: {e}")
        return

    print(f"✅ Dados carregados. Analisando {len(data)} entradas...\n")

    # Estruturas para estatísticas
    city_stats = defaultdict(lambda: {'total': 0, 'empty': 0, 'scripts': Counter()})
    global_script_counts = Counter()
    total_valid_text = 0

    # Loop principal
    for path, (script_vec, emb) in data.items():
        city = get_city_from_path(path).lower() # Normaliza para minúsculo
        stats = city_stats[city]
        
        stats['total'] += 1
        
        # Verifica se o embedding é zerado (indicador de "sem texto")
        if np.sum(emb) == 0:
            stats['empty'] += 1
        else:
            # Pega o script com maior probabilidade
            dom_idx = np.argmax(script_vec)
            confidence = script_vec[dom_idx]
            
            # Só conta se a confiança for > 0 (as vezes o vetor é [0,0,0,0,0,0] mas o embedding não é zero por ruído)
            if confidence > 0:
                script_name = SCRIPT_NAMES[dom_idx]
                stats['scripts'][script_name] += 1
                global_script_counts[script_name] += 1
                total_valid_text += 1
            else:
                # Caso raro onde tem embedding mas script_vec é tudo zero
                stats['empty'] += 1

    # --- RELATÓRIO GLOBAL ---
    print("="*80)
    print(f"🌍 ESTATÍSTICAS GLOBAIS DE OCR")
    print("="*80)
    print(f"Total de Imagens: {len(data)}")
    if len(data) > 0:
        print(f"Com Texto Útil:   {total_valid_text} ({total_valid_text/len(data)*100:.1f}%)")
        print(f"Sem Texto (Vazio): {len(data) - total_valid_text} ({(len(data) - total_valid_text)/len(data)*100:.1f}%)")
    print("-" * 40)
    print("Distribuição de Linguagens (Top 6):")
    for script, count in global_script_counts.most_common(6):
        pct = (count / total_valid_text * 100) if total_valid_text > 0 else 0
        print(f"  • {script:<12}: {count:<6} ({pct:.1f}%)")
    print("="*80)
    print("\n")

    # --- RELATÓRIO POR CIDADE ---
    # Cabeçalho
    print(f"{'CIDADE':<15} | {'TOTAL':<6} | {'VAZIO %':<8} | {'DOMINANTE (Script)'}")
    print("-" * 80)

    for city in sorted(city_stats.keys()):
        s = city_stats[city]
        total = s['total']
        if total == 0: continue

        empty_pct = (s['empty'] / total) * 100
        valid_count = total - s['empty']
        
        # Determina scripts dominantes
        if valid_count > 0 and s['scripts']:
            top_script, count = s['scripts'].most_common(1)[0]
            script_pct = (count / valid_count) * 100
            script_str = f"{top_script} ({script_pct:.0f}%)"
            
            # Mostra segundo script se for relevante (>10%)
            if len(s['scripts']) > 1:
                sec_script, sec_count = s['scripts'].most_common(2)[1]
                sec_pct = (sec_count / valid_count) * 100
                if sec_pct > 10:
                    script_str += f", {sec_script} ({sec_pct:.0f}%)"
        else:
            script_str = "---"

        print(f"{city.upper():<15} | {total:<6} | {empty_pct:<6.1f}%  | {script_str}")

    print("-" * 80)
    
    # --- DIAGNÓSTICO INTELIGENTE ---
    print("\n🕵️  DIAGNÓSTICO AUTOMÁTICO:")
    
    # 1. Moscou (Deve ser Cyrillic)
    moscow = city_stats.get('moscow') or city_stats.get('mo')
    if moscow:
        cyr = moscow['scripts']['Cyrillic']
        lat = moscow['scripts']['Latin']
        if cyr > lat:
            print("✅ MOSCOU: Detecção de Cirílico saudável.")
        else:
            print("⚠️ MOSCOU: Atenção! Mais Latin que Cirílico detectado. Verifique se o modelo 'cyrillic' rodou.")

    # 2. Tóquio (Deve ser Japanese)
    tokyo = city_stats.get('tokyo') or city_stats.get('tk')
    if tokyo:
        jap = tokyo['scripts']['Japanese']
        total_valid = tokyo['total'] - tokyo['empty']
        if total_valid > 0 and (jap / total_valid) > 0.2:
            print(f"✅ TÓQUIO: Japonês detectado em {jap/total_valid*100:.0f}% das placas.")
        else:
            print("⚠️ TÓQUIO: Pouco Japonês detectado. O modelo pode estar priorizando números/inglês.")

    # 3. Bengaluru (Deve ser Indic/Kannada)
    blr = city_stats.get('bengaluru') or city_stats.get('be')
    if blr:
        indic = blr['scripts']['Indic']
        if indic > 100:
            print("✅ BENGALURU: Script 'Indic' (Kannada) detectado consistentemente.")
        else:
            print("⚠️ BENGALURU: Baixa detecção de Indic. Verifique se o modelo 'ka' funcionou.")

    # 4. Bangkok (Esperado Latin devido ao fallback, mas ideal seria Thai)
    bk = city_stats.get('bangkok') or city_stats.get('bk')
    if bk:
        thai = bk['scripts']['Thai']
        lat = bk['scripts']['Latin']
        if thai == 0 and lat > 0:
            print("ℹ️  BANGKOK: Apenas Latin detectado (Esperado, pois usamos fallback 'latin' devido a erro no 'th').")
        elif thai > 0:
            print("✨ BANGKOK: Surpresa! Algum Thai foi detectado mesmo com modelo Latin.")

if __name__ == "__main__":
    main()