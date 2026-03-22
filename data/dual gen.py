import pandas as pd
import numpy as np

def generate_internalized_dataset(rows_per_variant=100000):
    # Mapping: The first character of the sequence identifies the Cas9 Variant
    # SpCas9 = 'A', StCas9 = 'C', SaCas9 = 'G'
    variants = {
        "SpCas9": {"tag": "A", "g_len": 20},
        "StCas9": {"tag": "C", "g_len": 20},
        "SaCas9": {"tag": "G", "g_len": 21}
    }
    
    bases = ['A', 'C', 'G', 'T']
    on_target_data = []
    off_target_data = []

    for name, specs in variants.items():
        print(f"🔄 Generating data for {name}...")
        tag = specs['tag']
        g_len = specs['g_len']

        for _ in range(rows_per_variant):
            # 1. Create original gRNA
            grna = "".join(np.random.choice(bases, size=g_len))
            
            # --- ON-TARGET LOGIC ---
            # Sequence: [Tag] + [gRNA] + [Padding to 74]
            on_seq = (tag + grna).ljust(74, 'N')
            gc_content = (grna.count('G') + grna.count('C')) / g_len
            efficiency = max(0.01, min(0.99, gc_content + np.random.normal(0, 0.12)))
            on_target_data.append([name, on_seq, efficiency])

            # --- OFF-TARGET LOGIC ---
            # Create a 'Target' with random mismatches
            target_list = list(grna)
            mismatches = 0
            if np.random.rand() > 0.6: # 40% chance of mismatches
                mismatches = np.random.randint(1, 4)
                idx_to_change = np.random.choice(range(g_len), size=mismatches, replace=False)
                for i in idx_to_change:
                    target_list[i] = np.random.choice([b for b in bases if b != grna[i]])
            
            target_seq = "".join(target_list)
            # Sequence for Siamese: [Tag] + [gRNA] vs [Tag] + [Target]
            off_grna_tagged = (tag + grna).ljust(23, 'N')
            off_target_tagged = (tag + target_seq).ljust(23, 'N')
            
            # Risk Score Logic based on mismatch count/position
            risk_score = 1.0 - (mismatches * 0.2)
            risk_score = max(0.01, min(0.99, risk_score + np.random.normal(0, 0.05)))
            label = 1 if risk_score > 0.5 else 0
            
            off_target_data.append([name, off_grna_tagged, off_target_tagged, risk_score, label])

    # 2. Create DataFrames
    df_on = pd.DataFrame(on_target_data, columns=['Variant', 'Sequence_Input', 'Efficiency_Score'])
    df_off = pd.DataFrame(off_target_data, columns=['Variant', 'gRNA_Input', 'Target_Input', 'Risk_Score', 'Label'])

    return df_on, df_off

# Run the generator
df_on, df_off = generate_internalized_dataset(100000)

# 3. Save to separate files
df_on.to_csv("on_target_multi_variant.csv", index=False)
df_off.to_csv("off_target_multi_variant.csv", index=False)

print("\n✅ Generation Complete!")
print(f"📁 Created: 'on_target_multi_variant.csv' ({len(df_on)} rows)")
print(f"📁 Created: 'off_target_multi_variant.csv' ({len(df_off)} rows)")