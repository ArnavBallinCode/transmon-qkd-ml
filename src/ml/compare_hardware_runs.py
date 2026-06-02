import os
import pandas as pd

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    hw_dir = os.path.join(project_root, "results", "hardware")
    tables_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    records = []
    
    for n_val in [100, 500, 1000, 3000]:
        file_path = os.path.join(hw_dir, f"N_{n_val}", f"hardware_qber_summary_{n_val}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            records.append({
                "sample_size": n_val,
                "num_runs": len(df),
                "total_sifted_bits": df["sifted_length"].sum(),
                "total_errors": df["errors"].sum(),
                "mean_qber": df["qber"].mean(),
                "std_qber": df["qber"].std()
            })
            
    if records:
        df_comp = pd.DataFrame(records)
        out_path = os.path.join(tables_dir, "hardware_validation_comparison.csv")
        df_comp.to_csv(out_path, index=False)
        print(f"Generated comparison table at {out_path}")
        print(df_comp)
    else:
        print("No data found to compare.")

if __name__ == "__main__":
    main()
