import os
from reporting.report import generate_report
from paths import RESULTS_DIR

def main():
    results_base_dir = RESULTS_DIR
    
    if not os.path.exists(results_base_dir):
        print(f"Results directory '{results_base_dir}' not found.")
        return

    datasets = os.listdir(results_base_dir)
    
    for dataset in datasets:
        results_dir = os.path.join(results_base_dir, dataset)
        
        if os.path.isdir(results_dir):
            results_csv_path = os.path.join(results_dir, "classification_results.csv")
            
            if os.path.exists(results_csv_path):
                print(f"Generating report for {dataset}...")
                try:
                    generate_report(results_csv_path, results_dir, readable_name=dataset)
                    print(f"Successfully generated report for {dataset}.\n")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error generating report for {dataset}: {e}\n")
            else:
                print(f"No classification_results.csv found for {dataset}, skipping.\n")

if __name__ == '__main__':
    main()
