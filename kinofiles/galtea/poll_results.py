import os
import time
from galtea import Galtea
from dotenv import load_dotenv

load_dotenv()
GALTEA_API_KEY = os.getenv("GALTEA_API_KEY")
galtea = Galtea(api_key=GALTEA_API_KEY)

def check_results():
    products = galtea.products.list()
    product = products[0]
    versions = galtea.versions.list(product_id=product.id)
    version = versions[0]
    
    print(f"Checking evaluations for version {version.id}...")
    
    while True:
        evals = galtea.evaluations.list(version_id=version.id)
        
        pending_count = sum(1 for e in evals if e.status.name == 'PENDING')
        finished_count = len(evals) - pending_count
        
        print(f"Progress: {finished_count}/{len(evals)} scored by judges...")
        
        if pending_count == 0:
            print("\nAll evaluations finished! Here are the failures:")
            failures = 0
            for e in evals:
                # If score is 0, it means the agent failed the metric
                if e.score == 0:
                    failures += 1
                    print(f"\n❌ Failed Metric ID: {e.metric_id}")
                    print(f"Reason: {e.reason}")
            
            if failures == 0:
                print("\n✅ Perfect score! No failures found.")
            else:
                print(f"\nFound {failures} failures we need to fix!")
            break
            
        time.sleep(5)

if __name__ == "__main__":
    check_results()
