import os
import json
from datetime import datetime
import pandas as pd
from jobspy import scrape_jobs
from playwright.sync_api import sync_playwright

def run_job_search():
    # 1. Search parameters
    search_terms = [
        "Java Developer", 
        "DevOps Engineer", 
        "Cloud Engineer", 
        "Full Stack Developer Java"
    ]
    
    location = "Karlsruhe, Germany"
    distance_miles = 35  # ~50 km radius around Karlsruhe
    results_per_term = 15
    
    # Supported sites in JobSpy: indeed, linkedin, glassdoor, zip_recruiter, etc.
    site_names = ["indeed", "linkedin", "glassdoor", "google"]
    
    all_jobs_list = []

    print(f"Starting job scrape for location: {location}...\n")

    for term in search_terms:
        print(f"Scraping roles for query: '{term}'...")
        try:
            # Execute JobSpy scraper
            jobs_df = scrape_jobs(
                site_name=site_names,
                search_term=term,
                location=location,
                distance=distance_miles,
                results_wanted=results_per_term,
                country_indeed="Germany",  # Targets indeed.de and glassdoor.de
                hours_old=72,              # Jobs posted in the last 3 days
                linkedin_fetch_description=True, # Fetch full description text
                description_format="markdown"
            )

            if not jobs_df.empty:
                print(f" Found {len(jobs_df)} jobs for '{term}'.")
                all_jobs_list.append(jobs_df)
            else:
                print(f" No jobs returned for '{term}'.")

        except Exception as e:
            print(f" Error scraping '{term}': {e}")

    if not all_jobs_list:
        print("\nNo jobs were found across all search queries.")
        return

    # 2. Combine and Deduplicate Results
    combined_df = pd.concat(all_jobs_list, ignore_index=True)
    
    # Drop duplicates based on job URL or title + company combination
    if "job_url" in combined_df.columns:
        combined_df.drop_duplicates(subset=["job_url"], inplace=True)
    else:
        combined_df.drop_duplicates(subset=["title", "company"], inplace=True)

    # Filter out empty job descriptions if needed
    combined_df = combined_df[combined_df['title'].notna()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "scraped_jobs"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Export to CSV
    csv_filename = os.path.join(output_dir, f"jobs_karlsruhe_{timestamp}.csv")
    combined_df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    print(f"\n Saved {len(combined_df)} unique job postings to CSV: {csv_filename}")

    # 4. Export to JSON (structured for LLM Agent processing)
    json_filename = os.path.join(output_dir, f"jobs_karlsruhe_{timestamp}.json")
    jobs_dict = combined_df.to_dict(orient="records")
    
    # Convert Timestamp/Date objects to strings for clean JSON serialization
    clean_jobs_dict = json.loads(json.dumps(jobs_dict, default=str))
    
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(clean_jobs_dict, f, ensure_ascii=False, indent=2)
    print(f" Saved structured JSON to: {json_filename}")

if __name__ == "__main__":
    run_job_search()     
