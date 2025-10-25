"""
========================================================================================
CAR AUCTION DATA SCRAPER - MAIN CONTROLLER
========================================================================================
This module orchestrates the web scraping pipeline for Cars & Bids auction data.
It coordinates link collection and multithreaded data extraction to build a
comprehensive dataset for machine learning price prediction.

Author: Akhileshwar Chauhan
Project: Car Price Prediction with Web Scraping
========================================================================================
"""

import logging
from scraper import init_driver, get_all_car_listing_links, scrape_all_cars_multithreaded

# ========================================================================================
# CONFIGURATION PARAMETERS
# ========================================================================================
# These settings control the scope and performance of the scraping operation

URL = "https://carsandbids.com/past-auctions/?body_style=1" # Target URL for sedan auctions
MAX_PAGES = 1           # Number of pages to scrape for links (1 page = 50 listings)
MAX_THREADS = 4          # Parallel threads for scraping (balance between speed and server load)
CHECKPOINT_EVERY = 10    # Frequency of data backups (saves progress every 10 cars)

# ========================================================================================
# LOGGING CONFIGURATION
# ========================================================================================
# Set up logging to track scraping progress, errors, and debugging information
# This creates a persistent log file for monitoring and troubleshooting

logging.basicConfig(
    filename='main.log',    # Log file location
    level=logging.INFO,     # Log level (INFO captures major events)
    format='%(asctime)s - %(levelname)s - %(message)s'      # Timestamp + severity + message
)

# ========================================================================================
# MAIN SCRAPING PIPELINE
# ========================================================================================

def main():
    """
    Main control flow for the Cars & Bids auction data scraper.

    This function executes a two-phase scraping strategy:

    PHASE 1 - Link Collection (Sequential):
        - Navigate through paginated auction listings
        - Extract individual car listing URLs
        - Sequential processing ensures stable pagination handling

    PHASE 2 - Data Extraction (Parallel):
        - Scrape detailed data from each car listing
        - Multithreaded execution for faster processing
        - Each thread uses independent Selenium driver (avoids race conditions)
        - Automatic checkpointing prevents data loss

    Returns:
        None (saves data to CSV file)

    Side Effects:
        - Creates 'scraped_car_data.csv' with complete dataset
        - Creates 'checkpoint.csv' for progress tracking
        - Logs all operations to 'main.log'
    """

    # --------------------------------------------------------------------------------
    # PHASE 1: COLLECT ALL CAR LISTING URLS
    # --------------------------------------------------------------------------------
    # Initialize Selenium WebDriver for automated browser control
    # This single driver handles pagination across all listing pages
    driver = init_driver()

    try:
        # Log the start of link collection phase
        logging.info("Starting to collect all car listing URLs...")

        # Scrape listing pages sequentially to gather all individual car URLs
        # Sequential approach prevents pagination errors and ensures completeness
        all_links = get_all_car_listing_links(driver, URL, MAX_PAGES)

        # Log the total number of links collected
        logging.info(f"Collected {len(all_links)} car listing URLs.")

        # Validate that links were found before proceeding
        if not all_links:
            print("No car links found. Exiting.")
            logging.warning("No car links found. Exiting.")
            return

    finally:
        # Always close the browser to free system resources
        # This executes even if an error occurs
        driver.quit()
        logging.info("Chrome browser closed.")

    # --------------------------------------------------------------------------------
    # PHASE 2: SCRAPE DETAILED DATA FROM EACH CAR LISTING
    # --------------------------------------------------------------------------------
    # Use multithreading to scrape individual car pages in parallel
    # This significantly reduces total scraping time

    print(f"Scraping data for {len(all_links)} cars using {MAX_THREADS} threads...")

    # Execute parallel scraping with automatic checkpointing
    # Each thread creates its own Selenium driver to avoid thread-safety issues
    all_car_data = scrape_all_cars_multithreaded(
        all_links,                          # List of car URLs to scrape
        checkpoint_every=CHECKPOINT_EVERY,  # Save progress periodically
        max_workers=MAX_THREADS             # Number of parallel threads
    )

    # Report completion status
    print(f"Scraping completed. {len(all_car_data)} car records saved to CSV.")

# ========================================================================================
# SCRIPT ENTRY POINT
# ========================================================================================
if __name__ == "__main__":
    """
        Entry point when script is run directly (not imported as a module).
        This ensures main() only executes when explicitly running this file.
    """
    main()