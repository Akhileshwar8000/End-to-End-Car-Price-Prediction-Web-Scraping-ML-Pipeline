"""
========================================================================================
CAR AUCTION WEB SCRAPER - DATA EXTRACTION MODULE
========================================================================================
This module contains all web scraping functions for extracting car auction data
from Cars & Bids. It implements robust parsing, error handling, and multithreading
to efficiently collect structured data for machine learning applications.

Key Features:
- Selenium-based dynamic page loading
- BeautifulSoup HTML parsing
- Multithreaded concurrent scraping
- Automatic checkpointing for data persistence
- Comprehensive error handling and logging

Author: Akhileshwar Chauhan
Project: Car Price Prediction with Web Scraping
========================================================================================
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ========================================================================================
# LOGGING CONFIGURATION
# ========================================================================================
# Configure logging to track scraping operations, errors, and debugging info
logging.basicConfig(
    filename='scraper1.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ========================================================================================
# SELENIUM CONFIGURATION
# ========================================================================================
# Configure Chrome WebDriver with headless mode and anti-detection settings
# Headless mode runs browser without GUI for better performance
CHROME_OPTIONS = webdriver.ChromeOptions()
CHROME_OPTIONS.add_argument("--headless")       # Run without GUI
CHROME_OPTIONS.add_argument("--disable-gpu")    # Disable GPU acceleration (for headless)
CHROME_OPTIONS.add_argument("--no-sandbox")     # Bypass OS security model (for containers)
CHROME_OPTIONS.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")       # Mimic real browser to avoid detection

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def safe_select_text(soup: BeautifulSoup, selector: str) -> str:
    """
    Safely extract text from HTML using CSS selector with None handling.

    Args:
        soup: BeautifulSoup object containing parsed HTML
        selector: CSS selector string to locate element

    Returns:
        Stripped text content if element exists, None otherwise

    Purpose:
        Prevents crashes from missing HTML elements during scraping
    """
    el = soup.select_one(selector)
    return el.get_text(strip=True) if el else None

def clean_text(value: str) -> Optional[str]:
    """
    Clean and normalize text values by removing whitespace and newlines.

    Args:
        value: Raw text string to clean

    Returns:
        Cleaned text with normalized whitespace, None if input is empty

    Purpose:
        Standardizes text data for consistent machine learning features
    """
    if not value:
        return None
    return value.strip().replace('\n', ' ').replace('\t', ' ')

def simplify_color(color_string: str) -> str:
    """
    Categorize detailed color descriptions into standard color groups.

    Args:
        color_string: Original color description (e.g., "Midnight Blue Metallic")

    Returns:
        Simplified color category (e.g., "Blue")

    Purpose:
        Reduces color feature cardinality for better model generalization

    Examples:
        "Alpine White" -> "White"
        "Nardo Gray" -> "Gray"
        "Estoril Blue" -> "Blue"
    """
    color_string = color_string.lower()

    # Map common color keywords to standard categories
    if "black" in color_string: return "Black"
    if "white" in color_string: return "White"
    if "silver" in color_string or "gray" in color_string: return "Gray"
    if "blue" in color_string: return "Blue"
    if "red" in color_string: return "Red"
    return "Other" # For less common colors

# ========================================================================================
# SELENIUM DRIVER INITIALIZATION
# ========================================================================================

def init_driver() -> webdriver.Chrome:
    """
    Initialize and configure a Selenium Chrome WebDriver instance.

    Returns:
        Configured Chrome WebDriver object ready for scraping

    Purpose:
        Creates browser automation instance with optimal settings for web scraping

    Note:
        Each thread in multithreaded scraping creates its own driver to avoid
        race conditions and thread-safety issues with Selenium
    """
    return webdriver.Chrome(options=CHROME_OPTIONS)

# ========================================================================================
# LINK COLLECTION (PHASE 1)
# ========================================================================================

def get_all_car_listing_links(driver: webdriver.Chrome, base_url: str, max_pages: int = 40) -> List[str]:
    """
    Scrape car listing URLs from paginated auction result pages.

    This function handles the sequential navigation through pagination to collect
    all individual car listing URLs before detailed scraping begins.

    Args:
        driver: Selenium WebDriver instance for browser control
        base_url: Starting URL for auction listings
        max_pages: Maximum number of pages to scrape (default: 40)

    Returns:
        List of complete URLs for individual car listings

    Strategy:
        - Sequential page loading (pagination requires stable state)
        - Explicit waits for dynamic content loading
        - Graceful handling of missing pages (stops at last available page)
    """
    all_links = []

    # Iterate through pagination
    for page_num in range(1, 1 + max_pages):
        # Construct paginated URL
        url = f"{base_url}&page={page_num}"
        logging.info(f"Loading page {page_num}: {url}")

        # Load page with Selenium
        driver.get(url)

        try:
            # Wait for car listing elements to load (up to 20 seconds)
            # This explicit wait ensures JavaScript has rendered the content
            WebDriverWait(driver, 20).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.auction-title > a"))
            )
        except TimeoutException:
            # If page doesn't load, log warning and stop pagination
            logging.warning(f"Timeout on page {page_num}: {url}")
            break

        # Parse the fully-loaded page HTML
        base_site_url = "https://carsandbids.com"
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        page_links = [
            base_site_url + a.get('href')
            for a in soup.select("div.auction-title > a")
            if a.get('href')
        ]

        logging.info(f"Found {len(page_links)} links on page {page_num}")
        all_links.extend(page_links)

        # If no links found, we've reached the end
        if not page_links:
            break

    logging.info(f"Total car links found: {len(all_links)}")
    return all_links

# ========================================================================================
# INDIVIDUAL CAR DATA SCRAPING (PHASE 2)
# ========================================================================================

def scrape_car_page(url: str) -> Dict[str, Any]:
    """
    Scrape complete data from a single car listing page.

    This is the main function called by each thread in multithreaded scraping.
    Each thread creates its own WebDriver to avoid thread-safety issues.

    Args:
        url: Complete URL of individual car listing

    Returns:
        Dictionary containing all extracted car attributes
        Empty dict if scraping fails

    Thread Safety:
        Each invocation creates and destroys its own WebDriver instance,
        making this function thread-safe for parallel execution
    """
    # Create dedicated driver for this thread
    driver = init_driver()

    try:
        # Delegate actual scraping to helper function
        return _get_individual_car_data(driver, url)
    finally:
        # Always close driver to prevent memory leaks
        driver.quit()

def _get_individual_car_data(driver: webdriver.Chrome, url: str) -> Dict[str, Any]:
    """
    Extract all car attributes from a listing page.

    This function orchestrates the extraction of different data sections:
    - Bid/price information
    - Technical specifications
    - Equipment and features
    - Condition and flaws
    - Modifications

    Args:
        driver: Selenium WebDriver instance
        url: URL of the car listing page

    Returns:
        Dictionary with all extracted attributes (27+ features)

    Error Handling:
        - Timeout if page doesn't load
        - Exception catching for parsing errors
    """
    logging.info(f"Loading listing page: {url}")
    driver.get(url)

    try:
        # Wait for main content to load (comments section is a reliable indicator)
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.comments"))
        )
    except TimeoutException:
        logging.error(f"Timeout: Core page content never loaded for {url}")
        return {}

    # Parse the loaded HTML once (reuse for all extraction functions)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    try:
        # Combine data from all extraction functions using dictionary unpacking
        car_data = {
            **_get_car_bids_data(soup),
            **_get_car_facts(soup),
            **_get_car_age(soup),
            **_get_equipment_details(soup),
            **_get_flaw_details(soup),
            **_get_modification_details(soup)
        }
    except Exception as e:
        # Log full exception traceback for debugging
        logging.exception(f"Error parsing car data for {url}: {e}")
        return {}

    return car_data

# ========================================================================================
# DATA EXTRACTION FUNCTIONS
# ========================================================================================

def _get_car_bids_data(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract auction results: selling price and sold status.

    Args:
        soup: BeautifulSoup object of car listing page

    Returns:
        Dict with 'Sold' (bool) and 'Selling Price' (int or None)

    Target Variable:
        'Selling Price' is our prediction target for the ML model
    """
    bid_attributes = {}
    bid_stats = soup.select_one("ul.bid-stats")

    if not bid_stats:
        logging.warning("No bid-stats section found.")
        return bid_attributes

    # Extract sold status (text contains "sold" or "reserve not met")
    sold_text = safe_select_text(bid_stats, "span.value")
    bid_attributes["Sold"] = "sold" in sold_text.lower() if sold_text else False

    # Extract selling price (remove $ and commas, convert to integer)
    price_text = safe_select_text(bid_stats, "span.bid-value")
    if price_text:
        try:
            bid_attributes["Selling Price"] = int(price_text.replace("$", "").replace(",", "").strip())
        except ValueError:
            bid_attributes["Selling Price"] = None
    else:
        bid_attributes["Selling Price"] = None

    return bid_attributes

def _get_car_facts(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract technical specifications and car details from the quick facts section.

    This is the most complex extraction function as it handles ~15 different
    attributes with varying formats and edge cases.

    Args:
        soup: BeautifulSoup object of car listing page

    Returns:
        Dictionary with specifications (make, model, mileage, engine, etc.)

    Extracted Features:
        - Make & Model (categorical)
        - Mileage (numeric)
        - Engine displacement, aspiration, cylinder config
        - Drivetrain (FWD/RWD/AWD)
        - Transmission & gears
        - Body style, colors
        - Title status, location, seller type
    """
    car_attributes = {}
    car_facts = soup.select_one("div.quick-facts")

    if car_facts:

        # Find all definition term (dt) tags in the facts section
        dt_tags = car_facts.find_all("dt")

        # Process each key-value pair
        for dt_tag in dt_tags:
            # Get corresponding value (dd = definition description)
            dd = dt_tag.find_next_sibling("dd")

            if dd:
                # Extract and clean key and value
                key = clean_text(dt_tag.get_text())
                value = clean_text(dd.get_text()).lower()

                # ================================================================================
                # ATTRIBUTE-SPECIFIC PARSING LOGIC
                # ================================================================================

                # --------------------------------------------------------------------------------
                # MODEL: Clean up extra text and handle compound names
                # --------------------------------------------------------------------------------
                if key == "Model":
                    # Remove "save" button text that sometimes appears
                    cleaned_model = value.replace("save", "").strip()

                    # Handle compound models like "M3/M4" -> take last part
                    if '/' in cleaned_model:
                        cleaned_model = cleaned_model.split('/')[-1].split(" ")[-1].strip()

                    car_attributes[key] = cleaned_model

                # --------------------------------------------------------------------------------
                # MILEAGE: Extract numeric value from "X,XXX mi" format
                # --------------------------------------------------------------------------------
                elif key == "Mileage":
                    try:
                        # Split on space, take first part, remove commas, convert to int
                        car_attributes[key] = int(value.split(" ")[0].replace(",", ""))
                    except (ValueError, IndexError):
                        car_attributes[key] = None

                # --------------------------------------------------------------------------------
                # TITLE STATUS: Extract first word (Clean/Salvage/Rebuilt)
                # --------------------------------------------------------------------------------
                elif key == "Title Status":
                    value = value.split(" ")[0]
                    car_attributes[key] = value

                # --------------------------------------------------------------------------------
                # LOCATION: Extract state abbreviation
                # --------------------------------------------------------------------------------
                elif key == "Location":
                    # Format: "City, ST ZIP" -> extract state
                    value = value.split(",")[1].split(" ")
                    if len(value) > 1:
                        car_attributes[key] = value[1]

                # --------------------------------------------------------------------------------
                # COLORS: Simplify to standard categories
                # --------------------------------------------------------------------------------
                elif key == "Exterior Color" or key == "Interior Color":
                    value = simplify_color(value)
                    car_attributes[key] = value

                # --------------------------------------------------------------------------------
                # TRANSMISSION: Extract type and number of gears
                # --------------------------------------------------------------------------------
                elif key == "Transmission":
                    values = value.split(" ")
                    car_attributes[key] = values[0]     # e.g., "manual" or "automatic"

                    # Extract gear count from format like "6-speed"
                    car_attributes["Gears"] = None
                    if len(values) > 1:
                        gears_text = values[1].replace("(", "").replace(")", "")
                        if 'speed' in gears_text.lower():
                            try:
                                car_attributes["Gears"] = int(gears_text.split('-')[0])
                            except (ValueError, IndexError):
                                pass  # Leave as None if parsing fails

                # --------------------------------------------------------------------------------
                # ENGINE: Parse displacement, aspiration, and cylinder configuration
                # --------------------------------------------------------------------------------
                elif key == "Engine":
                    parts = value.split(" ")
                    if not parts: continue

                    # Extract displacement and standardize to liters
                    size_str = parts[0].upper()
                    displacement_l = None

                    try:
                        if "L" in size_str:
                            # Already in liters (e.g., "3.0L")
                            displacement_l = float(size_str.replace("L", ""))
                        elif "CC" in size_str:
                            # Convert cubic centimeters to liters (e.g., "2000CC")
                            displacement_l = float(size_str.replace("CC", "")) / 1000
                        elif "CI" in size_str:
                            # Convert cubic inches to liters (e.g., "350CI")
                            displacement_l = float(size_str.replace("CI", "")) / 61.024
                    except ValueError:
                        displacement_l = None

                    if displacement_l:
                        car_attributes['Displacement (L)'] = round(displacement_l, 1)

                    #  Extract aspiration and cylinder configuration
                    if len(parts) > 1:
                        if len(parts) == 3:
                            # Format: "3.0L Turbocharged Inline-6"
                            car_attributes['Aspiration'] = parts[1]
                            car_attributes['Cylinder Config'] = parts[2]
                        elif len(parts) == 2:
                            # Format: "3.0L Inline-6" (naturally aspirated)
                            car_attributes['Aspiration'] = 'Naturally Aspirated'
                            car_attributes['Cylinder Config'] = parts[1]

                # --------------------------------------------------------------------------------
                # DRIVETRAIN: Standardize to FWD/RWD/AWD/4WD
                # --------------------------------------------------------------------------------
                elif key == "Drivetrain":
                    if "front" in value:
                        value = "FWD"  # Front-wheel drive
                    elif "rear" in value:
                        value = "RWD"  # Rear-wheel drive
                    elif "all" in value or "xdrive" in value or "quattro" in value:
                        value = "AWD"  # All-wheel drive
                    elif "four" in value:
                        value = "4WD"  # Four-wheel drive
                    car_attributes[key] = value.upper()

                # --------------------------------------------------------------------------------
                # SELLER TYPE: Remove parenthetical text
                # --------------------------------------------------------------------------------
                elif key == "Seller Type":
                    # Format: "Dealer (verified)" -> "Dealer"
                    value = value.split("(")[0].strip()
                    car_attributes[key] = value

                # --------------------------------------------------------------------------------
                # SKIP: Personal information fields
                # --------------------------------------------------------------------------------
                elif key == "Seller" or key == "VIN":
                    continue

                # --------------------------------------------------------------------------------
                # DEFAULT: Store other attributes as-is
                # --------------------------------------------------------------------------------
                else:
                    car_attributes[key] = value
            else:
                continue

    return car_attributes

def _get_car_age(soup: BeautifulSoup) -> Dict[str, Any]:
    """
        Calculate car age from model year (feature engineering).

        Args:
            soup: BeautifulSoup object of car listing page

        Returns:
            Dict with 'Model Year Age' (int): years since manufacture

        Rationale:
            Age is a strong predictor of car value (depreciation)
            Calculated feature: current_year - model_year
    """
    car_age = {}
    misc = soup.select_one("div.auction-title")

    if misc:
        # Title format: "2018 BMW M3" -> extract year
        age = datetime.now().year - int(misc.text.split()[0])
        car_age["Model Year Age"] = age

    return car_age

def _get_equipment_details(soup: BeautifulSoup) -> Dict[str, Any]:
    """
        Extract premium equipment features (binary indicators).

        These features significantly impact car value and are important
        predictors in the pricing model.

        Args:
            soup: BeautifulSoup object of car listing page

        Returns:
            Dict with 6 binary features (0 or 1):
            - Has Executive Package
            - Has Carbon Equipment
            - Has Lane Tracking Equipment
            - Has Leather Equipment
            - Has Premium Sound
            - Has Sunroof

        Purpose:
            Premium features add value and help differentiate similar vehicles
    """
    # Initialize all features to 0 (not present)
    car_equipment = {
        "Has Executive Package": 0,
        "Has Carbon Equipment": 0,
        "Has Lane Tracking Equipment": 0,
        "Has Leather Equipment": 0,
        "Has Premium Sound": 0,
        "Has Sunroof": 0,
    }

    # Find equipment section
    equipment_section = soup.select_one("div.detail-equipment")

    if equipment_section:
        equipment_list = equipment_section.find('ul')
        if equipment_list:
            # Check each list item for keywords
            for item in equipment_list.find_all('li'):
                text = item.get_text().lower()

                # Set flag to 1 if keyword found
                if 'executive package' in text:
                    car_equipment['Has Executive Package'] = 1
                if 'carbon' in text:
                    car_equipment['Has Carbon Equipment'] = 1
                if 'leather' in text:
                    car_equipment['Has Leather Equipment'] = 1
                if 'sunroof' in text:
                    car_equipment['Has Sunroof'] = 1
                if 'lane tracking' in text:
                    car_equipment['Has Lane Tracking Equipment'] = 1
                # Premium audio brands
                if 'burmester' in text or 'harman kardon' in text:
                    car_equipment['Has Premium Sound'] = 1

    return car_equipment

def _get_flaw_details(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Extract vehicle condition issues (negative value factors).

    Args:
        soup: BeautifulSoup object of car listing page

    Returns:
        Dict with 2 binary features (0 or 1):
        - Has Accident History
        - Has Cosmetic Flaw

    Impact on Model:
        These features typically decrease predicted price
        Important for realistic valuation
    """
    flaws ={
        "Has Accident History": 0,
        "Has Cosmetic Flaw": 0,
    }

    # Find known flaws section
    flaw_section = soup.select_one("div.detail-known_flaws")

    if flaw_section:
        flaw_list = flaw_section.find('ul')
        if flaw_list:
            for flaws_text in flaw_list.find_all('li'):
                flaws_text = flaws_text.get_text().lower()

                # Check for accident-related keywords
                if any(word in flaws_text for word in ['collision', 'damage', 'accident']):
                    flaws["Has Accident History"] = 1

                # Check for cosmetic damage keywords
                if any(word in flaws_text for word in ['chip', 'ding', 'scratch', 'wear']):
                    flaws["Has Cosmetic Flaw"] = 1

    return flaws

def _get_modification_details(soup: BeautifulSoup) -> Dict[str, Any]:
    """
        Check if car has aftermarket modifications.

        Args:
            soup: BeautifulSoup object of car listing page

        Returns:
            Dict with 1 binary feature:
            - Has Modifications (0 or 1)

        Note:
            Modifications can increase OR decrease value depending on type and quality.
            The model will learn the average effect across all modifications.
    """
    modifications = {
        "Has Modifications": 0,
    }

    # If modifications section exists, set flag to 1
    if soup.select_one("div.detail-modifications"):
        modifications["Has Modifications"] = 1

    return modifications

# ========================================================================================
# MULTITHREADED SCRAPING ORCHESTRATION
# ========================================================================================

def scrape_all_cars_multithreaded(all_links: List[str], checkpoint_every: int = 10, max_workers: int = 3):
    """
    Scrape all car listings using multithreading for performance.

    This function implements parallel scraping with the following features:
    - ThreadPoolExecutor for concurrent execution
    - Independent WebDriver per thread (thread-safe)
    - Periodic checkpointing (data persistence)
    - Robust error handling (failed scrapes don't crash entire job)

    Args:
        all_links: List of car listing URLs to scrape
        checkpoint_every: Save progress every N cars (default: 10)
        max_workers: Number of parallel threads (default: 3)

    Returns:
        List of dictionaries, each containing one car's complete data
    """
    all_car_data = []

    # Create thread pool with specified number of workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks to thread pool
        # Returns a dict mapping Future objects to their URLs
        future_to_url = {
            executor.submit(scrape_car_page, url):
                url for url in all_links
        }

        # Process completed tasks as they finish (not in submission order)
        for future in as_completed(future_to_url):
            url = future_to_url[future]

            try:
                # Get the result from the completed thread
                car_data = future.result()

                # Only add if scraping succeeded (non-empty dict)
                if car_data:
                    all_car_data.append(car_data)

                    # Periodic checkpoint: save progress to prevent data loss
                    if len(all_car_data) % checkpoint_every == 0:
                        pd.DataFrame(all_car_data).to_csv("checkpoint.csv", index=False)
                        logging.info(f"Checkpoint saved with {len(all_car_data)} cars.")

            except Exception as e:
                # Log error but continue scraping other cars
                logging.error(f"Failed to scrape {url}: {e}")

    # Final save: export all collected data to CSV
    if all_car_data:
        pd.DataFrame(all_car_data).to_csv("scraped_car_data.csv", index=False)
        logging.info(f"Saved final {len(all_car_data)} car records to scraped_car_data.csv")

    return all_car_data