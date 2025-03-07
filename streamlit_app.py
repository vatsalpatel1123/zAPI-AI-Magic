# streamlit_app.py

import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
import json
import re
import sys
import asyncio
# ---local imports---
from scraper import scrape_urls
from pagination import paginate_urls
from markdown import fetch_and_store_markdowns
from assets import MODELS_USED
from api_management import get_supabase_client
import os  # Import the 'os' module
from fastapi import FastAPI, HTTPException # Import FastAPI components
from fastapi.responses import JSONResponse # Import JSONResponse
import uvicorn # Import Uvicorn
import traceback # Import traceback

# Only use WindowsProactorEventLoopPolicy on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Initialize Streamlit app
st.set_page_config(page_title="zAPI AI-Web Scraper", page_icon="⚡")
supabase=get_supabase_client()

if supabase is None:
    st.error("🚨 **Supabase is not configured!** This project requires a Supabase database to function.")
    st.warning("Follow these steps to set it up:")

    st.markdown("""
    1. **[Create a free Supabase account](https://supabase.com/)**.
    2. **Create a new project** inside Supabase.
    3. **Create a table** in your project by running the following SQL command in the **SQL Editor**:
    
    ```sql
    CREATE TABLE IF NOT EXISTS scraped_data (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    unique_name TEXT NOT NULL,
    url TEXT,
    raw_data JSONB,        
    formatted_data JSONB, 
    pagination_data JSONB,
    api_link TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
    );
    ```

    4. **Go to Project Settings → API** and copy:
        - **Supabase URL**
        - **Anon Key**
    
    5. **Update your `.env` file** with these values:
    
    ```
    SUPABASE_URL=your_supabase_url_here
    SUPABASE_ANON_KEY=your_supabase_anon_key_here
    ```

    6. **Restart the project** close everything and reopen it, and you’re good to go! 🚀
    """)

st.title("zAPI AI-Web Scraper ⚡")

# Initialize session state variables
if 'scraping_state' not in st.session_state:
    st.session_state['scraping_state'] = 'idle'  # Possible states: 'idle', 'waiting', 'scraping', 'completed'
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'driver' not in st.session_state:
    st.session_state['driver'] = None
if 'api_link' not in st.session_state:
    st.session_state['api_link'] = None

# Model selection
model_selection = st.selectbox("Select Model", options=list(MODELS_USED.keys()), index=0)


with st.expander("API Keys", expanded=False):
    # Loop over every model in MODELS_USED
    for model, required_keys in MODELS_USED.items():
        # required_keys is a set (e.g. {"GEMINI_API_KEY"})
        for key_name in required_keys:
            # Create a password-type text input for each API key
            st.text_input(key_name,type="password",key=key_name)
    st.session_state['SUPABASE_URL'] = st.text_input("SUPABASE URL")
    st.session_state['SUPABASE_ANON_KEY'] = st.text_input("SUPABASE ANON KEY", type="password")


# URL Input Section
st.subheader("Enter URLs to Scrape")

# Ensure the session state for our URL list exists
if "urls_splitted" not in st.session_state:
    st.session_state["urls_splitted"] = []

col1, col2 = st.columns([3, 1], gap="small")

with col1:
    # A text area to paste multiple URLs at once
    if "text_temp" not in st.session_state:
        st.session_state["text_temp"] = ""

    url_text = st.text_area("Enter one or more URLs (space/tab/newline separated):",st.session_state["text_temp"], key="url_text_input", height=68)

with col2:
    if st.button("Add URLs"):
        if url_text.strip():
            new_urls = re.split(r"\s+", url_text.strip())
            new_urls = [u for u in new_urls if u]
            st.session_state["urls_splitted"].extend(new_urls)
            st.session_state["text_temp"] = ""
            st.rerun()
    if st.button("Clear URLs"):
        st.session_state["urls_splitted"] = []
        st.rerun()

# Show the URLs in an expander, each as a styled “bubble”
with st.expander("Added URLs", expanded=True):
    if st.session_state["urls_splitted"]:
        bubble_html = ""
        for url in st.session_state["urls_splitted"]:
            bubble_html += (
                f"<span style='"
                f"background-color: #E6F9F3;"  # Very Light Mint for contrast
                f"color: #0074D9;"            # Bright Blue for link-like appearance
                f"border-radius: 15px;"       # Slightly larger radius for smoother edges
                f"padding: 8px 12px;"         # Increased padding for better spacing
                f"margin: 5px;"               # Space between bubbles
                f"display: inline-block;"     # Ensures proper alignment
                f"text-decoration: none;"     # Removes underline if URLs are clickable
                f"font-weight: bold;"         # Makes text stand out
                f"font-family: Arial, sans-serif;"  # Clean and modern font
                f"box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'"  # Subtle shadow for depth
                f">{url}</span>"
            )
        st.markdown(bubble_html, unsafe_allow_html=True)
    else:
        st.write("No URLs added yet.")


# Fields to extract
show_tags = st.toggle("Enable Scraping")
fields = []
if show_tags:
    fields = st_tags(label='Enter Fields to Extract:',text='Press enter to add a field',value=[],suggestions=[],maxtags=-1,key='fields_input')


use_pagination = st.toggle("Enable Pagination")
pagination_details = ""
if use_pagination:
    pagination_details = st.text_input("Enter Pagination Details (optional)",help="Describe how to navigate through pages (e.g., 'Next' button class, URL pattern)")


# Main action button
if st.button("DO MAGIC 🪄", type="primary"):
    if st.session_state["urls_splitted"] == []:
        st.error("Please enter at least one URL.")
    elif show_tags and len(fields) == 0:
        st.error("Please enter at least one field to extract.")
    else:
        # Save user choices
        st.session_state['urls'] = st.session_state["urls_splitted"]
        st.session_state['fields'] = fields
        st.session_state['model_selection'] = model_selection
        st.session_state['use_pagination'] = use_pagination
        st.session_state['pagination_details'] = pagination_details
        
        # fetch or reuse the markdown for each URL
        unique_names = fetch_and_store_markdowns(st.session_state["urls_splitted"])
        st.session_state["unique_names"] = unique_names

        # Move on to "scraping" step
        st.session_state['scraping_state'] = 'scraping'

# FastAPI setup (only if SUPABASE URL and ANNON KEY exist)
if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_ANON_KEY"):
    app = FastAPI()

    @app.get("/api/data/{api_link}")
    async def get_data_by_api_link(api_link: str):  # Removed authentication
        try:
            response = supabase.table("scraped_data").select("*").eq("api_link", api_link).execute()
            data = response.data
            if data:
                return JSONResponse(content=data[0], media_type="application/json") # Explicit JSONResponse
            else:
                raise HTTPException(status_code=404, detail="Data not found for this API link")
        except Exception as e:
            traceback.print_exc()  # Print the traceback to the console
            raise HTTPException(status_code=500, detail=str(e))  # Provide error detail

    # Function to start FastAPI server
    def start_fastapi_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)  # Or another port

    # Run FastAPI in a separate thread
    import threading
    fastapi_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    fastapi_thread.start()
else:
    st.warning("FastAPI endpoint not available. Configure Supabase credentials.")

if st.session_state['scraping_state'] == 'scraping':
    try:
        with st.spinner("Processing..."):
            unique_names = st.session_state["unique_names"]  # from the LAUNCH step

            total_input_tokens = 0
            total_output_tokens = 0
            total_cost = 0
            
            # 1) Scraping logic
            all_data = []
            if show_tags:
                in_tokens_s, out_tokens_s, cost_s, parsed_data = scrape_urls(unique_names,st.session_state['fields'],st.session_state['model_selection'])
                total_input_tokens += in_tokens_s
                total_output_tokens += out_tokens_s
                total_cost += cost_s

                # Store or display parsed data 
                all_data = parsed_data # or rename to something consistent
                st.session_state['in_tokens_s'] = in_tokens_s
                st.session_state['out_tokens_s'] = out_tokens_s
                st.session_state['cost_s'] = cost_s
            # 2) Pagination logic
            pagination_info = None
            if st.session_state['use_pagination']:
                in_tokens_p, out_tokens_p, cost_p, page_results = paginate_urls(unique_names, st.session_state['model_selection'],st.session_state['pagination_details'],st.session_state["urls_splitted"])
                total_input_tokens += in_tokens_p
                total_output_tokens += out_tokens_s
                total_cost += cost_p

                # Example: store or display page_results
                # pagination_info can contain the final 'page_urls' from the LLM
                pagination_info = page_results
                # you'd parse the page_results to build 'page_urls'
                st.session_state['in_tokens_p'] = in_tokens_p
                st.session_state['out_tokens_p'] = out_tokens_s
                st.session_state['cost_p'] = cost_p

            # Find api link
            for data_item in all_data:
                if "api_link" in data_item:
                    st.session_state["api_link"] = data_item["api_link"]
                
            # 3) Save everything in session state
            st.session_state['results'] = {
                'data': all_data,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens,
                'total_cost': total_cost,
                'pagination_info': pagination_info
            }
            st.session_state['scraping_state'] = 'completed'
    except Exception as e:
        traceback.print_exc()  # Print the traceback to the console
        # Display the error message.
        st.error(f"An error occurred during scraping: {e}")

        # Reset the scraping state to 'idle' so that the app stays in an idle state.
        st.session_state['scraping_state'] = 'idle'

# Display results
if st.session_state['scraping_state'] == 'completed' and st.session_state['results']:
    results = st.session_state['results']
    all_data = results['data']
    total_input_tokens = results['input_tokens']
    total_output_tokens = results['output_tokens']
    total_cost = results['total_cost']
    pagination_info = results['pagination_info']

    if show_tags:
        st.subheader("Scraping Results")

        # Display token usage in a table format
        token_data = {'Metric': ['Input Tokens', 'Output Tokens', 'Total Cost'],
                      'Value': [total_input_tokens, total_output_tokens, total_cost]}  #Remove $ and avoid formatting here


        token_df = pd.DataFrame(token_data)

        # Format Total Cost as currency AFTER DataFrame creation
        token_df['Value'] = token_df['Value'].astype(float) # Ensure it's float
        token_df.loc[token_df['Metric'] == 'Total Cost', 'Value'] = token_df.loc[token_df['Metric'] == 'Total Cost', 'Value'].apply(lambda x: f"${x:.4f}")

        st.table(token_df.set_index('Metric'))
    

        # We'll accumulate all rows in this list
        all_rows = []

        # Loop over each data item in the "all_data" list
        for i, data_item in enumerate(all_data, start=1):

            # Usually data_item is something like:
            # {"unique_name": "...", "parsed_data": DynamicListingsContainer(...) or dict or str}

            # 1) Ensure data_item is a dict
            if not isinstance(data_item, dict):
                st.error(f"data_item is not a dict, skipping. Type: {type(data_item)}")
                continue

            # 2) If "parsed_data" is present and might be a Pydantic model or something
            if "parsed_data" in data_item:
                parsed_obj = data_item["parsed_data"]

                # Convert if it's a Pydantic model
                if hasattr(parsed_obj, "dict"):
                    parsed_obj = parsed_obj.model_dump()
                elif isinstance(parsed_obj, str):
                    # If it's a JSON string, attempt to parse
                    try:
                        parsed_obj = json.loads(parsed_obj)
                    except json.JSONDecodeError:
                        # fallback: just keep as raw string
                        pass

                # Now we have "parsed_obj" as a dict, list, or string
                data_item["parsed_data"] = parsed_obj

            # 3) If the "parsed_data" has a 'listings' key that is a list of items,
            #    we might want to treat them as multiple rows. 
            #    Otherwise, we treat the entire data_item as a single row.

            pd_obj = data_item["parsed_data"]

            # If it has 'listings' in parsed_data
            if isinstance(pd_obj, dict) and "listings" in pd_obj and isinstance(pd_obj["listings"], list):
                # We'll create one row per listing, plus carry over "unique_name" or other fields
                for listing in pd_obj["listings"]:
                    # Make a shallow copy so we don't mutate 'listing'
                    row_dict = dict(listing)
                    # You can also attach the unique_name or other top-level fields:
                    # row_dict["unique_name"] = data_item.get("unique_name", "")
                    all_rows.append(row_dict)
            else:
                # We'll just store the entire item as one row
                # Possibly flatten parsed_data => just store it as "parsed_data" field
                # e.g. if parsed_obj is a dict, embed it. Or keep it as string
                row_dict = dict(data_item)  # shallow copy
                all_rows.append(row_dict)

        # After collecting all rows from all_data in "all_rows", create one DataFrame
        if not all_rows:
            st.warning("No data rows to display.  Check your scraping fields.")
        else:
            df = pd.DataFrame(all_rows)
            st.dataframe(df, use_container_width=True)

        # Download options
        st.subheader("Download Extracted Data")
        col1, col2, col3 = st.columns(3)  # Define three columns

        with col1:
            json_data = json.dumps(all_data, default=lambda o: o.dict() if hasattr(o, 'dict') else str(o), indent=4)
            st.download_button("Download JSON", data=json_data, file_name="scraped_data.json")

        with col2:
            # Convert all data to a single DataFrame
            all_listings = []
            for data in all_data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                if isinstance(data, dict) and 'listings' in data:
                    all_listings.extend(data['listings'])
                elif hasattr(data, 'listings'):
                    all_listings.extend([item.dict() for item in data.listings])
                else:
                    all_listings.append(data)

            combined_df = pd.DataFrame(all_listings)
            st.download_button("Download CSV", data=combined_df.to_csv(index=False), file_name="scraped_data.csv")

        with col3:
            api_link = st.session_state.get("api_link")  # Get dynamic api_link
            
            if api_link:
                 st.markdown(f"""
         <style>
            .custom-button {{
                width: 135px;
                height: 35px;
                border: 0.1px solid #40434B;
                border-radius: 8px;
                background-color: #131720;
                color: white;
                cursor: pointer;
                text-align: center;
                display: inline-block;
                line-height: 35px;
                text-decoration: none; /* Remove hyperlink styling */
            }}
            .custom-button:hover {{
                border-color: #FF4B4B;
                color: #FF4B4B;
            }}
            .api-button {{ /* Specific style for the API button */
                color: white !important; /* Override any inherited color */
                text-decoration: none !important; /* Override link underline */
            }}
        </style>
        <a href="http://localhost:8000/api/data/{api_link}" target="_blank" class="custom-button api-button">Open API</a>
    """, unsafe_allow_html=True)
            else:
                st.warning("API link not available.")

    # Display pagination info
    if pagination_info:
        all_page_rows = []

        # Iterate through pagination_info, which contains multiple items
        for i, item in enumerate(pagination_info, start=1):

            # Ensure item is a dictionary
            if not isinstance(item, dict):
                st.error(f"item is not a dict, skipping. Type: {type(item)}")
                continue

            # Check if "pagination_data" exists
            if "pagination_data" in item:
                pag_obj = item["pagination_data"]

                # Convert if it's a Pydantic model
                if hasattr(pag_obj, "dict"):
                    pag_obj = pag_obj.model_dump()
                elif isinstance(pag_obj, str):
                    # If it's a JSON string, attempt to parse
                    try:
                        pag_obj = json.loads(pag_obj)
                    except json.JSONDecodeError:
                        # Fallback: keep it as raw string
                        pass

                # Now we have "pag_obj" as a dict, list, or string
                item["pagination_data"] = pag_obj

            # Process the extracted pagination_data
            pd_obj = item["pagination_data"]

            # If it contains "page_urls" and it's a list, extract individual rows
            if isinstance(pd_obj, dict) and "page_urls" in pd_obj and isinstance(pd_obj["page_urls"], list):
                for page_url in pd_obj["page_urls"]:
                    row_dict = {"page_url": page_url}
                    # Optionally, attach "unique_name" or other top-level fields
                    # row_dict["unique_name"] = item.get("unique_name", "")
                    all_page_rows.append(row_dict)
            else:
                # Otherwise, store the entire item as a single row
                row_dict = dict(item)  # Shallow copy
                all_page_rows.append(row_dict)

        # Create DataFrame and display it
        if not all_page_rows:
            st.warning("No page URLs found.")
        else:
            pagination_df = pd.DataFrame(all_page_rows)
            st.markdown("---")
            st.subheader("Pagination Information")
            st.write("**Page URLs:**")
            st.dataframe(pagination_df,column_config={"page_url": st.column_config.LinkColumn("Page URL")},use_container_width=True)

    # Reset scraping state
    if st.button("Clear Results"):
        st.session_state['scraping_state'] = 'idle'
        st.session_state['results'] = None

   # If both scraping and pagination were performed, show totals under the pagination table
    if show_tags and pagination_info:
        st.markdown("---")
        st.markdown("### Total Counts and Cost (Including Pagination)")
        st.markdown(f"**Total Input Tokens:** {total_input_tokens}")
        st.markdown(f"**Total Output Tokens:** {total_output_tokens}")
        st.markdown(f"**Total Combined Cost:** :rainbow-background[**${total_cost:.4f}**]")