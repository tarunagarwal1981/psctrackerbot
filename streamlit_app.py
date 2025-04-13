import streamlit as st
import requests
import pandas as pd
import json
import os
from openai import OpenAI

# Set page configuration
st.set_page_config(
    page_title="Vessel Data Assistant",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'vessel_data' not in st.session_state:
    st.session_state.vessel_data = None

if 'df' not in st.session_state:
    st.session_state.df = None

# Fetch OpenAI API key from Streamlit secrets (based on your structure)
openai_api_key = st.secrets["openai"]["openai_api_key"]
client = OpenAI(api_key=openai_api_key)

# Lambda function URL
LAMBDA_URL = "https://qescpqp626isx43ab5mnlyvayi0zvvsg.lambda-url.ap-south-1.on.aws/"

# Function to fetch data from Lambda
def fetch_vessel_data():
    try:
        response = requests.get(f"{LAMBDA_URL}/api/vessels")
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Create a detailed system prompt with column definitions
def create_system_prompt(df=None):
    system_prompt = """
    You are a specialized vessel data analyst assistant that helps maritime professionals get insights from vessel tracking data.
    
    # Data Schema
    You have access to vessel data with the following columns and their descriptions:
    
    - id: Unique identifier for each record in the database
    - vessel_name: The name of the vessel
    - imo_no: IMO number - International Maritime Organization unique vessel identifier
    - comments: Any additional notes or comments about the vessel
    - created_at: Timestamp when the record was created
    - updated_at: Timestamp when the record was last updated
    - owner: The company that owns the vessel
    - event_type: Current vessel status (e.g., "AT SEA", "IN PORT", "AT ANCHOR")
    - report_date: Date when this vessel data was reported
    - departure_port: Name of the port from which the vessel departed
    - departure_country: Country code of departure port
    - departure_date: Date when the vessel departed
    - arrival_port: Name of the port where the vessel is heading
    - arrival_country: Country code of arrival port
    - eta: Estimated Time of Arrival at destination
    - etb: Estimated Time of Berthing
    - etd: Estimated Time of Departure from destination
    - atd: Actual Time of Departure
    - lat: Current latitude coordinate of the vessel
    - lon: Current longitude coordinate of the vessel
    - speed: Current speed of the vessel in knots
    - heading: Direction the vessel is pointing in degrees
    - course: Direction the vessel is moving in degrees
    - port_match_flag: Flag indicating if there's a port match issue
    - multi_port_flag: Flag indicating if the vessel is visiting multiple ports
    - eta_check_flag: Flag indicating if there's an ETA check issue
    - distance_to_go: Distance remaining to destination in nautical miles
    - psc_last_inspection_date: Date of last Port State Control inspection
    - psc_last_inspection_port: Port where the last PSC inspection occurred
    - amsa_last_inspection_date: Date of last Australian Maritime Safety Authority inspection
    - amsa_last_inspection_port: Port where the last AMSA inspection occurred
    - dwh_load_date: Date when data was loaded into the data warehouse
    - status: Current operational status of the vessel
    - rds_load_date: Date when data was loaded into RDS
    - office_doc: Office documentation status
    - five_day_checklist: Status of the 5-day checklist (e.g., "Received", "Pending")
    - checklist_received: Alternative field for five_day_checklist status
    - sanz: SANZ certification status
    - BUILT_DATE: Date when the vessel was built
    - vessel_type: Type/category of the vessel
    - doc_id: Document identifier
    - fleet_type: Category of fleet the vessel belongs to
    
    # Processing Instructions
    
    1. For country-related questions:
       - Country codes are standard 2-letter codes (AU = Australia, SG = Singapore, etc.)
       - If a user asks about a country by name, look for the corresponding code in arrival_country or departure_country
    
    2. For date-related questions:
       - Timestamps are in ISO format
       - For questions about vessels arriving "soon" or "next week", compare the current date with eta
    
    3. For status questions:
       - Use event_type field to determine if vessels are at sea, in port, at anchor, etc.
       - Use checklist_received or five_day_checklist to check documentation status
    
    4. For counting questions:
       - When asked "how many vessels", filter the data based on criteria first, then count
       - For destination countries, filter by arrival_country
    
    5. For finding specific vessels:
       - Filter by vessel_name or imo_no
       - Return the specific details requested
    
    Always provide concise, accurate responses based exclusively on the data available.
    If the data doesn't contain information to answer a question, clearly state this limitation.
    """
    
    # Add data statistics if available
    if df is not None:
        vessel_count = len(df)
        unique_vessels = df['vessel_name'].nunique() if 'vessel_name' in df.columns else 'unknown'
        
        # Get event type distribution if available
        event_types = {}
        if 'event_type' in df.columns:
            event_types = df['event_type'].value_counts().to_dict()
            event_summary = ", ".join([f"{count} vessels {status.lower()}" for status, count in event_types.items() if pd.notna(status)])
        else:
            event_summary = "Event type data not available"
            
        # Get destination country distribution if available
        destinations = {}
        if 'arrival_country' in df.columns:
            destinations = df['arrival_country'].value_counts().to_dict()
            destination_summary = ", ".join([f"{count} vessels headed to {country}" for country, count in destinations.items() if pd.notna(country)])
        else:
            destination_summary = "Destination data not available"
        
        system_prompt += f"""
        # Current Data Statistics
        - Total records: {vessel_count}
        - Unique vessels: {unique_vessels}
        - Vessel status distribution: {event_summary}
        - Destination distribution: {destination_summary}
        """
    
    return system_prompt

# Function to query OpenAI with the appropriate context
def query_openai(query, df):
    # Create system message with schema definitions
    system_message = create_system_prompt(df)
    
    # Create example data message to show data structure
    # Only send a small sample to avoid token limits
    if df is not None and not df.empty:
        sample_data = df.sample(min(3, len(df))).to_dict(orient='records')
        data_message = f"Here's a sample of the current data structure:\n```json\n{json.dumps(sample_data, indent=2, default=str)}\n```"
    else:
        data_message = "No data available at the moment."
    
    # Combine user query with data context for better understanding
    augmented_query = f"""
    User Query: {query}
    
    Please answer based on the vessel data available. {data_message}
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": augmented_query}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better comprehension
            messages=messages,
            temperature=0.3,  # Lower temperature for more factual answers
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying OpenAI: {str(e)}"

# Define sidebar with metadata
with st.sidebar:
    st.image("https://www.marinetraffic.com/img/ico/apple-touch-icon-152x152.png", width=80)
    st.title("Vessel Assistant")
    st.markdown("---")
    
    st.subheader("About")
    st.write("""
    This application provides insights on vessel data through a chat interface.
    Ask questions about vessels, their status, locations, and more.
    """)
    
    # Add data refresh button
    if st.button("Refresh Data"):
        with st.spinner("Fetching latest vessel data..."):
            vessel_data = fetch_vessel_data()
            if vessel_data:
                st.session_state.vessel_data = vessel_data
                st.session_state.df = pd.DataFrame(vessel_data)
                st.session_state.data_loaded = True
                st.success("Data refreshed successfully!")
            else:
                st.error("Failed to refresh data.")
    
    # Add examples of questions users can ask
    st.markdown("### Example Questions")
    example_questions = [
        "How many vessels are going to Australia?",
        "Which vessels have pending checklists?",
        "What's the status of vessels with IMO number 9234567?",
        "How many vessels are currently at sea?",
        "Which vessel is expected to arrive next?"
    ]
    
    for q in example_questions:
        if st.button(q):
            # When example is clicked, send it as a query
            st.session_state.messages.append({"role": "user", "content": q})
            # Redirect to main area
            st.experimental_rerun()
    
    st.markdown("---")
    st.caption("© 2025 Vessel Data Assistant")

# Main area - Chat interface
st.title("🚢 Vessel Data Chatbot")

# Load data if not already loaded
if not st.session_state.data_loaded:
    with st.spinner("Loading vessel data..."):
        vessel_data = fetch_vessel_data()
        if vessel_data:
            st.session_state.vessel_data = vessel_data
            st.session_state.df = pd.DataFrame(vessel_data)
            st.session_state.data_loaded = True
        else:
            st.error("Failed to load initial data.")

# Display data summary
if st.session_state.data_loaded:
    with st.expander("Data Summary", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Vessels", len(st.session_state.vessel_data))
        
        with col2:
            # Count unique vessel names
            unique_vessels = st.session_state.df['vessel_name'].nunique()
            st.metric("Unique Vessels", unique_vessels)
        
        with col3:
            # Count vessels at sea
            at_sea = len(st.session_state.df[st.session_state.df['event_type'].str.contains('AT SEA', na=False, case=False)])
            st.metric("Vessels At Sea", at_sea)
        
        with col4:
            # Count vessels in port
            in_port = len(st.session_state.df[st.session_state.df['event_type'].str.contains('PORT', na=False, case=False)])
            st.metric("Vessels In Port", in_port)
        
        # Show a sample of the data
        st.dataframe(st.session_state.df.head(5), use_container_width=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask about vessel data..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.write(prompt)
    
    if st.session_state.data_loaded:
        # Get response from OpenAI with dataframe context
        with st.spinner("Analyzing vessel data..."):
            response = query_openai(prompt, st.session_state.df)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
    else:
        with st.chat_message("assistant"):
            st.write("I'm sorry, but I don't have access to vessel data right now. Please try refreshing the data or check your connection.")
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, but I don't have access to vessel data right now. Please try refreshing the data or check your connection."})
