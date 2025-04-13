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

# Fetch OpenAI API key from Streamlit secrets
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

# Function to prepare context for OpenAI
def prepare_context(data):
    # Create a summary of what data is available
    context = """
    I have access to vessel data with the following fields:
    - id: unique identifier for the vessel record
    - vessel_name: name of the vessel
    - imo_no: IMO number (unique vessel identifier)
    - comments: any comments about the vessel
    - event_type: current status like 'AT SEA', 'IN PORT', 'AT ANCHOR'
    - owner: vessel owner/company
    - report_date: date of the report
    - departure_port: port of departure
    - departure_country: country of departure
    - departure_date: date of departure
    - arrival_port: port of arrival
    - arrival_country: country of arrival
    - eta: estimated time of arrival
    - etb: estimated time of berthing
    - etd: estimated time of departure
    - atd: actual time of departure
    - lat: latitude of current position
    - lon: longitude of current position
    - speed: current speed
    - heading: current heading
    - course: current course
    - distance_to_go: distance remaining to destination
    - five_day_checklist: checklist status ("Received", "Pending", etc.)
    - checklist_received: alternative name for five_day_checklist
    - sanz: SANZ status information
    - vessel_type: type of vessel
    - fleet_type: fleet type classification
    - BUILT_DATE: date the vessel was built
    - psc_last_inspection_date: date of last Port State Control inspection
    - psc_last_inspection_port: port of last PSC inspection
    - amsa_last_inspection_date: date of last AMSA inspection
    - amsa_last_inspection_port: port of last AMSA inspection
    """
    
    # Add some sample data summary
    if data:
        vessel_count = len(data)
        vessel_types = set([v.get('vessel_type', 'Unknown') for v in data if v.get('vessel_type')])
        event_types = set([v.get('event_type', 'Unknown') for v in data if v.get('event_type')])
        
        context += f"""
        The dataset contains {vessel_count} vessel records.
        Vessel types include: {', '.join(vessel_types)}
        Event types include: {', '.join(event_types)}
        """
    
    return context

# Function to query OpenAI
def query_openai(query, context, sample_data):
    # Create a system message with context about the data
    system_message = f"""You are a vessel data assistant that helps users get information about vessels.
    {context}
    
    Based on the available data, answer the user's questions precisely and concisely.
    If you don't know the answer or if it requires data that's not available, be honest about it.
    When appropriate, include specific vessel names and numerical data in your responses.
    """
    
    # Include some sample data to help the model understand the structure
    sample_json = json.dumps(sample_data[:2], indent=2)
    system_message += f"\nHere's a sample of the data structure for reference (first 2 records):\n{sample_json}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # You can change this to a different model
            messages=messages,
            temperature=0.5,
            max_tokens=500
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
        # Prepare context for OpenAI
        context = prepare_context(st.session_state.vessel_data)
        
        # Get response from OpenAI
        with st.spinner("Thinking..."):
            response = query_openai(prompt, context, st.session_state.vessel_data)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
    else:
        with st.chat_message("assistant"):
            st.write("I'm sorry, but I don't have access to vessel data right now. Please try refreshing the data or check your connection.")
            st.session_state.messages.append({"role": "assistant", "content": "I'm sorry, but I don't have access to vessel data right now. Please try refreshing the data or check your connection."})
