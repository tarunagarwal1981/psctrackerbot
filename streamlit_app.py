import streamlit as st
import requests
import pandas as pd
import json
import os
from openai import OpenAI
from datetime import datetime

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

# Preprocess data to extract key statistics and findings
def analyze_data(df):
    analysis = {}
    
    # Fill missing values with appropriate placeholders for better analysis
    df_clean = df.copy()
    
    # Basic vessel counts
    analysis['total_vessels'] = len(df_clean)
    analysis['unique_vessels'] = df_clean['vessel_name'].nunique()
    
    # Event type analysis
    if 'event_type' in df_clean.columns:
        event_counts = df_clean['event_type'].value_counts().to_dict()
        analysis['event_counts'] = event_counts
        
        # Count vessels at sea
        at_sea_pattern = 'sea|transit|passage'
        analysis['vessels_at_sea'] = df_clean['event_type'].str.contains(
            at_sea_pattern, case=False, na=False).sum()
        
        # Count vessels in port
        in_port_pattern = 'port|berth|dock'
        analysis['vessels_in_port'] = df_clean['event_type'].str.contains(
            in_port_pattern, case=False, na=False).sum()
        
        # Count vessels at anchor
        at_anchor_pattern = 'anchor'
        analysis['vessels_at_anchor'] = df_clean['event_type'].str.contains(
            at_anchor_pattern, case=False, na=False).sum()
    
    # Destination country analysis
    if 'arrival_country' in df_clean.columns:
        country_counts = df_clean['arrival_country'].value_counts().to_dict()
        analysis['destination_countries'] = country_counts
        
        # Top destinations
        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis['top_destinations'] = top_countries
    
    # Origin country analysis
    if 'departure_country' in df_clean.columns:
        origin_counts = df_clean['departure_country'].value_counts().to_dict()
        analysis['origin_countries'] = origin_counts
    
    # Flag analysis
    for flag_col in ['port_match_flag', 'multi_port_flag', 'eta_check_flag']:
        if flag_col in df_clean.columns:
            analysis[f'{flag_col}_count'] = df_clean[flag_col].sum()
    
    # Checklist analysis
    if 'five_day_checklist' in df_clean.columns:
        checklist_counts = df_clean['five_day_checklist'].value_counts().to_dict()
        analysis['checklist_status'] = checklist_counts
        
        # Count pending checklists
        pending_pattern = 'pending|outstanding|not received'
        analysis['pending_checklists'] = df_clean['five_day_checklist'].str.contains(
            pending_pattern, case=False, na=True).sum()
    
    # Country-specific counts (for common queries)
    for country, code in [('Australia', 'AU'), ('New Zealand', 'NZ'), 
                          ('Singapore', 'SG'), ('Indonesia', 'ID')]:
        # Vessels going to country
        if 'arrival_country' in df_clean.columns:
            going_to = df_clean['arrival_country'].str.contains(
                f'^{code}$', case=False, na=False).sum()
            analysis[f'vessels_going_to_{country.lower()}'] = going_to
        
        # Vessels in country ports
        if 'arrival_country' in df_clean.columns and 'event_type' in df_clean.columns:
            in_port_mask = df_clean['event_type'].str.contains(
                'port|berth|dock', case=False, na=False)
            country_mask = df_clean['arrival_country'].str.contains(
                f'^{code}$', case=False, na=False)
            
            in_country_ports = (in_port_mask & country_mask).sum()
            analysis[f'vessels_in_{country.lower()}_ports'] = in_country_ports
    
    return analysis

# Create a detailed system prompt with column definitions and analysis
def create_system_prompt(df=None, analysis=None):
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    system_prompt = f"""
    You are a specialized vessel data analyst assistant that helps maritime professionals get insights from vessel tracking data.
    Today's date is {current_date}.
    
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
    - five_day_checklist: Status of the 5-day checklist (e.g., "Received", "Pending")
    - checklist_received: Alternative field for five_day_checklist status
    - sanz: SANZ certification status
    - BUILT_DATE: Date when the vessel was built
    - vessel_type: Type/category of the vessel
    """
    
    # Add pre-calculated analysis if available
    if analysis:
        system_prompt += "\n# Pre-calculated Data Analysis\n"
        
        # Format the analysis as bullet points
        for key, value in analysis.items():
            if isinstance(value, dict):
                system_prompt += f"- {key.replace('_', ' ').title()}:\n"
                for subkey, subvalue in value.items():
                    if subkey and subvalue:
                        system_prompt += f"  - {subkey}: {subvalue}\n"
            else:
                system_prompt += f"- {key.replace('_', ' ').title()}: {value}\n"
    
    system_prompt += """
    # IMPORTANT INSTRUCTIONS

    - Always provide direct answers based on the data. Don't say "I would filter by..." or "You would need to look at...".
    - For questions about specific countries, look at arrival_country and departure_country fields using the two-letter country codes.
    - For vessels "in" a country, check if arrival_country matches AND event_type contains "port", "berth", or "dock".
    - For vessels "going to" a country, check if arrival_country matches.
    - When asked about vessels in a specific status, filter the event_type field accordingly.
    - If a question has a clear numeric answer, start your response with the number.
    - Include specific vessel names when appropriate in your answers.
    - If the data doesn't contain enough information to answer precisely, say so clearly.
    - Answer questions using a confident, direct tone. You have the data and can give accurate answers.
    """
    
    return system_prompt

# Function to query OpenAI with the appropriate context
def query_openai(query, df):
    try:
        # Extract key statistics and pre-analyze data
        if df is not None and not df.empty:
            analysis = analyze_data(df)
        else:
            analysis = None
        
        # Create system message with schema definitions and analysis
        system_message = create_system_prompt(df, analysis)
        
        # Create example data message to show data structure
        if df is not None and not df.empty:
            # Extract a specific sample relevant to common queries
            # Sample 1: Vessels going to Australia
            aus_df = df[df['arrival_country'] == 'AU'].head(2) if 'arrival_country' in df.columns else pd.DataFrame()
            
            # Sample 2: Vessels in port
            in_port_mask = df['event_type'].str.contains('port|berth', case=False, na=False) if 'event_type' in df.columns else pd.Series([False] * len(df))
            port_df = df[in_port_mask].head(2)
            
            # Sample 3: Vessels with pending checklists
            if 'five_day_checklist' in df.columns:
                pending_mask = df['five_day_checklist'].str.contains('pending', case=False, na=False)
                checklist_df = df[pending_mask].head(2)
            else:
                checklist_df = pd.DataFrame()
            
            # Combine samples
            sample_dfs = [aus_df, port_df, checklist_df]
            sample_rows = []
            for sample_df in sample_dfs:
                if not sample_df.empty:
                    for _, row in sample_df.iterrows():
                        if row.to_dict() not in sample_rows:  # Avoid duplicates
                            sample_rows.append(row.to_dict())
            
            # Limit to 5 samples max
            sample_rows = sample_rows[:5]
            
            if sample_rows:
                data_message = f"Here are some relevant sample vessel records from the current data:\n```json\n{json.dumps(sample_rows, indent=2, default=str)}\n```"
            else:
                data_message = "No relevant sample data available at the moment."
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
        
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for better comprehension
            messages=messages,
            temperature=0.2,  # Lower temperature for more factual answers
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing your query: {str(e)}"

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
        "How many vessels are currently in New Zealand ports?",
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
            at_sea = len(st.session_state.df[st.session_state.df['event_type'].str.contains('AT SEA|TRANSIT|PASSAGE', na=False, case=False)])
            st.metric("Vessels At Sea", at_sea)
        
        with col4:
            # Count vessels in port
            in_port = len(st.session_state.df[st.session_state.df['event_type'].str.contains('PORT|BERTH|DOCK', na=False, case=False)])
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
