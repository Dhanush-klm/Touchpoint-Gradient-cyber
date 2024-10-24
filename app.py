import streamlit as st
import psycopg2
from openai import OpenAI
import os
from typing import Dict, List, Tuple
from datetime import datetime

# Database connection strings
BIO_DB_CONN = st.secrets["bio_db_connection"]
TP_DB_CONN = st.secrets["tp_db_connection"]

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["api_key"])

class QueryAnalyzer:
    def analyze_query(self, query: str) -> Dict:
        """Analyze the user query to extract company names and query intention"""
        try:
            prompt = f"""
Please analyze this query: "{query}"

Extract and return a JSON object with the following information:
1. Possible company names or partial names mentioned (including variations and abbreviations)
2. The main focus of the query (e.g., contract details, technical setup, meeting history, etc.)
3. The time frame of interest (if mentioned)
4. Any specific data points requested

Format the response as a JSON object with these exact keys:
{{
    "possible_company_names": [], # list of possible company names/variations
    "query_focus": "", # main topic of interest
    "time_frame": "", # time period of interest, if any
    "specific_data_points": [] # list of specific data points requested
}}

For example, if the query is "What did we discuss with First Summit last month?", the response should be:
{{
    "possible_company_names": ["First Summit", "First Summit Bank", "1st Summit"],
    "query_focus": "meeting_discussion",
    "time_frame": "last month",
    "specific_data_points": ["discussion points"]
}}
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            return eval(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing query: {str(e)}")
            return {
                "possible_company_names": [],
                "query_focus": "",
                "time_frame": "",
                "specific_data_points": []
            }

class DatabaseQuerier:
    def __init__(self):
        self.bio_conn = None
        self.tp_conn = None

    def connect_to_databases(self):
        """Create connections to both databases"""
        try:
            self.bio_conn = psycopg2.connect(BIO_DB_CONN)
            self.tp_conn = psycopg2.connect(TP_DB_CONN)
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            return False
        return True

    def close_connections(self):
        """Close database connections"""
        if self.bio_conn:
            self.bio_conn.close()
        if self.tp_conn:
            self.tp_conn.close()

    def get_all_companies(self) -> List[str]:
        """Get a list of all companies without any limit"""
        if not self.bio_conn:
            return []
        
        try:
            with self.bio_conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT company_name 
                    FROM companies 
                    ORDER BY company_name
                """)
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Error fetching companies: {str(e)}")
            return []

    def find_matching_companies(self, possible_names: List[str]) -> List[str]:
        """Find companies in the database that match any of the possible names"""
        if not self.bio_conn:
            return []
        
        matching_companies = []
        try:
            with self.bio_conn.cursor() as cur:
                for name in possible_names:
                    # Create variations of the name for flexible matching
                    search_term = f"%{name}%"
                    cur.execute("""
                        SELECT DISTINCT company_name
                        FROM companies
                        WHERE LOWER(company_name) LIKE LOWER(%s)
                    """, (search_term,))
                    
                    matches = [row[0] for row in cur.fetchall()]
                    matching_companies.extend(matches)
                
                return list(set(matching_companies))  # Remove duplicates
        except Exception as e:
            st.error(f"Error finding matching companies: {str(e)}")
            return []

    def get_company_details(self, company_name: str) -> List[Dict]:
        """Query company details from Bio-db"""
        if not self.bio_conn:
            return []
        
        try:
            with self.bio_conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        c.id,
                        c.company_name,
                        cd.tcv,
                        cd.arr,
                        cd.start_date,
                        cd.end_date,
                        cd.renewal_date,
                        pi.products,
                        pi.deployed_percentage,
                        si.sales_am,
                        si.partner,
                        td.data_retention,
                        td.tp_frequency,
                        i.fw,
                        i.edr,
                        i.switch,
                        i.isps
                    FROM companies c
                    LEFT JOIN contract_details cd ON c.id = cd.company_id
                    LEFT JOIN product_info pi ON c.id = pi.company_id
                    LEFT JOIN sales_info si ON c.id = si.company_id
                    LEFT JOIN technical_details td ON c.id = td.company_id
                    LEFT JOIN infrastructure i ON td.id = i.technical_details_id
                    WHERE c.company_name = %s
                """, (company_name,))
                
                columns = [desc[0] for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            st.error(f"Bio DB query error: {str(e)}")
            return []

    def get_touchpoint_history(self, company_name: str, time_frame: str = None) -> List[Dict]:
        """Query touchpoint history from TP-datas with optional time frame filtering"""
        if not self.tp_conn:
            return []
        
        try:
            with self.tp_conn.cursor() as cur:
                # Base query
                query = """
                    SELECT 
                        t.id,
                        t.meeting_date,
                        t.sentiment
                    FROM touchpoints t
                    WHERE t.company_name = %s
                """
                params = [company_name]

                # Add time frame filter if specified
                if time_frame:
                    if "last month" in time_frame.lower():
                        query += " AND t.meeting_date >= CURRENT_DATE - INTERVAL '1 month'"
                    elif "last year" in time_frame.lower():
                        query += " AND t.meeting_date >= CURRENT_DATE - INTERVAL '1 year'"

                query += " ORDER BY t.meeting_date DESC"
                cur.execute(query, params)
                
                touchpoints = []
                base_touchpoints = cur.fetchall()
                
                for tp_id, meeting_date, sentiment in base_touchpoints:
                    touchpoint_data = {
                        'meeting_date': meeting_date,
                        'sentiment': sentiment,
                        'attendees': {'gradient': [], 'customer': []},
                        'discussion_points': [],
                        'action_items': []
                    }
                    
                    # Get attendees
                    cur.execute("""
                        SELECT name, type
                        FROM attendees
                        WHERE touchpoint_id = %s
                    """, (tp_id,))
                    
                    for name, type_ in cur.fetchall():
                        if type_ == 'gradient':
                            touchpoint_data['attendees']['gradient'].append(name)
                        elif type_ == 'customer':
                            touchpoint_data['attendees']['customer'].append(name)
                    
                    # Get discussion points
                    cur.execute("""
                        SELECT point, type
                        FROM discussion_points
                        WHERE touchpoint_id = %s
                    """, (tp_id,))
                    
                    touchpoint_data['discussion_points'] = [
                        {'point': point, 'type': type_}
                        for point, type_ in cur.fetchall()
                    ]
                    
                    # Get action items
                    cur.execute("""
                        SELECT item
                        FROM action_items
                        WHERE touchpoint_id = %s
                    """, (tp_id,))
                    
                    touchpoint_data['action_items'] = [item[0] for item in cur.fetchall()]
                    
                    touchpoints.append(touchpoint_data)
                
                return touchpoints
        except Exception as e:
            st.error(f"TP DB query error: {str(e)}")
            return []

def get_company_clarification(companies: List[str]) -> str:
    """Get clarification from LLM when no company is specified"""
    return "Could you please specify which company you're interested in?"

def process_query(query: str, selected_company: str = None) -> Tuple[List[Dict], List[Dict], Dict]:
    """Process a natural language query and return relevant data"""
    # Initialize components
    analyzer = QueryAnalyzer()
    querier = DatabaseQuerier()
    
    # Connect to databases
    if not querier.connect_to_databases():
        return [], [], {}
    
    try:
        # If a company was selected from follow-up, add it to the query
        if selected_company:
            query = f"{query} for {selected_company}"

        # Analyze the query
        analysis = analyzer.analyze_query(query)
        
        # If no company is mentioned and none was selected
        if not analysis['possible_company_names'] and not selected_company:
            companies = querier.get_all_companies()
            return [], [], {
                "needs_clarification": True,
                "available_companies": companies,
                "original_query": query
            }
        
        # Find matching companies
        matching_companies = querier.find_matching_companies(analysis['possible_company_names'])
        
        if not matching_companies:
            return [], [], analysis
        
        company_name = matching_companies[0]
        company_data = querier.get_company_details(company_name)
        touchpoint_data = querier.get_touchpoint_history(company_name, analysis['time_frame'])
        
        return company_data, touchpoint_data, analysis
        
    finally:
        querier.close_connections()

def format_data_for_llm(company_data: List[Dict], touchpoint_data: List[Dict]) -> str:
    """Format the combined data for LLM consumption"""
    formatted_text = ""
    
    if company_data:
        company = company_data[0]
        formatted_text += f"""
COMPANY PROFILE:
Company: {company.get('company_name')}

Contract Information:
- Total Contract Value: {company.get('tcv', 'N/A')}
- Annual Recurring Revenue: {company.get('arr', 'N/A')}
- Contract Period: {company.get('start_date', 'N/A')} to {company.get('end_date', 'N/A')}
- Renewal Date: {company.get('renewal_date', 'N/A')}

Products and Deployment:
- Products: {company.get('products', 'N/A')}
- Deployment Status: {company.get('deployed_percentage', 'N/A')}

Sales Details:
- Account Manager: {company.get('sales_am', 'N/A')}
- Partner: {company.get('partner', 'N/A')}

Technical Configuration:
- Data Retention: {company.get('data_retention', 'N/A')}
- Touchpoint Frequency: {company.get('tp_frequency', 'N/A')}

Infrastructure:
- Firewall: {company.get('fw', 'N/A')}
- EDR Solution: {company.get('edr', 'N/A')}
- Network Switch: {company.get('switch', 'N/A')}
- ISP Details: {company.get('isps', 'N/A')}
"""

    if touchpoint_data:
        formatted_text += "\nMEETING HISTORY:\n"
        for tp in touchpoint_data:
            formatted_text += f"""
Date: {tp.get('meeting_date')}
Overall Sentiment: {tp.get('sentiment', 'N/A')}

Attendees:
- Gradient: {', '.join(tp['attendees']['gradient']) if tp['attendees']['gradient'] else 'N/A'}
- Customer: {', '.join(tp['attendees']['customer']) if tp['attendees']['customer'] else 'N/A'}

Discussion Points:
{chr(10).join(['- ' + point['point'] for point in tp['discussion_points']]) if tp['discussion_points'] else 'N/A'}

Action Items:
{chr(10).join(['- ' + item for item in tp['action_items']]) if tp['action_items'] else 'N/A'}
---
"""

    return formatted_text

def get_llm_response(query: str, data: str, system_instruction: str) -> str:
    try:
        prompt = f"""
{system_instruction}

Query: {query}

Available Data:
{data}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_instruction},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def get_system_instruction():
    if 'system_instruction' not in st.session_state:
        st.session_state.system_instruction = """You are a professional business analyst with access to company data and meeting history. Please analyze the following query and provide insights based on the available data:

1. Answer for the query using specific data points
2. Identify any relevant trends or patterns
3. Highlight important observations about the company's status
4. Note any potential areas of attention if relevant
5. Mention if any critical information is missing to fully answer the query

Format your response professionally and support your analysis with specific data points from the provided information."""
    
    return st.session_state.system_instruction

def main():
    st.set_page_config(page_title="Gradient Intelligence", layout="wide")
    
    # Sidebar for system instruction
    with st.sidebar:
        st.title("System Settings")
        system_instruction = st.text_area(
            "Edit System Instruction",
            value=get_system_instruction(),
            height=300
        )
        if st.button("Update System Instruction"):
            st.session_state.system_instruction = system_instruction
            st.success("System instruction updated successfully!")

    st.title("🔍 Touchpoint Data Insights")
    st.markdown("""
    Ask questions about company details, technical setup, or meeting history.
    The system will guide you through the process.
    """)
    
    # Initialize session state
    if 'awaiting_company' not in st.session_state:
        st.session_state.awaiting_company = False
    if 'original_query' not in st.session_state:
        st.session_state.original_query = None
    if 'available_companies' not in st.session_state:
        st.session_state.available_companies = []
    if 'selected_company' not in st.session_state:
        st.session_state.selected_company = None

    # Query input
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., 'What was discussed in the last meeting?' or 'Show me the technical setup'"
    )

    if query:
        # Reset state if it's a new query
        if query != st.session_state.original_query:
            st.session_state.awaiting_company = False
            st.session_state.selected_company = None

        with st.spinner("Processing your query..."):
            # Process the query
            company_data, touchpoint_data, analysis = process_query(query, st.session_state.selected_company)

            # Handle case where we need company clarification
            if analysis.get('needs_clarification'):
                st.session_state.awaiting_company = True
                st.session_state.original_query = query
                st.session_state.available_companies = analysis['available_companies']

                # Get and display clarification message
                clarification_msg = get_company_clarification(analysis['available_companies'])
                st.info(clarification_msg)

                # Company selection dropdown
                selected_company = st.selectbox(
                    "Select a company:",
                    options=analysis['available_companies']
                )

                if st.button("Proceed with Analysis"):
                    st.session_state.selected_company = selected_company
                    # Rerun the query with selected company
                    company_data, touchpoint_data, analysis = process_query(
                        st.session_state.original_query,
                        selected_company
                    )
                    st.session_state.awaiting_company = False

            # Display results if we have company data
            if company_data:
                # Format data and get LLM response
                formatted_data = format_data_for_llm(company_data, touchpoint_data)
                response = get_llm_response(query, formatted_data, get_system_instruction())

                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Analysis", "Company Details", "Meeting History"])

                with tab1:
                    st.markdown("### AI Analysis")
                    st.write(response)

                    # Quick metrics
                    if company_data:
                        company = company_data[0]
                        st.markdown("### Key Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Contract Value", company.get('tcv', 'N/A'))
                        with col2:
                            st.metric("ARR", company.get('arr', 'N/A'))
                        with col3:
                            st.metric("Deployment %", company.get('deployed_percentage', 'N/A'))

                with tab2:
                    st.markdown("### Company Information")
                    company = company_data[0]
                    
                    # Contract Details
                    st.subheader("Contract Details")
                    contract_col1, contract_col2 = st.columns(2)
                    with contract_col1:
                        st.write("**Contract Period:**", f"{company.get('start_date', 'N/A')} to {company.get('end_date', 'N/A')}")
                        st.write("**Renewal Date:**", company.get('renewal_date', 'N/A'))
                    with contract_col2:
                        st.write("**Products:**", company.get('products', 'N/A'))
                        st.write("**Deployment Status:**", company.get('deployed_percentage', 'N/A'))

                    # Technical Setup
                    st.subheader("Technical Configuration")
                    tech_col1, tech_col2 = st.columns(2)
                    with tech_col1:
                        st.write("**Data Retention:**", company.get('data_retention', 'N/A'))
                        st.write("**Touchpoint Frequency:**", company.get('tp_frequency', 'N/A'))
                    with tech_col2:
                        st.write("**Firewall:**", company.get('fw', 'N/A'))
                        st.write("**EDR Solution:**", company.get('edr', 'N/A'))

                    # Sales Information
                    st.subheader("Sales Information")
                    sales_col1, sales_col2 = st.columns(2)
                    with sales_col1:
                        st.write("**Account Manager:**", company.get('sales_am', 'N/A'))
                    with sales_col2:
                        st.write("**Partner:**", company.get('partner', 'N/A'))

                with tab3:
                    st.markdown("### Meeting History")
                    if touchpoint_data:
                        for tp in touchpoint_data:
                            with st.expander(
                                f"Meeting on {tp['meeting_date']} | Sentiment: {tp['sentiment']}"
                            ):
                                # Attendees
                                st.write("**Attendees**")
                                attendees_col1, attendees_col2 = st.columns(2)
                                with attendees_col1:
                                    st.write("Gradient Team:")
                                    for attendee in tp['attendees']['gradient']:
                                        st.write(f"- {attendee}")
                                with attendees_col2:
                                    st.write("Customer Team:")
                                    for attendee in tp['attendees']['customer']:
                                        st.write(f"- {attendee}")

                                # Discussion Points
                                st.write("**Discussion Points**")
                                for point in tp['discussion_points']:
                                    st.write(f"- {point['point']} ({point['type']})")

                                # Action Items
                                if tp['action_items']:
                                    st.write("**Action Items**")
                                    for item in tp['action_items']:
                                        st.write(f"- {item}")
                    else:
                        st.info("No meeting history available for the specified timeframe.")

            elif not st.session_state.awaiting_company:
                st.warning("No data found matching your query. Please try rephrasing or selecting a different company.")

if __name__ == "__main__":
    main()

