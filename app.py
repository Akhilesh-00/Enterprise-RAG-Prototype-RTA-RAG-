import os
import io
import json
import re
import base64
import textwrap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import google.generativeai as genai
from flask import Flask, request, jsonify

# Use a non-GUI backend for Matplotlib
matplotlib.use('Agg')

# --- 1. CONFIGURATION ---
# IMPORTANT: Replace with your actual API key
genai.configure(api_key="AIzaSyDK0_A0qcZRtWeTY3CMK8Ntn10295dWTas")
LLM_MODEL = "gemini-1.5-pro-latest"
DATA_FOLDER = "./dataset"
RIDERSHIP_FILE_PATH = r"D:\rta_rag\dataset\ridership.csv"
MONTH_FILES = {
    "June": "june_2025_data_full.xlsx",
    "July": "july_2025_data_full.xlsx",
    "August": "august_2025_data_full.xlsx"
}

# --- 2. INITIALIZE MODELS & LOAD DATA ---
print("⏳ Initializing models...")
llm = genai.GenerativeModel(LLM_MODEL)
print("✅ Models initialized.")

def load_and_prepare_data():
    all_dfs = []
    print("   -> Loading Excel files with station data...")
    for month, file_name in MONTH_FILES.items():
        file_path = os.path.join(DATA_FOLDER, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df.rename(columns={'station_name': 'station', 'transport_mode': 'mode'}, inplace=True)
            df['month'] = month
            all_dfs.append(df[['station', 'mode', 'trips', 'month']])
            print(f"       - Loaded {file_name}")

    print(f"   -> Loading ridership file ({os.path.basename(RIDERSHIP_FILE_PATH)})...")
    if os.path.exists(RIDERSHIP_FILE_PATH):
        try:
            ridership_df = pd.read_excel(RIDERSHIP_FILE_PATH)
            columns_to_drop = ['Metro']
            ridership_df.drop(columns=[col for col in columns_to_drop if col in ridership_df.columns], inplace=True, errors='ignore')
            ridership_df['date'] = pd.to_datetime(ridership_df['date'])
            ridership_df['month'] = ridership_df['date'].dt.strftime('%B')
            id_vars = ['date', 'month']
            value_vars = [col for col in ridership_df.columns if col not in id_vars]
            long_df = pd.melt(ridership_df, id_vars=id_vars, value_vars=value_vars, var_name='mode', value_name='trips')
            long_df['station'] = 'Citywide'
            all_dfs.append(long_df[['station', 'mode', 'trips', 'month']])
            print(f"       - Successfully loaded and processed ridership data.")
        except Exception as e:
            print(f"       - ERROR: Failed to read the ridership file. Error: {e}")
    else:
        print(f"       - Warning: Ridership file not found at {RIDERSHIP_FILE_PATH}, skipping.")

    if not all_dfs:
        raise FileNotFoundError("No data files were loaded. Please check your file paths.")

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df['month'] = master_df['month'].astype(str).str.strip()
    master_df['mode'] = master_df['mode'].astype(str).str.strip()
    master_df['station'] = master_df['station'].astype(str).str.strip()
    master_df['trips'] = pd.to_numeric(master_df['trips'], errors='coerce').fillna(0).astype(int)

    print(f"✅ Master DataFrame created with {len(master_df)} records from all sources.")
    return master_df

df = load_and_prepare_data()

# --- UTILITY FUNCTION ---
def extract_json_from_string(text):
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match: json_str = match.group(1)
    else:
        start = text.find('{'); end = text.rfind('}')
        if start != -1 and end != -1 and end > start: json_str = text[start:end+1]
        else: raise json.JSONDecodeError("No valid JSON object found.", text, 0)
    return json.loads(json_str)

# --- 3-STEP LOGIC ---
def create_query_plan(question):
    prompt = f"""
Analyze the user's question and convert it into a structured JSON plan.
- Your response MUST be a valid JSON object.
- All filters (like month) MUST be in a nested "filters" object.
- The "metric" key MUST always be "trips".
Valid "operation" values: 'sum', 'average', 'rank_modes', 'compare_modes', 'compare_average_by_mode', 'compare_months', 'rank_stations', 'compare_stations'.
Example: "busiest stations in june" -> {{"operation": "rank_stations", "filters": {{"month": "June"}}, "rank": "highest", "metric": "trips"}}
Question: "{question}"
Response:
"""
    response = llm.generate_content(prompt)
    return extract_json_from_string(response.text)

# --- Updated execute_pandas_query ---
def execute_pandas_query(plan, dataframe, question_text):
    question_lower = question_text.lower()

    # --- 1. Red/Green Line Metro restriction for June, July, August ---
    months_jja = ['june', 'july', 'august']
    if any(month in question_lower for month in months_jja) and \
       ('red line' in question_lower or 'green line' in question_lower):
        return {"error": "Red Line Metro and Green Line Metro data unavailable for the months June, July, and August."}

    # --- 2. Station-level restriction for Feb, Mar, Apr, May ---
    months_feb_may = ['feb', 'february', 'mar', 'march', 'apr', 'april', 'may']
    if any(month in question_lower for month in months_feb_may) and \
       ('station' in question_lower or 'stations' in question_lower):
        return {"error": "Station data unavailable for the months Feb, March, April, and May."}

    # --- Original query logic below ---
    operation = plan.get('operation')
    if not operation:
        return {"error": "I could not determine the main goal of your question. Please rephrase it."}

    filters = plan.get('filters', {})
    if not isinstance(filters, dict):
        return {"error": "Cannot process complex questions."}

    filtered_df = dataframe.copy()
    for key, value in filters.items():
        if key in filtered_df.columns:
            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[key].str.lower().isin([v.lower() for v in value])]
            else:
                filtered_df = filtered_df[filtered_df[key].str.lower() == str(value).lower()]

    if filtered_df.empty:
        return {"error": "No data found for the specified filters. Please broaden your query."}

    metric = 'trips'

    if operation in ['rank_stations', 'compare_stations']:
        station_data = filtered_df[filtered_df['station'] != 'Citywide']
        if station_data.empty:
            return {"error": "I don't have station data for these months. I can only provide city-wide totals for that period."}

        grouped = station_data.groupby('station')[metric].sum()
        if operation == 'rank_stations':
            rank_type = plan.get('rank', 'highest')
            ranked = grouped.nlargest(5) if rank_type == 'highest' else grouped.nsmallest(5)
            return {"result": ranked.to_dict()}
        else:
            return {"result": grouped.to_dict()}

    if operation == 'sum': return {"result": int(filtered_df[metric].sum())}
    if operation == 'average': return {"result": float(filtered_df[metric].mean())}
    if operation == 'rank_modes':
        grouped = filtered_df.groupby('mode')[metric].sum()
        rank_type = plan.get('rank', 'highest')
        ranked = grouped.nlargest(5) if rank_type == 'highest' else grouped.nsmallest(5)
        return {"result": ranked.to_dict()}
    if operation in ['compare_modes', 'compare_average_by_mode']:
        grouped = filtered_df.groupby('mode')[metric]
        result = grouped.sum() if operation == 'compare_modes' else grouped.mean()
        return {"result": result.apply(float).to_dict()}
    if operation == 'compare_months':
        grouped = filtered_df.groupby('month')[metric].sum()
        if 'month' in filters and isinstance(filters['month'], list):
            month_order = [m.capitalize() for m in filters['month']]
            return {"result": {month: int(grouped.get(month, 0)) for month in month_order}}
        return {"result": grouped.apply(int).to_dict()}

    return {"error": f"Unsupported operation: '{operation}'."}

def synthesize_answer(question, plan, calculation_result):
    prompt = f"""
You are a helpful data analyst. Convert the user's question and pre-calculated data into a natural language summary and chart data.
- Your response MUST be a valid JSON object and NOTHING else.
- You MUST ALWAYS generate a `chart_data` object, even for single-number results.
- CRUCIALLY: When mentioning any numbers in your summary, you MUST write out the full number with commas (e.g., 23,900,123) and not use abbreviations like 'million'.
User's Question: {question}
Query Plan: {json.dumps(plan)}
Calculation Result: {json.dumps(calculation_result)}
Example 1 (Multiple Values):
Calculation Result: {{"result": {{"Metro": 23900123, "Bus": 13400567}}}}
Response:
{{"summary": "In April, the total number of Metro trips was 23,900,123, while the total for Bus was 13,400,567.", "chart_data": {{"title": "Total Trips in April (Metro vs. Bus)", "labels": ["Metro", "Bus"], "values": [23900123, 13400567]}}}}
Example 2 (Single Value):
Calculation Result: {{"result": 13400567}}
Response:
{{"summary": "The total number of trips was 13,400,567.", "chart_data": {{"title": "Total Trips", "labels": ["Total"], "values": [13400567]}}}}
Now, generate the response for the provided data, following all rules.
Response:
"""
    generation_config = genai.types.GenerationConfig(temperature=0)
    response = llm.generate_content(prompt, generation_config=generation_config)
    return extract_json_from_string(response.text)

def create_chart(chart_data):
    if not chart_data or not chart_data.get('labels') or not chart_data.get('values'): return None
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    raw_labels = chart_data['labels']
    labels = [textwrap.fill(str(label).replace('_', ' ').title(), 15) for label in raw_labels]
    values = chart_data['values']
    if len(labels) > 1:
        sorted_pairs = sorted(zip(labels, values), key=lambda item: item[1], reverse=True)
        labels, values = zip(*sorted_pairs)
    bars = ax.bar(labels, values, color=['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'])
    ax.set_ylabel('Total Trips', fontsize=14, color='white', labelpad=15)
    ax.set_title(chart_data.get('title', 'Analysis Chart'), fontsize=18, fontweight='bold', color='white', pad=20)
    plt.xticks(rotation=30, ha='right', fontsize=12); plt.yticks(fontsize=12)
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    ax.set_facecolor('#1F2937'); fig.patch.set_facecolor('#111827')
    ax.grid(axis='y', linestyle='--', alpha=0.2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4B5563'); ax.spines['bottom'].set_color('#4B5563')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if values: ax.set_ylim(0, max(values) * 1.15)
    for bar in bars:
        yval = bar.get_height()
        offset = max(values) * 0.01 if values else 0
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + offset,
                f'{int(yval):,}', va='bottom', ha='center',
                fontsize=11, fontweight='bold', color='white')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# --- 5. FLASK APP ---
app = Flask(__name__)

# --- Updated /ask route ---
@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    print(f"\n🤔 Received Question: {user_question}")

    # --- IMMEDIATE BLOCKS BASED ON MONTHS ---
    question_lower = user_question.lower()

    # 1️⃣ Red/Green Line Metro for June, July, August
    if ('red line' in question_lower or 'green line' in question_lower) and \
       any(m in question_lower for m in ['june','july','august']):
        return jsonify({
            "summary": "Red Line Metro and Green Line Metro data unavailable for the months June, July, and August.",
            "graph": None
        })

    # 2️⃣ Station-level data for Feb, Mar, Apr, May
    if ('station' in question_lower or 'stations' in question_lower) and \
       any(m in question_lower for m in ['feb','march','mar','april','apr','may']):
        return jsonify({
            "summary": "Station data unavailable for the months Feb, March, April, and May.",
            "graph": None
        })
    # --- END BLOCKS ---

    # All other questions continue normally
    try:
        plan = create_query_plan(user_question)
        print(f"   📊 Plan: {plan}")

        calculation_result = execute_pandas_query(plan, df, user_question)
        print(f"   🔢 Result: {calculation_result}")

        if "error" in calculation_result:
            return jsonify({"summary": calculation_result["error"], "graph": None})

        final_result = synthesize_answer(user_question, plan, calculation_result)
        chart_image = create_chart(final_result.get('chart_data'))
        print("✅ Responded successfully.")
        return jsonify({"summary": final_result.get('summary'), "graph": chart_image})

    except json.JSONDecodeError:
        return jsonify({"summary": "I had trouble understanding that. Could you rephrase?", "graph": None})
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e}")
        return jsonify({"error": "An internal error occurred. See server logs."}), 500

# --- 6. STARTUP LOGIC ---
if __name__ == "__main__":
    print("🚀 Starting Flask API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
