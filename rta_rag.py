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
genai.configure(api_key="AIzaSyDK0_A0qcZRtWeTY3CMK8Ntn10295dWTas")
LLM_MODEL = "gemini-1.5-pro-latest"
DATA_FOLDER = "./dataset"
RIDERSHIP_FILE_PATH = r"D:\rta_rag\dataset\ridership.xlsx"
MONTH_FILES = {}



# --- 2. INITIALIZE MODELS & LOAD DATA ---
print("⏳ Initializing models...")
llm = genai.GenerativeModel(LLM_MODEL)
print("✅ Models initialized.")



MONTH_MAP = {
    'jan':'january','feb':'february','mar':'march','apr':'april','may':'may',
    'jun':'june','jul':'july','aug':'august','sep':'september','oct':'october',
    'nov':'november','dec':'december'
}



def normalize_month(m):
    m = str(m).strip().lower()[:3]
    return MONTH_MAP.get(m, m)



def load_and_prepare_data():
    all_dfs = []

    print("   -> Loading Excel files with station data...")
    for month, file_name in MONTH_FILES.items():
        file_path = os.path.join(DATA_FOLDER, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            df.rename(columns={'station_name': 'station', 'transport_mode': 'mode'}, inplace=True)
            df['month'] = normalize_month(month)
            all_dfs.append(df[['station', 'mode', 'trips', 'month']])
            print(f"       - Loaded {file_name}")

    print(f"   -> Loading ridership file ({os.path.basename(RIDERSHIP_FILE_PATH)})...")
    if os.path.exists(RIDERSHIP_FILE_PATH):
        try:
            ridership_df = pd.read_excel(RIDERSHIP_FILE_PATH)
            ridership_df['date'] = pd.to_datetime(ridership_df['date'], dayfirst=False, errors='coerce')
            ridership_df = ridership_df.dropna(subset=['date']).copy()
            ridership_df['month'] = ridership_df['date'].dt.strftime('%B').str.lower()
            ridership_df['station'] = 'Citywide'

            mode_columns = ['Metro Red Line','Metro Green Line','Tram','Bus','Marine','E-Hail','Car-Share','Bus on Demand','Taxi','Metro']
            existing_modes = [col for col in mode_columns if col in ridership_df.columns]

            if existing_modes:
                id_vars = [c for c in ridership_df.columns if c not in existing_modes]
                ridership_df = ridership_df.melt(
                    id_vars=id_vars,
                    value_vars=existing_modes,
                    var_name='mode',
                    value_name='trips'
                )
                ridership_df['trips'] = pd.to_numeric(ridership_df['trips'], errors='coerce').fillna(0).astype(int)

            all_dfs.append(ridership_df)
            print(f"       - Successfully loaded ridership data.")
        except Exception as e:
            print(f"       - ERROR: Failed to read the ridership file. Error: {e}")
    else:
        print(f"       - Warning: Ridership file not found at {RIDERSHIP_FILE_PATH}, skipping.")

    if not all_dfs:
        raise FileNotFoundError("No data files were loaded. Please check your file paths.")

    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df['month'] = master_df['month'].apply(normalize_month)
    master_df['station'] = master_df['station'].astype(str).str.strip()
    if 'trips' in master_df.columns:
        master_df['trips'] = pd.to_numeric(master_df['trips'], errors='coerce').fillna(0).astype(int)
    else:
        master_df['trips'] = 0

    return master_df



df = load_and_prepare_data()



# --- 3. UTILITY FUNCTIONS ---
def extract_json_from_string(text):
    match = re.search(r"``````", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = text.find('{'); end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end+1]
        else:
            raise json.JSONDecodeError("No valid JSON object found.", text, 0)
    return json.loads(json_str)



def create_query_plan(question):
    prompt = f"""
Analyze the user's question and convert it into a structured JSON plan.
- Your response MUST be a valid JSON object.
- All filters (like month) MUST be in a nested "filters" object.
- The "metric" key MUST always be "trips".
Question: "{question}"
Response:
"""
    response = llm.generate_content(prompt)
    try:
        return extract_json_from_string(response.text)
    except:
        # fallback default
        return {"operation": "sum", "filters": {}}



def execute_pandas_query(plan, dataframe, question_text):
    question_lower = question_text.lower()

    # --- BLOCK JUNE/JULY/AUGUST ---
    if any(m in question_lower for m in ['june','july','august']):
        return {"error": "Data unavailable for the months June, July, and August."}

    # --- BLOCK STATION-LEVEL DATA FOR FEB-MAY ---
    if ('station' in question_lower or 'stations' in question_lower) and \
       any(m in question_lower for m in ['feb','march','mar','april','apr','may']):
        return {"error": "Station data unavailable for the months Feb, March, April, and May."}

    operation = plan.get('operation') or 'sum'
    filters = plan.get('filters', {}).copy()
    filtered_df = dataframe.copy()

    # --- Ensure month filter from text if planner omitted it ---
    month_keywords = {
        'january':'jan','february':'feb','march':'mar','april':'apr','may':'may',
        'june':'jun','july':'jul','august':'aug','september':'sep','october':'oct',
        'november':'nov','december':'dec'
    }
    if 'month' not in filters:
        q_words = re.findall(r"\b[a-z]+\b", question_lower)
        for full, abbr in month_keywords.items():
            if full in question_lower or abbr in q_words:
                filters['month'] = full
                break

    # --- Apply month filter (supports single str or list) ---
    if 'month' in filters:
        if isinstance(filters['month'], list):
            months = [normalize_month(v) for v in filters['month']]
            filtered_df = filtered_df[filtered_df['month'].isin(months)]
        else:
            month_val = normalize_month(filters['month'])
            filtered_df = filtered_df[filtered_df['month'] == month_val]

    # Helpers for melted data
    def sum_mode(mode_name):
        return int(filtered_df.loc[filtered_df['mode'] == mode_name, 'trips'].sum())

    # --- Red/Green priority, then Metro fallback ---
    wants_red = 'red line' in question_lower
    wants_green = 'green line' in question_lower
    wants_metro = 'metro' in question_lower
    wants_average = 'average' in question_lower

    if wants_red or wants_green:
        result = {}
        if wants_red:
            mask = (filtered_df['mode'] == 'Metro Red Line')
            if mask.any():
                val = float(filtered_df.loc[mask, 'trips'].mean()) if wants_average else int(filtered_df.loc[mask, 'trips'].sum())
                result['Metro Red Line'] = val
            else:
                return {"error": "Metro Red Line data not available."}
        if wants_green:
            mask = (filtered_df['mode'] == 'Metro Green Line')
            if mask.any():
                val = float(filtered_df.loc[mask, 'trips'].mean()) if wants_average else int(filtered_df.loc[mask, 'trips'].sum())
                result['Metro Green Line'] = val
            else:
                return {"error": "Metro Green Line data not available."}
        return {"result": result if len(result) > 1 else list(result.values())[0]}

    if wants_metro:
        mask_metro = (filtered_df['mode'] == 'Metro')
        if mask_metro.any():
            if wants_average:
                return {"result": float(filtered_df.loc[mask_metro, 'trips'].mean())}
            return {"result": int(filtered_df.loc[mask_metro, 'trips'].sum())}
        # Fallback to lines if aggregate Metro missing
        mask_red = (filtered_df['mode'] == 'Metro Red Line')
        mask_green = (filtered_df['mode'] == 'Metro Green Line')
        if mask_red.any() or mask_green.any():
            if wants_average:
                combined = filtered_df.loc[mask_red | mask_green].groupby('date', as_index=False)['trips'].sum()
                return {"result": float(combined['trips'].mean())}
            total = int(filtered_df.loc[mask_red, 'trips'].sum()) + int(filtered_df.loc[mask_green, 'trips'].sum())
            return {"result": total}
        return {"error": "Metro data not available."}

    metric = 'trips'

    if 'compare' in question_lower:
        known_modes = ['Metro','Metro Red Line','Metro Green Line','Tram','Bus','Marine','E-Hail','Car-Share','Bus on Demand','Taxi']
        requested = [m for m in known_modes if m.lower() in question_lower]
        if 'e hail' in question_lower and 'E-Hail' not in requested: requested.append('E-Hail')
        if 'car share' in question_lower and 'Car-Share' not in requested: requested.append('Car-Share')
        if not requested:
            requested = known_modes
        comp = {m: sum_mode(m) for m in requested}
        return {"result": comp}

    if 'busiest' in question_lower or ('most' in question_lower and 'mode' in question_lower):
        by_mode = filtered_df.groupby('mode')[metric].sum()
        if by_mode.empty:
            return {"error": "No data available for the selected filter."}
        busiest_mode = by_mode.idxmax()
        busiest_value = int(by_mode.max())
        return {"result": {busiest_mode: busiest_value}}
    
    single_modes = ['Tram','Bus','Marine','E-Hail','Car-Share','Bus on Demand','Taxi']
    for m in single_modes:
        if m.lower() in question_lower:
            return {"result": int(filtered_df.loc[filtered_df['mode'] == m, 'trips'].sum())}

    if operation == 'sum':
        return {"result": int(filtered_df[metric].sum())}
    if operation == 'average':
        return {"result": float(filtered_df[metric].mean())}
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
            month_order = [normalize_month(m) for m in filters['month']]
            return {"result": {m: int(grouped.get(m, 0)) for m in month_order}}
        return {"result": grouped.apply(int).to_dict()}

    return {"error": f"Unsupported operation: '{operation}'."}



def synthesize_answer(question, plan, calculation_result):
    prompt = f"""
You are a helpful data analyst. Convert the user's question and pre-calculated data into a natural language summary and chart data.
- Your response MUST be a valid JSON object and NOTHING else.
User's Question: {question}
Query Plan: {json.dumps(plan)}
Calculation Result: {json.dumps(calculation_result)}
Response:
"""
    generation_config = genai.types.GenerationConfig(temperature=0)
    try:
        response = llm.generate_content(prompt, generation_config=generation_config)
        return extract_json_from_string(response.text)
    except:
        return {"summary": f"Result: {calculation_result.get('result')}"}



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



@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    print(f"\n🤔 Received Question: {user_question}")

    question_lower = user_question.lower()

    if any(month in question_lower for month in ['june','july','august']):
        return jsonify({"summary": "Data unavailable for the months June, July, and August.", "graph": None})

    if ('station' in question_lower or 'stations' in question_lower) and \
       any(m in question_lower for m in ['feb','march','mar','april','apr','may']):
        return jsonify({"summary": "Station data unavailable for the months Feb, March, April, and May.", "graph": None})

    try:
        plan = create_query_plan(user_question)
        calculation_result = execute_pandas_query(plan, df, user_question)
        if "error" in calculation_result:
            return jsonify({"summary": calculation_result["error"], "graph": None})

        final_result = synthesize_answer(user_question, plan, calculation_result)

        calc_result = calculation_result.get('result')
        chart_data = None
        if isinstance(calc_result, dict):
            chart_data = {
                "title": "Analysis Result",
                "labels": list(calc_result.keys()),
                "values": list(calc_result.values())
            }
        elif isinstance(calc_result, int) or isinstance(calc_result, float):
            chart_data = {
                "title": "Analysis Result",
                "labels": ["Total"],
                "values": [calc_result]
            }

        chart_image = create_chart(chart_data)
        return jsonify({"summary": final_result.get('summary'), "graph": chart_image})

    except json.JSONDecodeError:
        return jsonify({"summary": "I had trouble understanding that. Could you rephrase?", "graph": None})
    except Exception as e:
        print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e}")
        return jsonify({"error": "An internal error occurred. See server logs."}), 500



# --- 6. STARTUP LOGIC ---
if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
